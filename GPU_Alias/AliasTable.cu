#include "AliasTable.hu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "cub/cub.cuh"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d [%s]: %s\n",            \
                    __FILE__, __LINE__, #call,                             \
                    cudaGetErrorString(_e));                               \
            fflush(stderr); exit(EXIT_FAILURE);                           \
        }                                                                  \
    } while (0)


#define MAX_LOCAL_DEGREE 64
#define MAX_BLOCKS       512
#define WARP_SIZE        32
#define ROWS_PER_BLOCK   8    // 8 warps per block = 256 threads

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static __host__ CSRInput csr_to_device(CSRInput host) {
    CSRInput dev;
    dev.size  = host.size;
    dev.edges = host.edges;

    CUDA_CHECK(cudaMalloc((void**)&dev.row_ptr, (host.size + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev.col_ind,  host.edges     * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&dev.values,   host.edges     * sizeof(float)));

    cudaMemcpy(dev.row_ptr, host.row_ptr, (host.size + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.col_ind, host.col_ind,  host.edges     * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.values,  host.values,   host.edges     * sizeof(float),     cudaMemcpyHostToDevice);

    return dev;
}

static __host__ void csr_free_device(CSRInput dev) {
    cudaFree(dev.row_ptr);
    cudaFree(dev.col_ind);
    cudaFree(dev.values);
}

static __host__ arrayOfAliasTable* device_array_alloc(int size, int edges) {
    arrayOfAliasTable* ret = (arrayOfAliasTable*) malloc(sizeof(arrayOfAliasTable));
    ret->N = size;
    ret->E = edges;
    CUDA_CHECK(cudaMalloc((void**)&ret->sum,   size      * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&ret->cum,  (size + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ret->prob,  edges     * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&ret->alias, edges     * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&ret->col,   edges     * sizeof(int)));
    return ret;
}

// ---------------------------------------------------------------------------
// max_large_degree_kernel — unchanged, used only to size scratch buffers
// ---------------------------------------------------------------------------

__global__ void max_large_degree_kernel(
    const int* __restrict__ row_ptr,
    int*       __restrict__ out,
    int                     num_rows)
{
    typedef cub::BlockReduce<int, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;

    int val = 0;
    for (int r = threadIdx.x; r < num_rows; r += blockDim.x) {
        int deg = row_ptr[r + 1] - row_ptr[r];
        if (deg > MAX_LOCAL_DEGREE) val = max(val, deg);
    }
    val = BlockReduce(temp).Reduce(val, cub::Max());
    if (threadIdx.x == 0) *out = val;
}

int get_max_large_N(const int* row_ptr, int num_rows) {
    int* d_out;
    cudaMalloc(&d_out, sizeof(int));
    max_large_degree_kernel<<<1, 256>>>(row_ptr, d_out, num_rows);
    int result = 0;
    cudaMemcpy(&result, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    return result;
}

__global__ void build_alias_table_small_rows(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float*     __restrict__ values,
    int                     num_rows,
    float*                        d_sum,
    const int* __restrict__ d_cum,
    float*                        d_prob,
    int*                    d_alias,
    int*                    d_col,
    int                           /* max_small_deg – kept for ABI compat */)
{
    extern __shared__ float smem[];   // allocated as ROWS_PER_BLOCK * MAX_LOCAL_DEGREE * sizeof(float)

    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int lane           = threadIdx.x % WARP_SIZE;
    const int global_warp    = blockIdx.x * ROWS_PER_BLOCK + warp_in_block;

    float* scaled = smem + warp_in_block * MAX_LOCAL_DEGREE; 

    int r = (int)global_warp;
    if (r >= num_rows) return;

    const int start = __ldg(&row_ptr[r]);
    const int       N     = (int)(__ldg(&row_ptr[r + 1]) - start);
    if (N == 0 || N > MAX_LOCAL_DEGREE) return;

    const int base = __ldg(&d_cum[r]);

    // ---- Step 1: parallel warp-reduce sum --------------------------------
    float sum = 0.0f;
    for (int i = lane; i < N; i += WARP_SIZE)
        sum += __ldg(&values[start + i]);

    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    sum = __shfl_sync(0xffffffff, sum, 0);

    if (lane == 0) d_sum[r] = sum;

    const float inv = (float)N / sum;

    // ---- Step 2: parallel normalize → smem + col copy → global ----------
    for (int i = lane; i < N; i += WARP_SIZE) {
        scaled[i]          = __ldg(&values[start + i]) * inv;
        d_col[base + i]    = __ldg(&col_ind[start + i]);
    }
    __syncwarp();

    // ---- Steps 3-4: Vose alias construction ------------------------------

#if 1
    if (N <= WARP_SIZE) {
        // Every lane owns exactly one element (lane >= N → sentinel 2.0f).
        // We cooperate via __ballot_sync to build the small/large queues,
        // then run the Vose pairing loop collectively.

        float sv = (lane < N) ? scaled[lane] : 2.0f;

        // Iterative Vose: each iteration pairs one small with one large.
        // We keep a "live" mask of unfinished indices.
        float     my_prob   = sv;
        int my_alias  = -1LL;

        // Repeat until all lanes are settled (prob rounded to 0 or 1).
        // In the worst case N iterations, but typically ~N/2 rounds.
        for (int iter = 0; iter < N && N > 1; ++iter) {
            // Find one small and one large via ballot
            unsigned small_mask = __ballot_sync(0xffffffff, (lane < N) && (my_prob < 1.0f - 1e-7f) && (my_alias < 0LL));
            unsigned large_mask = __ballot_sync(0xffffffff, (lane < N) && (my_prob >= 1.0f - 1e-7f) && (my_alias < 0LL));

            if (!small_mask || !large_mask) break;  // both queues cleared

            int small_lane = __ffs(small_mask) - 1;   // lowest set bit
            int large_lane = __ffs(large_mask) - 1;

            if (small_lane == large_lane) {
                // Same lane can't be both; skip (numerical edge case)
                break;
            }

            if (lane == small_lane) {
                // Pair this small with large_lane — base can exceed INT_MAX on
                // large graphs (e.g. Orkut), so keep as int.
                my_alias = base + (int)large_lane;
            }
            if (lane == large_lane) {
                // Absorb the small's deficit
                float small_prob = __shfl_sync(0xffffffff, my_prob, small_lane);
                my_prob = (my_prob + small_prob) - 1.0f;
            }
        }

        // Write results: every active lane writes its own entry
        if (lane < N) {
            d_prob[base + lane]  = (my_alias >= 0LL) ? my_prob : 1.0f;
            d_alias[base + lane] = (my_alias >= 0LL) ? my_alias
                                                    : (base + lane);
        }
        return;
    }
#endif

    // ---- Sequential Vose pass for 32 < N <= 64 (lane 0 only) ------------
    if (lane == 0) {
        int small_q[MAX_LOCAL_DEGREE * 2], large_q[MAX_LOCAL_DEGREE * 2];
        int ns = 0, nl = 0;

        for (int i = 0; i < N; i++)
            (scaled[i] < 1.0f) ? (small_q[ns++] = i) : (large_q[nl++] = i);

        int si = 0, li = 0;
        while (si < ns && li < nl) {
            int L = small_q[si++], H = large_q[li++];
            d_prob [base + L] = scaled[L];
            d_alias[base + L] = base + H;
            scaled[H] = (scaled[H] + scaled[L]) - 1.0f;
            if (scaled[H] < 1.0f) small_q[ns++] = H;
            else                   large_q[nl++] = H;
        }
        while (li < nl) d_prob[base + large_q[li++]] = 1.0f;
        while (si < ns) d_prob[base + small_q[si++]] = 1.0f;
    }
}

__global__ void build_alias_table_kernel_large(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_ind,
    const float*     __restrict__ values,
    const int* __restrict__ large_row_ids,
    int                     num_large_rows,
    float*                        d_sum,
    const int* __restrict__ d_cum,
    float*                        d_prob,
    int*                    d_alias,
    int*                    d_col,
    float*                        d_scaled,
    int*                    d_scratch_ll,
    const int* __restrict__ d_scratch_f_offsets,
    const int* __restrict__ d_scratch_ll_offsets)
{
    int r     = __ldg(&large_row_ids[blockIdx.x]);
    int start = __ldg(&row_ptr[r]);
    int N     = __ldg(&row_ptr[r + 1]) - start;
    int base  = __ldg(&d_cum[r]);

    float*     scaled  = d_scaled     + __ldg(&d_scratch_f_offsets[blockIdx.x]);
    int* small_q = d_scratch_ll + __ldg(&d_scratch_ll_offsets[blockIdx.x]);
    int* large_q = small_q + N;

    // ---------- Step 1: parallel sum via block reduction ----------
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        local_sum += __ldg(&values[start + i]);

    __shared__ float sdata[512];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    __shared__ float total_sum;
    if (threadIdx.x == 0) { total_sum = sdata[0]; d_sum[r] = sdata[0]; }
    __syncthreads();

    float inv = (float)N / total_sum;

    // ---------- Step 2: parallel normalize + write col ----------
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        scaled[i]       = __ldg(&values[start + i]) * inv;
        d_col[base + i] = __ldg(&col_ind[start + i]);
    }
    __syncthreads();

    // ---------- Step 3: partition via parallel prefix scan ----------
    __shared__ int ns_total, nl_total;
    __shared__ int scan_small[512], scan_large[512];

    int local_ns = 0, local_nl = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x)
        (scaled[i] < 1.0f) ? local_ns++ : local_nl++;

    scan_small[threadIdx.x] = local_ns;
    scan_large[threadIdx.x] = local_nl;
    __syncthreads();

    for (int s = 1; s < blockDim.x; s <<= 1) {
        int vs = (threadIdx.x >= s) ? scan_small[threadIdx.x - s] : 0;
        int vl = (threadIdx.x >= s) ? scan_large[threadIdx.x - s] : 0;
        __syncthreads();
        scan_small[threadIdx.x] += vs;
        scan_large[threadIdx.x] += vl;
        __syncthreads();
    }
    int off_s = scan_small[threadIdx.x] - local_ns;
    int off_l = scan_large[threadIdx.x] - local_nl;
    if (threadIdx.x == blockDim.x - 1) {
        ns_total = scan_small[threadIdx.x];
        nl_total = scan_large[threadIdx.x];
    }
    __syncthreads();

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (scaled[i] < 1.0f) small_q[off_s++] = i;
        else                   large_q[off_l++] = i;
    }
    __syncthreads();

    // ---------- Step 4: sequential Vose fill (thread 0) ----------
    if (threadIdx.x == 0) {
        int si = 0, li = 0;
        int ns = ns_total, nl = nl_total;
        while (si < ns && li < nl) {
            int L = small_q[si++], H = large_q[li++];
            d_prob [base + L] = scaled[L];
            d_alias[base + L] = base + H;
            scaled[H] = (scaled[H] + scaled[L]) - 1.0f;
            if (scaled[H] < 1.0f) small_q[ns++] = H;
            else                   large_q[nl++] = H;
        }
        while (li < nl) d_prob[base + large_q[li++]] = 1.0f;
        while (si < ns) d_prob[base + small_q[si++]] = 1.0f;
    }
}

__host__ arrayOfAliasTable* generate_array_aliastable_gpu(CSRInput host_csr) {
    CSRInput dev_csr = csr_to_device(host_csr);
    arrayOfAliasTable* ret = device_array_alloc(host_csr.size, host_csr.edges);

    cudaMemcpy(ret->cum, host_csr.row_ptr,
               (host_csr.size + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    int num_small_rows = 0;
    int       max_small_deg  = 0;
    std::vector<int> large_ids; 

    for (int i = 0; i < host_csr.size; i++) {
        int deg = host_csr.row_ptr[i+1] - host_csr.row_ptr[i];
        if (deg <= MAX_LOCAL_DEGREE) {
            num_small_rows++;
            if ((int)deg > max_small_deg) max_small_deg = (int)deg;
        } else {
            large_ids.push_back(i);
        }
    }

    cudaStream_t stream_small, stream_large;
    cudaStreamCreate(&stream_small);
    cudaStreamCreate(&stream_large);

    // --- Small-row kernel (one warp per row) ------------------------------
    if (num_small_rows > 0) {
        const int threads = ROWS_PER_BLOCK * WARP_SIZE;   // 256
        const int blocks = (host_csr.size + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;

        const size_t smem_bytes = ROWS_PER_BLOCK * MAX_LOCAL_DEGREE * sizeof(float);

        build_alias_table_small_rows<<<blocks, threads, smem_bytes, stream_small>>>(
            dev_csr.row_ptr, dev_csr.col_ind, dev_csr.values,
            host_csr.size,
            ret->sum, ret->cum, ret->prob, ret->alias, ret->col,
            max_small_deg
        );
    }

    // --- Large-row kernel -------------------------------------------------
    // get_max_large_N runs on the default stream (or stream_large); it is
    // cheap (one block) and its result is needed before the large kernel.
    int max_large_N = get_max_large_N(dev_csr.row_ptr, dev_csr.size);

    float*     d_scratch_f  = nullptr;
    int* d_scratch_ll = nullptr;

    if (!large_ids.empty()) {
        int num_large = (int)large_ids.size();

        // Build per-block scratch offsets from actual degrees (no wasted memory).
        // f_offsets[i]  = prefix sum of deg[i]      → float scratch offset per block
        // ll_offsets[i] = prefix sum of 2*deg[i]    → ll scratch offset per block
        //                 (factor of 2: one slot per element for small_q + large_q)
        std::vector<int> h_f_off(num_large + 1, 0);
        std::vector<int> h_ll_off(num_large + 1, 0);
        for (int i = 0; i < num_large; i++) {
            int deg = host_csr.row_ptr[large_ids[i] + 1] - host_csr.row_ptr[large_ids[i]];
            h_f_off [i + 1] = h_f_off [i] + deg;
            h_ll_off[i + 1] = h_ll_off[i] + 2 * deg;
        }
        int total_f  = h_f_off [num_large];
        int total_ll = h_ll_off[num_large];

        int* d_large_ids;
        CUDA_CHECK(cudaMalloc(&d_large_ids, num_large * sizeof(int)));
        cudaMemcpy(d_large_ids, large_ids.data(), num_large * sizeof(int), cudaMemcpyHostToDevice);

        int *d_f_off, *d_ll_off;
        CUDA_CHECK(cudaMalloc(&d_f_off,  (num_large + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_ll_off, (num_large + 1) * sizeof(int)));
        cudaMemcpy(d_f_off,  h_f_off.data(),  (num_large + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ll_off, h_ll_off.data(), (num_large + 1) * sizeof(int), cudaMemcpyHostToDevice);

        CUDA_CHECK(cudaMalloc(&d_scratch_f,  total_f  * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_scratch_ll, total_ll * sizeof(int)));

        build_alias_table_kernel_large<<<num_large, 256, 0, stream_large>>>(
            dev_csr.row_ptr, dev_csr.col_ind, dev_csr.values,
            d_large_ids, num_large,
            ret->sum, ret->cum, ret->prob, ret->alias, ret->col,
            d_scratch_f, d_scratch_ll,
            d_f_off, d_ll_off
        );

        CUDA_CHECK(cudaStreamSynchronize(stream_large));
        cudaFree(d_large_ids);
        cudaFree(d_scratch_f);
        cudaFree(d_scratch_ll);
        cudaFree(d_f_off);
        cudaFree(d_ll_off);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_small));

    cudaStreamDestroy(stream_small);
    cudaStreamDestroy(stream_large);

    csr_free_device(dev_csr);

    arrayOfAliasTable* d_ret = push_struct_to_device(ret);
    free(ret);
    return d_ret;
}

// ---------------------------------------------------------------------------
// Unchanged utility functions
// ---------------------------------------------------------------------------

__host__ arrayOfAliasTable* push_struct_to_device(arrayOfAliasTable* host_shell) {
    arrayOfAliasTable* d_ptr = nullptr;
    cudaMalloc((void**)&d_ptr, sizeof(arrayOfAliasTable));
    cudaMemcpy(d_ptr, host_shell, sizeof(arrayOfAliasTable), cudaMemcpyHostToDevice);
    return d_ptr;
}

__host__ void free_array_aliastable_gpu(arrayOfAliasTable* d_ptr) {
    arrayOfAliasTable host_shell;
    cudaMemcpy(&host_shell, d_ptr, sizeof(arrayOfAliasTable), cudaMemcpyDeviceToHost);
    cudaFree(host_shell.sum);
    cudaFree(host_shell.cum);
    cudaFree(host_shell.prob);
    cudaFree(host_shell.alias);
    cudaFree(host_shell.col);
    cudaFree(d_ptr);
}

__host__ void print_AliasTable1f(arrayOfAliasTable* mat) {
    arrayOfAliasTable* test = (arrayOfAliasTable*) malloc(sizeof(arrayOfAliasTable));
    cudaMemcpy(test, mat, sizeof(arrayOfAliasTable), cudaMemcpyDeviceToHost);

    printf("Alias Table — %d nodes, %d edges:\n", test->N, test->E);

    float*     h_sum   = (float*)     malloc(test->N          * sizeof(float));
    int* h_cum   = (int*) malloc((test->N + 1)    * sizeof(int));
    float*     h_prob  = (float*)     malloc(test->E          * sizeof(float));
    int* h_alias = (int*) malloc(test->E          * sizeof(int));
    int* h_col   = (int*) malloc(test->E          * sizeof(int));

    cudaMemcpy(h_sum,   test->sum,   test->N       * sizeof(float),     cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum,   test->cum,  (test->N + 1)  * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_prob,  test->prob,  test->E       * sizeof(float),     cudaMemcpyDeviceToHost);
    cudaMemcpy(h_alias, test->alias, test->E       * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_col,   test->col,   test->E       * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < test->N; i++)  printf("%f ",   h_sum[i]);   printf("\n");
    for (int i = 0; i <= test->N; i++) printf("%d ",  h_cum[i]);   printf("\n");
    for (int i = 0; i < test->E; i++)  printf("%f ",   h_prob[i]);  printf("\n");
    for (int i = 0; i < test->E; i++)  printf("%d ", h_alias[i]); printf("\n");
    for (int i = 0; i < test->E; i++)  printf("%d ", h_col[i]);   printf("\n");

    free(h_sum); free(h_cum); free(h_prob); free(h_alias); free(h_col);
    free(test);
}
