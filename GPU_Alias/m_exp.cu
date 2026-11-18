#include <stdio.h>
#include <stdlib.h>
#include "m_exp.hu"
#include <omp.h>
#include "cub/cub.cuh"

#define N_THREADS 256 // TODO: adjust these parameters
#define MAX_BLOCKS 512

#include <cuda_runtime.h>
#include <iostream>

void checkCudaError()
{
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        std::cout << "CUDA error: "
                  << cudaGetErrorString(err)
                  << std::endl;
    }
}

// Add initialization kernel
__global__ void init_curand_states(curandState* states, unsigned long long seed, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

__host__ float* expMatrix(arrayOfAliasTable* mat, float* vector, int n_series, int n_plays, int size, int* series) {
    printf("Exponential a matrix of size %d, with %d series and %d plays\n", size, n_series, n_plays);

    int num_blocks = MAX_BLOCKS;
    int threads_per_block = N_THREADS;
    int total_threads = num_blocks * threads_per_block;
    
    // printf("Allocating %d blocks of %d threads \n", num_blocks, threads_per_block);

    // Allocate and initialize random states
    curandState* rand_states;
    cudaMalloc(&rand_states, total_threads * sizeof(curandState)); // this might be to much memory but idk
    // unsigned int seed = 3;
    unsigned long long seed = (unsigned long long)time(NULL) ^ ((unsigned long long)rand() << 32);
    init_curand_states<<<num_blocks, threads_per_block>>>(rand_states, seed, total_threads);
    cudaDeviceSynchronize();
    checkCudaError();

    float* d_ret;
    cudaMalloc(&d_ret, size * sizeof(float));

    calculate_all_rows<<<num_blocks, threads_per_block>>>(reinterpret_cast<const arrayOfAliasTableConst*>(mat), n_series, n_plays, d_ret, vector, rand_states, series);
    checkCudaError();

    // Copy `ret` from device to host so we can safely access it on the CPU.
    float* ret = (float*) malloc(size * sizeof(float));
    cudaMemcpy(ret, d_ret, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(rand_states);
    cudaFree(d_ret);

    return ret;
}

__global__ void calculate_all_rows(const arrayOfAliasTableConst* __restrict__ mat, int n_series, int n_plays, float* __restrict__ ret, const float* __restrict__ vector, curandState* rand_states, int* series) { // restrict no rand states might help e no vector
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    // sera q eu posso medir shared memory do block reduce e ocupar o resto com o vector?

    typedef cub::BlockReduce<float, N_THREADS> BlockReduceT; 
    __shared__ typename BlockReduceT::TempStorage temp_storage;

    // Get this thread's random state
    curandState localState = rand_states[global_tid]; // still checkar esta porra


    for (int row = blockIdx.x; row < mat->N; row += gridDim.x) {
        float cum = 0;

        if (mat->cum[row + 1] - mat->cum[row] == 0) {
            ret[row] = vector[row];
            continue;
        }

        // Step 1: Each thread accumulates its portion
        for (int i = tid; i < n_plays; i += blockDim.x) {
            float value = 1.0;
            int currentRow = row;

            for (int k = 1; k < n_series; k++) {
                value = value * __ldg(&mat->sum[currentRow]);
                currentRow = weighted_random(mat, currentRow, &localState);
                cum += value * vector[currentRow] / n_plays / series[k];
            }
        }
        
        // CUB Block Reduction
        float block_sum = BlockReduceT(temp_storage).Sum(cum);

        if (tid == 0)
            ret[row] = block_sum + vector[row]; // add first series k=0 A^0 = I (1 * vector[row])
    }

    // Save the state back
    rand_states[global_tid] = localState;
}

__device__ int weighted_random(const arrayOfAliasTableConst* __restrict__ mat, int row, curandState* state) {
    // Generate random integer in range [0, mat->Ns[row])
    int Ns = __ldg(&mat->cum[row + 1]) - __ldg(&mat->cum[row]);
    int k = (curand(state) % Ns) + __ldg(&mat->cum[row]);
    // Generate random float in [0, 1) // here it should be f
    float u = curand_uniform(state); // here it should be f
    
    int chosen = (u < __ldg(&mat->prob[k])) ? k : __ldg(&mat->alias[k]); // would they do this branchless
    return __ldg(&mat->col[chosen]);
}

__host__ int factorial(int N) {
    int fact = 1;
    for (int i = 1; i <= N; i++) {
        fact *= i;
    }
    return fact;
}