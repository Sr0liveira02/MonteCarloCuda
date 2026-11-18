#include "m_exp.hu"
#include "AliasTable.hu"
#include "matrix_init.hu"
#include "matrix_utils.hu"
// #include "watts_strogatz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "sys/time.h"

static void derive_output_filename(const char *input, char *output, size_t out_size) { // change this shit to something more reasonable wtf ai
    if (!input || !output || out_size == 0) return;
    const char *suffix = "_vec.txt";
    size_t suf_len = strlen(suffix);
    if (out_size <= suf_len) {
        output[0] = '\0';
        return;
    }
    size_t max_base = out_size - suf_len - 1; /* room for null */
    const char *dot = strrchr(input, '.');
    size_t input_base_len = dot ? (size_t)(dot - input) : strlen(input);
    size_t copy_len = input_base_len < max_base ? input_base_len : max_base;
    memcpy(output, input, copy_len);
    output[copy_len] = '\0';
    /* append suffix safely */
    strncat(output, suffix, out_size - strlen(output) - 1);
}

int main(int argc, char *argv[]) {
    // Input = ./m_exp <series_length> <number_plays> [matrix_file]
    if (argc == 2 && !strcmp(argv[1], "help")) {
        printf("Use: m_exp <number of series> <number of plays> [matrix_file]\n");
        exit(1);
    }
    if (argc != 3 && argc != 4) {
        printf("Invalid input, please use:\n\tm_exp <number of series> <number of plays> [matrix_file]\n");
        exit(1);
    }
    int n_series = atoi(argv[1]);
    int n_plays = atoi(argv[2]);
    char filename[256] = "watts_strogatz.h5";
    char output[256];

    if (argc == 4) {
        strncpy(filename, argv[3], sizeof(filename) - 1);
        filename[sizeof(filename) - 1] = '\0';
    }
    /* derive output filename */
    derive_output_filename(filename, output, sizeof(output));

    CSRInput mat = load_csr_hdf5(filename);
    matrix_divide(mat);
    int size = mat.size;
    // printf("Main matrix of size %d and %d edges:\n", size, edges);
    // print_matrixf(mat);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record start event
	cudaEventRecord(start);
    arrayOfAliasTable* device_mat = generate_array_aliastable_gpu(mat);
    cudaDeviceSynchronize();

    checkCudaError();

	// Record end event
	cudaEventRecord(stop);

	// Wait for the end event to complete
	cudaEventSynchronize(stop);

	// Calculate elapsed time in milliseconds
	float acmilliseconds = 0; // here it should be f
	cudaEventElapsedTime(&acmilliseconds, start, stop);
    cudaDeviceSynchronize();
    checkCudaError();
    // print_AliasTable1f(device_mat);

    // Code that creates Vector with full 1
    float* h_vector = (float*) malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) h_vector[i] = 1;
    float* vector;
    cudaMalloc(&vector, size * sizeof(float));
    cudaMemcpy(vector, h_vector, size * sizeof(float), cudaMemcpyHostToDevice);
    free(h_vector);

    int* h_series = (int*) malloc(n_series * sizeof(int));
    for (int i = 0; i < n_series; i++) h_series[i] = factorial(i);
    int* d_series;
    cudaMalloc(&d_series, n_series * sizeof(int));
    cudaMemcpy(d_series, h_series, n_series * sizeof(int), cudaMemcpyHostToDevice);
    free(h_series);
    
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record start event
	cudaEventRecord(start);
    float* result = expMatrix(device_mat, vector, n_series, n_plays, size, d_series);
    // wait
	cudaDeviceSynchronize();

	// Record end event
	cudaEventRecord(stop);

	// Wait for the end event to complete
	cudaEventSynchronize(stop);

	// Calculate elapsed time in milliseconds
    float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    printf("Alias MATH %f s\n", (acmilliseconds) / 1000);
	printf("GPU MATH %f s\n", (milliseconds) / 1000);
	fflush(stdout);

    // Change this to the new output
    // printf("Result matrix of size %d:\n", size);
    // print_matrixf(result, size);
    // write_matrix_to_file(result, size, output);
    // printf("Demorou: %.1fs\n", exec_time);
    // write_vector_to_file(result, size, output);

    free_array_aliastable_gpu(device_mat);
    cudaFree(vector);
    cudaFree(d_series);
    free_csr(mat);
    free(result);
}
