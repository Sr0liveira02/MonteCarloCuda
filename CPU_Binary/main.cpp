#include "m_exp.hpp"
#include "AliasTable.hpp"
#include "matrix_init.hpp"
#include "matrix_utils.hpp"
// #include "watts_strogatz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int factorial(int N) {
    int fact = 1;
    for (int i = 1; i <= N; i++) {
        fact *= i;
    }
    return fact;
}

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

    float exec_time = -omp_get_wtime();
    arrayOfAliasTable* matrix_information = generate_array_aliastable(mat);
    exec_time += omp_get_wtime();
    // print_AliasTable1f(matrix_information);

    // Code that creates Vector with full 1
    float* vector = (float*) malloc(mat.size * sizeof(float));
    for (int i = 0; i < mat.size; i++) vector[i] = 1;
    
    float* series = (float*) malloc(n_series * sizeof(float));
    for (int i = 0; i < n_series; i++) series[i] = 1.0f / factorial(i);

    float exec_time2 = -omp_get_wtime();
    float* result = expMatrix(matrix_information, vector, n_series, n_plays, size, series);
    exec_time2 += omp_get_wtime();

	printf("GPU MATH %f s\n", (exec_time + exec_time2));
	fflush(stdout);

    // Change this to the new output
    // printf("Result matrix of size %d:\n", size);
    // print_matrixf(result, size);
    // write_matrix_to_file(result, size, output);
    // printf("Demorou: %.1fs\n", exec_time);
    // write_vector_to_file(result, size, output);

    free(matrix_information);
    free(result);
    free(vector);
}
