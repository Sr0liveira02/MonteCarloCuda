#include <stdio.h>
#include <stdlib.h>
#include "matrix_init.hpp"

CSRInput load_csr_hdf5(const char* path)
{
    CSRInput A;

    hid_t file  = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t group = H5Gopen(file, "/CSR", H5P_DEFAULT);

    // --- attributes ---
    H5Aread(H5Aopen(group, "n_rows", H5P_DEFAULT),
            H5T_NATIVE_INT, &A.size);

    int n_cols;
    H5Aread(H5Aopen(group, "n_cols", H5P_DEFAULT),
            H5T_NATIVE_INT, &n_cols);

    // --- row_ptr ---
    hid_t dset  = H5Dopen(group, "row_ptr", H5P_DEFAULT);
    hid_t space = H5Dget_space(dset);

    hsize_t dims[1];
    H5Sget_simple_extent_dims(space, dims, nullptr);
    A.row_ptr = (int*) malloc(dims[0] * sizeof(int));

    H5Dread(dset, H5T_NATIVE_INT,
            H5S_ALL, H5S_ALL, H5P_DEFAULT,
            A.row_ptr);

    H5Sclose(space);
    H5Dclose(dset);

    A.edges = A.row_ptr[dims[0] - 1];   // replaces .back()

    // --- col_ind ---
    A.col_ind = (int*)malloc(A.edges * sizeof(int));
    dset = H5Dopen(group, "col_ind", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_INT,
            H5S_ALL, H5S_ALL, H5P_DEFAULT,
            A.col_ind);
    H5Dclose(dset);

    // --- values ---
    A.values = (float*)malloc(A.edges * sizeof(float));
    dset = H5Dopen(group, "values", H5P_DEFAULT);
    H5Dread(dset, H5T_NATIVE_FLOAT,
            H5S_ALL, H5S_ALL, H5P_DEFAULT,
            A.values);
    H5Dclose(dset);

    H5Gclose(group);
    H5Fclose(file);

    return A;
}


void write_vector_to_file(float* vector, int size, char* filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    fprintf(file, "%d\n", size);

    for (int r = 0; r < size; r++) {
        fprintf(file, "%.10f ", vector[r]);
    }
    fprintf(file, "\n");
    
    fclose(file);
}

/*
float* file_alloc_matrix(int size) {
    float* mat = (float *) malloc(size * size * sizeof(float)); 
    return mat;
}

__host__ void alloc_matrix(float** matrix, int size) {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("CUDA device count: %d\n", count);
    if (count == 0) {
		fprintf(stderr, "No CUDA devices found!\n");
		exit(1);
	}

    cudaMallocManaged(matrix, size * size * sizeof(float));

    int device = -1;
	cudaError_t err = cudaGetDevice(&device);
	printf("cudaGetDevice returned %d, device = %d\n", err, device);
	
	cudaMemPrefetchAsync(*matrix, size * size * sizeof(int), device, 0);
}

__host__ void calloc_matrix(float** matrix, int size) {
    int count = 0;
    cudaGetDeviceCount(&count);
    printf("CUDA device count: %d\n", count);
    if (count == 0) {
		fprintf(stderr, "No CUDA devices found!\n");
		exit(1);
	}

    cudaMallocManaged(matrix, size * size * sizeof(float));

    int device = -1;
	cudaError_t err = cudaGetDevice(&device);
	printf("cudaGetDevice returned %d, device = %d\n", err, device);
	
	cudaMemPrefetchAsync(*matrix, size * size * sizeof(int), device, 0);

    cudaMemset(*matrix, 0, size * size * sizeof(int));
}

__host__ void delete_matrix(float* matrix) {
	cudaFree(matrix);
}

__host__ void parse_matrix(char* path, float** matrix, int* size) { // This is NOT USED rn so i dont need to predict the number of edges maybe also write them
    FILE* fpm = fopen(path, "r");
    if(fpm == NULL) {
        fprintf(stderr, "No such file exists: %s.\n", path);
        exit(-2);
    }

    // Read Size
    int a = fscanf(fpm, "%d", size);
    if (a == 0)
        printf("????\n");

    float* mat = file_alloc_matrix(*size);
    for(int r = 0; r < *size; r++)
        for(int c = 0; c < *size; c++)
	        a = fscanf(fpm, "%lf", &mat[r * *size + c]);

    cudaMemcpy(matrix, mat, *size * *size * sizeof(float), cudaMemcpyHostToDevice);
    free(mat);

    float* mat;
    alloc_matrix(&mat, *size);
    for(int r = 0; r < *size; r++)
        for(int c = 0; c < *size; c++)
	        a = fscanf(fpm, "%lf", &mat[r * *size + c]);
}

__host__ void write_matrix_to_file(float** matrix, int size, char* filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error: Could not open file %s for writing\n", filename);
        return;
    }

    // float* mat = file_alloc_matrix(size);
    // cudaMemcpy(mat, matrix, size * size * sizeof(float), cudaMemcpyDeviceToHost);

    fprintf(file, "%d\n", size);

    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            fprintf(file, "%.10f", *matrix[r * size + c]);
            if (c < size - 1) fprintf(file, " ");
        }
        fprintf(file, "\n");
    }
    
    // free(mat);
    fclose(file);
}

*/
