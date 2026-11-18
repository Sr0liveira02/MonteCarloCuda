#include <stdio.h>
#include <stdlib.h>
#include "matrix_init.hu"

__host__ CSRInput load_csr_hdf5(const char* path)
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

__host__ void free_csr(CSRInput& A) {
    free(A.row_ptr);
    free(A.col_ind);
    free(A.values);
    A.row_ptr = nullptr;
    A.col_ind = nullptr;
    A.values  = nullptr;
}

__host__ void write_vector_to_file(float* vector, int size, char* filename) {
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