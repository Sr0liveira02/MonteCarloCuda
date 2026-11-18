#ifndef MATRIX_INIT_H
#define MATRIX_INIT_H

#include <vector>
extern "C" {
#include <hdf5.h>
}

struct CSRInput {
    int size;
    int edges;
    int* row_ptr;
    int* col_ind;
    float* values;
};

CSRInput load_csr_hdf5(const char* path);
void write_vector_to_file(float* matrix, int size, char* filename);

/*__host__ float* file_alloc_matrix(int size);
__host__ void alloc_matrix(float** matrix, int size);
__host__ void calloc_matrix(float** matrix, int size);
__host__ void delete_matrix(float* matrix);
__host__ void parse_matrix(char* path, float** matrix, int size);
__host__ void write_matrix_to_file(float** matrix, int size, char* filename);
*/

#endif // MATRIX_INIT_H