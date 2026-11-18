#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include "matrix_init.hpp"

void print_matrixf(CSRInput mat);
void print_matrix(float* mat, int size);
// __device__ void matrix_sum_row(float* mat, float* row, int size, int n_row);
void calculate_row_sum(CSRInput& mat, float* precalcualted_rows);
// __host__ void normalize_matrix(float* mat, float* precalculated_rows, int size);
void matrix_divide(CSRInput& mat);

#endif // MATRIX_UTILS_H