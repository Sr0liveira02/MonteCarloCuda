#ifndef MATRIX_EXP_HU
#define MATRIX_EXP_HU

#include "matrix_init.hpp"
#include "matrix_utils.hpp"
#include "AliasTable.hpp"

// TODO exp Matrix global e calculate row device as it is to be optimized as hell
float* expMatrix(arrayOfAliasTable* mat, float* vector, int n_series, int n_plays, int size, float* series);
void calculate_row(const float* __restrict__ sum, const int* __restrict__ cum, const float* __restrict__ values, const int* __restrict__ col, int n_row, int n_series, int n_plays, const float* vector, float* ret, unsigned int* seed, const float* __restrict__ series);
int weighted_random(const int* __restrict__ cum, const float* __restrict__ values, const int* __restrict__ col, int row, unsigned int* seed);

#endif // MATRIX_EXP_HU