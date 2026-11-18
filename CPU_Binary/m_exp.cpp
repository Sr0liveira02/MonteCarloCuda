#include <stdio.h>
#include <stdlib.h>
#include "m_exp.hpp"
#include <omp.h>

float* expMatrix(arrayOfAliasTable* mat, float* vector, int n_series, int n_plays, int size, float* series) {
    printf("Exponential a matrix of size %d, with %d series and %d plays\n", size, n_series, n_plays);

    float* ret = (float*) calloc(mat->N, sizeof(float));

    #pragma omp parallel
    {
        unsigned int seed = 3 + (unsigned int) omp_get_thread_num();
        #pragma omp for
        for (int r = 0; r < mat->N; r++) { // For each row
            calculate_row(mat->sum, mat->cum, mat->values, mat->col, r, n_series, n_plays, vector, ret, &seed, series);
        }
    }
    
    // Transformar isto na multiplicaçao e somar a ret
    for (int r = 0; r < mat->N; r++) {
        ret[r] += 1.0 * vector[r]; // add first series k=0 A^0 = I (0! = 1)
    }
    
    return ret;
}

void calculate_row(const float* __restrict__ sum, const int* __restrict__ cum, const float* __restrict__ values, const int* __restrict__ col,
    int n_row, int n_series, int n_plays, const float* vector, float* ret, unsigned int* seed, const float* __restrict__ series) {
    
    if (cum[n_row + 1] - cum[n_row] == 0) {
        ret[n_row] = 0;
        return;
    }
    
    float inv = 1.0f / n_plays;

    for (int m = 1; m < n_plays; m++) { // each play
        float value = 1;
        int currentRow = n_row;

        for (int k = 1; k < n_series; k++) { // For each part of the series of length(n_series)
            value = value * sum[currentRow];
            currentRow = weighted_random(cum, values, col, currentRow, seed);
            ret[n_row] += value * vector[currentRow] * inv * series[k];
        }
    }
}

int weighted_random(const int* __restrict__ cum, const float* __restrict__ values, const int* __restrict__ col,
    int row, unsigned int* seed) {
    
    float u = rand_r(seed) / (float)RAND_MAX;
    int start = cum[row];
    int end = cum[row + 1];
    
    // Binary search for the smallest index where mat->values[i] >= u
    int low = start;
    int high = end - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (values[mid] < u) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    
    return col[low];
}
