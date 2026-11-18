#include "matrix_utils.hpp"
#include <stdlib.h>
#include <stdio.h>

void print_matrixf(CSRInput mat) {
    printf("Alias Table de %d Nodes e %d Edges:\n", mat.size, mat.edges);
    for (int i = 0; i < mat.edges; i++)
        printf("%f ", mat.values[i]);
    printf("\n");
    for (int i = 0; i < mat.edges; i++)
        printf("%d ", mat.col_ind[i]);
    printf("\n");
    for (int i = 0; i < mat.size; i++)
        printf("%d ", mat.row_ptr[i]);
    printf("\n");
}

void print_matrix(float* mat, int size) {
    for (int r = 0; r < size; r++) {
        printf("   ");
        for (int c = 0; c < size; c++)
            printf("%2.0f ", mat[r * size + c]);
        printf("\n");    
    }
}

/*__device__ void matrix_sum_row(float* mat, float* row, int size, int n_row) {
    for (int c = 0; c < size; c++)
            mat[n_row * size + c] = mat[n_row * size + c] + row[c];
}*/

void calculate_row_sum(CSRInput& mat, float* precalculated_rows) { // might not be needed
    for (int r = 0; r < mat.size; r++) precalculated_rows[r] = 0;
    for (int e = 0, r = 0; e < mat.edges; e++) {
        while (e >= mat.row_ptr[r + 1]) r++;  // Handle multiple empty rows
        precalculated_rows[r] += mat.values[e];
    }
}

/*__host__ void normalize_matrix(float* mat_aux, float* precalculated_sum_rows_abs, int size) {
    for (int r = 0; r < size; r++) {
        for (int c = 0; c < size; c++) {
            mat_aux[r * size + c] /= precalculated_sum_rows_abs[r];
            if (mat_aux[r * size + c] < 0)
                mat_aux[r * size + c] *= -1;
        }
    }
}*/

float max_value(float* myArray, int size) {
    float maxValue = myArray[0];

    for (int i = 1; i < size; i++) {
        if ( myArray[i] > maxValue ) {
            maxValue = myArray[i];
        }
    }
    return maxValue;
}

void matrix_divide(CSRInput& mat) {
    float* precalculated_rows = (float*) malloc(mat.size * sizeof(float)); 
    calculate_row_sum(mat, precalculated_rows);
    float dividend = max_value(precalculated_rows, mat.size);
    // printf("O valor por dividir é: %f\n", dividend);
    for (int e = 0; e < mat.edges; e++)
        mat.values[e] = mat.values[e] / dividend;
    free(precalculated_rows);
}
