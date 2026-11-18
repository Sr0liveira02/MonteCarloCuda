#include "matrix_utils.hu"
#include <stdlib.h>
#include <stdio.h>

__host__ void print_matrixf(CSRInput mat) {
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

__host__ void print_matrix(float* mat, int size) {
    for (int r = 0; r < size; r++) {
        printf("   ");
        for (int c = 0; c < size; c++)
            printf("%2.0f ", mat[r * size + c]);
        printf("\n");    
    }
}

__host__ void calculate_row_sum(CSRInput& mat, float* precalculated_rows) { // might not be needed
    for (int r = 0; r < mat.size; r++) precalculated_rows[r] = 0;
    for (int e = 0, r = 0; e < mat.edges; e++) {
        while (e >= mat.row_ptr[r + 1]) r++;  // Handle multiple empty rows
        precalculated_rows[r] += mat.values[e];
    }
}

float max_value(float* myArray, int size) {
    float maxValue = myArray[0];

    for (int i = 1; i < size; i++) {
        if ( myArray[i] > maxValue ) {
            maxValue = myArray[i];
        }
    }
    return maxValue;
}

__host__ void matrix_divide(CSRInput& mat) {
    float* precalculated_rows = (float*) malloc(mat.size * sizeof(float)); 
    calculate_row_sum(mat, precalculated_rows);
    float dividend = max_value(precalculated_rows, mat.size);
    // printf("O valor por dividir é: %f\n", dividend);
    for (int e = 0; e < mat.edges; e++)
        mat.values[e] = mat.values[e] / dividend;
    free(precalculated_rows);
}
