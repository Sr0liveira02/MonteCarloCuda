#include <cstdlib>
#include "AliasTable.hpp"
#include <cstdio>

arrayOfAliasTable* generate_array_aliastable(CSRInput csr) { // FUCK THIS CODE
    arrayOfAliasTable* ret = host_array_alloc(csr.size, csr.edges);

    // Copy row_ptr to cum and calculate Ns for each row
    for (int r = 0; r < csr.edges; r++) {
        ret->col[r] = csr.col_ind[r];
        ret->values[r] = csr.values[r];
        // printf("%f ", ret->values[r]);
    }
    for (int r = 0; r < csr.size; r++) {
        ret->cum[r] = csr.row_ptr[r];
        int row_start = csr.row_ptr[r];
        int row_end = csr.row_ptr[r + 1];
        float sum = 0;
        for (int j = row_start; j < row_end; j++) sum += ret->values[j];
        ret->sum[r] = sum;
    }

    // get max degree
    int deg = 0;
    for (int r = 0; r < csr.size; r++) {
        if (deg < csr.row_ptr[r + 1] - csr.row_ptr[r]) deg = csr.row_ptr[r + 1] - csr.row_ptr[r];
    }

    for (int r = 0; r < csr.edges; r++) {
        ret->values[r] /= deg;
        // printf("%f ", ret->values[r]);
    }

    // Step 1: Normalize and cum
    for (int r = 0; r < csr.size; r++) {
        int row_start = csr.row_ptr[r];
        int row_end = csr.row_ptr[r + 1];
        float sum = 0;
        for (int j = row_start; j < row_end; j++) sum += ret->values[j];
        for (int j = row_start; j < row_end; j++) ret->values[j] /= sum;
        float cum = 0;
        for (int j = row_start; j < row_end; j++) {
            cum += ret->values[j];
            ret->values[j] = cum;
        }
    }

    return ret;
}

arrayOfAliasTable* host_array_alloc(int size, int edges) {
    arrayOfAliasTable* ret = (arrayOfAliasTable*) malloc(sizeof(arrayOfAliasTable));

    ret->N = size;
    ret->E = edges;
    ret->sum = (float*) malloc(size * sizeof(float));
    ret->cum = (int*)   malloc((size + 1) * sizeof(int));
    ret->values = (float*)   malloc(edges * sizeof(float));
    ret->col = (int*)     malloc(edges * sizeof(int));
    return ret;
}

void print_AliasTable1f(arrayOfAliasTable* mat) {
    printf("Alias Table de %d Nodes e %d Edges:\n", mat->N, mat->E);
    for (int i = 0; i < mat->N; i++)
        printf("%f ", mat->sum[i]);
    printf("\n");
    for (int i = 0; i < mat->N; i++)
        printf("%d ", mat->cum[i]);
    printf("\n");
    for (int i = 0; i < mat->E; i++)
        printf("%f ", mat->values[i]);
    printf("\n");
    for (int i = 0; i < mat->E; i++)
        printf("%d ", mat->col[i]);
    printf("\n");
}