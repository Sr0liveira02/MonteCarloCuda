#ifndef ALIASTABLE_H
#define ALIASTABLE_H

#include "matrix_init.hpp"

typedef struct {
    int N;
    int E;
    float* sum;
    // CSR
    float* values;
    int* col;
    int* cum;
} arrayOfAliasTable;

arrayOfAliasTable* generate_array_aliastable(CSRInput csr);
arrayOfAliasTable* host_array_alloc(int size, int edges);
void print_AliasTable1f(arrayOfAliasTable* matrix_information);

#endif