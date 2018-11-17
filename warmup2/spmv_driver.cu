/*
 * spmv_driver.cu
 * Copyright (C) 2018 
 * 	P Sadayappan (saday) <psaday@gmail.com>
 * 	Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>

void check_vec(double* a, double *b, unsigned int n, bool quit_on_err = true ) {
    for (unsigned int i = 0; i < n; ++i) {
        if(std::abs(a[i] - b[i]) > 1e-1) {
            std::cerr << "Possible error at " << i << std::endl;
            if(quit_on_err)
                exit(-1);
        }
    }
}

void init_vec(double *a, unsigned int n, double offset) {
    for (unsigned int i = 0; i < n; ++i) {
        a[i]  = i + offset;
    }
}

void host_spmv(CSR mat, double* vec_in, double *vec_out  ) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        unsigned int row_start = mat.row_indx[r];
        unsigned int row_end = mat.row_indx[r + 1];
        vec_out[r] = 0;

        for (unsigned int j = row_start; j < row_end; ++j) {

            unsigned int col_id = mat.col_id[j];
            double val = mat.values[j];
            vec_out[r] += val * vec_in[col_id];
        }

    }
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        std::cerr << "Missing filename" << std::endl;
        exit(-1);
    }

    CSR mat = read_matrix_market_to_CSR(argv[1]);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << std::endl;

    double *vec_in = (double*)malloc(mat.ncols * sizeof(double));
    double *vec_out = (double*)malloc(mat.nrows * sizeof(double));

    init_vec(vec_in, mat.ncols, 1.0);

    host_spmv(mat, vec_in, vec_out);


    std::cout << "replace one argument to the below function with the values from gpu " << std::endl;
    check_vec(vec_out, vec_out, mat.nrows);


    free(mat.row_indx);
    free(mat.col_id);
    free(mat.values);
    return 0;
}
