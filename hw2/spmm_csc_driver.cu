/*
 * spmm_csc_driver.cu
 * Copyright (C) 2018
 *  P Sadayappan (saday) <psaday@gmail.com>
 *  Aravind SUKUMARAN RAJAM (asr) <aravind_sr@outlook.com>
 *
 * Distributed under terms of the GNU LGPL3 license.
 */

#include "mm_helper.hpp"
#include "sparse_representation.hpp"
#include <iostream>

void check_dmat(double* a, double *b, unsigned int n, unsigned int K, bool quit_on_err = true ) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            if(std::abs(a[i * K + k] - b[i * K + k]) > 1e-1) {
                std::cerr << "Possible error at " << i << std::endl;

                if(quit_on_err) {
                    exit(-1);
                }
            }
        }
    }
    if(quit_on_err)
        std::cout << "Verification succeeded\n";
    else
        std::cout << "Check error messages to see if verification succeeded. (No error msg == success)\n";
}

static unsigned int g_seed = 0X4B1D;
inline int fastrand() {
    g_seed = (214013 * g_seed + 2531011);
    return (g_seed >> 16) & 0x7FFF;
}

void init_dmat(double *a, unsigned int n, unsigned int K, double offset) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int k = 0; k < K; ++k) {
            a[i * K + k]  = i * K + k + offset;
            //a[i * K + j]  = fastrand() + offset;
        }
    }
}

void print_dmat(double *a, unsigned int n, unsigned int K) {
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < K; ++j) {
            std::cout << a[i * K + j]   << ' ';
        }
        std::cout << '\n';
    }
}

void host_csc_spmm(CSC mat, double * dmat_in, double * dmat_out, unsigned int K) {
    for (unsigned int r = 0; r < mat.nrows; ++r) {
        for (unsigned int k = 0; k < K; ++k) {
            dmat_out[r * K + k] = 0;
        }
    }
    for (unsigned int c = 0; c < mat.ncols; ++c) {
        unsigned int col_start = mat.col_indx[c];
        unsigned int col_end = mat.col_indx[c + 1];

        for (unsigned int r = col_start; r < col_end; ++r) {
            unsigned int row_id = mat.row_id[r];
            double val = mat.values[r];

            for (unsigned int k = 0; k < K; ++k) {
                dmat_out[row_id * K + k] += val * dmat_in[c * K + k];
            }
        }

    }
}

__global__ void dev_csc_spmm(double *D, double *O, unsigned int *row_indx, unsigned int *col_id, double *values, unsigned int nrows, unsigned int ncols, unsigned int nnz, int K){
	int id = blockIdx.x * blockDim.x + threadIdx.x;

	if(id >= ncols) return;

	int end_idx = col_indx[id+1];
	if(id+1 >= ncols){
		end_idx = nnz;
	}

	for(int j=col_indx[id];j<end_idx;j++){
		for(int k=0;k<K;k++){
			O[id * K + k] += values[j] * D[col_id[j] * K + k];
		}
	}

}

int main(int argc, char *argv[]) {
    if(argc < 3) {
        std::cerr << "usage ./exec inputfile K  " << std::endl;
        exit(-1);
    }

    unsigned int K = std::atoi(argv[2]);
    CSC mat = read_matrix_market_to_CSC(argv[1]);
    std::cout << mat.nrows << ' ' << mat.ncols << ' ' << mat.nnz << ' ' << K << '\n';

    CSC *mat_pinned;
    cudaMallocHost(&mat_pinned, sizeof(CSR));

    memcpy(&(mat_pinned->col_indx), &(mat.col_indx), sizeof(mat.col_indx));
    memcpy(&(mat_pinned->row_id), &(mat.row_id), sizeof(mat.row_id));
    memcpy(&(mat_pinned->values), &(mat.values), sizeof(mat.values));
    memcpy(&(mat_pinned->nrows), &(mat.nrows), sizeof(mat.nrows));
    memcpy(&(mat_pinned->ncols), &(mat.ncols), sizeof(mat.ncols));
    memcpy(&(mat_pinned->nnz), &(mat.nnz), sizeof(mat.nnz));

    //double *dmat_in = (double*)malloc(mat.ncols * K  * sizeof(double));
    //double *dmat_out = (double*)malloc(mat.nrows * K * sizeof(double));

    double *dmat_in, *dmat_out, *dmat_result;
    cudaMallocHost(&dmat_in, mat.ncols * K  * sizeof(double));
    cudaMallocHost(&dmat_out, mat.nrows * K * sizeof(double));
    cudaMallocHost(&dmat_result, mat.nrows * K * sizeof(double));


    init_dmat(dmat_in, mat.ncols, K, 1.0);
    //print_dmat(dmat_in, mat.ncols, K);

    host_csc_spmm(mat, dmat_in, dmat_out, K);

    double *dmat_in_d, *dmat_out_d, *val_d;
    unsigned int *row_idx_d, *col_idx_d;
 
    /* Allocate memory for device variables and move variables to device*/
    const float num_threads = 128;
    dim3 threads(num_threads);
    dim3 grid(ceil(mat.ncol/num_threads));

    cudaStream_t stream;

    cudaMalloc(&dmat_in_d, mat.ncols * K * sizeof(double));
    cudaMalloc(&dmat_out_d, mat.nrows * K * sizeof(double));
    cudaMalloc(&row_idx_d, mat.nnz * sizeof(unsigned int));
    cudaMalloc(&col_idx_d, mat.ncols * sizeof(unsigned int));
    cudaMalloc(&val_d, mat.nnz * sizeof(double));

    cudaStreamCreate(&stream);
    cudaMemcpyAsync(dmat_in_d, dmat_in, mat.ncols * K * sizeof(double), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(row_idx_d, mat_pinned->row_id, mat.nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(col_idx_d, mat_pinned->col_indx, mat.ncols * sizeof(unsigned int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(val_d, mat_pinned->values, mat.nnz * sizeof(double), cudaMemcpyHostToDevice, stream);

    /* Compute product */

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    dev_csc_spmm<<<grid, threads,0,stream>>>(dmat_in_d, dmat_out_d, row_idx_d, col_idx_d, val_d, mat_pinned->nrows, mat_pinned->ncols, mat_pinned->nnz, K);
    cudaEventRecord(stop);

    /* Move result back to host */
    cudaMemcpyAsync(dmat_result, dmat_out_d, mat.nrows * K * sizeof(double), cudaMemcpyDeviceToHost); 

    cudaEventRecord(stop);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);


    std::cout << "replace one argument to the below function with the values from gpu " << std::endl;
    check_dmat(dmat_out, dmat_out, mat.nrows, K);

    //print_dmat(dmat_out, mat.nrows, K);


    free(mat.col_indx);
    free(mat.row_id);
    free(mat.values);
    return 0;
}
