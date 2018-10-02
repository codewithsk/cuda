/* objective
 * 	C = A*B  // A[m][k], B[k][n], C[m][n]
 * compile: nvcc --gpu-architecture=compute_60 --gpu-code=sm_60 -O3 matmul_double.cu -o matmul_double
 */

/* Required answers
 * Name of executable must be matmul_double
 * The executable must accept m, n, k as parameters
 * Report GFLOPS for:
   *  for i in 4, 8, 10, 11
       for j in 4, 8, 10, 11 m,n=power(2,i)
         k = power(2,j)
   * m=4095,n=4097,k=125
 * Report number of following for m=2047, n = 2049, k=125:
   * Read DRAM transactions
   * Write DRAM transactions
   * Shared memory reads/transaction
   * Shared memory write/transaction
 * Report number of following for m=2048, n = 2048, k=128:
   * Read DRAM transactions
   * Write DRAM transactions
   * Shared memory reads/transaction
   * Shared memory write/transaction
 */


#include <iostream>
#include <cstdlib>


#define EC(ans) { chkerr((ans), __FILE__, __LINE__); }
inline void chkerr(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) << " File: " << file << " Line: " << line << '\n';
        exit(-1);
    }
}

void init (double *A, double *B, int M , int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < K; ++j)
        {
            A[i * K + j] = i * K + j;
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = i * N + j + 1;
        }
    }

}


void matmul_double_host(double* A, double* B, double* C, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            double tmp = 0;

            for (int k = 0; k < K; ++k)
            {
                tmp += A[i * K + k] * B[k * N + j];
            }

            C[i * N + j] = tmp;
        }
    }
}

__global__ void matmul_double(double* A, double* B , double* C, int M, int N, int K)
{
    /// complete code
}

void validate (double *host, double *gpu, int M, int N)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if(std::abs(host[i * N + j] - gpu[i * N + j]) > 1e-3)
            {
                std::cerr << "possible error at position " << i << ',' << j << " host: " << host[i * N + j] << " device " << gpu[i * N + j] << '\n';
            }

        }
    }
}


int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        std::cerr << "Usage: ./matmul_double M N K\n";
        exit(-1);
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    /* Host alloc */
    double *hA = (double*) malloc (M * K * sizeof(double));
    double *hB = (double*) malloc (K * N * sizeof(double));
    double *hC = (double*) malloc (M * N * sizeof(double));
    double *dtohC = (double*) malloc (M * N * sizeof(double));

    /* Device alloc */
    /// complete code

    /* Initialize host memory*/
    init(hA, hB, M, N, K);

    /* host compute */
    matmul_double_host(hA, hB, hC, M, N, K);


    /* Copy from host to device */
    /// complete code


    /* call gpu kernel */
    /// complete code

    /* Copy from device to host (dC -> dtohC) */
    /// complete code


    /* host vs device validation */
    validate(hC, dtohC, M, N);


    /* be clean */
    free(hA);
    free(hB);
    free(hC);
    free(dtohC);

    /// add code to free gpu memory


    return 0;
}




