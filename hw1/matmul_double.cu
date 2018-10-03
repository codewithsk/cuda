/* objective
 * 	C = A*B  // A[m][k], B[k][n], C[m][n]
 * compile: nvcc --gpu-architecture=compute_60 --gpu-code=sm_60 -O3 matmul_double.cu -o matmul_double
 */

#include <iostream>
#include <cstdlib>
#include <math.h>

# define BLK_SIZE 32
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
            A[i * K + j] = i; //i * K + j;
        }
    }

    for (int i = 0; i < K; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            B[i * N + j] = j; //i * N + j + 1;
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

__global__ void matmul_double(double* A, double* B , double* C, int M, int N, int K)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

   
    if(row >= M || col >= N)
	    return;

    __shared__ float SA[BLK_SIZE][BLK_SIZE];
    __shared__ float SB[BLK_SIZE][BLK_SIZE];
    
    double temp = 0;

    for(int tilek=0;tilek<K;tilek+=BLK_SIZE){
      SA[ty][tx] = A[row*K + (tilek + tx)];
      SB[ty][tx] = B[(tilek+ty) * N + col];
      __syncthreads();
      for(int i=0;i<BLK_SIZE;i++){
        temp+= SA[ty][i] * SB[i][tx];
      }
      __syncthreads();
    }
    int id = row * N + col;
    C[id] = temp;
 
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
    double *dA, *dB, *dC;
    cudaMalloc((void**) &dA, M*K*sizeof(double));    
    cudaMalloc((void**) &dB, K*N*sizeof(double));    
    cudaMalloc((void**) &dC, M*N*sizeof(double));    

    /* Initialize host memory*/
    init(hA, hB, M, N, K);

    /* host compute */
    matmul_double_host(hA, hB, hC, M, N, K);

    /* Copy from host to device */
    cudaMemcpy(dA, hA, M*K*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(double), cudaMemcpyHostToDevice);

    /* call gpu kernel */
    dim3 threads(BLK_SIZE, BLK_SIZE);
    dim3 grid(ceil(N/float(BLK_SIZE)),ceil(M/float(BLK_SIZE)));

    printf("Number of threads in a block %dx%d\n",(int)BLK_SIZE, (int)BLK_SIZE);
    printf("Number of blocks in a grid %dx%d\n",(int)ceil(N/float(BLK_SIZE)),(int)ceil(M/float(BLK_SIZE)));

    matmul_double<<<grid, threads>>>(dA, dB, dC, M, N, K);
    
    std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
    
    /* Copy from device to host (dC -> dtohC) */
    cudaMemcpy(dtohC, dC, M*N*sizeof(double), cudaMemcpyDeviceToHost);


    /* host vs device validation */
    validate(hC, dtohC, M, N);

    /* be clean */
    free(hA);
    free(hB);
    free(hC);
    free(dtohC);

    /// add code to free gpu memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return 0;
}




