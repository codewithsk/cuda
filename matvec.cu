/* objective
 * 	c = A*b  // A[m][n] is a matrix, b[n] and c[m] are vectors
 * compile: nvcc --gpu-architecture=compute_60 --gpu-code=sm_60 -O3 matvec.cu -o matvec
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

void init (double *mat, double *vec, int m , int n)
{
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            mat[i * n + j] = i * m + j;
        }
    }

    for (int j = 0; j < n; ++j)
    {
        vec[j] = j;
    }
}


void matvec_h(double* mat, double*vec_in , double* vec_out, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        vec_out[i] = 0;

        for (int j = 0; j < n; ++j)
        {
            vec_out[i] += mat[i * n + j] * vec_in[j];
        }
    }
}

__global__ void matvec_d(double* mat, double*vec_in , double* vec_out, int m, int n)
{
    /// complete code
}

void validate (double *host, double *gpu, int m)
{

    for (int i = 0; i < m; ++i)
    {
        if(std::abs(host[i] - gpu[i]) > 1e-3)
        {
            std::cerr << "possible error at position " << i << " host: " << host[i] << " device " << gpu[i] << '\n';
        }
    }
}


int main(int argc, char *argv[])
{
    if(argc < 2)
    {
        std::cerr << "Usage: ./matvec M N\n";
        exit(-1);
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);

    /* Host alloc */
    double *mat = (double*) malloc (m * n * sizeof(double));
    double *vec_in = (double*) malloc (n * sizeof(double));
    double *vec_out = (double*) malloc (m * sizeof(double));

    /* Device alloc */
    /// complete code

    /* Initialize host memory*/
    init(mat, vec_in, m, n);

    /* host compute */
    matvec_h(mat, vec_in, vec_out, m, n);


    /* Copy from host to device */
    /// complete code


    /* call gpu kernel */
    /// complete code

    /* Copy from device to host */
    /// complete code


    /* host vs device validation */
    /// REPLACE one vec_out with the result array that you moved from device to host
    validate(vec_out, vec_out, m);


    /* be clean */
    free(mat);
    free(vec_in);
    free(vec_out);

    /// add code to free gpu memory


    return 0;
}




