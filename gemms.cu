#include <cuda_runtime.h>
#include <stdint.h>

#define CEIL_DIV(X, Y) (((X) + (Y) - 1) / (Y))


__global__ void gemm_kernel_1(
    int M, int N, int K, float alpha, const float *A,
                                const float *B, float beta, float *C
    ) {
        const uint x = blockIdx.x * blockDim.x + threadIdx.x;
        const uint y = blockIdx.y * blockDim.y + threadIdx.y;


        if (x < M && y < N) {
            float tmp = 0.0;
            for(int i = 0; i < K; ++i) {
                tmp += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = alpha * tmp + beta * C[N * x + y];
        }
    }


extern "C" int gemm_launch_1(
    const float* A, const float* B, float* C,
    int M, int N, int K, float alpha, float beta
) {
    // create as many blocks as necessary to map all of C
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32, 1);

    //kernel launch
    gemm_kernel_1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

    cudaError_t err = cudaGetLastError();
    return int(err);
}

