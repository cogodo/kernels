#include <__clang_cuda_builtin_vars.h>
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))

__global__ void naive_kernel(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    int BLOCK_SIZE = 32;

    const int row = BLOCK_SIZE * blockIdx.x + (threadIdx.x / BLOCK_SIZE);
    const int col = BLOCK_SIZE * blockIdx.y + (threadIdx.x % BLOCK_SIZE);

    if(row < M && col < N) {
        float tmp = 0.0f;

        for(int i = 0; i < K; ++i) {
            tmp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = alpha * tmp + beta * C[row * N + col];
    }
}

void launch_naive_gemm(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
    // create as many blocks as necessary to map all of C
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32 * 32);
    // launch the asynchronous execution of the kernel on the device
    // the function call returns immediately on the host
    naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}