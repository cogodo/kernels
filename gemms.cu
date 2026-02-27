#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdint.h>
// #include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

#define CEIL_DIV(X, Y) (((X) + (Y) - 1) / (Y))

const int WARPSIZE = 32;

// naive kernel, (almost) as simple as can be
__global__ void gemm_kernel_1(int M, int N, int K, float alpha, const float *A,
                              const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = alpha * tmp + beta * C[N * x + y];
  }
}

extern "C" int gemm_launch_1(const float *A, const float *B, float *C, int M,
                             int N, int K, float alpha, float beta) {
  // create as many blocks as necessary to map all of C
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  // 32 * 32 = 1024 thread per block
  dim3 blockDim(32, 32, 1);

  // kernel launch
  gemm_kernel_1<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);

  cudaError_t err = cudaGetLastError();
  return int(err);
}

namespace wt {

template <const int BM, const int BN, const int BK, const int rowStrideA,
          const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                             float *As, float *Bs, int innerRowA, int innerColA,
                             int innerRowB, int innerColB) {

  // first, load from A - trickier bc A is transposed
  for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(offset + innerRowA) * K + innerColA * 4])[0];
    As[(offset + innerColA + 0) * BN + innerRowA * 4] = tmp.x;
    As[(offset + innerColA + 1) * BN + innerRowA * 4] = tmp.y;
    As[(offset + innerColA + 2) * BN + innerRowA * 4] = tmp.z;
    As[(offset + innerColA + 3) * BN + innerRowA * 4] = tmp.w;
  }
  // then, load from B - easy

  for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(
        &Bs[(offset + innerRowB) * N + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[(offset + innerRowB) * BN + innerColB * 4])[0];
  }
}

__device__ void writeToGmem(int N, int K, int innerRowA, int innerColA, int innerRowB, int innerColB, float* C) {
  // do we write with the float4 thing as well?
  
}

} // namespace ws

// warp tile kernel, should be much faster
/*
 * @tparam BM The threadblock size for M dimension SMEM caching.
 * @tparam BN The threadblock size for N dimension SMEM caching.
 * @tparam BK The threadblock size for K dimension SMEM caching.
 * @tparam WM M dim of continuous tile computed by each warp
 * @tparam WN N dim of continuous tile computed by each warp
 * @tparam WMITER The number of subwarp tiling steps in M dimension.
 * @tparam WNITER The number of subwarp tiling steps in N dimension.
 * @tparam TM The per-thread tile size for M dimension.
 * @tparam TN The per-thread tile size for N dimension.
 */
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WNITER, const int TM, const int TN, const int NUM_THREADS>
__global__ void __launch_bounds__(NUM_THREADS)
    gemm_kernel_2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

  // the overall goal of this section is to get the indexing of our data correct
  // row/col within the output tile in C
  int cRow = threadIdx.x;
  int cCol = threadIdx.y;

  // TO DEFINE:
  int rowStrideA;
  int rowStrideB;
  int innerRowA;
  int innerColA;
  int innerRowB;
  int innerColB;



  // 1. get warp indexing
  

  // 2. get warp subtile size


  // get thread placement within the subtile


  // shared mem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BN * BK];

  // load the data into the smem with our function
  wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(N, N, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
  // using the smem tiles, use our indices + a short loop on TM, TN to do our acc
  float acc[TM][TN];

  for(int k=0; k < BK; ++k) {
    //load fragment of As
    float* A_frag;
    //load fragment of Bs
    float* B_frag;
    // do outer prod into acc
    for(int m_sub = 0; m_sub < TM; ++m_sub) {
      for (int n_sub = 0; n_sub < TN; ++n_sub) {
        acc[m_sub][n_sub] = A_frag[m_sub] * B_frag[n_sub];
    }
  }
  }
  
  // use a function to write from the acc to C

  // Done!
}

extern "C" int gemm_launch_2(const float *A, const float *B, float *C, int M,
                             int N, int K, float alpha, float beta) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
  dim3 blockDim(32, 32, 1);

  // gemm_kernel_2<128, 128, 16, ><<<gridDim, blockDim>>>(M, N, K, alpha, A, B,
  // beta, C);

  cudaError_t err = cudaGetLastError();
  return int(err);
}