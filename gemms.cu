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

template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
          const int TM, const int TN>
__device__ void processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
  // here is where we read into As, Bs, and do the warp tiling itself
  
  // outer loop - need to loop over all k in BK
  for(int dotIdx = 0; dotIdx < BK; ++dotIdx) {

    //1. read into regM
    // WMITER is basically the number of times we need the warp on axis M to perform its action
    for(int wSubtileRowIdx = 0; wSubtileRowIdx < WMITER; ++wSubtileRowIdx) {
      for(int i = 0; i < WSUBM; ++i) {
        /* this is a lot so let me break it down: 
        As is BM * BK, and stored transposed. This means BM rows, BK cols.
        what is in As? The tile that the current block is!
        so we have a whole tile in As. Thus, first we must index along the BK dimension,
        as this is 
        dotidx * BM = we're at the start of the tile and getting to the warp that we care about 
        warpRow * WM = we're at the start of the warp we care about getting to the row in the warp where our thread is
        wSubtileRowIdx * WSUBM = we're at the col (bc transpose) in warp where our thread is, and want to TODO: i need to think / read more on this
        */
        regM[wSubtileRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubtileRowIdx * WSUBM + threadRowInWarp * TM + i];
    }
  }

  //2. read into regN
  // WSUBN = WN/WNITER = 16 (used to iterate over slices)
  for(int wSubtileColIdx = 0; wSubtileColIdx < WNITER; ++wSubtileColIdx) {
    for(int i = 0; i < WSUBN; ++i) {

      regN[wSubtileColIdx * TN + i] = Bs[(dotIdx * BN) + warCol * WN + wSubtileColIdx * WSUBN + threadColInWarp * TN + i];
    }
  }



  // from here, pretending the index calcs + data loading is all done - 
  // time to do the actual warp-tile!

  for(int wSubtileRowIdx = 0; wSubtileRowIdx < WMITER; ++wSubtileRowIdx) {
    for(int wSubtileColIdx = 0; wSubtileColIdx < WNITER; ++wSubtileColIdx) {
      for(int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for(int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          // TODO: annotate the indexing here
          threadResults[(wSubtileRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubtileColIdx * TN) + resIdxN] = 
          regM[(wSubtileRowIdx * TM) + resIdxM] * regN[(wSubtileColIdx * TN) + resIdxN];
        }
      } 
    }

  }

  
}
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



  // from the top: block to warp to thread indexing

  //tile size == block size? NO! actually idk TODO: figure out the discrepancy that i am missing here
  // run through the basics: 1 block can have 4 warps active = 128 threads at a time.int
  // consider 1024 x 1024 matrix.
  // split into 128 x 128 tiles, 8 x 8 grid of tiles
  // each SM gets 1 tile because there are more sms than tiles
  // TODO: think about above as well
  int blockIdxM = blockIdx.x;
  int blockIdxN = blockIdx.y;

  int numWarpsN = BN / WN;
  int warpIdx = threadIdx.x / WARPSIZE;
  int warpCol = warpIdx % numWarpsN;
  int warpRow = warpIdx / numWarpsN;

  //TODO: grasp what this even means and annotate it
  constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr uint WSUBM = WM / WMITER; // 64/2=32
  constexpr uint WSUBN = WN / WNITER; // 32/2=16
  
  int numThreadsPerSubtile = WSUBN / TN;
  int threadIdxInWarp = threadIdx.x % warpIdx;
  int threadColInWarp = threadIdxInWarp % numThreadsPerSubtile;
  int threadRowInWarp = threadIdxInWarp / numThreadsPerSubtile;


  // shared mem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BN * BK];

  //set A, B to correct offset
  A += blockIdxM * BM * K;
  B += blockIdxN * BN * K;

  // set C to warp output tile
  C += (blockIdxM * BM + warpRow * WM) + blockIdxN * BN + warpCol * WN;

  // set all the inner stuff + strides:
  // remember 4 float vectorized loads
  int ARowSizeInf4 = BK / 4;
  int BRowSizeInf4 = BN / 4;
  int rowStrideA = NUM_THREADS  / ARowSizeInf4;
  int rowStrideB = NUM_THREADS / BRowSizeInf4;
  int innerRowA = threadIdx.x / ARowSizeInf4;
  int innerColA = threadIdx.x % ARowSizeInf4;
  int innerRowB = threadIdx.x / BRowSizeInf4;
  int innerColB = threadIdx.x % BRowSizeInf4;

  // using the smem tiles, use our indices + a short loop on TM, TN to do our acc
  float threadResults[TM * WMITER * TN * WNITER] = {0.0}

  float regM[WMITER * TM];
  float regN[WNITER * TN];

  for(int blockKIdx=0; blockKIdx < K; blockKIdx += BK) {
    //populate As, Bs
    wt::loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    // go from As, Bs to registers, do the warp tiling and write it back to C
    wt::processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM, TN>
    (regM, regN, threadResults, As, Bs, warpRow, warpCol, threadRowInWarp, threadColInWarp);
    // move A and B to next block
    A += BK;
    B += BK * N;
    __syncthreads(); 
  }
  
  // write from the acc to C
  // more for loops!
  
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