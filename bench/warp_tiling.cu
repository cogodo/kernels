#include "../include/gemm_common.cuh"
#include "../include/warp_tiling.cuh"

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
gemm_kernel_warp_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

/*
tile size == block size? NO! in practice our block is 256 threads, but a tile can be far larger
In our case, tile size is BM * BN = 128 * 128 = 16384 output elts, so obv each thread has to do more work than just 1 elt
How do we split it up?
Consider our case: 16384 / 256 = 64 output elts / thread. each gets 2 8 * 4 subtiles to compute to thread results,
and then push back to C. 
*/

int blockIdxM = blockIdx.x;
int blockIdxN = blockIdx.y;

int numWarpsN = BN / WN;
int warpIdx = threadIdx.x / WARPSIZE;
int warpCol = warpIdx % numWarpsN;
int warpRow = warpIdx / numWarpsN;

/*
This indexing feels weird to me, but it just fills the relationship of 
WM * WN = WARPSIZE * (WMITER * TM) * (WNITER * TN).                                                                                                       
In other words, this is saying the size of the warptile is equivalent to 
the size of the work all threads in the warp do.
*/
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
constexpr uint WSUBM = WM / WMITER; // 64/2=32
constexpr uint WSUBN = WN / WNITER; // 32/2=16

int numThreadsPerSubtile = WSUBN / TN;
int threadIdxInWarp = threadIdx.x % WARPSIZE;
int threadColInWarp = threadIdxInWarp % numThreadsPerSubtile;
int threadRowInWarp = threadIdxInWarp / numThreadsPerSubtile;


// shared mem
__shared__ float As[BM * BK];
__shared__ float Bs[BN * BK];

//set A, B to correct offset
A += blockIdxM * BM * K;
B += blockIdxN * BN;

// set C to warp output tile
C += (blockIdxM * BM + warpRow * WM) * N + blockIdxN * BN + warpCol * WN;

// set all the inner stuff + strides:
// remember 4 float vectorized loads
constexpr int ARowSizeInf4 = BK / 4;
constexpr int BRowSizeInf4 = BN / 4;
constexpr uint rowStrideA = NUM_THREADS  / ARowSizeInf4;
constexpr uint rowStrideB = NUM_THREADS / BRowSizeInf4;
const int innerRowA = threadIdx.x / ARowSizeInf4;
const int innerColA = threadIdx.x % ARowSizeInf4;
const int innerRowB = threadIdx.x / BRowSizeInf4;
const int innerColB = threadIdx.x % BRowSizeInf4;

// using the smem tiles, use our indices + a short loop on TM, TN to do our acc
float threadResults[TM * WMITER * TN * WNITER] = {0.0};

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
for(int wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
for(int wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
// set C pointer
float *C_moved = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
for(int rthreadRow = 0; rthreadRow < TM; ++rthreadRow) {
for(int rthreadCol = 0; rthreadCol < TN; rthreadCol += 4) {
 float4 tmp = reinterpret_cast<float4 *>(&C_moved[(threadRowInWarp * TM + rthreadRow) * N + threadColInWarp * TN + rthreadCol])[0];

 const int i = (wSubRowIdx * TM + rthreadRow) * (WNITER * TN) + wSubColIdx * TN + rthreadCol;

 tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
 tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
 tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
 tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;

 reinterpret_cast<float4 *>(
   &C_moved[(threadRowInWarp * TM + rthreadRow) * N + threadColInWarp * TN + rthreadCol])[0] = tmp;
 
}
}
}
}
// Done!
}

extern "C" int gemm_launch_warp_tiling(const float *A, const float *B, float *C, int M,
                    int N, int K, float alpha, float beta) {
const int BM = 128;
const int BN = 128;
const int BK = 16;
const int WM = 64;
const int WN = 32;
const int WNITER = 2;
const int TM = 8;
const int TN = 4;
const int NUM_THREADS = 256;

dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1);
dim3 blockDim(NUM_THREADS, 1, 1);

gemm_kernel_warp_tiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS><<<gridDim, blockDim>>>(M, N, K, alpha, A, B,
beta, C);

cudaError_t err = cudaGetLastError();
return int(err);
}