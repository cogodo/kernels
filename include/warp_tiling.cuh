#pragma once

#include "gemm_common.cuh"

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
      As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }
    // then, load from B - easy
  
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
      reinterpret_cast<float4 *>(
          &Bs[(offset + innerRowB) * BN + innerColB * 4])[0] =
          reinterpret_cast<const float4 *>(
              &B[(offset + innerRowB) * N + innerColB * 4])[0];
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
        for(int i = 0; i < TM; ++i) {
          /* this is a lot so let me break it down: 
          As is BM * BK, and stored transposed. This means BM rows, BK cols.
          what is in As? The tile that the current block is!
          so we have a whole tile in As. Thus, first we must index along the BK dimension.
          left to right: in regM, need to fill in row wSubtileRowIdx, col i, so TM is thread reg row len, classic row-major indexing.
          Then for As: dotIdx is row number in block of M axis, mult with BM is more row major. now down to "row" that current warp is responsible for:
          warpRow * WM is row major of the "row" that the warp is current working on (by our design, one row at a time).
          wSubtileRowIdx * WSUBM is again "row" within the current data that the warp is working on, again based on how big our register acc is
          threadRowInWarp * TM is more obvious,  warp is working on the "row" of data the warp is working on is imagined in 2D and each thread gets a chunk. 
          Then add i and get the spot we care about!
          */
          regM[wSubtileRowIdx * TM + i] = As[(dotIdx * BM) + warpRow * WM + wSubtileRowIdx * WSUBM + threadRowInWarp * TM + i];
      }
    }
  
    //2. read into regN
    // WSUBN = WN/WNITER = 16 (used to iterate over slices)
    for(int wSubtileColIdx = 0; wSubtileColIdx < WNITER; ++wSubtileColIdx) {
      for(int i = 0; i < TN; ++i) {
  
        regN[wSubtileColIdx * TN + i] = Bs[(dotIdx * BN) + warpCol * WN + wSubtileColIdx * WSUBN + threadColInWarp * TN + i];
      }
    }
  
    // time to do the actual warp-tile!
    for(int wSubtileRowIdx = 0; wSubtileRowIdx < WMITER; ++wSubtileRowIdx) {
      for(int wSubtileColIdx = 0; wSubtileColIdx < WNITER; ++wSubtileColIdx) {
        for(int resIdxM = 0; resIdxM < TM; ++resIdxM) {
          for(int resIdxN = 0; resIdxN < TN; ++resIdxN) {
            /*
            Indexing explanation for understanding:
            wSubtileRowIdx is the row index in 2D-land of what chunk of the current assigned work our warp is working on.
            The work generally exceeds what is possible to do in one step, so we turn each tile into subtiles for the warp.
            Additionally, on the M axis, we make the math work so that each subtile is just a "row" out of the tile (subtiles ought to divide tiles nicely, so makes sense)
            Next, thread results is of size WMITER * TM * WNITER * TN, essentially all iterations needed for the warp to cover the data it needs to handle.
            Thus, wSubtileRowIdx * TM + resIdxM is indexing to the coord specifie by M matrix axis, and similar for N matrix axis.
            Finally, WNITER * TN is multiplied with the M part because we have WNITER * WMITER subtiles and WMITER is 1, 
            TN is for the row width for each subtile entry.
            */
            threadResults[(wSubtileRowIdx * TM + resIdxM) * (WNITER * TN) + (wSubtileColIdx * TN) + resIdxN] +=
            regM[(wSubtileRowIdx * TM) + resIdxM] * regN[(wSubtileColIdx * TN) + resIdxN];
            }
          } 
        }
      }
    }
  }
} // namespace wt