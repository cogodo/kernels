#pragma once

extern "C" int gemm_launch_naive(const float *A, const float *B, float *C,
                                 int M, int N, int K, float alpha,
                                 float beta);

extern "C" int gemm_launch_warp_tiling(const float *A, const float *B, float *C,
                                       int M, int N, int K, float alpha,
                                       float beta);
