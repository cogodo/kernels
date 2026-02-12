#pragma once
#include <cuda_runtime.h>

void launch_naive_gemm(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C);