#pragma once
#include <cuda_runtime.h>

void gemm_launch_1(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

void gemm_launch_2(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);
