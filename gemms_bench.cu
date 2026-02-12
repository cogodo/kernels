#include <cuda_runtime.h>
#include <cstdio>
#include "kernels/gemms.cuh"

int main() {
    // alloc, init, etc...
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  launch_naive_gemm(1024, 1024, 1024, 1, nullptr,  nullptr, 0,  nullptr);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
}
