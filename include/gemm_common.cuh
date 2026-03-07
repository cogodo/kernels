#pragma once

#ifdef __CLANGD__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_runtime_wrapper.h>
#endif

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>

#include <stdint.h>

#define CEIL_DIV(X, Y) (((X) + (Y) - 1) / (Y))

constexpr int WARPSIZE = 32;
