// g4_compat.hpp - what the Gemma 4 engine's shared code needs to be compiled by nvcc (host + device), by a plain C++ compiler
// (the CPU reference, the tests) and by the CPU emulation of CUDA (g4_emu.hpp, see tools/gemma4/README.md).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#define G4_HD __host__ __device__ __forceinline__
#define G4_DEV __device__ __forceinline__
#else
#define G4_HD inline
#define G4_DEV inline
#endif

#if defined(__CUDA_ARCH__)
#define G4_ON_DEVICE 1
#endif
