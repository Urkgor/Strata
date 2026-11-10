// g4_cuda_compat.hpp - the CUDA runtime, or its CPU emulation (STRATA_G4_EMU), and the one way the engine launches a kernel.
#pragma once

#include "g4_compat.hpp"

#ifdef STRATA_G4_EMU
#include "g4_emu.hpp"
#define G4_K(...) __VA_ARGS__
#define G4_LAUNCH(kernel, grid, block, stream, ...) g4emu::launch(dim3(grid), dim3(block), [&] { kernel(__VA_ARGS__); })
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define G4_K(...) __VA_ARGS__
#define G4_LAUNCH(kernel, grid, block, stream, ...) kernel<<<dim3(grid), dim3(block), 0, stream>>>(__VA_ARGS__)
#endif
