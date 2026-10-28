// src/prefill/mmq_compat.cuh - the one place that knows how llama.cpp's mmq.cuh host interface changed between the commit
// the engine was written for (third_party/ggml/VERSION.txt) and llama.cpp v0.8.0.  Include it AFTER mmq.cuh.
//
//   generation 1 (the pin ... v0.6.0)   template <ggml_type> mul_mat_q_case(ctx, args, stream); mmq_args ends in ncols_max, ncols_opt
//                                       (mul_mat_q_switch_J picks the tile width from ncols_opt); ggml_cuda_mmq_get_config(type, J,
//                                       fallback, cc).
//   generation 2 (v0.8.0)               template <ggml_type, ggml_prec> mul_mat_q_case; mmq_args ends in ncols_max, J_best, fusion (the
//                                       CALLER picks the tile width: mmq.cu's static ggml_cuda_mmq_get_J_best); ggml_cuda_mmq_get_config
//                                       takes the precision of src1 as a fifth argument.
//
// STRATA_GGML_MMQ_GEN is defined by CMake from what it finds in the llama.cpp tree's mmq.cuh (CMakeLists.txt, "which generation
// of ggml-cuda's host interface"), never by hand; undefined = 1.  Engine code calls the STRATA_MMQ_* names below and
// never mul_mat_q_case / ggml_cuda_mmq_get_config directly, so one source builds against either tree.
#pragma once

#ifndef STRATA_GGML_MMQ_GEN
#define STRATA_GGML_MMQ_GEN 1
#endif

#if STRATA_GGML_MMQ_GEN >= 2

#define STRATA_PREC_Q8 GGML_PREC_Q8
#define STRATA_PREC_Q4 GGML_PREC_Q4
/// fn<type[, prec]>(ctx, args, stream): fn is mul_mat_q_case or a renamed copy of it (the NVFP4 units)
#define STRATA_MMQ_CASE(fn, type, prec, ctx, args, stream) fn<type, prec>(ctx, args, stream)
/// the tile configuration of `type` at tile width J
#define STRATA_MMQ_CONFIG(type, J, fallback, cc, prec) ggml_cuda_mmq_get_config(type, J, fallback, cc, (ggml_prec) (prec))

/// The last field before the fusion block of mmq_args (generation 2: the tile width J_best), for a product of `ncols` columns on
/// `device` - ggml_cuda_mmq_get_J_best of mmq.cu: the J whose config fits the card's shared memory with the fewest column tiles.
/// Aborts like mul_mat_q_switch_J does when no tile fits (moe_mmq.cu's fits() has said so before).
static inline int strata_mmq_tile_arg(int device, ggml_type type, int64_t nrows_x, int64_t ncols, int prec) {
    const ggml_cuda_device_info& info = ggml_cuda_info();
    const int cc = info.devices[device].cc;
    const size_t smpbo = info.devices[device].smpbo;
    const bool fallback = ggml_cuda_mmq_needs_fallback(nrows_x);
    int best = 0, ntiles_best = 0x7fffffff;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(type, J, fallback, cc, (ggml_prec) prec);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > smpbo) continue;
        const int ntiles = (int) ((ncols + c.J - 1) / c.J);
        if (ntiles < ntiles_best) { best = J; ntiles_best = ntiles; }
    }
    GGML_ASSERT(best > 0);
    return best;
}

#else  // generation 1

#define STRATA_PREC_Q8 0
#define STRATA_PREC_Q4 1
#define STRATA_MMQ_CASE(fn, type, prec, ctx, args, stream) fn<type>(ctx, args, stream)
#define STRATA_MMQ_CONFIG(type, J, fallback, cc, prec) ggml_cuda_mmq_get_config(type, J, fallback, cc)

/// generation 1: mul_mat_q_switch_J chooses the tile width itself, from the number of columns
static inline int64_t strata_mmq_tile_arg(int, ggml_type, int64_t, int64_t ncols, int) { return ncols; }

#endif
