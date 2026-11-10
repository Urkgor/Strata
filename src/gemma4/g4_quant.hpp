// g4_quant.hpp - the ggml block formats the Gemma 4 engine reads, and their dequantization, for the host (the CPU reference, the loader's
// embedding lookup) and the device (the GEMV kernels): one implementation, so the two cannot differ.  Every function is a port of ggml's
// dequantize_row_* (ggml/src/ggml-quants.c) split at the granularity a GPU lane works at: a UNIT of 32 values.  tests/gemma4/g4_quant_test.cpp
// compares all of them with ggml's own dequantization on random blocks.
//
//   unit u of a row: values [32 u, 32 u + 32) of the row.  The row is `n_in` values, n_in a multiple of the type's block (32, or 256 for
//   the K-quants and IQ4_XS).  `row` points at the first block of the row.
#pragma once

#include "g4_compat.hpp"

namespace g4 {

// ggml_type ids (GGUF's tensor types)
enum QType : int {
    T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q4_1 = 3, T_Q5_0 = 6, T_Q5_1 = 7, T_Q8_0 = 8, T_Q2_K = 10, T_Q3_K = 11, T_Q4_K = 12, T_Q5_K = 13,
    T_Q6_K = 14, T_IQ4_NL = 20, T_IQ4_XS = 23, T_BF16 = 30,
};

/// elements per block and bytes per block of a type, 0 when the engine's kernels do not read it
G4_HD int block_elems(int t) {
    switch (t) {
        case T_F32: case T_F16: case T_BF16: return 32;     // (no blocks: 32 values form a unit)
        case T_Q4_0: case T_Q4_1: case T_Q5_0: case T_Q5_1: case T_Q8_0: case T_IQ4_NL: return 32;
        case T_Q2_K: case T_Q3_K: case T_Q4_K: case T_Q5_K: case T_Q6_K: case T_IQ4_XS: return 256;
        default: return 0;
    }
}
G4_HD int block_bytes(int t) {
    switch (t) {
        case T_F32: return 128;
        case T_F16: case T_BF16: return 64;
        case T_Q4_0: case T_IQ4_NL: return 18;
        case T_Q4_1: return 20;
        case T_Q5_0: return 22;
        case T_Q5_1: return 24;
        case T_Q8_0: return 34;
        case T_Q2_K: return 84;
        case T_Q3_K: return 110;
        case T_Q4_K: return 144;
        case T_Q5_K: return 176;
        case T_Q6_K: return 210;
        case T_IQ4_XS: return 136;
        default: return 0;
    }
}
/// bytes of a row of n values (n a multiple of block_elems)
G4_HD long long row_bytes(int t, long long n) { return n / block_elems(t) * block_bytes(t); }
/// the name of a type for messages
inline const char* type_name(int t) {
    switch (t) {
        case T_F32: return "F32"; case T_F16: return "F16"; case T_BF16: return "BF16"; case T_Q4_0: return "Q4_0"; case T_Q4_1: return "Q4_1";
        case T_Q5_0: return "Q5_0"; case T_Q5_1: return "Q5_1"; case T_Q8_0: return "Q8_0"; case T_Q2_K: return "Q2_K"; case T_Q3_K: return "Q3_K";
        case T_Q4_K: return "Q4_K"; case T_Q5_K: return "Q5_K"; case T_Q6_K: return "Q6_K"; case T_IQ4_NL: return "IQ4_NL"; case T_IQ4_XS: return "IQ4_XS";
        default: return "?";
    }
}

// ---- IEEE half and bfloat16 to float: the hardware conversion on the device, bit arithmetic (exact) elsewhere
G4_HD float half_to_float(uint16_t h) {
#if defined(__CUDA_ARCH__)
    return __half2float(__ushort_as_half(h));
#else
    uint32_t u;
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) {
        if (m == 0) {
            u = s;
        } else {
            int sh = 0;
            while (!(m & 0x400)) { m <<= 1; ++sh; }
            m &= 0x3ff;
            u = s | ((113u - (uint32_t) sh) << 23) | (m << 13);
        }
    } else if (e == 31) {
        u = s | 0x7f800000u | (m << 13);
    } else {
        u = s | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}
G4_HD float bf16_to_float(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
G4_HD uint16_t rd16(const uint8_t* p) { return (uint16_t) (p[0] | (p[1] << 8)); }
G4_HD float rdh(const uint8_t* p) { return half_to_float(rd16(p)); }

// IQ4's non-linear 4-bit values (ggml's kvalues_iq4nl), packed so that the device needs no constant memory
G4_HD int kv_iq4nl(int i) {
    const unsigned long long lo = 0xF6EADDCFBFAD9881ull, hi = 0x7159453526190D01ull;   // -127,-104,-83,-65,-49,-35,-22,-10 | 1,13,25,38,53,69,89,113
    const unsigned long long w = i < 8 ? lo : hi;
    return (int) (signed char) ((w >> (8 * (i & 7))) & 0xff);
}

/// ggml's get_scale_min_k4: the 6-bit scale and min j (0..7) of a Q4_K / Q5_K superblock
G4_HD void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

/// Q3_K's 16 signed 6-bit scales (already minus 32) from the 12 packed bytes
G4_HD void q3k_scales(const uint8_t* s, int* out) {
    for (int i = 0; i < 4; ++i) {
        out[i]      = ((s[i] & 0xF) | (((s[8 + i] >> 0) & 3) << 4)) - 32;
        out[4 + i]  = ((s[4 + i] & 0xF) | (((s[8 + i] >> 2) & 3) << 4)) - 32;
        out[8 + i]  = (((s[i] >> 4) & 0xF) | (((s[8 + i] >> 4) & 3) << 4)) - 32;
        out[12 + i] = (((s[4 + i] >> 4) & 0xF) | (((s[8 + i] >> 6) & 3) << 4)) - 32;
    }
}

/// Dequantize unit u (values 32 u .. 32 u + 31) of the row starting at `row` into out[0..31].
template <int T>
G4_HD void dequant32(const uint8_t* row, long long u, float* out) {
    if constexpr (T == T_F32) {
        const uint8_t* p = row + u * 128;
        for (int j = 0; j < 32; ++j) { float f; std::memcpy(&f, p + 4 * j, 4); out[j] = f; }
    } else if constexpr (T == T_F16) {
        const uint8_t* p = row + u * 64;
        for (int j = 0; j < 32; ++j) out[j] = half_to_float(rd16(p + 2 * j));
    } else if constexpr (T == T_BF16) {
        const uint8_t* p = row + u * 64;
        for (int j = 0; j < 32; ++j) out[j] = bf16_to_float(rd16(p + 2 * j));
    } else if constexpr (T == T_Q8_0) {
        const uint8_t* p = row + u * 34;
        const float d = rdh(p);
        for (int j = 0; j < 32; ++j) out[j] = (float) (int) (signed char) p[2 + j] * d;
    } else if constexpr (T == T_Q4_0) {
        const uint8_t* p = row + u * 18;
        const float d = rdh(p);
        for (int j = 0; j < 16; ++j) {
            out[j] = (float) ((int) (p[2 + j] & 0x0F) - 8) * d;
            out[j + 16] = (float) ((int) (p[2 + j] >> 4) - 8) * d;
        }
    } else if constexpr (T == T_Q4_1) {
        const uint8_t* p = row + u * 20;
        const float d = rdh(p), m = rdh(p + 2);
        for (int j = 0; j < 16; ++j) {
            out[j] = (float) (p[4 + j] & 0x0F) * d + m;
            out[j + 16] = (float) (p[4 + j] >> 4) * d + m;
        }
    } else if constexpr (T == T_Q5_0) {
        const uint8_t* p = row + u * 22;
        const float d = rdh(p);
        const uint32_t qh = (uint32_t) p[2] | ((uint32_t) p[3] << 8) | ((uint32_t) p[4] << 16) | ((uint32_t) p[5] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = (int) (((qh >> j) << 4) & 0x10), xh1 = (int) ((qh >> (j + 12)) & 0x10);
            out[j] = (float) (((p[6 + j] & 0x0F) | xh0) - 16) * d;
            out[j + 16] = (float) (((p[6 + j] >> 4) | xh1) - 16) * d;
        }
    } else if constexpr (T == T_Q5_1) {
        const uint8_t* p = row + u * 24;
        const float d = rdh(p), m = rdh(p + 2);
        const uint32_t qh = (uint32_t) p[4] | ((uint32_t) p[5] << 8) | ((uint32_t) p[6] << 16) | ((uint32_t) p[7] << 24);
        for (int j = 0; j < 16; ++j) {
            const int xh0 = (int) (((qh >> j) << 4) & 0x10), xh1 = (int) ((qh >> (j + 12)) & 0x10);
            out[j] = (float) ((p[8 + j] & 0x0F) | xh0) * d + m;
            out[j + 16] = (float) ((p[8 + j] >> 4) | xh1) * d + m;
        }
    } else if constexpr (T == T_IQ4_NL) {
        const uint8_t* p = row + u * 18;
        const float d = rdh(p);
        for (int j = 0; j < 16; ++j) {
            out[j] = d * (float) kv_iq4nl(p[2 + j] & 0xf);
            out[j + 16] = d * (float) kv_iq4nl(p[2 + j] >> 4);
        }
    } else if constexpr (T == T_Q4_K) {
        const uint8_t* p = row + (u >> 3) * 144;
        const int sub = (int) (u & 7);
        const float d = rdh(p), dmin = rdh(p + 2);
        int sc, mn;
        scale_min_k4(sub, p + 4, sc, mn);
        const uint8_t* q = p + 16 + 32 * (sub >> 1);
        const int sh = 4 * (sub & 1);
        const float d1 = d * (float) sc, m1 = dmin * (float) mn;
        for (int l = 0; l < 32; ++l) out[l] = d1 * (float) ((q[l] >> sh) & 0xF) - m1;
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* p = row + (u >> 3) * 176;
        const int sub = (int) (u & 7);
        const float d = rdh(p), dmin = rdh(p + 2);
        int sc, mn;
        scale_min_k4(sub, p + 4, sc, mn);
        const uint8_t* qh = p + 16;
        const uint8_t* ql = p + 48 + 32 * (sub >> 1);
        const int sh = 4 * (sub & 1), mask = 1 << sub;      // u1 = 1, u2 = 2, then x4 per 64 values: bit `sub`
        const float d1 = d * (float) sc, m1 = dmin * (float) mn;
        for (int l = 0; l < 32; ++l) out[l] = d1 * (float) (((ql[l] >> sh) & 0xF) + ((qh[l] & mask) ? 16 : 0)) - m1;
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* p = row + (u >> 3) * 210;
        const int sub = (int) (u & 7), n = sub >> 2, g = sub & 3;
        const float d = rdh(p + 208);
        const uint8_t* ql = p + 64 * n;
        const uint8_t* qh = p + 128 + 32 * n;
        const signed char* sc = (const signed char*) (p + 192) + 8 * n;
        for (int l = 0; l < 32; ++l) {
            const int lo = (g & 1) ? (ql[l + 32] >> (g >> 1 ? 4 : 0)) : (ql[l] >> (g >> 1 ? 4 : 0));
            const int hi = (qh[l] >> (2 * g)) & 3;
            const int q = ((lo & 0xF) | (hi << 4)) - 32;
            out[l] = d * (float) sc[(l >> 4) + 2 * g] * (float) q;
        }
    } else if constexpr (T == T_Q2_K) {
        const uint8_t* p = row + (u >> 3) * 84;
        const int sub = (int) (u & 7), n = sub >> 2, j = sub & 3;
        const float d = rdh(p + 80), dmin = rdh(p + 82);
        const uint8_t* q = p + 16 + 32 * n;
        for (int l = 0; l < 32; ++l) {
            const int sc = p[8 * n + 2 * j + (l >> 4)];
            out[l] = d * (float) (sc & 0xF) * (float) ((q[l] >> (2 * j)) & 3) - dmin * (float) (sc >> 4);
        }
    } else if constexpr (T == T_Q3_K) {
        const uint8_t* p = row + (u >> 3) * 110;
        const int sub = (int) (u & 7), n = sub >> 2, j = sub & 3;
        const float d_all = rdh(p + 108);
        int scales[16];
        q3k_scales(p + 96, scales);
        const uint8_t* hm = p;
        const uint8_t* q = p + 32 + 32 * n;
        const int m = 1 << (4 * n + j);
        for (int l = 0; l < 32; ++l) {
            const float dl = d_all * (float) scales[8 * n + 2 * j + (l >> 4)];
            out[l] = dl * (float) (((q[l] >> (2 * j)) & 3) - ((hm[l] & m) ? 0 : 4));
        }
    } else if constexpr (T == T_IQ4_XS) {
        const uint8_t* p = row + (u >> 3) * 136;
        const int ib = (int) (u & 7);
        const float d = rdh(p);
        const int scales_h = rd16(p + 2);
        const int ls = ((p[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* qs = p + 8 + 16 * ib;
        for (int j = 0; j < 16; ++j) {
            out[j] = dl * (float) kv_iq4nl(qs[j] & 0xf);
            out[j + 16] = dl * (float) kv_iq4nl(qs[j] >> 4);
        }
    } else {
        for (int j = 0; j < 32; ++j) out[j] = 0.f;
    }
}

/// One scale of Q3_K's 16 (minus 32) from the 12 packed bytes: what q3k_scales computes, for index i only
G4_HD int q3k_scale(const uint8_t* s, int i) {
    if (i < 4) return ((s[i] & 0xF) | (((s[8 + i] >> 0) & 3) << 4)) - 32;
    if (i < 8) return ((s[i] & 0xF) | (((s[4 + i] >> 2) & 3) << 4)) - 32;           // (s[8 + (i - 4)])
    if (i < 12) return (((s[i - 8] >> 4) & 0xF) | (((s[i] >> 4) & 3) << 4)) - 32;    // (s[8 + (i - 8)])
    return (((s[i - 8] >> 4) & 0xF) | (((s[i - 4] >> 6) & 3) << 4)) - 32;            // (s[8 + (i - 12)])
}

/// Element `l` (0..31) of unit u: what dequant32 puts in out[l].  The GPU kernels use this form: the 32 lanes of a warp take the 32 values
/// of a unit, so every load of the weights and of the activations is coalesced and the rows are summed once, at the end.
template <int T>
G4_HD float dequant_elem(const uint8_t* row, long long u, int l) {
    if constexpr (T == T_F32) {
        float f;
        std::memcpy(&f, row + u * 128 + 4 * l, 4);
        return f;
    } else if constexpr (T == T_F16) {
        return half_to_float(rd16(row + u * 64 + 2 * l));
    } else if constexpr (T == T_BF16) {
        return bf16_to_float(rd16(row + u * 64 + 2 * l));
    } else if constexpr (T == T_Q8_0) {
        const uint8_t* p = row + u * 34;
        return (float) (int) (signed char) p[2 + l] * rdh(p);
    } else if constexpr (T == T_Q4_0) {
        const uint8_t* p = row + u * 18;
        const int b = p[2 + (l & 15)];
        return (float) ((l < 16 ? (b & 0x0F) : (b >> 4)) - 8) * rdh(p);
    } else if constexpr (T == T_Q4_1) {
        const uint8_t* p = row + u * 20;
        const int b = p[4 + (l & 15)];
        return (float) (l < 16 ? (b & 0x0F) : (b >> 4)) * rdh(p) + rdh(p + 2);
    } else if constexpr (T == T_Q5_0) {
        const uint8_t* p = row + u * 22;
        const uint32_t qh = (uint32_t) p[2] | ((uint32_t) p[3] << 8) | ((uint32_t) p[4] << 16) | ((uint32_t) p[5] << 24);
        const int b = p[6 + (l & 15)];
        const int xh = l < 16 ? (int) (((qh >> l) << 4) & 0x10) : (int) ((qh >> (l - 4)) & 0x10);
        return (float) (((l < 16 ? (b & 0x0F) : (b >> 4)) | xh) - 16) * rdh(p);
    } else if constexpr (T == T_Q5_1) {
        const uint8_t* p = row + u * 24;
        const uint32_t qh = (uint32_t) p[4] | ((uint32_t) p[5] << 8) | ((uint32_t) p[6] << 16) | ((uint32_t) p[7] << 24);
        const int b = p[8 + (l & 15)];
        const int xh = l < 16 ? (int) (((qh >> l) << 4) & 0x10) : (int) ((qh >> (l - 4)) & 0x10);
        return (float) ((l < 16 ? (b & 0x0F) : (b >> 4)) | xh) * rdh(p) + rdh(p + 2);
    } else if constexpr (T == T_IQ4_NL) {
        const uint8_t* p = row + u * 18;
        const int b = p[2 + (l & 15)];
        return rdh(p) * (float) kv_iq4nl(l < 16 ? (b & 0xf) : (b >> 4));
    } else if constexpr (T == T_Q4_K) {
        const uint8_t* p = row + (u >> 3) * 144;
        const int sub = (int) (u & 7);
        int sc, mn;
        scale_min_k4(sub, p + 4, sc, mn);
        const int q = (p[16 + 32 * (sub >> 1) + l] >> (4 * (sub & 1))) & 0xF;
        return rdh(p) * (float) sc * (float) q - rdh(p + 2) * (float) mn;
    } else if constexpr (T == T_Q5_K) {
        const uint8_t* p = row + (u >> 3) * 176;
        const int sub = (int) (u & 7);
        int sc, mn;
        scale_min_k4(sub, p + 4, sc, mn);
        const int q = ((p[48 + 32 * (sub >> 1) + l] >> (4 * (sub & 1))) & 0xF) + ((p[16 + l] & (1 << sub)) ? 16 : 0);
        return rdh(p) * (float) sc * (float) q - rdh(p + 2) * (float) mn;
    } else if constexpr (T == T_Q6_K) {
        const uint8_t* p = row + (u >> 3) * 210;
        const int sub = (int) (u & 7), n = sub >> 2, g = sub & 3;
        const int lo = p[64 * n + l + 32 * (g & 1)] >> (4 * (g >> 1));
        const int hi = (p[128 + 32 * n + l] >> (2 * g)) & 3;
        const int q = ((lo & 0xF) | (hi << 4)) - 32;
        return rdh(p + 208) * (float) (int) (signed char) p[192 + 8 * n + (l >> 4) + 2 * g] * (float) q;
    } else if constexpr (T == T_Q2_K) {
        const uint8_t* p = row + (u >> 3) * 84;
        const int sub = (int) (u & 7), n = sub >> 2, j = sub & 3;
        const int sc = p[8 * n + 2 * j + (l >> 4)];
        return rdh(p + 80) * (float) (sc & 0xF) * (float) ((p[16 + 32 * n + l] >> (2 * j)) & 3) - rdh(p + 82) * (float) (sc >> 4);
    } else if constexpr (T == T_Q3_K) {
        const uint8_t* p = row + (u >> 3) * 110;
        const int sub = (int) (u & 7), n = sub >> 2, j = sub & 3;
        const float dl = rdh(p + 108) * (float) q3k_scale(p + 96, 8 * n + 2 * j + (l >> 4));
        return dl * (float) (((p[32 + 32 * n + l] >> (2 * j)) & 3) - ((p[l] & (1 << (4 * n + j))) ? 0 : 4));
    } else if constexpr (T == T_IQ4_XS) {
        const uint8_t* p = row + (u >> 3) * 136;
        const int ib = (int) (u & 7);
        const int ls = ((p[4 + (ib >> 1)] >> (4 * (ib & 1))) & 0xf) | (((rd16(p + 2) >> (2 * ib)) & 3) << 4);
        const int b = p[8 + 16 * ib + (l & 15)];
        return rdh(p) * (float) (ls - 32) * (float) kv_iq4nl(l < 16 ? (b & 0xf) : (b >> 4));
    } else {
        return 0.f;
    }
}

/// the same with the type as a value (the host paths)
G4_HD void dequant32_t(int t, const uint8_t* row, long long u, float* out) {
    switch (t) {
        case T_F32: dequant32<T_F32>(row, u, out); break;
        case T_F16: dequant32<T_F16>(row, u, out); break;
        case T_BF16: dequant32<T_BF16>(row, u, out); break;
        case T_Q4_0: dequant32<T_Q4_0>(row, u, out); break;
        case T_Q4_1: dequant32<T_Q4_1>(row, u, out); break;
        case T_Q5_0: dequant32<T_Q5_0>(row, u, out); break;
        case T_Q5_1: dequant32<T_Q5_1>(row, u, out); break;
        case T_Q8_0: dequant32<T_Q8_0>(row, u, out); break;
        case T_Q2_K: dequant32<T_Q2_K>(row, u, out); break;
        case T_Q3_K: dequant32<T_Q3_K>(row, u, out); break;
        case T_Q4_K: dequant32<T_Q4_K>(row, u, out); break;
        case T_Q5_K: dequant32<T_Q5_K>(row, u, out); break;
        case T_Q6_K: dequant32<T_Q6_K>(row, u, out); break;
        case T_IQ4_NL: dequant32<T_IQ4_NL>(row, u, out); break;
        case T_IQ4_XS: dequant32<T_IQ4_XS>(row, u, out); break;
        default: for (int j = 0; j < 32; ++j) out[j] = 0.f;
    }
}

/// A whole row of n values (the host's embedding lookup, the reference's matrix-vector product)
inline void dequant_row(int t, const uint8_t* row, long long n, float* out) {
    for (long long u = 0; u < n / 32; ++u) dequant32_t(t, row, u, out + 32 * u);
}

}  // namespace g4
