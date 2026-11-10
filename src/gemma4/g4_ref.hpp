// g4_ref.hpp - the Gemma 4 forward pass on the CPU, written for clarity: one token at a time, float arithmetic, weights dequantized as they are read
// (g4_quant.hpp).  It is the engine's reference: tests/gemma4 compares it with llama.cpp (tools/gemma4/llama_ref.cpp), and the CUDA engine with
// it.  Slow (the real model: about a second a token and a half on four cores); `strata-gemma4 --cpu` serves with it when there is no GPU.
//
// The layer, as llama.cpp's llama_model_gemma4::graph builds it (src/models/gemma4.cpp):
//   h -> n = rms(h) * attn_norm;  q = Wq n;  k = Wk n;  v = Wv n (or k itself when the layer has no attn_v: V = K)
//        q, k: rms per head * q_norm / k_norm, then rope (NEOX pairs i, i + n_rot/2; the global layers divide the angles by rope_freqs);
//        v: rms per head, no weight, no rope.   Attention scale 1 (the norms carry the scale); a sliding-window layer sees the last `window` cells.
//        h1 = h + rms(Wo attn) * post_attention_norm
//   dense MLP: m = rms(Wdown(gelu(Wgate x) * Wup x)) * post_ffw_norm_1  with x = rms(h1) * ffn_norm        (when the layer has experts)
//   experts:   r = softmax(router . (rms(h1) / sqrt(n_embd) * router_scale)); the n_used best, weights renormalized; for each:
//              Wdown_e(gelu(gate_e y) * up_e y) * down_scale_e * weight_e  with y = rms(h1) * pre_ffw_norm_2;  summed; rms(.) * post_ffw_norm_2
//   h = h1 + rms(m + experts) * post_ffw_norm;  h *= layer_output_scale.
// Embeddings are scaled by sqrt(n_embd); the output is rms(h) * output_norm times the tied token_embd, then n * tanh(logit / n) (n = softcap).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "g4_model.hpp"

namespace g4 {

namespace refmath {
inline uint16_t float_to_half(float f) {      // round to nearest even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint16_t s = (uint16_t) ((u >> 16) & 0x8000u);
    const int e = (int) ((u >> 23) & 0xff) - 127 + 15;
    uint32_t m = u & 0x7fffffu;
    if (((u >> 23) & 0xff) == 0xff) return (uint16_t) (s | 0x7c00u | (m ? 0x200u : 0u));
    if (e >= 31) return (uint16_t) (s | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return s;
        m |= 0x800000u;
        const int sh = 14 - e;
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (s | r);
}
inline float gelu_exact(float x) {            // the tanh form (gelu_pytorch_tanh), as ggml_gelu_f32
    const float coef = 0.044715f, sqrt_2_over_pi = 0.79788456080286535588f;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + coef * x * x)));
}
/// ggml-cpu's GELU when built with GGML_GELU_FP16: a 64K-entry half table indexed by the half of x (zero below -10, x above 10)
inline float gelu_table(float x) {
    static const std::vector<uint16_t> table = [] {
        std::vector<uint16_t> t(65536);
        for (int i = 0; i < 65536; ++i) t[(size_t) i] = float_to_half(gelu_exact(half_to_float((uint16_t) i)));
        return t;
    }();
    if (x <= -10.0f) return 0.0f;
    if (x >= 10.0f) return x;
    return half_to_float(table[float_to_half(x)]);
}
}  // namespace refmath

struct RefOpts {
    bool gelu_table = false;     // ggml-cpu's half-table GELU (to compare with llama.cpp's CPU backend bit for bit); default: the exact tanh form
    int threads = 1;
    bool cache_f16 = false;      // keep K / V as halves (what the CUDA engine does) instead of floats
};

class RefEngine {
public:
    RefEngine(const Model& m, int ctx, RefOpts opt = RefOpts()) : m_(m), hp_(m.hp), ctx_(ctx), opt_(opt) {
        const int E = hp_.n_embd;
        kc_.resize((size_t) hp_.n_layer);
        vc_.resize((size_t) hp_.n_layer);
        for (int il = 0; il < hp_.n_layer; ++il) {
            const size_t n = (size_t) hp_.kv_heads(il) * (size_t) ctx_ * (size_t) hp_.head_dim(il);
            kc_[(size_t) il].assign(n, 0.f);
            vc_[(size_t) il].assign(n, 0.f);
        }
        h_.resize((size_t) E);
        if (const char* t = std::getenv("STRATA_G4_THREADS")) opt_.threads = std::max(1, std::atoi(t));
    }
    int ctx() const { return ctx_; }
    int64_t pos() const { return pos_; }
    void reset() { pos_ = 0; }
    /// keep the first p tokens (the cells past them are never read again: attention reads cells up to the current position)
    void truncate(int64_t p) { pos_ = std::min(pos_, std::max<int64_t>(0, p)); }

    /// Append `n` tokens at the current position.  `last`: the logits of the final token; `all`: those of every token, n * n_vocab floats.
    /// False when the context is full.
    bool forward(const int32_t* ids, int n, std::vector<float>* last, std::vector<float>* all = nullptr) {
        if (pos_ + n > ctx_) return false;
        if (all) all->assign((size_t) n * (size_t) hp_.n_vocab, 0.f);
        for (int t = 0; t < n; ++t) {
            const bool want = all != nullptr || (last != nullptr && t == n - 1);
            std::vector<float>* dst = nullptr;
            std::vector<float> tmp;
            if (want) {
                dst = &tmp;
            }
            token(ids[t], dst);
            if (all) std::memcpy(all->data() + (size_t) t * (size_t) hp_.n_vocab, tmp.data(), tmp.size() * 4);
            if (last && t == n - 1) *last = tmp;
            ++pos_;
        }
        return true;
    }

private:
    const Model& m_;
    const Hparams& hp_;
    int ctx_;
    RefOpts opt_;
    int64_t pos_ = 0;
    std::vector<std::vector<float>> kc_, vc_;
    std::vector<float> h_;

    float gelu(float x) const { return opt_.gelu_table ? refmath::gelu_table(x) : refmath::gelu_exact(x); }

    template <class F>
    void parallel_for(long long n, F fn) const {
        const int T = (int) std::min<long long>(opt_.threads, n);
        if (T <= 1) { for (long long i = 0; i < n; ++i) fn(i); return; }
        std::vector<std::thread> ts;
        for (int k = 0; k < T; ++k)
            ts.emplace_back([&, k] { for (long long i = k; i < n; i += T) fn(i); });
        for (auto& t : ts) t.join();
    }

    /// y[r] = row r of the matrix (expert e) . x, for r in [0, n_out)
    void matvec(const Mat& w, long long e, const float* x, float* y) const {
        parallel_for(w.n_out, [&](long long r) {
            const uint8_t* row = w.row(r, e);
            float tmp[32];
            float acc = 0.f;
            for (long long u = 0; u < w.n_in / 32; ++u) {
                dequant32_t(w.type, row, u, tmp);
                float a = 0.f;
                for (int j = 0; j < 32; ++j) a += tmp[j] * x[32 * u + j];
                acc += a;
            }
            y[r] = acc;
        });
    }

    static void rms(const float* x, int n, float eps, const float* w, float* y) {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double) (x[i] * x[i]);
        const float scale = 1.0f / sqrtf((float) (ss / n) + eps);
        for (int i = 0; i < n; ++i) y[i] = w ? x[i] * scale * w[i] : x[i] * scale;
    }

    /// NEOX rope on the first n_rot dims of a head: angle i = pos * base^(-2 i / n_rot) / ff[i]
    void rope(float* x, int n_rot, float base, const float* ff, int64_t pos) const {
        const int half = n_rot / 2;
        for (int i = 0; i < half; ++i) {
            float theta = (float) pos * powf(base, -2.0f * (float) i / (float) n_rot);
            if (ff) theta /= ff[i];
            const float c = cosf(theta), s = sinf(theta);
            const float x0 = x[i], x1 = x[i + half];
            x[i] = x0 * c - x1 * s;
            x[i + half] = x0 * s + x1 * c;
        }
    }

    void token(int32_t id, std::vector<float>* logits) {
        const int E = hp_.n_embd, nh = hp_.n_head;
        std::vector<float> h((size_t) E), n((size_t) E), tmp((size_t) E);
        // embedding: the row of token_embd, scaled by sqrt(n_embd)
        {
            std::vector<float> row((size_t) E);
            dequant_row(m_.tok_embd.type, m_.tok_embd.row(id), E, row.data());
            const float sc = sqrtf((float) E);
            for (int i = 0; i < E; ++i) h[(size_t) i] = row[(size_t) i] * sc;
        }
        for (int il = 0; il < hp_.n_layer; ++il) {
            const LayerW& w = m_.L[(size_t) il];
            const int hd = hp_.head_dim(il), nkv = hp_.kv_heads(il), group = nh / nkv;
            const bool swa = hp_.swa[(size_t) il] != 0;
            const float* ff = swa ? nullptr : m_.rope_freqs;
            rms(h.data(), E, hp_.eps, w.attn_norm, n.data());
            std::vector<float> q((size_t) nh * hd), k((size_t) nkv * hd), v((size_t) nkv * hd);
            matvec(w.wq, 0, n.data(), q.data());
            matvec(w.wk, 0, n.data(), k.data());
            if (w.has_v) matvec(w.wv, 0, n.data(), v.data());
            else v = k;                                         // no attn_v: V is the raw K projection
            for (int hh = 0; hh < nh; ++hh) {
                float* qh = q.data() + (size_t) hh * hd;
                rms(qh, hd, hp_.eps, w.q_norm, qh);
                rope(qh, hp_.rot(il), hp_.base(il), ff, pos_);
            }
            for (int kk = 0; kk < nkv; ++kk) {
                float* kh = k.data() + (size_t) kk * hd;
                float* vh = v.data() + (size_t) kk * hd;
                rms(kh, hd, hp_.eps, w.k_norm, kh);
                rope(kh, hp_.rot(il), hp_.base(il), ff, pos_);
                rms(vh, hd, hp_.eps, nullptr, vh);
                float* kdst = kc_[(size_t) il].data() + ((size_t) kk * ctx_ + (size_t) pos_) * hd;
                float* vdst = vc_[(size_t) il].data() + ((size_t) kk * ctx_ + (size_t) pos_) * hd;
                for (int d = 0; d < hd; ++d) {
                    kdst[d] = opt_.cache_f16 ? half_to_float(refmath::float_to_half(kh[d])) : kh[d];
                    vdst[d] = opt_.cache_f16 ? half_to_float(refmath::float_to_half(vh[d])) : vh[d];
                }
            }
            // attention
            const int64_t lo = swa ? std::max<int64_t>(0, pos_ - hp_.window + 1) : 0;
            std::vector<float> ao((size_t) nh * hd, 0.f), sc((size_t) pos_ + 1);
            for (int hh = 0; hh < nh; ++hh) {
                const int kk = hh / group;
                const float* qh = q.data() + (size_t) hh * hd;
                const float* kbase = kc_[(size_t) il].data() + (size_t) kk * ctx_ * hd;
                const float* vbase = vc_[(size_t) il].data() + (size_t) kk * ctx_ * hd;
                float mx = -INFINITY;
                for (int64_t j = lo; j <= pos_; ++j) {
                    float s = 0.f;
                    for (int d = 0; d < hd; ++d) s += qh[d] * kbase[(size_t) j * hd + d];
                    sc[(size_t) j] = s;                          // attention scale 1
                    mx = std::max(mx, s);
                }
                double sum = 0.0;
                for (int64_t j = lo; j <= pos_; ++j) { sc[(size_t) j] = expf(sc[(size_t) j] - mx); sum += sc[(size_t) j]; }
                const float inv = (float) (1.0 / sum);
                float* oh = ao.data() + (size_t) hh * hd;
                for (int64_t j = lo; j <= pos_; ++j) {
                    const float p = sc[(size_t) j] * inv;
                    for (int d = 0; d < hd; ++d) oh[d] += p * vbase[(size_t) j * hd + d];
                }
            }
            std::vector<float> proj((size_t) E);
            matvec(w.wo, 0, ao.data(), proj.data());
            rms(proj.data(), E, hp_.eps, w.post_attn_norm, proj.data());
            std::vector<float> h1((size_t) E);
            for (int i = 0; i < E; ++i) h1[(size_t) i] = h[(size_t) i] + proj[(size_t) i];
            // feed-forward
            std::vector<float> g((size_t) hp_.n_ff), u((size_t) hp_.n_ff), mlp((size_t) E);
            rms(h1.data(), E, hp_.eps, w.ffn_norm, n.data());
            matvec(w.ffn_gate, 0, n.data(), g.data());
            matvec(w.ffn_up, 0, n.data(), u.data());
            for (int i = 0; i < hp_.n_ff; ++i) g[(size_t) i] = gelu(g[(size_t) i]) * u[(size_t) i];
            matvec(w.ffn_down, 0, g.data(), mlp.data());
            std::vector<float> cur((size_t) E);
            if (w.has_moe) {
                rms(mlp.data(), E, hp_.eps, w.post_ffw_norm_1, mlp.data());
                // router
                std::vector<float> r((size_t) E), logit((size_t) hp_.n_expert);
                rms(h1.data(), E, hp_.eps, nullptr, r.data());
                const float inv_sqrt = 1.0f / sqrtf((float) E);
                for (int i = 0; i < E; ++i) r[(size_t) i] = r[(size_t) i] * inv_sqrt * w.router_scale[i];
                for (int e = 0; e < hp_.n_expert; ++e) {
                    float s = 0.f;
                    for (int i = 0; i < E; ++i) s += w.router[(size_t) e * E + i] * r[(size_t) i];
                    logit[(size_t) e] = s;
                }
                float mx = *std::max_element(logit.begin(), logit.end());
                double sum = 0.0;
                std::vector<float> prob((size_t) hp_.n_expert);
                for (int e = 0; e < hp_.n_expert; ++e) { prob[(size_t) e] = expf(logit[(size_t) e] - mx); sum += prob[(size_t) e]; }
                for (int e = 0; e < hp_.n_expert; ++e) prob[(size_t) e] = (float) (prob[(size_t) e] / sum);
                std::vector<int> order((size_t) hp_.n_expert);
                for (int e = 0; e < hp_.n_expert; ++e) order[(size_t) e] = e;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return prob[(size_t) a] > prob[(size_t) b]; });
                float wsum = 0.f;
                for (int k2 = 0; k2 < hp_.n_used; ++k2) wsum += prob[(size_t) order[(size_t) k2]];
                wsum = std::max(wsum, 6.103515625e-5f);
                std::vector<float> y((size_t) E), moe((size_t) E, 0.f), gu((size_t) 2 * hp_.n_ff_exp), act((size_t) hp_.n_ff_exp), d((size_t) E);
                rms(h1.data(), E, hp_.eps, w.pre_ffw_norm_2, y.data());
                for (int k2 = 0; k2 < hp_.n_used; ++k2) {
                    const int e = order[(size_t) k2];
                    const float wt = prob[(size_t) e] / wsum;
                    if (w.gate_up_exps) {
                        matvec(w.gate_up_exps, e, y.data(), gu.data());
                    } else {
                        matvec(w.gate_exps, e, y.data(), gu.data());
                        matvec(w.up_exps, e, y.data(), gu.data() + hp_.n_ff_exp);
                    }
                    for (int i = 0; i < hp_.n_ff_exp; ++i) act[(size_t) i] = gelu(gu[(size_t) i]) * gu[(size_t) hp_.n_ff_exp + (size_t) i];
                    matvec(w.down_exps, e, act.data(), d.data());
                    const float sc2 = w.down_scale ? w.down_scale[e] : 1.f;
                    for (int i = 0; i < E; ++i) moe[(size_t) i] += d[(size_t) i] * sc2 * wt;
                }
                rms(moe.data(), E, hp_.eps, w.post_ffw_norm_2, moe.data());
                for (int i = 0; i < E; ++i) cur[(size_t) i] = mlp[(size_t) i] + moe[(size_t) i];
            } else {
                cur = mlp;
            }
            rms(cur.data(), E, hp_.eps, w.post_ffw_norm, cur.data());
            for (int i = 0; i < E; ++i) h[(size_t) i] = (cur[(size_t) i] + h1[(size_t) i]) * w.out_scale;
        }
        if (!logits) return;
        rms(h.data(), E, hp_.eps, m_.output_norm, tmp.data());
        logits->assign((size_t) hp_.n_vocab, 0.f);
        parallel_for(hp_.n_vocab, [&](long long r) {
            const uint8_t* row = m_.tok_embd.row(r);
            float t32[32];
            float acc = 0.f;
            for (long long u2 = 0; u2 < E / 32; ++u2) {
                dequant32_t(m_.tok_embd.type, row, u2, t32);
                float a = 0.f;
                for (int j = 0; j < 32; ++j) a += t32[j] * tmp[(size_t) (32 * u2 + j)];
                acc += a;
            }
            (*logits)[(size_t) r] = acc;
        });
        if (hp_.softcap > 0.f)
            for (float& x : *logits) x = hp_.softcap * tanhf(x / hp_.softcap);
    }
};

}  // namespace g4
