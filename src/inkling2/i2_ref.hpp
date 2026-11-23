// i2_ref.hpp - the Inkling 2 Mini forward pass on the CPU, written for clarity: one token at a time, float arithmetic, weights dequantized as they
// are read.  The engine's reference: tools/inkling2 compares it with llama.cpp (llama_ref), and the CUDA engine with it.  Slow; not for serving.
//
// One token, as llama.cpp's llama_model_inkling2_loop::graph builds it (src/models/inkling2.cpp, inkling2-loop.cpp):
//   cur = rms(embd[id]) * token_embd_norm;  blocks = []
//   visits: layers 0 .. start-1; then `passes` times layers start .. end-1 (pass 0 from the prelude output, every later pass again from the
//   prelude output with cross-pass messages); then layers end .. n_layer-1.  Every visit has its own K / V cache and convolution history.
//   layer(ip, cur):
//     inputs[0..2] = cur, or (layers > 0) mix(blocks, cur, attn_res_query) : one mixed vector for a local layer, three (q, k, v) for a global one
//     pass >= 1: inputs[j] = control(inputs[j], message_j)
//     local layer: n_j = rms(inputs[j]) * attn_norm;  global: rms(inputs[0]) * q_inp_norm, rms(inputs[1]) * k_inp_norm, rms(inputs[2]) * v_inp_norm
//     q = Wq n0;  k = conv(Wk n1) (K history 0);  v = conv(Wv n2) (V history 1);  r = Wrel n0 (n_rel per head)
//       conv(x) = (x + sum_k w[k] x_{t-3+k} [+ sum_lag tap[lag] x_{t-lag}, global layers]) / sqrt(n_conv (2 if taps) + 1)
//     q = clip(q * q_clip, qk_clip), k = clip(k, qk_clip), v = rms(v) (no weight), r = clip(r * rel_clip, rel_clip_threshold)
//       clip(x, T) = x * min(1, T / (|x| + 1e-6)) per head vector;  global layers: q, r *= 1 + alpha log(max(1, (pos + 1) / floor))
//     logit_j = q.k_j / sqrt(hd) + (bias_table(d) . r) - sink,  d = pos - pos_j (the bias row of d >= extent is zero), keys j <= pos, local: d < window
//     weights = exp(logit_j) / (1 + sum exp(logit))   (the sink: one more key with logit 0 and no value)
//     a = (sum_j w_j v_j) * sigmoid(Wgate n0);  cur = conv(Wo a) (history 2)
//     the first layer of a block of `block` (ip % block == 0): blocks += [inpSA]; else cur = inpSA + cur.   ffn_inp = cur
//     x = mix(blocks, ffn_inp, ffn_res_query); (pass >= 1: control with the ffn message);  x = rms(x) * ffn_norm
//     experts: p = sigmoid(router . x); top n_used of p + bias; w = p / (sum + 1e-20) * route_scale;
//              sum_e w_e down_e(situ(gate_e x, up_e x)),  situ(g, u) = beta tanh(g / beta) sigmoid(g) * lb tanh(u / lb)
//     cur = ffn_inp + conv(experts) (history 3)
//   mix(blocks, cur, queries) = for each query: softmax_s(rms(state_s) . query / sqrt(n_embd)) weighting the rms-normed states (blocks, then cur)
//   control(cur, m) = cur - tanh(forget(g)) cur + tanh(gain(g)) m,  g = down([rms(cur), rms(m)])
//   pass >= 1 messages for layer ip: mix(bank, bank_out, msg_query[ip]) (4 queries: q, k, v, ffn): the previous pass's blocks after the first core
//   block, and its output.
//   output: x = rms(mix(blocks, cur, output_res_query)) * output_norm;  logits = softcap * tanh(output x / softcap), the first n_vocab rows.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "i2_model.hpp"

namespace i2 {

struct RefOpts {
    int threads = 1;
    bool cache_f16 = false;      // keep K / V as halves (what the CUDA engine does) instead of floats
    int kvq = 0;                 // the global layers' K / V as the CUDA engine's --kv: 1 q8_0 (blocks of 32, half scale amax / 127), 2 4 bits (amax / 7); 0 as cache_f16 says
};

class RefEngine {
public:
    RefEngine(const Model& m, int ctx, RefOpts opt = RefOpts()) : m_(m), hp_(m.hp), ctx_(ctx), opt_(opt) {
        const int V = hp_.visits();
        kc_.resize((size_t) V);
        vc_.resize((size_t) V);
        ci_.resize((size_t) V);
        for (int v = 0; v < V; ++v) {
            kc_[(size_t) v].assign((size_t) hp_.n_kv * (size_t) ctx_ * (size_t) hp_.hd, 0.f);
            vc_[(size_t) v].assign((size_t) hp_.n_kv * (size_t) ctx_ * (size_t) hp_.hd, 0.f);
            ci_[(size_t) v].assign((size_t) ctx_ * (size_t) hp_.conv_channels(), 0.f);
        }
        if (const char* t = std::getenv("STRATA_I2_THREADS")) opt_.threads = std::max(1, std::atoi(t));
    }
    int ctx() const { return ctx_; }
    int64_t pos() const { return pos_; }
    void reset() { pos_ = 0; }
    /// keep the first p tokens (K / V and the convolution inputs are stored per position, so a step back is free)
    void truncate(int64_t p) { pos_ = std::min(pos_, std::max<int64_t>(0, p)); }
    /// `passes` override (the variable-loop checkpoint runs 1 .. passes; a fixed loop keeps its own)
    void set_passes(int p) { passes_ = p; }

    /// Append `n` tokens.  `last`: the logits of the final token; `all`: those of every token, n * n_vocab floats.  False when the context is full.
    bool forward(const int32_t* ids, int n, std::vector<float>* last, std::vector<float>* all = nullptr) {
        if (pos_ + n > ctx_) return false;
        if (all) all->assign((size_t) n * (size_t) hp_.n_vocab, 0.f);
        for (int t = 0; t < n; ++t) {
            const bool want = all != nullptr || (last != nullptr && t == n - 1);
            std::vector<float> tmp;
            token(ids[t], want ? &tmp : nullptr);
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
public:
    std::vector<std::vector<int>>* sel_trace = nullptr;
    std::vector<std::vector<float>>* trace = nullptr;     // test hook: the stream after every visit, token by token
private:
    int passes_ = 0;
    int64_t pos_ = 0;
    std::vector<std::vector<float>> kc_, vc_, ci_;       // per visit: K / V [kv head][ctx][hd]; convolution inputs [pos][channels]

    int passes() const { return passes_ > 0 ? passes_ : hp_.passes; }

    template <class F>
    void parallel_for(long long n, F fn) const {
        const int T = (int) std::min<long long>(opt_.threads, n);
        if (T <= 1) { for (long long i = 0; i < n; ++i) fn(i); return; }
        std::vector<std::thread> ts;
        for (int k = 0; k < T; ++k)
            ts.emplace_back([&, k] { for (long long i = k; i < n; i += T) fn(i); });
        for (auto& t : ts) t.join();
    }

    /// y[r] = row r of the matrix (expert e) . x, for r in [0, rows)
    void matvec(const Mat& w, long long e, const float* x, float* y, long long rows = -1) const {
        if (rows < 0) rows = w.n_out;
        parallel_for(rows, [&](long long r) {
            const uint8_t* row = w.row(r, e);
            float tmp[32];
            float acc = 0.f;
            for (long long u = 0; u < w.n_in / 32; ++u) {
                g4::dequant32_t(w.type, row, u, tmp);
                float a = 0.f;
                for (int j = 0; j < 32; ++j) a += tmp[j] * x[32 * u + j];
                acc += a;
            }
            y[r] = acc;
        });
    }
    /// y = W x for a float matrix [n_out][n_in]
    static void matvec_f(const float* w, long long n_out, long long n_in, const float* x, float* y) {
        for (long long r = 0; r < n_out; ++r) {
            float acc = 0.f;
            for (long long i = 0; i < n_in; ++i) acc += w[(size_t) (r * n_in + i)] * x[i];
            y[r] = acc;
        }
    }
    static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

    static void rms(const float* x, int n, float eps, const float* w, float* y) {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double) (x[i] * x[i]);
        const float scale = 1.0f / sqrtf((float) (ss / n) + eps);
        for (int i = 0; i < n; ++i) y[i] = w ? x[i] * scale * w[i] : x[i] * scale;
    }
    /// x * min(1, thr / (|x| + 1e-6)), x scaled per channel first when `scale`
    static void clip(float* x, int n, const float* scale, float thr) {
        if (scale)
            for (int i = 0; i < n; ++i) x[i] *= scale[i];
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double) (x[i] * x[i]);
        const float norm = sqrtf((float) ss) + 1e-6f;
        const float f = std::min(1.0f, std::max(0.0f, thr / norm));
        for (int i = 0; i < n; ++i) x[i] *= f;
    }

    /// attention over the depth: the rms-normed states (blocks, then cur) weighted by a softmax of their products with each query
    void mix(const std::vector<std::vector<float>>& blocks, const std::vector<float>& cur, const float* queries, int nq, std::vector<float>& out) const {
        const int E = hp_.n_embd;
        const size_t S = blocks.size() + 1;
        std::vector<std::vector<float>> st(S, std::vector<float>((size_t) E));
        for (size_t s = 0; s < blocks.size(); ++s) rms(blocks[s].data(), E, 1e-6f, nullptr, st[s].data());
        rms(cur.data(), E, 1e-6f, nullptr, st[S - 1].data());
        out.assign((size_t) nq * (size_t) E, 0.f);
        const float scale = 1.0f / sqrtf((float) E);
        std::vector<float> sc(S);
        for (int q = 0; q < nq; ++q) {
            const float* qv = queries + (size_t) q * (size_t) E;
            float mx = -INFINITY;
            for (size_t s = 0; s < S; ++s) {
                float a = 0.f;
                for (int e = 0; e < E; ++e) a += st[s][(size_t) e] * qv[e];
                sc[s] = a * scale;
                mx = std::max(mx, sc[s]);
            }
            float sum = 0.f;
            for (size_t s = 0; s < S; ++s) { sc[s] = expf(sc[s] - mx); sum += sc[s]; }
            float* o = out.data() + (size_t) q * (size_t) E;
            for (size_t s = 0; s < S; ++s) {
                const float wgt = sc[s] / sum;
                for (int e = 0; e < E; ++e) o[e] += wgt * st[s][(size_t) e];
            }
        }
    }

    /// the cross-pass controller: cur - forget cur + gain m
    void control(std::vector<float>& cur, const float* m, const LoopW& lw, int c) const {
        const int E = hp_.n_embd, rk = hp_.gate_rank;
        std::vector<float> g((size_t) 2 * (size_t) E), gate((size_t) rk), ga((size_t) E), fo((size_t) E);
        rms(cur.data(), E, 1e-6f, nullptr, g.data());
        rms(m, E, 1e-6f, nullptr, g.data() + E);
        matvec_f(lw.down[c], rk, 2 * E, g.data(), gate.data());
        matvec_f(lw.gain[c], E, rk, gate.data(), ga.data());
        matvec_f(lw.forget[c], E, rk, gate.data(), fo.data());
        for (int i = 0; i < E; ++i) cur[(size_t) i] = cur[(size_t) i] - tanhf(fo[(size_t) i]) * cur[(size_t) i] + tanhf(ga[(size_t) i]) * m[i];
    }

    /// the causal short convolution with a residual, on the channels [off, off + n) of the visit's stored inputs; `taps`: per-head input taps (global K / V)
    void convolve(int visit, int off, const float* in, int n, const float* w, const float* taps, int n_heads, std::vector<float>& out) {
        const int C = hp_.n_conv, ch = hp_.conv_channels();
        std::vector<float>& ci = ci_[(size_t) visit];
        float* cur = ci.data() + (size_t) pos_ * (size_t) ch + (size_t) off;
        std::memcpy(cur, in, (size_t) n * 4);
        out.assign((size_t) n, 0.f);
        const int per_head = taps ? n / n_heads : 1;
        for (int c = 0; c < n; ++c) {
            float acc = 0.f;
            for (int k = 0; k < C; ++k) {
                const int64_t p = pos_ - (C - 1) + k;
                const float xv = p >= 0 ? ci[(size_t) p * (size_t) ch + (size_t) off + (size_t) c] : 0.f;
                acc += w[(size_t) c * (size_t) C + (size_t) k] * xv;
            }
            if (taps) {
                const int h = c / per_head;
                for (int lag = 0; lag < C; ++lag) {
                    const int64_t p = pos_ - lag;
                    const float xv = p >= 0 ? ci[(size_t) p * (size_t) ch + (size_t) off + (size_t) c] : 0.f;
                    acc += taps[(size_t) h * (size_t) C + (size_t) lag] * xv;
                }
            }
            out[(size_t) c] = (in[c] + acc) / sqrtf((float) ((taps ? 2 : 1) * C + 1));
        }
    }

    /// one layer visit; `msg`: the four messages of a later pass (null: none); returns the new cur, may append to `blocks`
    void layer(int visit, int ip, std::vector<float>& cur, std::vector<std::vector<float>>& blocks, const std::vector<float>* msg, const LoopW* lw) {
        const LayerW& w = m_.L[(size_t) ip];
        const int E = hp_.n_embd, H = hp_.n_head, KV = hp_.n_kv, hd = hp_.hd, kd = KV * hd, qd = H * hd, R = hp_.n_rel, group = H / KV;
        const bool swa = hp_.swa[(size_t) ip] != 0;
        const int ch = hp_.conv_channels();
        (void) ch;
        std::vector<float> inpSA = cur;
        std::vector<float> in[3] = {cur, cur, cur};
        if (w.attn_res_query) {
            std::vector<float> mixed;
            mix(blocks, cur, w.attn_res_query, swa ? 1 : 3, mixed);
            for (int j = 0; j < 3; ++j) in[j].assign(mixed.begin() + (size_t) (swa ? 0 : j) * (size_t) E, mixed.begin() + (size_t) ((swa ? 0 : j) + 1) * (size_t) E);
        }
        if (msg)
            for (int j = 0; j < 3; ++j) control(in[j], msg->data() + (size_t) j * (size_t) E, *lw, j);
        std::vector<float> n[3] = {std::vector<float>((size_t) E), std::vector<float>((size_t) E), std::vector<float>((size_t) E)};
        if (swa) {
            for (int j = 0; j < 3; ++j) rms(in[j].data(), E, hp_.eps, w.attn_norm, n[j].data());
        } else {
            rms(in[0].data(), E, hp_.eps, w.q_inp_norm, n[0].data());
            rms(in[1].data(), E, hp_.eps, w.k_inp_norm, n[1].data());
            rms(in[2].data(), E, hp_.eps, w.v_inp_norm, n[2].data());
        }
        // projections
        std::vector<float> q((size_t) qd), kr((size_t) kd), vr((size_t) kd), r((size_t) R * (size_t) H), sink((size_t) H), gate((size_t) qd);
        matvec(w.wq, 0, n[0].data(), q.data());
        matvec(w.wk, 0, n[1].data(), kr.data());
        matvec(w.wv, 0, n[2].data(), vr.data());
        matvec(w.wrel, 0, n[0].data(), r.data());
        matvec(w.wsink, 0, n[0].data(), sink.data());
        matvec(w.wgate, 0, n[0].data(), gate.data());
        std::vector<float> ktap, vtap;
        if (!swa) {
            ktap.resize((size_t) hp_.n_conv * (size_t) KV);
            vtap.resize((size_t) hp_.n_conv * (size_t) KV);
            matvec(w.kconv_proj, 0, n[1].data(), ktap.data());
            matvec(w.vconv_proj, 0, n[2].data(), vtap.data());
        }
        std::vector<float> k, v;
        convolve(visit, 0, kr.data(), kd, w.kconv, swa ? nullptr : ktap.data(), KV, k);
        convolve(visit, kd, vr.data(), kd, w.vconv, swa ? nullptr : vtap.data(), KV, v);
        for (int h = 0; h < H; ++h) clip(q.data() + (size_t) h * hd, hd, w.q_clip + (size_t) h * hd, hp_.qk_clip);
        for (int h = 0; h < KV; ++h) {
            clip(k.data() + (size_t) h * hd, hd, nullptr, hp_.qk_clip);
            rms(v.data() + (size_t) h * hd, hd, hp_.eps, nullptr, v.data() + (size_t) h * hd);
        }
        for (int h = 0; h < H; ++h) clip(r.data() + (size_t) h * R, R, w.rel_clip + (size_t) h * R, hp_.rel_clip);
        if (!swa) {
            const float tau = 1.0f + hp_.log_alpha * logf(std::max(1.0f, (float) (pos_ + 1) / hp_.log_floor));
            for (auto& x : q) x *= tau;
            for (auto& x : r) x *= tau;
        }
        // cache the new K / V
        std::vector<float>& kc = kc_[(size_t) visit];
        std::vector<float>& vc = vc_[(size_t) visit];
        if (opt_.kvq > 0 && !swa) {                                  // quantized and back, as the engine's cache holds them
            auto qdq = [&](std::vector<float>& a) {
                const float qmax = opt_.kvq == 1 ? 127.f : 7.f;
                for (size_t b0 = 0; b0 + 32 <= a.size(); b0 += 32) {
                    float amax = 0.f;
                    for (size_t i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(a[b0 + i]));
                    const float d = g4::half_to_float(g4::float_to_half(amax / qmax));
                    for (size_t i = 0; i < 32; ++i) {
                        int qi = d > 0.f ? (int) std::nearbyint(a[b0 + i] / d) : 0;
                        qi = opt_.kvq == 1 ? std::max(-127, std::min(127, qi)) : std::max(-8, std::min(7, qi));
                        a[b0 + i] = (float) qi * d;
                    }
                }
            };
            qdq(k);
            qdq(v);
        } else if (opt_.cache_f16) {
            for (auto& x : k) x = g4::half_to_float(g4::float_to_half(x));
            for (auto& x : v) x = g4::half_to_float(g4::float_to_half(x));
        }
        for (int h = 0; h < KV; ++h) {
            std::memcpy(kc.data() + ((size_t) h * (size_t) ctx_ + (size_t) pos_) * (size_t) hd, k.data() + (size_t) h * hd, (size_t) hd * 4);
            std::memcpy(vc.data() + ((size_t) h * (size_t) ctx_ + (size_t) pos_) * (size_t) hd, v.data() + (size_t) h * hd, (size_t) hd * 4);
        }
        // attention
        const int extent = swa ? hp_.window : hp_.rel_extent;
        const int64_t first = swa ? std::max<int64_t>(0, pos_ - hp_.window + 1) : 0;
        const float scale = 1.0f / sqrtf((float) hd);
        std::vector<float> att((size_t) qd, 0.f);
        parallel_for(H, [&](long long h) {
            const int kvh = (int) h / group;
            const float* qh = q.data() + (size_t) h * hd;
            const float* rh = r.data() + (size_t) h * R;
            std::vector<float> lg((size_t) (pos_ - first + 1));
            float mx = 0.f;                                    // the sink's logit is 0
            for (int64_t j = first; j <= pos_; ++j) {
                const float* kj = kc.data() + ((size_t) kvh * (size_t) ctx_ + (size_t) j) * (size_t) hd;
                float s = 0.f;
                for (int d = 0; d < hd; ++d) s += qh[d] * kj[d];
                const int64_t dist = pos_ - j;
                float b = 0.f;
                if (dist < extent) {
                    const float* row = w.rel_bias + (size_t) dist * (size_t) R;
                    for (int d = 0; d < R; ++d) b += row[d] * rh[d];
                }
                lg[(size_t) (j - first)] = s * scale + b - sink[(size_t) h];
                mx = std::max(mx, lg[(size_t) (j - first)]);
            }
            float sum = expf(0.0f - mx);
            for (auto& x : lg) { x = expf(x - mx); sum += x; }
            float* o = att.data() + (size_t) h * hd;
            for (int64_t j = first; j <= pos_; ++j) {
                const float wgt = lg[(size_t) (j - first)] / sum;
                const float* vj = vc.data() + ((size_t) kvh * (size_t) ctx_ + (size_t) j) * (size_t) hd;
                for (int d = 0; d < hd; ++d) o[d] += wgt * vj[d];
            }
        });
        for (int i = 0; i < qd; ++i) att[(size_t) i] *= sigmoid(gate[(size_t) i]);
        std::vector<float> ao((size_t) E);
        matvec(w.wo, 0, att.data(), ao.data());
        std::vector<float> aoc;
        convolve(visit, 2 * kd, ao.data(), E, w.post_conv, nullptr, 1, aoc);
        // residual stream
        if (ip % hp_.block == 0) {
            blocks.push_back(inpSA);
            cur = aoc;
        } else {
            for (int i = 0; i < E; ++i) cur[(size_t) i] = inpSA[(size_t) i] + aoc[(size_t) i];
        }
        const std::vector<float> ffn_inp = cur;
        // experts
        std::vector<float> x;
        mix(blocks, ffn_inp, w.ffn_res_query, 1, x);
        if (msg) control(x, msg->data() + (size_t) 3 * (size_t) E, *lw, 3);
        std::vector<float> xn((size_t) E);
        rms(x.data(), E, hp_.eps, w.ffn_norm, xn.data());
        const int NE = hp_.n_expert, NU = hp_.n_used, F = hp_.n_ff_exp;
        std::vector<float> probs((size_t) NE), sel((size_t) NE);
        matvec_f(w.router, NE, E, xn.data(), probs.data());
        for (int e = 0; e < NE; ++e) {
            probs[(size_t) e] = sigmoid(probs[(size_t) e]);
            sel[(size_t) e] = probs[(size_t) e] + (w.exp_bias ? w.exp_bias[e] : 0.f);
        }
        std::vector<int> order((size_t) NE);
        for (int e = 0; e < NE; ++e) order[(size_t) e] = e;
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return sel[(size_t) a] > sel[(size_t) b]; });
        if (sel_trace) sel_trace->push_back(std::vector<int>(order.begin(), order.begin() + NU));
        float wsum = 0.f;
        for (int i = 0; i < NU; ++i) wsum += probs[(size_t) order[(size_t) i]];
        wsum += 1e-20f;
        std::vector<float> moe((size_t) E, 0.f), gu((size_t) 2 * (size_t) F), act((size_t) F), y((size_t) E);
        for (int i = 0; i < NU; ++i) {
            const int e = order[(size_t) i];
            const float wt = probs[(size_t) e] / wsum * hp_.route_scale;
            matvec(w.gate_up, e, xn.data(), gu.data());
            if (w.gu_scale) for (float& v : gu) v *= w.gu_scale[e];
            for (int j = 0; j < F; ++j) {
                const float g = gu[(size_t) j], u = gu[(size_t) F + (size_t) j];
                const float a = hp_.situ_beta * tanhf(g / hp_.situ_beta) * sigmoid(g);
                act[(size_t) j] = a * (hp_.situ_lin * tanhf(u / hp_.situ_lin));
            }
            matvec(w.down, e, act.data(), y.data());
            const float dsc = w.dn_scale ? w.dn_scale[e] : 1.f;
            for (int j = 0; j < E; ++j) moe[(size_t) j] += wt * dsc * y[(size_t) j];
        }
        std::vector<float> fo;
        convolve(visit, 2 * kd + E, moe.data(), E, w.ffn_post_conv, nullptr, 1, fo);
        for (int i = 0; i < E; ++i) cur[(size_t) i] = ffn_inp[(size_t) i] + fo[(size_t) i];
        if (trace) trace->push_back(cur);
    }

    void token(int32_t id, std::vector<float>* logits) {
        const int E = hp_.n_embd, S = hp_.loop_start, Ee = hp_.loop_end;
        std::vector<float> cur((size_t) E);
        {
            std::vector<float> row((size_t) E);
            dequant_row(m_.tok_embd.type, m_.tok_embd.row(id), E, row.data());
            rms(row.data(), E, hp_.eps, m_.tok_norm, cur.data());
        }
        std::vector<std::vector<float>> blocks;
        const int np = hp_.has_loop() ? passes() : 1;
        const int core0 = hp_.has_loop() ? S : hp_.n_layer, core1 = hp_.has_loop() ? Ee : hp_.n_layer;
        for (int il = 0; il < core0; ++il) layer(il, il, cur, blocks, nullptr, nullptr);
        if (hp_.has_loop()) {
            const std::vector<float> prelude = cur;
            const std::vector<std::vector<float>> prelude_blocks = blocks;
            std::vector<std::vector<float>> bank;
            std::vector<float> bank_out;
            for (int pass = 0; pass < np; ++pass) {
                if (pass > 0) {
                    bank.assign(blocks.begin() + (long) prelude_blocks.size() + 1, blocks.end());
                    bank_out = cur;
                    cur = prelude;
                    blocks = prelude_blocks;
                }
                for (int ip = S; ip < Ee; ++ip) {
                    const int visit = pass == 0 ? ip : hp_.n_layer + (pass - 1) * (Ee - S) + (ip - S);
                    if (pass == 0) { layer(visit, ip, cur, blocks, nullptr, nullptr); continue; }
                    const LoopW& lw = m_.loop[(size_t) (ip - S)];
                    std::vector<float> msg;
                    mix(bank, bank_out, lw.msg_query, 4, msg);
                    layer(visit, ip, cur, blocks, &msg, &lw);
                }
            }
        }
        for (int il = core1; il < hp_.n_layer; ++il) layer(il, il, cur, blocks, nullptr, nullptr);
        if (!logits) return;
        std::vector<float> mixed, x((size_t) E);
        mix(blocks, cur, m_.output_res_query, 1, mixed);
        rms(mixed.data(), E, hp_.eps, m_.output_norm, x.data());
        logits->assign((size_t) hp_.n_vocab, 0.f);
        matvec(m_.output, 0, x.data(), logits->data(), hp_.n_vocab);
        if (hp_.softcap > 0.f)
            for (auto& l : *logits) l = hp_.softcap * tanhf(l / hp_.softcap);
    }
};

}  // namespace i2
