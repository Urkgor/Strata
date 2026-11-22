// i2_cuda.cu - the Inkling 2 Mini engine on CUDA, one card: the weights as the file stores them (read by the kernels of g4_kernels.cuh and
// i2_kernels.cuh), K / V as halves.  The math is the CPU reference's (i2_ref.hpp); tests/inkling2/i2_emu_test.cpp runs this very file on the CPU emulation
// of CUDA against it.
//
// A chunk of T tokens goes visit by visit (24 physical layers, the middle 12 visited three times: 48 visits), every visit over all T tokens:
//   cur [T][E] is the residual stream; `sa` its copy at the start of a visit; the residual blocks (every 6th layer starts one) live in a pool of buffers
//   (the prelude's, the loop core's twice - one set for each parity of the pass, so that a pass reads the one before it - and the postlude's).
// Experts: a layer's experts are on the card, or - when the card cannot hold them all - in host memory (pinned), the ones a visit needs copied into the
// staging slots after its router has chosen (physical layers outside the loop first: a loop layer is visited three times).
#include "i2_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <thread>
#include <type_traits>

#include "../gemma4/g4_cuda_compat.hpp"
#include "i2_kernels.cuh"
#include "i2_ref.hpp"
#include "../gemma4/g4_serve.hpp"

#ifndef STRATA_I2_VERSION
#define STRATA_I2_VERSION "0.1.0-inkling2"
#endif

namespace i2cuda {

namespace {

#define CK(call)                                                                                                       \
    do {                                                                                                               \
        const cudaError_t e_ = (call);                                                                                 \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-inkling2: CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

template <class F>
void with_type(int t, F&& f) {
    using namespace g4;
    switch (t) {
        case T_F32: f(std::integral_constant<int, T_F32>{}); break;
        case T_F16: f(std::integral_constant<int, T_F16>{}); break;
        case T_BF16: f(std::integral_constant<int, T_BF16>{}); break;
        case T_Q4_0: f(std::integral_constant<int, T_Q4_0>{}); break;
        case T_Q4_1: f(std::integral_constant<int, T_Q4_1>{}); break;
        case T_Q5_0: f(std::integral_constant<int, T_Q5_0>{}); break;
        case T_Q5_1: f(std::integral_constant<int, T_Q5_1>{}); break;
        case T_Q8_0: f(std::integral_constant<int, T_Q8_0>{}); break;
        case T_Q2_K: f(std::integral_constant<int, T_Q2_K>{}); break;
        case T_Q3_K: f(std::integral_constant<int, T_Q3_K>{}); break;
        case T_Q4_K: f(std::integral_constant<int, T_Q4_K>{}); break;
        case T_Q5_K: f(std::integral_constant<int, T_Q5_K>{}); break;
        case T_Q6_K: f(std::integral_constant<int, T_Q6_K>{}); break;
        case T_IQ4_NL: f(std::integral_constant<int, T_IQ4_NL>{}); break;
        case T_IQ4_XS: f(std::integral_constant<int, T_IQ4_XS>{}); break;
        case T_IQ2_XXS: f(std::integral_constant<int, T_IQ2_XXS>{}); break;
        case T_IQ2_XS: f(std::integral_constant<int, T_IQ2_XS>{}); break;
        case T_IQ2_S: f(std::integral_constant<int, T_IQ2_S>{}); break;
        case T_IQ3_XXS: f(std::integral_constant<int, T_IQ3_XXS>{}); break;
        case T_IQ3_S: f(std::integral_constant<int, T_IQ3_S>{}); break;
        case T_MXFP4: f(std::integral_constant<int, T_MXFP4>{}); break;
        default: std::fprintf(stderr, "strata-inkling2: no kernel for the type %d\n", t); std::exit(1);
    }
}

constexpr int kExpertTile = 16;     // (token, expert) pairs of one expert that a block reads the weights for at once, in prompt reading

struct DevMat {
    int type = -1;
    long long n_in = 0, n_out = 0, n_expert = 1;
    size_t row_bytes = 0, expert_bytes = 0, bytes = 0;
    uint8_t* d = nullptr;
    explicit operator bool() const { return d != nullptr; }
};

struct DevLayer {
    DevMat wq, wk, wv, wo, wgate, wrel, wsink, kconv_proj, vconv_proj, router, gate_up, down;
    float *attn_norm = nullptr, *q_inp = nullptr, *k_inp = nullptr, *v_inp = nullptr, *q_clip = nullptr, *rel_clip = nullptr, *rel_bias = nullptr;
    float *kconv = nullptr, *vconv = nullptr, *post_conv = nullptr, *ffn_post_conv = nullptr, *attn_res_query = nullptr, *ffn_res_query = nullptr;
    float *ffn_norm = nullptr, *exp_bias = nullptr;
    bool streamed = false;
    uint8_t *h_gu = nullptr, *h_dn = nullptr;       // the experts in pinned host memory (a streamed layer)
};

struct DevLoop {
    float* msg_query = nullptr;
    float *down[4] = {}, *gain[4] = {}, *forget[4] = {};
};

struct Visit {
    uint16_t *kc = nullptr, *vc = nullptr;           // [kv head][cells][hd]
    int cells = 0;
    float* cring = nullptr;                          // [kRing][conv channels]
};

double ms_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        v.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return v;
}

}  // namespace

struct GpuEngine::Impl {
    const i2::Model& m;
    const i2::Hparams& hp;
    int ctx, chunk = 1;
    GpuOptions opt;
    cudaStream_t st = nullptr;
    std::vector<void*> allocs;
    std::vector<std::pair<void*, bool>> host_allocs;     // (pointer, pinned)
    size_t host_bytes = 0;
    std::vector<DevLayer> L;
    std::vector<DevLoop> loop;
    std::vector<Visit> vis;
    DevMat head;
    float *tok_norm = nullptr, *output_norm = nullptr, *output_res_query = nullptr;
    std::vector<char> is_streamed;
    int stage_slots = 32;
    // activations of a chunk
    float *emb = nullptr, *cur = nullptr, *sa = nullptr, *mixed = nullptr, *n0 = nullptr, *n1 = nullptr, *n2 = nullptr, *msgs = nullptr, *prel = nullptr;
    float *proj = nullptr, *moe = nullptr, *q = nullptr, *kraw = nullptr, *vraw = nullptr, *r = nullptr, *sink = nullptr, *gate = nullptr, *ktap = nullptr, *vtap = nullptr,
          *att = nullptr, *rlog = nullptr, *topk_w = nullptr, *gu = nullptr, *eact = nullptr, *ed = nullptr, *xh = nullptr, *mixh = nullptr;
    int *topk_idx = nullptr, *grp_expert = nullptr, *grp_first = nullptr, *grp_n = nullptr, *ent_tok = nullptr, *ent_row = nullptr;
    int *dec_first = nullptr, *dec_n = nullptr, *dec_tok = nullptr, *dec_row = nullptr, *dec_slots = nullptr;
    std::vector<float*> pre_blk, post_blk, core_blk[2], outbuf;
    uint8_t *stgA = nullptr, *stgC = nullptr;
    // the expert cache of the streamed layers (decode): device pools of `slots` experts (gate+up, down) at the largest layer's size; a slot holds the
    // expert `key` = layer * n_expert + expert.  The slot kept is the one with the highest decayed use count (every use adds `inc`, which grows by 1 / decay
    // at each decoded token: the same order as multiplying all counts by decay, without touching them).
    struct ExpertCache {
        int slots = 0;
        uint8_t *a = nullptr, *c = nullptr;
        std::vector<int> key, where;
        std::vector<double> score;
        double inc = 1.0;
        long long hits = 0, misses = 0;
    } ec;
    size_t cache_budget = 0;
    int* cache_slot_dev = nullptr;
    float *logits = nullptr, *pv = nullptr;
    int *pi = nullptr, *amax = nullptr;
    int64_t pos = 0;
    bool logits_ready = false;
    std::vector<float> hbuf;
    std::vector<std::vector<float>>* trace = nullptr;
    std::vector<std::vector<int>>* sel_trace = nullptr;

    Impl(const i2::Model& mm, int c, GpuOptions o) : m(mm), hp(mm.hp), ctx(c), opt(std::move(o)) {
        chunk = std::max(1, opt.chunk);
        stage_slots = std::max(1, std::min(hp.n_expert, std::max(opt.stage_slots, 1)));
    }

    ~Impl() {
        for (void* p : allocs) cudaFree(p);
        if (st) cudaStreamDestroy(st);
        for (auto& h : host_allocs) {
            if (h.second) cudaFreeHost(h.first);
            else std::free(h.first);
        }
    }

    template <class T>
    T* alloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, std::max<size_t>(n * sizeof(T), 16)));
        CK(cudaMemset(p, 0, std::max<size_t>(n * sizeof(T), 16)));
        allocs.push_back(p);
        return static_cast<T*>(p);
    }
    float* uvec(const float* src, size_t n) {
        float* d = alloc<float>(n);
        CK(cudaMemcpy(d, src, n * 4, cudaMemcpyHostToDevice));
        return d;
    }
    DevMat upload(const i2::Mat& a, long long rows = -1) {
        DevMat d;
        if (!a) return d;
        d.type = a.type;
        d.n_in = a.n_in;
        d.n_out = rows >= 0 ? rows : a.n_out;
        d.n_expert = a.n_expert;
        d.row_bytes = a.row_bytes;
        d.expert_bytes = a.row_bytes * (size_t) d.n_out;
        d.bytes = d.expert_bytes * (size_t) d.n_expert;
        d.d = alloc<uint8_t>(d.bytes);
        CK(cudaMemcpy(d.d, a.data, d.bytes, cudaMemcpyHostToDevice));
        return d;
    }
    DevMat meta_of(const i2::Mat& a) {
        DevMat d;
        d.type = a.type;
        d.n_in = a.n_in;
        d.n_out = a.n_out;
        d.n_expert = a.n_expert;
        d.row_bytes = a.row_bytes;
        d.expert_bytes = a.expert_bytes;
        d.bytes = a.bytes;
        return d;
    }
    uint8_t* host_copy(const i2::Mat& a) {
        void* p = nullptr;
        bool pinned = true;
        if (cudaHostAlloc(&p, a.bytes, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError();
            p = std::malloc(a.bytes);
            pinned = false;
        }
        if (!p) { std::fprintf(stderr, "strata-inkling2: no host memory for %.2f GiB of experts\n", (double) a.bytes / (1 << 30)); std::exit(1); }
        std::memcpy(p, a.data, a.bytes);
        host_allocs.push_back({p, pinned});
        host_bytes += a.bytes;
        return static_cast<uint8_t*>(p);
    }
    bool host_pinned() const {
        for (const auto& h : host_allocs) if (!h.second) return false;
        return true;
    }

    // ------------------------------------------------------------------------------------------------------------------------ sizes
    static size_t mat_bytes(const i2::Mat& a) { return a ? a.bytes : 0; }
    size_t layer_expert_bytes(int ip) const { return mat_bytes(m.L[(size_t) ip].gate_up) + mat_bytes(m.L[(size_t) ip].down); }
    // the layers' experts can have different types (the IQ4_NL file keeps the last 4 layers' in Q8_0): the staging slots take the largest
    size_t max_gu_bytes() const { size_t b = 0; for (const auto& l : m.L) b = std::max<size_t>(b, l.gate_up.expert_bytes); return b; }
    size_t max_dn_bytes() const { size_t b = 0; for (const auto& l : m.L) b = std::max<size_t>(b, l.down.expert_bytes); return b; }
    size_t per_expert_bytes() const { return max_gu_bytes() + max_dn_bytes(); }
    size_t layer_dense_bytes(int ip) const {
        const i2::LayerW& w = m.L[(size_t) ip];
        size_t b = mat_bytes(w.wq) + mat_bytes(w.wk) + mat_bytes(w.wv) + mat_bytes(w.wo) + mat_bytes(w.wgate) + mat_bytes(w.wrel) + mat_bytes(w.wsink) +
                   mat_bytes(w.kconv_proj) + mat_bytes(w.vconv_proj);
        const size_t E = (size_t) hp.n_embd, H = (size_t) hp.n_head, R = (size_t) hp.n_rel, C = (size_t) hp.n_conv, kd = (size_t) hp.kv_dim();
        const size_t extent = (size_t) (hp.swa[(size_t) ip] ? hp.window : hp.rel_extent);
        b += 4 * (H * (size_t) hp.hd + extent * R + H * R + 2 * kd * C + 2 * E * C + 4 * E + (size_t) hp.n_expert * E + (size_t) hp.n_expert + 4 * E);
        return b;
    }
    size_t loop_bytes() const { return hp.has_loop() ? (size_t) (hp.loop_end - hp.loop_start) * 4 * (4 * (size_t) hp.n_embd + 4 * (size_t) (3 * hp.n_embd * hp.gate_rank)) : 0; }
    size_t head_bytes() const { return (size_t) hp.n_vocab * m.output.row_bytes; }
    int local_cells() const { return (int) std::min<int64_t>(ctx, (int64_t) hp.window + chunk + std::max(0, opt.slack)); }
    size_t kv_bytes() const {
        size_t b = 0;
        for (int v = 0; v < hp.visits(); ++v) {
            const int cells = hp.swa[(size_t) hp.phys(v)] ? local_cells() : ctx;
            b += (size_t) 2 * (size_t) hp.n_kv * (size_t) cells * (size_t) hp.hd * 2;
            b += (size_t) i2k::kRing * (size_t) hp.conv_channels() * 4;
        }
        return b;
    }
    size_t activation_bytes() const {
        const size_t T = (size_t) chunk, E = (size_t) hp.n_embd;
        const int nb = (int) (pre_blocks() + 2 * core_blocks() + post_blocks());
        size_t f = T * E * (size_t) (14 + nb + 2 * (hp.has_loop() ? 1 : 0)) + T * (size_t) hp.q_dim() * 3 + T * (size_t) hp.kv_dim() * 2 + T * (size_t) hp.n_head * (size_t) hp.n_rel +
                   T * (size_t) hp.n_head + T * (size_t) hp.n_conv * (size_t) hp.n_kv * 2 + T * (size_t) hp.n_expert + T * (size_t) hp.n_used * 2;
        f += T * (size_t) hp.n_used * ((size_t) 2 * (size_t) hp.n_ff_exp + (size_t) hp.n_ff_exp + E) + (size_t) 3 * (size_t) hp.n_vocab + E * 8;
        return f * 4 + (size_t) (1 << 20);
    }
    size_t stage_bytes() const { return per_expert_bytes() * (size_t) stage_slots + (1 << 20); }
    int pre_blocks() const { return (hp.has_loop() ? hp.loop_start : hp.n_layer + hp.block - 1) / hp.block + (hp.has_loop() ? 0 : 0); }
    int core_blocks() const { return hp.has_loop() ? (hp.loop_end - hp.loop_start) / hp.block : 0; }
    int post_blocks() const { return hp.has_loop() ? (hp.n_layer - hp.loop_end + hp.block - 1) / hp.block : 0; }

    int streamed_layers_count() const { int c = 0; for (char x : is_streamed) c += x != 0; return c; }

    /// which physical layers keep their experts in host memory: the ones outside the loop first (a loop layer is visited `passes` times), from the last
    std::string place() {
        const int n = hp.n_layer;
        is_streamed.assign((size_t) n, 0);
        size_t fr = 0, tot = 0;
        CK(cudaSetDevice(opt.device));
        CK(cudaMemGetInfo(&fr, &tot));
        long long budget = (long long) fr - ((long long) opt.reserve_mib << 20) - (long long) activation_bytes() - (long long) kv_bytes() - (long long) head_bytes() -
                           (long long) loop_bytes() - (long long) 2 * (long long) m.tok_embd.row_bytes;
        size_t dense = 0, experts = 0;
        for (int ip = 0; ip < n; ++ip) { dense += layer_dense_bytes(ip); experts += layer_expert_bytes(ip); }
        budget -= (long long) dense;
        std::vector<int> order;                                    // the preference of streaming
        for (int ip = n - 1; ip >= 0; --ip)
            if (!hp.has_loop() || ip < hp.loop_start || ip >= hp.loop_end) order.push_back(ip);
        if (hp.has_loop())
            for (int ip = hp.loop_end - 1; ip >= hp.loop_start; --ip) order.push_back(ip);
        long long need = (long long) experts;
        size_t k = 0;
        const size_t forced = (size_t) std::min<int>(n, std::max(0, opt.host_experts));
        while (k < order.size() && (need > budget - (k > 0 || forced > 0 ? (long long) stage_bytes() : 0) || k < forced)) {
            is_streamed[(size_t) order[k]] = 1;
            need -= (long long) layer_expert_bytes(order[k]);
            ++k;
        }
        cache_budget = 0;
        if (k > 0 && opt.cache_mib != 0) {
            long long left = budget - (long long) stage_bytes() - need;
            left -= left / 32;                                                // a margin for the allocator
            size_t streamed = 0;
            for (int ip = 0; ip < n; ++ip) if (is_streamed[(size_t) ip]) streamed += layer_expert_bytes(ip);
            cache_budget = (size_t) std::max<long long>(0, left);
            if (opt.cache_mib > 0) cache_budget = std::min<size_t>(cache_budget, (size_t) opt.cache_mib << 20);
            cache_budget = std::min(cache_budget, streamed);
        }
        if (need > budget - (k > 0 ? (long long) stage_bytes() : 0)) {
            char buf[420];
            std::snprintf(buf, sizeof buf, "even with every expert in host memory the model needs %.2f GiB of VRAM (attention and dense weights %.2f, K / V and convolution state at a context of %d %.2f, "
                          "the output head %.2f) and %.2f GiB are free after the reserve and the buffers: use a shorter --max-context, --swa-slack 0, --vram-reserve-mib 300, or another card",
                          (double) (dense + kv_bytes() + head_bytes()) / (1 << 30), (double) dense / (1 << 30), ctx, (double) kv_bytes() / (1 << 30), (double) head_bytes() / (1 << 30),
                          (double) (fr > ((size_t) opt.reserve_mib << 20) ? fr - ((size_t) opt.reserve_mib << 20) : 0) / (1 << 30));
            return buf;
        }
        return "";
    }

    // ------------------------------------------------------------------------------------------------------------------------ init
    std::string init() {
        if (hp.hd > 128 || hp.hd % 32 != 0 || hp.n_rel > 128 || hp.gate_rank > 256 || hp.n_expert > 512 || hp.rel_extent > i2k::kMaxExtent || hp.window > i2k::kMaxExtent)
            return "a dimension the kernels do not take (head size 32..128 in steps of 32, relative width <= 128, loop gate rank <= 256, <= 512 experts, bias tables <= 1024)";
        if (hp.n_conv > 8) return "convolution kernel size above 8";
        std::string err = place();
        if (!err.empty()) return err;
        CK(cudaSetDevice(opt.device));
        CK(cudaStreamCreate(&st));
        const int E = hp.n_embd, H = hp.n_head, KV = hp.n_kv, hd = hp.hd, R = hp.n_rel, C = hp.n_conv, qd = hp.q_dim(), kd = hp.kv_dim();
        tok_norm = uvec(m.tok_norm, (size_t) E);
        output_norm = uvec(m.output_norm, (size_t) E);
        output_res_query = uvec(m.output_res_query, (size_t) E);
        head = upload(m.output, hp.n_vocab);
        L.assign((size_t) hp.n_layer, DevLayer());
        for (int ip = 0; ip < hp.n_layer; ++ip) {
            const i2::LayerW& w = m.L[(size_t) ip];
            DevLayer& d = L[(size_t) ip];
            const bool swa = hp.swa[(size_t) ip] != 0;
            const size_t extent = (size_t) (swa ? hp.window : hp.rel_extent);
            d.wq = upload(w.wq);
            d.wk = upload(w.wk);
            d.wv = upload(w.wv);
            d.wo = upload(w.wo);
            d.wgate = upload(w.wgate);
            d.wrel = upload(w.wrel);
            d.wsink = upload(w.wsink);
            if (!swa) { d.kconv_proj = upload(w.kconv_proj); d.vconv_proj = upload(w.vconv_proj); }
            d.q_clip = uvec(w.q_clip, (size_t) H * hd);
            d.rel_clip = uvec(w.rel_clip, (size_t) H * R);
            d.rel_bias = uvec(w.rel_bias, extent * (size_t) R);
            d.kconv = uvec(w.kconv, (size_t) kd * C);
            d.vconv = uvec(w.vconv, (size_t) kd * C);
            d.post_conv = uvec(w.post_conv, (size_t) E * C);
            d.ffn_post_conv = uvec(w.ffn_post_conv, (size_t) E * C);
            if (swa) {
                d.attn_norm = uvec(w.attn_norm, (size_t) E);
            } else {
                d.q_inp = uvec(w.q_inp_norm, (size_t) E);
                d.k_inp = uvec(w.k_inp_norm, (size_t) E);
                d.v_inp = uvec(w.v_inp_norm, (size_t) E);
            }
            if (w.attn_res_query) d.attn_res_query = uvec(w.attn_res_query, (size_t) E * (swa ? 1 : 3));
            d.ffn_res_query = uvec(w.ffn_res_query, (size_t) E);
            d.ffn_norm = uvec(w.ffn_norm, (size_t) E);
            if (w.exp_bias) d.exp_bias = uvec(w.exp_bias, (size_t) hp.n_expert);
            {   // the router as an F32 matrix [n_expert][E]
                d.router.type = g4::T_F32;
                d.router.n_in = E;
                d.router.n_out = hp.n_expert;
                d.router.row_bytes = (size_t) E * 4;
                d.router.expert_bytes = d.router.row_bytes * (size_t) hp.n_expert;
                d.router.bytes = d.router.expert_bytes;
                d.router.d = reinterpret_cast<uint8_t*>(uvec(w.router, (size_t) E * (size_t) hp.n_expert));
            }
            d.streamed = is_streamed[(size_t) ip] != 0;
            if (d.streamed) {
                d.gate_up = meta_of(w.gate_up);
                d.down = meta_of(w.down);
                d.h_gu = host_copy(w.gate_up);
                d.h_dn = host_copy(w.down);
            } else {
                d.gate_up = upload(w.gate_up);
                d.down = upload(w.down);
            }
        }
        if (hp.has_loop()) {
            loop.assign((size_t) (hp.loop_end - hp.loop_start), DevLoop());
            for (int i = 0; i < hp.loop_end - hp.loop_start; ++i) {
                const i2::LoopW& w = m.loop[(size_t) i];
                DevLoop& d = loop[(size_t) i];
                d.msg_query = uvec(w.msg_query, (size_t) 4 * E);
                for (int c = 0; c < 4; ++c) {
                    d.down[c] = uvec(w.down[c], (size_t) 2 * E * hp.gate_rank);
                    d.gain[c] = uvec(w.gain[c], (size_t) E * hp.gate_rank);
                    d.forget[c] = uvec(w.forget[c], (size_t) E * hp.gate_rank);
                }
            }
        }
        // the visits' caches
        vis.assign((size_t) hp.visits(), Visit());
        for (int v = 0; v < hp.visits(); ++v) {
            Visit& V = vis[(size_t) v];
            V.cells = hp.swa[(size_t) hp.phys(v)] ? local_cells() : ctx;
            V.kc = alloc<uint16_t>((size_t) KV * (size_t) V.cells * (size_t) hd);
            V.vc = alloc<uint16_t>((size_t) KV * (size_t) V.cells * (size_t) hd);
            V.cring = alloc<float>((size_t) i2k::kRing * (size_t) hp.conv_channels());
        }
        // activations
        const size_t T = (size_t) chunk, TE = T * (size_t) E;
        emb = alloc<float>(TE);
        cur = alloc<float>(TE);
        sa = alloc<float>(TE);
        mixed = alloc<float>(3 * TE);
        n0 = alloc<float>(TE);
        n1 = alloc<float>(TE);
        n2 = alloc<float>(TE);
        msgs = alloc<float>(4 * TE);
        prel = alloc<float>(TE);
        proj = alloc<float>(TE);
        moe = alloc<float>(TE);
        q = alloc<float>(T * (size_t) qd);
        att = alloc<float>(T * (size_t) qd);
        gate = alloc<float>(T * (size_t) qd);
        kraw = alloc<float>(T * (size_t) kd);
        vraw = alloc<float>(T * (size_t) kd);
        r = alloc<float>(T * (size_t) H * (size_t) R);
        sink = alloc<float>(T * (size_t) H);
        ktap = alloc<float>(T * (size_t) C * (size_t) KV);
        vtap = alloc<float>(T * (size_t) C * (size_t) KV);
        rlog = alloc<float>(T * (size_t) hp.n_expert);
        topk_idx = alloc<int>(T * (size_t) hp.n_used);
        topk_w = alloc<float>(T * (size_t) hp.n_used);
        gu = alloc<float>(T * (size_t) hp.n_used * (size_t) 2 * (size_t) hp.n_ff_exp);
        eact = alloc<float>(T * (size_t) hp.n_used * (size_t) hp.n_ff_exp);
        ed = alloc<float>(T * (size_t) hp.n_used * (size_t) E);
        const size_t ng = std::min<size_t>(T * (size_t) hp.n_used, (size_t) hp.n_expert) * (size_t) ((T * hp.n_used + kExpertTile - 1) / kExpertTile + 1);
        grp_expert = alloc<int>(ng + 1);
        grp_first = alloc<int>(ng + 1);
        grp_n = alloc<int>(ng + 1);
        ent_tok = alloc<int>(T * (size_t) hp.n_used + 16);
        ent_row = alloc<int>(T * (size_t) hp.n_used + 16);
        {   // the decode step's constant group tables (one token: its n_used experts are its groups)
            const int nu = hp.n_used;
            std::vector<int> first((size_t) nu + 1), one((size_t) nu, 1), zero((size_t) nu, 0), rows((size_t) nu), slots((size_t) nu);
            for (int k = 0; k < nu; ++k) { first[(size_t) k] = k; rows[(size_t) k] = k; slots[(size_t) k] = k; }
            dec_first = alloc<int>((size_t) nu + 1);
            dec_n = alloc<int>((size_t) nu);
            dec_tok = alloc<int>((size_t) nu);
            dec_row = alloc<int>((size_t) nu);
            dec_slots = alloc<int>((size_t) nu);
            CK(cudaMemcpy(dec_first, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dec_n, one.data(), one.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dec_tok, zero.data(), zero.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dec_row, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(dec_slots, slots.data(), slots.size() * 4, cudaMemcpyHostToDevice));
        }
        bool any_streamed = false;
        for (char s : is_streamed) any_streamed |= s != 0;
        if (any_streamed) {
            stgA = alloc<uint8_t>(max_gu_bytes() * (size_t) stage_slots);
            stgC = alloc<uint8_t>(max_dn_bytes() * (size_t) stage_slots);
            int want = opt.cache_slots >= 0 ? opt.cache_slots : (int) std::min<size_t>(cache_budget / std::max<size_t>(per_expert_bytes(), 1), 1u << 20);
            int streamed_experts = 0;
            for (int ip = 0; ip < hp.n_layer; ++ip) if (is_streamed[(size_t) ip]) streamed_experts += hp.n_expert;
            want = std::min(want, streamed_experts);
            if (want >= hp.n_used) {                       // (fewer slots than one step needs: no cache)
                ec.slots = want;
                ec.a = alloc<uint8_t>(max_gu_bytes() * (size_t) want);
                ec.c = alloc<uint8_t>(max_dn_bytes() * (size_t) want);
                ec.key.assign((size_t) want, -1);
                ec.score.assign((size_t) want, 0.0);
                ec.where.assign((size_t) hp.n_layer * (size_t) hp.n_expert, -1);
                cache_slot_dev = alloc<int>((size_t) hp.n_used);
            }
        }
        // the residual blocks' pool, the loop's outputs
        for (int i = 0; i < pre_blocks(); ++i) pre_blk.push_back(alloc<float>(TE));
        for (int i = 0; i < post_blocks(); ++i) post_blk.push_back(alloc<float>(TE));
        for (int p = 0; p < 2; ++p) {
            for (int i = 0; i < core_blocks(); ++i) core_blk[p].push_back(alloc<float>(TE));
            if (hp.has_loop()) outbuf.push_back(alloc<float>(TE));
        }
        xh = alloc<float>((size_t) E);
        mixh = alloc<float>((size_t) E);
        logits = alloc<float>((size_t) hp.n_vocab);
        pv = alloc<float>((size_t) (hp.n_vocab / 1024 + 2));
        pi = alloc<int>((size_t) (hp.n_vocab / 1024 + 2));
        amax = alloc<int>(1);
        CK(cudaDeviceSynchronize());
        if (opt.verbose) {
            int ns = 0;
            for (char s : is_streamed) ns += s;
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            std::fprintf(stderr, "strata-inkling2: 1 CUDA device [device %d: %d layers (%d visits), %.2f GiB free after loading]; context %d, chunk %d\n", opt.device, hp.n_layer, hp.visits(),
                         (double) fr / (1 << 30), ctx, chunk);
            if (ns && ec.slots)
                std::fprintf(stderr, "strata-inkling2: expert cache: %d slots (%.2f GiB, %.0f%% of the streamed layers' experts), decayed use count (x%.2f a token)\n", ec.slots,
                             (double) ec.slots * (double) per_expert_bytes() / (1 << 30), 100.0 * ec.slots / std::max(1, streamed_layers_count() * hp.n_expert), opt.cache_decay);
            if (ns)
                std::fprintf(stderr, "strata-inkling2: the experts of %d of %d layers do not fit the VRAM: they stay in host memory (%.2f GiB%s) and the ones a visit needs are copied to the card "
                             "(about %.0f MiB a visit at 8 experts); --host-experts N chooses how many layers do it\n", ns, hp.n_layer, (double) host_bytes / (1 << 30),
                             host_pinned() ? ", pinned" : ", pageable: pinning failed", (double) hp.n_used * (double) per_expert_bytes() / (1 << 20));
        }
        return "";
    }

    // ------------------------------------------------------------------------------------------------------------------------ launches
    static int tile_of(int Tn) { return Tn <= 1 ? 1 : Tn <= 4 ? 4 : Tn <= 8 ? 8 : 16; }

    /// y[t][y_off + r] = W row r . x[t] for t < T
    void mm(const DevMat& W, const float* x, int ldx, int Tn, float* y, int ldy, int y_off = 0) {
        const int NT = tile_of(Tn);
        const dim3 grid((unsigned) ((W.n_out + g4k::kMmWarps - 1) / g4k::kMmWarps), (unsigned) ((Tn + NT - 1) / NT));
        const dim3 block(g4k::kMmThreads);
        with_type(W.type, [&](auto tag) {
            constexpr int TY = decltype(tag)::value;
            if (NT == 1) G4_LAUNCH(G4_K(g4k::k_mm<TY, 1>), grid, block, st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, Tn, y, ldy, y_off);
            else if (NT == 4) G4_LAUNCH(G4_K(g4k::k_mm<TY, 4>), grid, block, st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, Tn, y, ldy, y_off);
            else if (NT == 8) G4_LAUNCH(G4_K(g4k::k_mm<TY, 8>), grid, block, st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, Tn, y, ldy, y_off);
            else G4_LAUNCH(G4_K(g4k::k_mm<TY, 16>), grid, block, st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, Tn, y, ldy, y_off);
        });
    }

    void mm_exp(const DevMat& W, const float* x, int ldx, int n_groups, bool decode, const int* grp_exp, const int* grp_fst, const int* grp_cnt, const int* ent_in,
                const int* ent_out, float* y, int ldy, int y_off = 0) {
        const dim3 grid((unsigned) ((W.n_out + g4k::kMmWarps - 1) / g4k::kMmWarps), (unsigned) n_groups);
        const dim3 block(g4k::kMmThreads);
        with_type(W.type, [&](auto tag) {
            constexpr int TY = decltype(tag)::value;
            if (decode) G4_LAUNCH(G4_K(g4k::k_mm_exp<TY, 1>), grid, block, st, W.d, (long long) W.expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx,
                                  grp_exp, grp_fst, grp_cnt, ent_in, ent_out, y, ldy, y_off);
            else G4_LAUNCH(G4_K(g4k::k_mm_exp<TY, kExpertTile>), grid, block, st, W.d, (long long) W.expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx,
                           grp_exp, grp_fst, grp_cnt, ent_in, ent_out, y, ldy, y_off);
        });
    }

    void copy_f(float* dst, const float* src, size_t n) { CK(cudaMemcpyAsync(dst, src, n * 4, cudaMemcpyDeviceToDevice, st)); }

    void rms(const float* x, const float* w, float* y, int Tn) {
        G4_LAUNCH(g4k::k_rms, dim3(Tn), dim3(g4k::kThreads), st, x, hp.n_embd, w, y, hp.n_embd, hp.n_embd, hp.eps);
    }

    /// out[q][t][i] for nq queries: the attention over the depth of `states` (blocks, then the stream)
    void mix(const std::vector<const float*>& states, int Tn, const float* queries, int nq, float* out) {
        const float* s[i2k::kMaxStates] = {nullptr, nullptr, nullptr, nullptr, nullptr};
        const int S = (int) states.size();
        if (S > i2k::kMaxStates) { std::fprintf(stderr, "strata-inkling2: %d residual states (at most %d)\n", S, i2k::kMaxStates); std::exit(1); }
        for (int i = 0; i < S; ++i) s[i] = states[(size_t) i];
        G4_LAUNCH(i2k::k_mix, dim3(Tn), dim3(i2k::kMixThreads), st, s[0], s[1], s[2], s[3], s[4], S, hp.n_embd, queries, nq, out, Tn, 1e-6f);
    }
    void control(float* c, const float* msg, int Tn, const DevLoop& lw, int k) {
        G4_LAUNCH(i2k::k_control, dim3(Tn), dim3(i2k::kMixThreads), st, c, msg, hp.n_embd, hp.gate_rank, (const float*) lw.down[k], (const float*) lw.gain[k], (const float*) lw.forget[k],
                  1e-6f);
    }
    void ring_store(const float* buf, int n, float* cring, int seg_off, int Tn, int pos0) {
        G4_LAUNCH(i2k::k_ring_store, dim3((unsigned) ((n + 127) / 128), (unsigned) Tn), dim3(g4k::kThreads), st, buf, n, n, cring, hp.conv_channels(), seg_off, pos0, Tn);
    }
    void conv_res(const float* in, const float* w, const float* cring, int seg_off, int Tn, int pos0, const float* resid, float* out) {
        const int E = hp.n_embd;
        G4_LAUNCH(i2k::k_conv_res, dim3((unsigned) ((E + 127) / 128), (unsigned) Tn), dim3(g4k::kThreads), st, in, E, w, hp.n_conv, cring, hp.conv_channels(), seg_off, pos0, resid, out);
    }

    void attention(int ip, const Visit& V, int Tn, int pos0) {
        const DevLayer& dl = L[(size_t) ip];
        const bool swa = hp.swa[(size_t) ip] != 0;
        const int window = swa ? hp.window : 0, extent = swa ? hp.window : hp.rel_extent;
        const dim3 grid((unsigned) Tn, (unsigned) hp.n_head), block(g4k::kThreads);
        switch (hp.hd / 32) {
            case 1: G4_LAUNCH(G4_K(i2k::k_attn<1>), grid, block, st, q, r, dl.rel_bias, sink, gate, V.kc, V.vc, att, hp.n_head, hp.n_kv, hp.n_rel, extent, pos0, V.cells, window); break;
            case 2: G4_LAUNCH(G4_K(i2k::k_attn<2>), grid, block, st, q, r, dl.rel_bias, sink, gate, V.kc, V.vc, att, hp.n_head, hp.n_kv, hp.n_rel, extent, pos0, V.cells, window); break;
            case 3: G4_LAUNCH(G4_K(i2k::k_attn<3>), grid, block, st, q, r, dl.rel_bias, sink, gate, V.kc, V.vc, att, hp.n_head, hp.n_kv, hp.n_rel, extent, pos0, V.cells, window); break;
            default: G4_LAUNCH(G4_K(i2k::k_attn<4>), grid, block, st, q, r, dl.rel_bias, sink, gate, V.kc, V.vc, att, hp.n_head, hp.n_kv, hp.n_rel, extent, pos0, V.cells, window); break;
        }
    }

    // ------------------------------------------------------------------------------------------------------------------------ experts
    void exp_ffn(const DevMat& gu_m, const DevMat& dn, int ng, bool decode, const int* ge, const int* gf, const int* gn, const int* et, const int* er, int rows,
                 const int* situ_rows) {
        const int E = hp.n_embd, F = hp.n_ff_exp;
        mm_exp(gu_m, n0, E, ng, decode, ge, gf, gn, et, er, gu, 2 * F);
        G4_LAUNCH(i2k::k_situ, dim3((unsigned) ((F + 127) / 128), (unsigned) rows), dim3(g4k::kThreads), st, (const float*) gu, eact, F, hp.situ_beta, hp.situ_lin, situ_rows);
        mm_exp(dn, eact, F, ng, decode, ge, gf, gn, er, er, ed, E);
    }
    void upload_groups(const std::vector<int>& gex, const std::vector<int>& gfi, const std::vector<int>& gnn, const std::vector<int>& ent_t, const std::vector<int>& ent_r) {
        CK(cudaMemcpy(grp_expert, gex.data(), gex.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(grp_first, gfi.data(), gfi.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(grp_n, gnn.data(), gnn.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ent_tok, ent_t.data(), ent_t.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ent_row, ent_r.data(), ent_r.size() * 4, cudaMemcpyHostToDevice));
    }
    /// the experts of a layer on the card (xn = n0): gate+up, SITU, down for every (token, expert) pair, then the weighted sum into `moe`
    void experts(const DevLayer& dl, int ip, int Tn) {
        const int nu = hp.n_used, E = hp.n_embd;
        mm(dl.router, n0, E, Tn, rlog, hp.n_expert);
        G4_LAUNCH(i2k::k_topk, dim3(Tn), dim3(g4k::kThreads), st, (const float*) rlog, (const float*) dl.exp_bias, hp.n_expert, nu, hp.route_scale, topk_idx, topk_w);
        const int rows = Tn * nu;
        if (sel_trace) {
            CK(cudaStreamSynchronize(st));
            std::vector<int> h((size_t) rows);
            CK(cudaMemcpy(h.data(), topk_idx, (size_t) rows * 4, cudaMemcpyDeviceToHost));
            sel_trace->push_back(h);
        }
        if (dl.streamed) {
            experts_streamed(dl, ip, Tn);
        } else if (Tn == 1) {                      // one token: the router's own choice is the group table, no round trip to the host
            exp_ffn(dl.gate_up, dl.down, nu, true, topk_idx, dec_first, dec_n, dec_tok, dec_row, rows, nullptr);
        } else {                                   // prompt reading: group the pairs by expert on the host (the weights are then read once per 16 pairs)
            std::vector<int> idx((size_t) rows);
            CK(cudaMemcpy(idx.data(), topk_idx, (size_t) rows * 4, cudaMemcpyDeviceToHost));
            std::vector<std::vector<int>> by((size_t) hp.n_expert);
            for (int rr = 0; rr < rows; ++rr) by[(size_t) idx[(size_t) rr]].push_back(rr);
            std::vector<int> gex, gfi, gnn, ent_t, ent_r;
            for (int e = 0; e < hp.n_expert; ++e) {
                const auto& v = by[(size_t) e];
                for (size_t s = 0; s < v.size(); s += kExpertTile) {
                    const size_t k = std::min<size_t>(kExpertTile, v.size() - s);
                    gex.push_back(e);
                    gfi.push_back((int) ent_t.size());
                    gnn.push_back((int) k);
                    for (size_t j = 0; j < k; ++j) { ent_r.push_back(v[s + j]); ent_t.push_back(v[s + j] / nu); }
                }
            }
            upload_groups(gex, gfi, gnn, ent_t, ent_r);
            exp_ffn(dl.gate_up, dl.down, (int) gex.size(), false, grp_expert, grp_first, grp_n, ent_tok, ent_row, rows, nullptr);
        }
        G4_LAUNCH(g4k::k_expert_sum, dim3((unsigned) ((E + 127) / 128), (unsigned) Tn), dim3(g4k::kThreads), st, (const float*) ed, (const int*) topk_idx, (const float*) topk_w,
                  (const float*) nullptr, nu, E, moe);
    }
    /// The experts of a layer in host memory: the router's choice is read back, the experts it names are copied into the staging slots (the n_used of a decode
    /// step; for a prompt chunk the experts in batches of stage_slots, each with the groups of pairs that use it).
    void experts_streamed(const DevLayer& dl, int ip, int Tn) {
        const int nu = hp.n_used;
        const int rows = Tn * nu;
        std::vector<int> idx((size_t) rows);
        CK(cudaMemcpy(idx.data(), topk_idx, (size_t) rows * 4, cudaMemcpyDeviceToHost));       // (also the barrier: the router has run)
        auto copy_in = [&](int slot, int e) {
            CK(cudaMemcpyAsync(stgA + (size_t) slot * dl.gate_up.expert_bytes, dl.h_gu + (size_t) e * dl.gate_up.expert_bytes, dl.gate_up.expert_bytes, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(stgC + (size_t) slot * dl.down.expert_bytes, dl.h_dn + (size_t) e * dl.down.expert_bytes, dl.down.expert_bytes, cudaMemcpyHostToDevice, st));
        };
        DevMat vgu = dl.gate_up, vdn = dl.down;
        vgu.d = stgA;
        vdn.d = stgC;
        if (Tn == 1 && ec.slots >= nu) {                   // decode: the experts the cache has are used where they are, the others replace its least used
            std::vector<int> sl((size_t) nu, -1);
            for (int k = 0; k < nu; ++k) {                         // the hits first: a miss must not take the slot of an expert this step uses
                sl[(size_t) k] = ec.where[(size_t) (ip * hp.n_expert + idx[(size_t) k])];
                ec.hits += sl[(size_t) k] >= 0;
            }
            for (int k = 0; k < nu; ++k) {
                const int e = idx[(size_t) k], key = ip * hp.n_expert + e;
                int s = sl[(size_t) k];
                if (s < 0) {
                    ++ec.misses;
                    double best = 1e300;
                    for (int j = 0; j < ec.slots; ++j) {
                        bool taken = false;
                        for (int q = 0; q < nu; ++q) taken |= sl[(size_t) q] == j;
                        if (!taken && ec.score[(size_t) j] < best) { best = ec.score[(size_t) j]; s = j; }
                    }
                    if (ec.key[(size_t) s] >= 0) ec.where[(size_t) ec.key[(size_t) s]] = -1;
                    ec.key[(size_t) s] = key;
                    ec.where[(size_t) key] = s;
                    ec.score[(size_t) s] = 0.0;
                    CK(cudaMemcpyAsync(ec.a + (size_t) s * max_gu_bytes(), dl.h_gu + (size_t) e * dl.gate_up.expert_bytes, dl.gate_up.expert_bytes, cudaMemcpyHostToDevice, st));
                    CK(cudaMemcpyAsync(ec.c + (size_t) s * max_dn_bytes(), dl.h_dn + (size_t) e * dl.down.expert_bytes, dl.down.expert_bytes, cudaMemcpyHostToDevice, st));
                    sl[(size_t) k] = s;
                }
                ec.score[(size_t) s] += ec.inc;
            }
            CK(cudaMemcpy(cache_slot_dev, sl.data(), (size_t) nu * 4, cudaMemcpyHostToDevice));
            DevMat cgu = dl.gate_up, cdn = dl.down;
            cgu.d = ec.a; cgu.expert_bytes = max_gu_bytes();
            cdn.d = ec.c; cdn.expert_bytes = max_dn_bytes();
            exp_ffn(cgu, cdn, nu, true, cache_slot_dev, dec_first, dec_n, dec_tok, dec_row, rows, nullptr);
            return;
        }
        if (Tn == 1) {
            for (int k = 0; k < nu; ++k) copy_in(k, idx[(size_t) k]);
            exp_ffn(vgu, vdn, nu, true, dec_slots, dec_first, dec_n, dec_tok, dec_row, rows, nullptr);
            return;
        }
        std::vector<std::vector<int>> by((size_t) hp.n_expert);
        for (int rr = 0; rr < rows; ++rr) by[(size_t) idx[(size_t) rr]].push_back(rr);
        std::vector<int> used;
        for (int e = 0; e < hp.n_expert; ++e)
            if (!by[(size_t) e].empty()) used.push_back(e);
        for (size_t b0 = 0; b0 < used.size(); b0 += (size_t) stage_slots) {
            const size_t nb = std::min<size_t>((size_t) stage_slots, used.size() - b0);
            std::vector<int> gex, gfi, gnn, ent_t, ent_r;
            for (size_t k = 0; k < nb; ++k) {
                const int e = used[b0 + k];
                copy_in((int) k, e);
                const auto& v = by[(size_t) e];
                for (size_t s = 0; s < v.size(); s += kExpertTile) {
                    const size_t n = std::min<size_t>(kExpertTile, v.size() - s);
                    gex.push_back((int) k);                                    // the slot: what the kernels index experts by here
                    gfi.push_back((int) ent_t.size());
                    gnn.push_back((int) n);
                    for (size_t j = 0; j < n; ++j) { ent_r.push_back(v[s + j]); ent_t.push_back(v[s + j] / nu); }
                }
            }
            upload_groups(gex, gfi, gnn, ent_t, ent_r);                          // (a copy on the legacy stream: it waits for the batch before, which reads these tables)
            exp_ffn(vgu, vdn, (int) gex.size(), false, grp_expert, grp_first, grp_n, ent_tok, ent_row, (int) ent_r.size(), ent_row);
        }
    }

    // ------------------------------------------------------------------------------------------------------------------------ a visit
    /// One visit of physical layer ip for the Tn tokens at positions pos0 ..: cur -> cur.  `blocks`: the residual blocks so far (read by the mixes; a block
    /// started by this layer is copied to `push_dst` and appended).  `msgs_in` / `lw`: the cross-pass messages [4][Tn][E] of a later pass (null: none).
    void visit(int v, int ip, int Tn, int pos0, const float* msgs_in, const DevLoop* lw, std::vector<const float*>& blocks, float* push_dst) {
        const DevLayer& dl = L[(size_t) ip];
        const Visit& V = vis[(size_t) v];
        const int E = hp.n_embd, qd = hp.q_dim(), kd = hp.kv_dim();
        const size_t TE = (size_t) Tn * (size_t) E;
        const bool swa = hp.swa[(size_t) ip] != 0;
        const dim3 b1(g4k::kThreads);
        copy_f(sa, cur, TE);
        float* in[3] = {cur, cur, cur};
        if (dl.attn_res_query) {
            std::vector<const float*> states = blocks;
            states.push_back(cur);
            mix(states, Tn, dl.attn_res_query, swa ? 1 : 3, mixed);
            if (swa) {
                in[0] = mixed;
                if (msgs_in) { copy_f(mixed + TE, mixed, TE); copy_f(mixed + 2 * TE, mixed, TE); in[1] = mixed + TE; in[2] = mixed + 2 * TE; }
                else { in[1] = in[2] = mixed; }
            } else {
                for (int j = 0; j < 3; ++j) in[j] = mixed + (size_t) j * TE;
            }
        }
        if (msgs_in)
            for (int j = 0; j < 3; ++j) control(in[j], msgs_in + (size_t) j * TE, Tn, *lw, j);
        const float* nq0 = n0;
        const float* nk = n0;
        const float* nv = n0;
        if (swa) {
            rms(in[0], dl.attn_norm, n0, Tn);
            if (msgs_in) { rms(in[1], dl.attn_norm, n1, Tn); rms(in[2], dl.attn_norm, n2, Tn); nk = n1; nv = n2; }
        } else {
            rms(in[0], dl.q_inp, n0, Tn);
            rms(in[1], dl.k_inp, n1, Tn);
            rms(in[2], dl.v_inp, n2, Tn);
            nk = n1;
            nv = n2;
        }
        mm(dl.wq, nq0, E, Tn, q, qd);
        mm(dl.wk, nk, E, Tn, kraw, kd);
        mm(dl.wv, nv, E, Tn, vraw, kd);
        mm(dl.wrel, nq0, E, Tn, r, hp.n_head * hp.n_rel);
        mm(dl.wsink, nq0, E, Tn, sink, hp.n_head);
        mm(dl.wgate, nq0, E, Tn, gate, qd);
        if (!swa) {
            mm(dl.kconv_proj, nk, E, Tn, ktap, hp.n_conv * hp.n_kv);
            mm(dl.vconv_proj, nv, E, Tn, vtap, hp.n_conv * hp.n_kv);
        }
        G4_LAUNCH(i2k::k_prep, dim3((unsigned) Tn, (unsigned) (2 * hp.n_head + 2 * hp.n_kv)), b1, st, q, (const float*) kraw, (const float*) vraw, r, (const float*) ktap, (const float*) vtap,
                  (const float*) dl.kconv, (const float*) dl.vconv, (const float*) dl.q_clip, (const float*) dl.rel_clip, (const float*) V.cring, hp.conv_channels(), V.kc, V.vc, hp.n_head, hp.n_kv,
                  hp.hd, hp.n_rel, hp.n_conv, pos0, V.cells, !swa, hp.qk_clip, hp.rel_clip, hp.eps, hp.log_alpha, hp.log_floor);
        ring_store(kraw, kd, V.cring, 0, Tn, pos0);
        ring_store(vraw, kd, V.cring, kd, Tn, pos0);
        attention(ip, V, Tn, pos0);
        mm(dl.wo, att, qd, Tn, proj, E);
        const bool push = ip % hp.block == 0;
        conv_res(proj, dl.post_conv, V.cring, 2 * kd, Tn, pos0, push ? nullptr : sa, cur);
        ring_store(proj, E, V.cring, 2 * kd, Tn, pos0);
        if (push) {
            copy_f(push_dst, sa, TE);
            blocks.push_back(push_dst);
        }
        // the experts
        {
            std::vector<const float*> states = blocks;
            states.push_back(cur);
            mix(states, Tn, dl.ffn_res_query, 1, mixed);
        }
        if (msgs_in) control(mixed, msgs_in + (size_t) 3 * TE, Tn, *lw, 3);
        rms(mixed, dl.ffn_norm, n0, Tn);
        experts(dl, ip, Tn);
        conv_res(moe, dl.ffn_post_conv, V.cring, 2 * kd + E, Tn, pos0, cur, cur);
        ring_store(moe, E, V.cring, 2 * kd + E, Tn, pos0);
        if (trace) {
            CK(cudaStreamSynchronize(st));
            std::vector<float> h(TE);
            CK(cudaMemcpy(h.data(), cur, TE * 4, cudaMemcpyDeviceToHost));
            trace->push_back(h);
        }
    }

    // ------------------------------------------------------------------------------------------------------------------------ a chunk
    void forward_chunk(const int32_t* ids, int Tn, bool want_logits) {
        if (ec.slots && Tn == 1) {                                 // a decoded token: the use counts age by one step
            ec.inc /= opt.cache_decay;
            if (ec.inc > 1e50) { for (double& x : ec.score) x /= ec.inc; ec.inc = 1.0; }
        }
        const int E = hp.n_embd;
        hbuf.resize((size_t) Tn * (size_t) E);
        for (int t = 0; t < Tn; ++t) g4::dequant_row(m.tok_embd.type, m.tok_embd.row(ids[t]), E, hbuf.data() + (size_t) t * (size_t) E);
        CK(cudaMemcpy(emb, hbuf.data(), (size_t) Tn * (size_t) E * 4, cudaMemcpyHostToDevice));
        rms(emb, tok_norm, cur, Tn);
        const size_t TE = (size_t) Tn * (size_t) E;
        const int S = hp.loop_start, Ee = hp.loop_end;
        const int core0 = hp.has_loop() ? S : hp.n_layer, core1 = hp.has_loop() ? Ee : hp.n_layer;
        const int np = hp.has_loop() ? hp.passes : 1;
        std::vector<const float*> blocks;
        for (int ip = 0; ip < core0; ++ip) visit(ip, ip, Tn, (int) pos, nullptr, nullptr, blocks, ip % hp.block == 0 ? pre_blk[(size_t) (ip / hp.block)] : nullptr);
        if (hp.has_loop()) {
            copy_f(prel, cur, TE);
            const std::vector<const float*> prelude_blocks = blocks;
            for (int pass = 0; pass < np; ++pass) {
                std::vector<const float*> bank;
                if (pass > 0) {
                    for (int k = 1; k < core_blocks(); ++k) bank.push_back(core_blk[(pass - 1) % 2][(size_t) k]);
                    bank.push_back(outbuf[(size_t) ((pass - 1) % 2)]);
                    copy_f(cur, prel, TE);
                    blocks = prelude_blocks;
                }
                for (int ip = S; ip < Ee; ++ip) {
                    const int vi = pass == 0 ? ip : hp.n_layer + (pass - 1) * (Ee - S) + (ip - S);
                    float* push_dst = ip % hp.block == 0 ? core_blk[pass % 2][(size_t) ((ip - S) / hp.block)] : nullptr;
                    if (pass == 0) {
                        visit(vi, ip, Tn, (int) pos, nullptr, nullptr, blocks, push_dst);
                    } else {
                        const DevLoop& lw = loop[(size_t) (ip - S)];
                        mix(bank, Tn, lw.msg_query, 4, msgs);
                        visit(vi, ip, Tn, (int) pos, msgs, &lw, blocks, push_dst);
                    }
                }
                copy_f(outbuf[(size_t) (pass % 2)], cur, TE);
            }
        }
        for (int ip = core1; ip < hp.n_layer; ++ip)
            visit(ip, ip, Tn, (int) pos, nullptr, nullptr, blocks, ip % hp.block == 0 ? post_blk[(size_t) ((ip - core1) / hp.block)] : nullptr);
        if (want_logits) {
            std::vector<const float*> states;
            for (const float* b : blocks) states.push_back(b + (size_t) (Tn - 1) * (size_t) E);
            states.push_back(cur + (size_t) (Tn - 1) * (size_t) E);
            mix(states, 1, output_res_query, 1, mixh);
            rms(mixh, output_norm, xh, 1);
            mm(head, xh, E, 1, logits, hp.n_vocab);
            if (hp.softcap > 0.f) G4_LAUNCH(g4k::k_softcap, dim3((unsigned) ((hp.n_vocab + 127) / 128)), dim3(g4k::kThreads), st, logits, hp.n_vocab, hp.softcap);
            logits_ready = true;
        } else {
            logits_ready = false;
        }
        CK(cudaStreamSynchronize(st));
        pos += Tn;
    }
};

// ============================================================================================================ the public class
GpuEngine::GpuEngine(const i2::Model& m, int ctx, GpuOptions opt) : p_(new Impl(m, ctx, std::move(opt))) {}
GpuEngine::~GpuEngine() = default;
std::string GpuEngine::init() { return p_->init(); }
int GpuEngine::ctx() const { return p_->ctx; }
int GpuEngine::n_vocab() const { return p_->hp.n_vocab; }
int64_t GpuEngine::pos() const { return p_->pos; }
void GpuEngine::cache_stats(int* slots, long long* hits, long long* misses) const { *slots = p_->ec.slots; *hits = p_->ec.hits; *misses = p_->ec.misses; }
void GpuEngine::set_trace(std::vector<std::vector<float>>* t, std::vector<std::vector<int>>* s) { p_->trace = t; p_->sel_trace = s; }
void GpuEngine::reset() { p_->pos = 0; p_->logits_ready = false; }
void GpuEngine::truncate(int64_t q) { p_->pos = std::min(p_->pos, std::max<int64_t>(0, q)); p_->logits_ready = false; }
int GpuEngine::streamed_layers() const { return (int) std::count(p_->is_streamed.begin(), p_->is_streamed.end(), (char) 1); }
bool GpuEngine::can_keep(int64_t q) const {
    // Going back to q: the next token (position q) needs the keys of the sliding-window layers from q - window + 1 on, and the convolution inputs of
    // the positions q - 3 .. q - 1.  A local ring of `cells` cells holds the last `cells` positions, the convolution rings the last kRing.
    if (q > p_->pos) return false;
    const int64_t cells = p_->local_cells();
    const bool kv_ok = p_->pos <= cells || q - p_->hp.window + 1 >= p_->pos - cells;
    const bool conv_ok = p_->pos <= i2k::kRing || q - (p_->hp.n_conv - 1) >= p_->pos - i2k::kRing;
    return kv_ok && conv_ok;
}
bool GpuEngine::process(const int32_t* ids, int64_t n, const std::function<bool(int64_t)>& on_piece, int64_t* done) {
    if (p_->pos + n > p_->ctx) return false;
    *done = 0;
    for (int64_t i = 0; i < n; i += p_->chunk) {
        const int Tn = (int) std::min<int64_t>(p_->chunk, n - i);
        p_->forward_chunk(ids + i, Tn, i + Tn == n);
        *done = i + Tn;
        if (!on_piece(*done)) return *done >= n;
    }
    return true;
}
void GpuEngine::logits(std::vector<float>& out) {
    Impl& I = *p_;
    out.resize((size_t) I.hp.n_vocab);
    CK(cudaMemcpy(out.data(), I.logits, out.size() * 4, cudaMemcpyDeviceToHost));
}
int GpuEngine::greedy() {
    Impl& I = *p_;
    const int n = I.hp.n_vocab, np = (n + 1023) / 1024;
    G4_LAUNCH(g4k::k_argmax_part, dim3((unsigned) np), dim3(g4k::kThreads), I.st, (const float*) I.logits, n, I.pv, I.pi);
    G4_LAUNCH(g4k::k_argmax_final, dim3(1), dim3(g4k::kThreads), I.st, (const float*) I.pv, (const int*) I.pi, np, I.amax);
    int r = 0;
    CK(cudaMemcpy(&r, I.amax, 4, cudaMemcpyDeviceToHost));
    return r;
}

// ============================================================================================================ the program
int run(const i2::Model& m, int ctx, bool serve, const std::string& tokens, int n_gen, const std::string& dump, const std::string& gpus, const std::set<int32_t>& eos,
        const std::vector<std::string>& extra) {
    GpuOptions o;
    if (!gpus.empty()) o.device = parse_list(gpus)[0];
    bool check = false;
    int bench_pp = 0, bench_tg = 0;
    for (size_t i = 0; i < extra.size(); ++i) {
        const std::string& a = extra[i];
        auto val = [&]() -> std::string { return i + 1 < extra.size() ? extra[++i] : std::string(); };
        if (a == "--prefill" || a == "--chunk") { const std::string v = val(); if (v != "auto") o.chunk = std::max(1, std::atoi(v.c_str())); }
        else if (a == "--host-experts") o.host_experts = std::max(0, std::atoi(val().c_str()));
        else if (a == "--expert-cache-mib") o.cache_mib = std::max(0, std::atoi(val().c_str()));
        else if (a == "--cache-decay") o.cache_decay = std::min(1.0, std::max(0.5, std::atof(val().c_str())));
        else if (a == "--swa-slack") o.slack = std::max(0, std::atoi(val().c_str()));
        else if (a == "--vram-reserve-mib") o.reserve_mib = std::max(0, std::atoi(val().c_str()));
        else if (a == "--check") check = true;
        else if (a == "--bench-pp") bench_pp = std::atoi(val().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(val().c_str());
        else if (a == "--kv") { const std::string v = val(); if (v != "f16") std::fprintf(stderr, "strata-inkling2: --kv %s: this engine keeps K / V as f16\n", v.c_str()); }
        else std::fprintf(stderr, "strata-inkling2: ignoring %s (not an option of this engine)\n", a.c_str());
    }
    GpuEngine eng(m, ctx, o);
    const std::string err = eng.init();
    if (!err.empty()) { std::fprintf(stderr, "strata-inkling2: %s\n", err.c_str()); return 1; }
    if (serve) {
        g4::Serve<GpuEngine> sv(eng, eos, STRATA_I2_VERSION);
        return sv.run();
    }
    auto noop = [](int64_t) { return true; };
    int64_t done = 0;
    if (bench_pp > 0 || bench_tg > 0) {
        std::vector<int32_t> ids((size_t) std::max(bench_pp, 16));
        uint32_t s = 12345;
        for (auto& x : ids) { s = s * 1664525u + 1013904223u; x = 106 + (int) (s >> 16) % (m.hp.n_vocab - 200); }
        if (bench_pp > 0) {
            eng.reset();
            const double t0 = ms_now();
            if (!eng.process(ids.data(), std::min<int64_t>(bench_pp, ctx), noop, &done)) { std::fprintf(stderr, "strata-inkling2: the prompt does not fit the context\n"); return 1; }
            std::vector<float> lg;
            eng.logits(lg);
            const double ms = ms_now() - t0;
            std::printf("prompt: %lld tokens in %.0f ms: %.1f tok/s\n", (long long) done, ms, 1000.0 * (double) done / ms);
        }
        if (bench_tg > 0) {
            eng.reset();
            eng.process(ids.data(), 16, noop, &done);
            const double t0 = ms_now();
            int n = 0;
            for (; n < bench_tg && eng.pos() < ctx; ++n) {
                const int32_t nx = eng.greedy();
                int64_t d = 0;
                eng.process(&nx, 1, noop, &d);
            }
            const double ms = ms_now() - t0;
            std::printf("generation: %d tokens in %.0f ms: %.1f tok/s\n", n, ms, 1000.0 * n / ms);
            int cs = 0;
            long long ch = 0, cm = 0;
            eng.cache_stats(&cs, &ch, &cm);
            if (cs) std::printf("expert cache: %d slots, %lld hits, %lld misses (%.1f%% hits)\n", cs, ch, cm, 100.0 * (double) ch / (double) std::max<long long>(1, ch + cm));
        }
        return 0;
    }
    if (tokens.empty()) { std::fprintf(stderr, "strata-inkling2: --tokens ID,ID,... or --serve\n"); return 2; }
    const std::vector<int32_t> ids = g4::parse_ids(tokens.c_str());
    if (!eng.process(ids.data(), (int64_t) ids.size(), noop, &done)) { std::fprintf(stderr, "strata-inkling2: the prompt does not fit the context\n"); return 1; }
    std::vector<float> lg;
    eng.logits(lg);
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        if (f) { std::fwrite(lg.data(), 4, lg.size(), f); std::fclose(f); }
    }
    int rc = 0;
    if (check) {   // the same prompt on the CPU reference (K / V rounded to half as here): the engine's own gate, for a machine with a card
        i2::RefOpts ro;
        ro.cache_f16 = true;
        ro.threads = (int) std::max(1u, std::thread::hardware_concurrency());
        i2::RefEngine ref(m, (int) ids.size() + 1, ro);
        std::vector<float> rl;
        const double t0 = ms_now();
        ref.forward(ids.data(), (int) ids.size(), &rl);
        double d2 = 0, s2 = 0, mean = 0;
        for (float x : rl) mean += x;
        mean /= (double) rl.size();
        for (size_t i = 0; i < rl.size(); ++i) { d2 += (double) (lg[i] - rl[i]) * (lg[i] - rl[i]); s2 += (double) (rl[i] - mean) * (rl[i] - mean); }
        const double rel = std::sqrt(d2 / std::max(s2, 1e-30));
        const int a = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()), b = (int) (std::max_element(rl.begin(), rl.end()) - rl.begin());
        std::printf("check: %zu tokens, GPU against the CPU reference (%.0f ms): logits differ by %.2e (rms / spread), best token %d / %d\n", ids.size(), ms_now() - t0, rel, a, b);
        rc = (rel < 2e-3 && a == b) ? 0 : 1;
        std::printf("check: %s\n", rc == 0 ? "ok" : "FAILED (more than 2e-3, or another best token)");
    }
    std::printf("generated:");
    for (int i = 0; i < n_gen; ++i) {
        const int nx = eng.greedy();
        std::printf(" %d", nx);
        if (eos.count(nx) || eng.pos() >= ctx) break;
        int64_t d = 0;
        const int32_t one = nx;
        eng.process(&one, 1, noop, &d);
    }
    std::printf("\n");
    return rc;
}

}  // namespace i2cuda
