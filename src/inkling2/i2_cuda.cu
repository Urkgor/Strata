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
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>

#include "../gemma4/g4_cuda_compat.hpp"
#include "i2_cpu_pool.hpp"
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
        case T_NVFP4: f(std::integral_constant<int, T_NVFP4>{}); break;
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
    float *ffn_norm = nullptr, *exp_bias = nullptr, *gu_scale = nullptr, *dn_scale = nullptr;
    bool streamed = false;
    const float* gu_scale_h = nullptr;               // the host copy of gu_scale (the CPU pool applies it; the down scale is the GPU's, in k_expert_sum)
    uint8_t *h_gu = nullptr, *h_dn = nullptr;       // the experts in pinned host memory (a streamed layer)
};

struct DevLoop {
    float* msg_query = nullptr;
    float *down[4] = {}, *gain[4] = {}, *forget[4] = {};
};

struct Visit {
    uint16_t *kc = nullptr, *vc = nullptr;           // [kv head][cells][hd] halves (kvq 0)
    uint8_t *kq = nullptr, *vq = nullptr;            // [kv head][cells][hd] int8 (kvq 1) or [hd / 2] bytes of two 4-bit values (kvq 2)
    uint16_t *ks = nullptr, *vs = nullptr;           // [kv head][cells][hd / 32] half scales of the quantized ones
    int kvq = 0;
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
    // a prompt chunk's streamed experts: a copy stream, two halves of the staging slots and of the group tables, the events that order them
    cudaStream_t stc = nullptr;
    cudaEvent_t ev_copied[2] = {nullptr, nullptr}, ev_used[2] = {nullptr, nullptr};
    int* dtab[2][5] = {};
    int* htab[2] = {nullptr, nullptr};
    size_t tab_cap_g = 0, tab_cap_e = 0;
    // the expert cache of the streamed layers (decode): device pools of `slots` experts (gate+up, down) at the largest layer's size; a slot holds the
    // expert `key` = layer * n_expert + expert.  The slot kept is the one with the highest decayed use count (every use adds `inc`, which grows by 1 / decay
    // at each decoded token: the same order as multiplying all counts by decay, without touching them).
    struct ExpertCache {
        int slots = 0;
        uint8_t *a = nullptr, *c = nullptr;
        std::vector<int> key, where;
        double inc = 1.0;
        long long hits = 0, misses = 0;
    } ec;
    // every (layer, expert)'s decayed use count, in the cache or not: what decides who stays in the cache and who is computed on the CPU
    std::vector<double> freq;
    std::unique_ptr<i2cpu::Pool> pool;
    bool cpu_on = false;
    long long cpu_evals = 0, shared_evals = 0;
    double t_cpu_ms = 1.0, t_copy_ms = 1.0;      // an expert on the CPU pool / over PCIe, measured at start
    float *xh_host = nullptr, *cpu_y = nullptr;            // pinned: the CPU pool's input (one token) and outputs (n_used rows)
    int* slot_pin = nullptr;                               // pinned: the slots of a decode step, copied to the card without waiting
    // --profile: where a decoded token's time goes (the stream is synchronized at each boundary, so the total is a little more than without)
    struct Prof { double res = 0, str = 0, wait = 0, host = 0, cpu = 0, total = 0; long long nres = 0, nstr = 0, tokens = 0, chunks = 0; double lap[2][6] = {}, enq[2][6] = {}; };
    Prof pfk[2];                                        // [0] decoded tokens, [1] prompt chunks
    int prof_kind = 0;
    double lap_t = 0;
    void lap_(bool prof, int i, bool streamed) {          // --profile: the time since the last lap, into phase i of a resident / streamed visit
        if (!prof) return;
        const double t_enq = ms_now();                        // (the host has queued the phase's work: how long that took, against how long the card took)
        CK(cudaStreamSynchronize(st));
        const double t = ms_now();
        pfk[prof_kind].enq[streamed ? 1 : 0][i] += t_enq - lap_t;
        pfk[prof_kind].lap[streamed ? 1 : 0][i] += t - lap_t;
        lap_t = t;
    }
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
    /// the bytes of one cell of one KV head, K or V: halves, q8_0 (a half scale per 32 values) or 4 bits
    size_t kv_cell_bytes(int kvq) const {
        return kvq == 0 ? (size_t) hp.hd * 2 : kvq == 1 ? (size_t) hp.hd + (size_t) hp.hd / 16 : (size_t) hp.hd / 2 + (size_t) hp.hd / 16;
    }
    size_t kv_bytes() const {
        size_t b = 0;
        for (int v = 0; v < hp.visits(); ++v) {
            const bool local = hp.swa[(size_t) hp.phys(v)] != 0;
            const int cells = local ? local_cells() : ctx;
            b += (size_t) 2 * (size_t) hp.n_kv * (size_t) cells * kv_cell_bytes(local ? 0 : opt.kvq);
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

    /// The integer decode kernels, where they are faster: a Q8_0 dense matrix and an NVFP4 expert matrix of the model timed with the float kernel and the integer one.
    void setup_dp() {
        dp_ok[0] = dp_ok[1] = dp_any = false;
        pf_ok[0] = pf_ok[1] = pf_any = false;
        if (opt.dp4a == 0) return;
        const DevMat *m8 = nullptr, *m4 = nullptr;
        bool m4_streamed = false;
        for (int ip = 0; ip < hp.n_layer; ++ip) {
            if (!m8 && L[(size_t) ip].wq.type == g4::T_Q8_0) m8 = &L[(size_t) ip].wq;
            if (!m4 && L[(size_t) ip].gate_up.type == g4::T_NVFP4) { m4 = &L[(size_t) ip].gate_up; m4_streamed = L[(size_t) ip].streamed; }
        }
        const int E = hp.n_embd;
        auto bench = [&](const DevMat& W0, bool staged, double* t_f, double* t_i) {
            DevMat W = W0;
            if (staged) W.d = stgA;                                  // (a streamed layer's experts are not on the card: the staging slots hold the same shape)
            float* out = gu;
            prep(n0, E, 0);
            auto timed = [&](auto fn) {
                for (int i = 0; i < 300; ++i) fn();                  // (a warm-up long enough for the card to leave its idle clocks)
                CK(cudaStreamSynchronize(st));
                const double t0 = ms_now();
                for (int i = 0; i < 100; ++i) fn();
                CK(cudaStreamSynchronize(st));
                return (ms_now() - t0) / 100.0;
            };
            *t_f = timed([&] { mm(W, n0, E, 1, out, (int) W.n_out); });
            *t_i = timed([&] { mv(W, qslot(0), dslot(0), 0, nullptr, 1, out, (int) W.n_out, 0, 0); });
        };
        // reading a prompt: 16 tokens through the float kernel and through k_mmq
        auto bench_pf = [&](const DevMat& W0, bool staged, double* t_f, double* t_i) {
            DevMat W = W0;
            if (staged) W.d = stgA;
            const int Tn = std::min(16, chunk);
            float* out = gu;
            if (Tn < 2) { *t_f = *t_i = 0; return; }
            G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((E / 32 + 7) / 8), (unsigned) Tn), dim3(256), st, (const float*) n0, E, pslot(0), pdslot(0));
            auto timed = [&](auto fn) {
                for (int i = 0; i < 60; ++i) fn();
                CK(cudaStreamSynchronize(st));
                const double t0 = ms_now();
                for (int i = 0; i < 30; ++i) fn();
                CK(cudaStreamSynchronize(st));
                return (ms_now() - t0) / 30.0;
            };
            *t_f = timed([&] { mm(W, n0, E, Tn, out, (int) W.n_out); });
            *t_i = timed([&] { mmq(W, pslot(0), pdslot(0), 1, nullptr, nullptr, nullptr, nullptr, nullptr, Tn, out, (int) W.n_out, 0, 0); });
        };
        double f8 = 0, i8 = 0, f4 = 0, i4 = 0, pf8 = 0, pi8 = 0, pf4 = 0, pi4 = 0;
        if (m8 && (int) m8->n_in == E) { bench(*m8, false, &f8, &i8); dp_ok[0] = opt.dp4a == 1 || i8 < 0.9 * f8; }
        if (m4 && (int) m4->n_in == E && !(m4_streamed && !stgA)) { bench(*m4, m4_streamed, &f4, &i4); dp_ok[1] = opt.dp4a == 1 || i4 < 0.9 * f4; }
        if (m8 && (int) m8->n_in == E) { bench_pf(*m8, false, &pf8, &pi8); pf_ok[0] = std::min(16, chunk) >= 2 && (opt.dp4a == 1 || pi8 < 0.9 * pf8); }
        if (m4 && (int) m4->n_in == E && !(m4_streamed && !stgA)) { bench_pf(*m4, m4_streamed, &pf4, &pi4); pf_ok[1] = std::min(16, chunk) >= 2 && (opt.dp4a == 1 || pi4 < 0.9 * pf4); }
        dp_any = dp_ok[0] || dp_ok[1];
        pf_any = pf_ok[0] || pf_ok[1];
        if (opt.verbose && (m8 || m4)) {
            std::fprintf(stderr, "strata-inkling2: integer decode kernels (--dp4a on|off|auto):");
            if (m8) std::fprintf(stderr, " Q8_0 matrix %.3f ms (float kernel %.3f): %s;", i8, f8, dp_ok[0] ? "used" : "not used");
            if (m4) std::fprintf(stderr, " NVFP4 expert matrix %.3f ms (float kernel %.3f): %s;", i4, f4, dp_ok[1] ? "used" : "not used");
            std::fprintf(stderr, "\n");
            std::fprintf(stderr, "strata-inkling2: integer prompt kernels, 16 tokens:");
            if (m8) std::fprintf(stderr, " Q8_0 matrix %.3f ms (float kernel %.3f): %s;", pi8, pf8, pf_ok[0] ? "used" : "not used");
            if (m4) std::fprintf(stderr, " NVFP4 expert matrix %.3f ms (float kernel %.3f): %s;", pi4, pf4, pf_ok[1] ? "used" : "not used");
            std::fprintf(stderr, "\n");
        }
    }

    /// The CPU expert pool, if it pays: a decode step's cold expert is computed on the CPU instead of copied to the card when that is faster here (measured once: an
    /// expert of a streamed layer evaluated by the pool against the same bytes copied over PCIe).
    void setup_cpu_pool() {
        if (ec.slots == 0 || opt.cpu_threads == 0) return;
        int ip = -1;
        for (int i = 0; i < hp.n_layer && ip < 0; ++i) if (is_streamed[(size_t) i]) ip = i;
        if (ip < 0) return;
        const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
        const int nt = opt.cpu_threads > 0 ? opt.cpu_threads : (int) std::max(2u, std::min(8u, hw - 1));
        pool.reset(new i2cpu::Pool(nt));
        const DevLayer& dl = L[(size_t) ip];
        const int E = hp.n_embd, nj = std::min(4, hp.n_expert);
        std::vector<float> x((size_t) E), y((size_t) nj * (size_t) E);
        for (int i = 0; i < E; ++i) x[(size_t) i] = std::sin(0.37f * (float) i);
        std::vector<i2cpu::Job> jobs;
        for (int j = 0; j < nj; ++j)
            jobs.push_back({dl.h_gu + (size_t) j * dl.gate_up.expert_bytes, dl.gate_up.type, dl.gate_up.row_bytes, dl.h_dn + (size_t) j * dl.down.expert_bytes, dl.down.type,
                            dl.down.row_bytes, y.data() + (size_t) j * (size_t) E, 1.f});
        double t_cpu = 1e300, t_copy = 1e300;
        for (int rep = 0; rep < 4; ++rep) {
            const double t0 = ms_now();
            pool->run(x.data(), E, hp.n_ff_exp, hp.situ_beta, hp.situ_lin, jobs.data(), nj);
            t_cpu = std::min(t_cpu, (ms_now() - t0) / nj);
            const double t1 = ms_now();
            CK(cudaMemcpy(stgA, dl.h_gu, dl.gate_up.expert_bytes, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(stgC, dl.h_dn, dl.down.expert_bytes, cudaMemcpyHostToDevice));
            t_copy = std::min(t_copy, ms_now() - t1);
        }
        cpu_on = opt.cpu_force || t_cpu < t_copy;
        t_cpu_ms = opt.cpu_force ? 1.0 : t_cpu;
        t_copy_ms = opt.cpu_force ? 1.0 : t_copy;
        if (opt.verbose)
            std::fprintf(stderr, "strata-inkling2: CPU expert pool (%d threads%s): an expert takes %.2f ms on the CPU, %.2f ms to copy to the card: %s\n", nt,
                         have_avx2_s() ? ", AVX2" : ", portable code", t_cpu, t_copy, cpu_on ? "cold experts are computed on the CPU" : "off, the copy is faster here");
        if (!cpu_on) pool.reset();
    }
    static bool have_avx2_s() { return i2cpu::have_avx2(); }

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
            if (w.gu_scale) { d.gu_scale = uvec(w.gu_scale, (size_t) hp.n_expert); d.gu_scale_h = w.gu_scale; }
            if (w.dn_scale) d.dn_scale = uvec(w.dn_scale, (size_t) hp.n_expert);
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
            const bool local = hp.swa[(size_t) hp.phys(v)] != 0;
            V.cells = local ? local_cells() : ctx;
            V.kvq = local ? 0 : opt.kvq;                        // (the sliding-window layers' rings are small: halves)
            const size_t cn = (size_t) KV * (size_t) V.cells;
            if (V.kvq == 0) {
                V.kc = alloc<uint16_t>(cn * (size_t) hd);
                V.vc = alloc<uint16_t>(cn * (size_t) hd);
            } else {
                V.kq = alloc<uint8_t>(cn * (size_t) (V.kvq == 1 ? hd : hd / 2));
                V.vq = alloc<uint8_t>(cn * (size_t) (V.kvq == 1 ? hd : hd / 2));
                V.ks = alloc<uint16_t>(cn * (size_t) (hd / 32));
                V.vs = alloc<uint16_t>(cn * (size_t) (hd / 32));
            }
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
        mixstat = alloc<float>(T * (size_t) i2k::kMaxStates * (size_t) (1 + i2k::kMaxQueries));
        n_qw = ((std::max(qd, E) + 255) / 256) * 256;                                   // the integer kernels' activations: 4 slots of one input, then the experts' F-wide rows
        const size_t qe = (size_t) std::max(8, hp.n_used) * (size_t) hp.n_ff_exp;
        xq_ = alloc<int8_t>((size_t) 4 * (size_t) n_qw + std::max<size_t>(qe, (size_t) n_qw) + 64);
        xd_ = alloc<float>(((size_t) 4 * (size_t) n_qw + std::max<size_t>(qe, (size_t) n_qw)) / 32 + 16);
        {   // a prompt chunk's: 4 slots of chunk rows, and the experts' pair rows
            const size_t sl = (size_t) chunk * (size_t) n_qw, pe = (size_t) chunk * (size_t) hp.n_used * (size_t) hp.n_ff_exp;
            pxq_ = alloc<int8_t>(4 * sl + 64);
            pxd_ = alloc<float>(4 * sl / 32 + 16);
            pxe_ = alloc<int8_t>(pe + 64);
            pxed_ = alloc<float>(pe / 32 + 16);
        }
        ctrl_gate = alloc<float>(T * (size_t) std::max(hp.gate_rank, 1));
        pm_ = alloc<float>((size_t) H * (size_t) std::max(1, opt.attn_max_split));
        pl_ = alloc<float>((size_t) H * (size_t) std::max(1, opt.attn_max_split));
        pacc_ = alloc<float>((size_t) H * (size_t) std::max(1, opt.attn_max_split) * (size_t) hd);
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
            CK(cudaStreamCreateWithFlags(&stc, cudaStreamNonBlocking));
            for (int q = 0; q < 2; ++q) {
                CK(cudaEventCreateWithFlags(&ev_copied[q], cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&ev_used[q], cudaEventDisableTiming));
                CK(cudaEventRecord(ev_copied[q], st));
                CK(cudaEventRecord(ev_used[q], st));
            }
            tab_cap_g = (size_t) (T * (size_t) hp.n_used / kExpertTile + (size_t) stage_slots + 8);
            tab_cap_e = (size_t) (T * (size_t) hp.n_used + 16 * (size_t) stage_slots + 16);
            for (int q = 0; q < 2; ++q) {
                for (int k = 0; k < 3; ++k) dtab[q][k] = alloc<int>(tab_cap_g);
                for (int k = 3; k < 5; ++k) dtab[q][k] = alloc<int>(tab_cap_e);
                void* hp_ = nullptr;
                CK(cudaHostAlloc(&hp_, (3 * tab_cap_g + 2 * tab_cap_e) * 4 + 64, cudaHostAllocPortable));
                host_allocs.push_back({hp_, true});
                htab[q] = static_cast<int*>(hp_);
            }
            int want = opt.cache_slots >= 0 ? opt.cache_slots : (int) std::min<size_t>(cache_budget / std::max<size_t>(per_expert_bytes(), 1), 1u << 20);
            int streamed_experts = 0;
            for (int ip = 0; ip < hp.n_layer; ++ip) if (is_streamed[(size_t) ip]) streamed_experts += hp.n_expert;
            want = std::min(want, streamed_experts);
            if (want >= hp.n_used) {                       // (fewer slots than one step needs: no cache)
                ec.slots = want;
                ec.a = alloc<uint8_t>(max_gu_bytes() * (size_t) (want + hp.n_used));       // (+ n_used transient slots: cold experts the PCIe takes its share of)
                ec.c = alloc<uint8_t>(max_dn_bytes() * (size_t) (want + hp.n_used));
                ec.key.assign((size_t) want, -1);
                freq.assign((size_t) hp.n_layer * (size_t) hp.n_expert, 0.0);
                ec.where.assign((size_t) hp.n_layer * (size_t) hp.n_expert, -1);
                cache_slot_dev = alloc<int>((size_t) hp.n_used);
                {   // pinned host buffers of the decode path (a copy from pageable memory is staged, and ordered behind what is queued)
                    void* p1 = nullptr;
                    CK(cudaHostAlloc(&p1, (size_t) (hp.n_embd + hp.n_used * hp.n_embd) * 4 + 4096, cudaHostAllocPortable));
                    host_allocs.push_back({p1, true});
                    xh_host = static_cast<float*>(p1);
                    cpu_y = xh_host + hp.n_embd;
                    void* p2 = nullptr;
                    CK(cudaHostAlloc(&p2, (size_t) hp.n_used * 4 + 64, cudaHostAllocPortable));
                    host_allocs.push_back({p2, true});
                    slot_pin = static_cast<int*>(p2);
                }
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
        setup_dp();
        setup_cpu_pool();
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

    // ---- integer decode kernels (Q8_0 and NVFP4 matrices, one token): the activations quantized once per input, then dp4a rows
    int8_t* xq_ = nullptr;                                // [5 slots of n_qw values] + the experts' activation rows
    float* xd_ = nullptr;
    int n_qw = 0;
    bool dp_ok[2] = {false, false};                       // Q8_0, NVFP4: the integer kernel is used (a speed check at start decides, --dp4a on|off forces)
    bool dp_any = false;
    bool dp_type(int t) const { return hp.n_embd <= 4096 && hp.q_dim() <= 4096 && ((t == g4::T_Q8_0 && dp_ok[0]) || (t == g4::T_NVFP4 && dp_ok[1])); }
    bool pf_ok[2] = {false, false};                       // the same for reading a prompt (k_mmq): Q8_0, NVFP4
    bool pf_any = false;
    bool pf_type(int t) const { return hp.n_embd <= 4096 && hp.q_dim() <= 4096 && ((t == g4::T_Q8_0 && pf_ok[0]) || (t == g4::T_NVFP4 && pf_ok[1])); }
    int8_t* pxq_ = nullptr;                               // a prompt chunk's quantized activations: 4 slots of chunk * n_qw values, then the experts' pair rows
    float* pxd_ = nullptr;
    int cur_Tn = 1;
    int8_t* pxe_ = nullptr;                               // the experts' activation rows of a prompt chunk (one per pair)
    float* pxed_ = nullptr;
    int8_t* pslot(int i) const { return pxq_ + (size_t) i * (size_t) chunk * (size_t) n_qw; }
    float* pdslot(int i) const { return pxd_ + (size_t) i * (size_t) chunk * (size_t) (n_qw / 32); }
    int8_t* qslot(int i) const { return xq_ + (size_t) i * (size_t) n_qw; }
    float* dslot(int i) const { return xd_ + (size_t) i * (size_t) (n_qw / 32); }
    void prep(const float* x, int n, int slot) {         // x[n] -> slot `slot`
        G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((n / 32 + 7) / 8), 1u), dim3(256), st, x, n, qslot(slot), dslot(slot));
    }
    void mv(const DevMat& W, const int8_t* xq, const float* xd, long long x_stride, const int* ge, int entries, float* y, int ldy, int y_off, long long expert_bytes) {
        const dim3 grid((unsigned) ((W.n_out + 7) / 8), (unsigned) entries), block(256);
        if (W.type == g4::T_Q8_0)
            G4_LAUNCH(G4_K(i2k::k_mv<g4::T_Q8_0>), grid, block, st, W.d, expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, xq, xd, x_stride, ge, y, ldy, y_off);
        else
            G4_LAUNCH(G4_K(i2k::k_mv<g4::T_NVFP4>), grid, block, st, W.d, expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, xq, xd, x_stride, ge, y, ldy, y_off);
    }
    void prep_any(const float* x, int n, int Tn, int slot) {      // the Tn tokens of x into slot `slot` of the decode or the prompt buffers
        if (Tn == 1) prep(x, n, slot);
        else G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((n / 32 + 7) / 8), (unsigned) Tn), dim3(256), st, x, n, pslot(slot), pdslot(slot));
    }
    bool q_on(int Tn) const { return Tn == 1 ? dp_any : pf_any; }
    template <int T>
    void mmq_t(const DevMat& W, const int8_t* xq, const float* xd, int ng, const int* ge, const int* gf, const int* gn, const int* ei, const int* eo, int ntok, float* y, int ldy, int y_off,
               long long expert_bytes) {
        G4_LAUNCH(G4_K((i2k::k_mmq<T, 16>)), dim3((unsigned) ((W.n_out + 15) / 16), (unsigned) ng), dim3(256), st, W.d, expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, xq, xd, ge, gf, gn, ei, eo,
                  ntok, y, ldy, y_off);
    }
    void mmq(const DevMat& W, const int8_t* xq, const float* xd, int ng, const int* ge, const int* gf, const int* gn, const int* ei, const int* eo, int ntok, float* y, int ldy, int y_off,
             long long expert_bytes) {
        if (W.type == g4::T_Q8_0) mmq_t<g4::T_Q8_0>(W, xq, xd, ng, ge, gf, gn, ei, eo, ntok, y, ldy, y_off, expert_bytes);
        else mmq_t<g4::T_NVFP4>(W, xq, xd, ng, ge, gf, gn, ei, eo, ntok, y, ldy, y_off, expert_bytes);
    }
    /// y = W x for the Tn tokens of x (ldx), whose activations are in `slot` (-1: not quantized): one token and a type with an integer kernel: that; a prompt chunk and k_mmq: that; else the float kernel
    void mmd(const DevMat& W, const float* x, int ldx, int Tn, float* y, int ldy, int slot, int y_off = 0) {
        if (slot >= 0 && Tn == 1 && dp_type(W.type)) mv(W, qslot(slot), dslot(slot), 0, nullptr, 1, y, ldy, y_off, 0);
        else if (slot >= 0 && Tn > 1 && pf_type(W.type)) mmq(W, pslot(slot), pdslot(slot), (Tn + 15) / 16, nullptr, nullptr, nullptr, nullptr, nullptr, Tn, y, ldy, y_off, 0);
        else mm(W, x, ldx, Tn, y, ldy, y_off);
    }

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
        G4_LAUNCH(i2k::k_mix_stats, dim3((unsigned) S, (unsigned) Tn), dim3(i2k::kMixThreads), st, s[0], s[1], s[2], s[3], s[4], S, hp.n_embd, queries, nq, mixstat);
        G4_LAUNCH(i2k::k_mix_apply, dim3((unsigned) ((hp.n_embd + 255) / 256), (unsigned) Tn), dim3(i2k::kMixThreads), st, s[0], s[1], s[2], s[3], s[4], S, hp.n_embd, nq,
                  (const float*) mixstat, out, Tn, 1e-6f);
    }
    float *mixstat = nullptr, *ctrl_gate = nullptr;            // the mix's per-state sums, the controller's gate (per token)
    void control(float* c, const float* msg, int Tn, const DevLoop& lw, int k) {
        G4_LAUNCH(i2k::k_ctrl_gate, dim3((unsigned) ((hp.gate_rank + 3) / 4), (unsigned) Tn), dim3(128), st, (const float*) c, msg, hp.n_embd, hp.gate_rank, (const float*) lw.down[k], ctrl_gate,
                  1e-6f);
        G4_LAUNCH(i2k::k_ctrl_apply, dim3((unsigned) ((hp.n_embd + 255) / 256), (unsigned) Tn), dim3(i2k::kMixThreads), st, c, msg, hp.n_embd, hp.gate_rank, (const float*) lw.gain[k],
                  (const float*) lw.forget[k], (const float*) ctrl_gate);
    }
    void ring_store(const float* buf, int n, float* cring, int seg_off, int Tn, int pos0) {
        G4_LAUNCH(i2k::k_ring_store, dim3((unsigned) ((n + 127) / 128), (unsigned) Tn), dim3(g4k::kThreads), st, buf, n, n, cring, hp.conv_channels(), seg_off, pos0, Tn);
    }
    void conv_res(const float* in, const float* w, const float* cring, int seg_off, int Tn, int pos0, const float* resid, float* out) {
        const int E = hp.n_embd;
        G4_LAUNCH(i2k::k_conv_res, dim3((unsigned) ((E + 127) / 128), (unsigned) Tn), dim3(g4k::kThreads), st, in, E, w, hp.n_conv, cring, hp.conv_channels(), seg_off, pos0, resid, out);
    }

    template <int NU, int KVQ>
    void attention_nu(const DevLayer& dl, const Visit& V, int Tn, int pos0, int window, int extent, int nsplit) {
        const dim3 grid((unsigned) Tn, (unsigned) hp.n_head, (unsigned) nsplit), block(g4k::kThreads);
        G4_LAUNCH(G4_K((i2k::k_attn<NU, KVQ>)), grid, block, st, q, r, dl.rel_bias, sink, gate, (const uint16_t*) V.kc, (const uint16_t*) V.vc, (const uint8_t*) V.kq, (const uint8_t*) V.vq,
                  (const uint16_t*) V.ks, (const uint16_t*) V.vs, att, hp.n_head, hp.n_kv, hp.n_rel, extent, pos0, V.cells, window, nsplit, pm_, pl_, pacc_);
        if (nsplit > 1) G4_LAUNCH(G4_K(i2k::k_attn_merge<NU>), dim3((unsigned) hp.n_head), dim3(g4k::kThreads), st, (const float*) gate, (const float*) pm_, (const float*) pl_, (const float*) pacc_, att, nsplit);
    }
    template <int KVQ>
    void attention_kvq(const DevLayer& dl, const Visit& V, int Tn, int pos0, int window, int extent, int nsplit) {
        switch (hp.hd / 32) {
            case 1: attention_nu<1, KVQ>(dl, V, Tn, pos0, window, extent, nsplit); break;
            case 2: attention_nu<2, KVQ>(dl, V, Tn, pos0, window, extent, nsplit); break;
            case 3: attention_nu<3, KVQ>(dl, V, Tn, pos0, window, extent, nsplit); break;
            default: attention_nu<4, KVQ>(dl, V, Tn, pos0, window, extent, nsplit); break;
        }
    }
    void attention(int ip, const Visit& V, int Tn, int pos0) {
        const DevLayer& dl = L[(size_t) ip];
        const bool swa = hp.swa[(size_t) ip] != 0;
        const int window = swa ? hp.window : 0, extent = swa ? hp.window : hp.rel_extent;
        // one decoded token over many keys: the keys are cut into parts, a block each (24 blocks alone cannot fill a card), then merged
        int nsplit = 1;
        if (Tn == 1) {
            const int nkeys = window > 0 ? std::min(window, pos0 + 1) : pos0 + 1;
            nsplit = std::max(1, std::min(opt.attn_max_split, nkeys / std::max(1, opt.attn_split_keys)));
        }
        switch (V.kvq) {
            case 0: attention_kvq<0>(dl, V, Tn, pos0, window, extent, nsplit); break;
            case 1: attention_kvq<1>(dl, V, Tn, pos0, window, extent, nsplit); break;
            default: attention_kvq<2>(dl, V, Tn, pos0, window, extent, nsplit); break;
        }
    }
    float *pm_ = nullptr, *pl_ = nullptr, *pacc_ = nullptr;     // the parts of a split attention (per head and part)

    // ------------------------------------------------------------------------------------------------------------------------ experts
    const float* cur_gu_scale = nullptr;                 // the layer's per-expert scalar on its gate+up product (null: none)
    void exp_ffn(const DevMat& gu_m, const DevMat& dn, int ng, bool decode, const int* ge, const int* gf, const int* gn, const int* et, const int* er, int rows,
                 const int* situ_rows) {
        const int E = hp.n_embd, F = hp.n_ff_exp;
        if (decode && dp_type(gu_m.type) && dp_type(dn.type) && F % 32 == 0) {    // one token: the experts' rows by integer arithmetic (group g is entry g: the tables are identities)
            prep(n0, E, 0);
            mv(gu_m, qslot(0), dslot(0), 0, ge, ng, gu, 2 * F, 0, (long long) gu_m.expert_bytes);
            G4_LAUNCH(i2k::k_situ, dim3((unsigned) ((F + 127) / 128), (unsigned) rows), dim3(g4k::kThreads), st, (const float*) gu, eact, F, hp.situ_beta, hp.situ_lin, situ_rows,
                      (const float*) cur_gu_scale, (const int*) topk_idx);
            int8_t* eq = qslot(4);
            float* ed_ = dslot(4);
            G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((F / 32 + 7) / 8), (unsigned) ng), dim3(256), st, (const float*) eact, F, eq, ed_);
            mv(dn, eq, ed_, (long long) F, ge, ng, ed, E, 0, (long long) dn.expert_bytes);
            return;
        }
        if (!decode && pf_type(gu_m.type) && pf_type(dn.type) && F % 32 == 0) {          // a prompt chunk: the experts' groups by k_mmq
            prep_any(n0, E, cur_Tn, 0);
            mmq(gu_m, pslot(0), pdslot(0), ng, ge, gf, gn, et, er, cur_Tn, gu, 2 * F, 0, (long long) gu_m.expert_bytes);
            G4_LAUNCH(i2k::k_situ, dim3((unsigned) ((F + 127) / 128), (unsigned) rows), dim3(g4k::kThreads), st, (const float*) gu, eact, F, hp.situ_beta, hp.situ_lin, situ_rows,
                      (const float*) cur_gu_scale, (const int*) topk_idx);
            G4_LAUNCH(i2k::k_quant_x, dim3((unsigned) ((F / 32 + 7) / 8), (unsigned) (cur_Tn * hp.n_used)), dim3(256), st, (const float*) eact, F, pxe_, pxed_);
            mmq(dn, pxe_, pxed_, ng, ge, gf, gn, er, er, cur_Tn * hp.n_used, ed, E, 0, (long long) dn.expert_bytes);
            return;
        }
        mm_exp(gu_m, n0, E, ng, decode, ge, gf, gn, et, er, gu, 2 * F);
        G4_LAUNCH(i2k::k_situ, dim3((unsigned) ((F + 127) / 128), (unsigned) rows), dim3(g4k::kThreads), st, (const float*) gu, eact, F, hp.situ_beta, hp.situ_lin, situ_rows, (const float*) cur_gu_scale, (const int*) topk_idx);
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
        cur_gu_scale = dl.gu_scale;
        cur_Tn = Tn;
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
                  (const float*) dl.dn_scale, nu, E, moe);
    }
    /// The experts of a layer in host memory: the router's choice is read back, the experts it names are copied into the staging slots (the n_used of a decode
    /// step; for a prompt chunk the experts in batches of stage_slots, each with the groups of pairs that use it).
    void experts_streamed(const DevLayer& dl, int ip, int Tn) {
        const int nu = hp.n_used;
        const int rows = Tn * nu;
        std::vector<int> idx((size_t) rows);
        const double tw0 = opt.profile ? ms_now() : 0.0;
        CK(cudaMemcpy(idx.data(), topk_idx, (size_t) rows * 4, cudaMemcpyDeviceToHost));       // (also the barrier: the router has run)
        const double tw1 = opt.profile ? ms_now() : 0.0;
        if (opt.profile && Tn == 1) pfk[0].wait += tw1 - tw0;
        auto copy_in = [&](int slot, int e) {
            CK(cudaMemcpyAsync(stgA + (size_t) slot * dl.gate_up.expert_bytes, dl.h_gu + (size_t) e * dl.gate_up.expert_bytes, dl.gate_up.expert_bytes, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(stgC + (size_t) slot * dl.down.expert_bytes, dl.h_dn + (size_t) e * dl.down.expert_bytes, dl.down.expert_bytes, cudaMemcpyHostToDevice, st));
        };
        DevMat vgu = dl.gate_up, vdn = dl.down;
        vgu.d = stgA;
        vdn.d = stgC;
        if (Tn == 1 && ec.slots >= nu) {                   // decode: the experts the cache has are used where they are; each other one is either copied in (it is used more than
                                                           // the cache's least used) or computed on the CPU, from the host copy, while the GPU runs the rest
            const int E = hp.n_embd;
            std::vector<int> sl((size_t) nu, -1);
            std::vector<char> cpu((size_t) nu, 0);
            for (int k = 0; k < nu; ++k) {
                const int key = ip * hp.n_expert + idx[(size_t) k];
                freq[(size_t) key] += ec.inc;
                sl[(size_t) k] = ec.where[(size_t) key];                // the hits first: a miss must not take the slot of an expert this step uses
                ec.hits += sl[(size_t) k] >= 0;
            }
            auto taken = [&](int j) { for (int q = 0; q < nu; ++q) if (sl[(size_t) q] == j && !cpu[(size_t) q]) return true; return false; };
            int ncpu = 0, nprom = 0, nshare = 0;
            std::vector<std::pair<int, int>> todo;                     // (slot, expert) copies to make, queued only after the CPU's input has been read back (below)
            for (int k = 0; k < nu; ++k) {
                if (sl[(size_t) k] >= 0) continue;
                ++ec.misses;
                const int e = idx[(size_t) k], key = ip * hp.n_expert + e;
                double best = 1e300;
                int s = -1;
                for (int j = 0; j < ec.slots; ++j) {
                    if (taken(j)) continue;
                    const double sc = ec.key[(size_t) j] >= 0 ? freq[(size_t) ec.key[(size_t) j]] : 0.0;
                    if (sc < best) { best = sc; s = j; }
                }
                if (cpu_on && s >= 0 && freq[(size_t) key] <= best) {   // not used more than the one it would replace: not worth a place
                    if ((double) (nprom + nshare + 1) * t_copy_ms < (double) (ncpu + 1) * t_cpu_ms) {
                        // the PCIe finishes it sooner than the CPU would (its queue counted): it takes this one, into a transient slot, outside the cache's books
                        const int ts = ec.slots + nshare++;
                        todo.push_back({ts, e});
                        sl[(size_t) k] = ts;
                        ++shared_evals;
                        continue;
                    }
                    cpu[(size_t) k] = 1;
                    sl[(size_t) k] = 0;                                  // (the kernels compute something for it; its row is overwritten by the CPU's)
                    ++ncpu;
                    ++cpu_evals;
                    continue;
                }
                if (ec.key[(size_t) s] >= 0) ec.where[(size_t) ec.key[(size_t) s]] = -1;
                ec.key[(size_t) s] = key;
                ec.where[(size_t) key] = s;
                todo.push_back({s, e});
                sl[(size_t) k] = s;
                ++nprom;
            }
            // The CPU's input first: a synchronous read-back waits for everything queued before it, the expert copies included, and the CPU would start only when the PCIe was done.
            if (ncpu) CK(cudaMemcpy(xh_host, n0, (size_t) E * 4, cudaMemcpyDeviceToHost));
            for (const auto& c : todo) {
                CK(cudaMemcpyAsync(ec.a + (size_t) c.first * max_gu_bytes(), dl.h_gu + (size_t) c.second * dl.gate_up.expert_bytes, dl.gate_up.expert_bytes, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(ec.c + (size_t) c.first * max_dn_bytes(), dl.h_dn + (size_t) c.second * dl.down.expert_bytes, dl.down.expert_bytes, cudaMemcpyHostToDevice, st));
            }
            for (int k = 0; k < nu; ++k) slot_pin[(size_t) k] = sl[(size_t) k];
            CK(cudaMemcpyAsync(cache_slot_dev, slot_pin, (size_t) nu * 4, cudaMemcpyHostToDevice, st));
            DevMat cgu = dl.gate_up, cdn = dl.down;
            cgu.d = ec.a; cgu.expert_bytes = max_gu_bytes();
            cdn.d = ec.c; cdn.expert_bytes = max_dn_bytes();
            exp_ffn(cgu, cdn, nu, true, cache_slot_dev, dec_first, dec_n, dec_tok, dec_row, rows, nullptr);
            if (ncpu) {                                              // (the GPU is busy with the launches above while the CPU works)
                std::vector<i2cpu::Job> jobs;
                std::vector<int> ks;
                for (int k = 0; k < nu; ++k) {
                    if (!cpu[(size_t) k]) continue;
                    const int e = idx[(size_t) k];
                    jobs.push_back({dl.h_gu + (size_t) e * dl.gate_up.expert_bytes, dl.gate_up.type, dl.gate_up.row_bytes, dl.h_dn + (size_t) e * dl.down.expert_bytes, dl.down.type,
                                    dl.down.row_bytes, cpu_y + (size_t) k * (size_t) E, dl.gu_scale_h ? dl.gu_scale_h[e] : 1.f});
                    ks.push_back(k);
                }
                const double tc0 = opt.profile ? ms_now() : 0.0;
                pool->run(xh_host, E, hp.n_ff_exp, hp.situ_beta, hp.situ_lin, jobs.data(), (int) jobs.size());
                if (opt.profile) pfk[0].cpu += ms_now() - tc0;
                for (int k : ks) CK(cudaMemcpyAsync(ed + (size_t) k * (size_t) E, cpu_y + (size_t) k * (size_t) E, (size_t) E * 4, cudaMemcpyHostToDevice, st));
            }
            return;
        }
        if (Tn == 1) {
            for (int k = 0; k < nu; ++k) copy_in(k, idx[(size_t) k]);
            exp_ffn(vgu, vdn, nu, true, dec_slots, dec_first, dec_n, dec_tok, dec_row, rows, nullptr);
            return;
        }
        // A prompt chunk: the experts the chunk uses, in batches of half the staging slots.  Two halves take turns: the copy stream fills one (and uploads its group tables)
        // while the compute stream works on the other, so that the PCIe and the card overlap.
        std::vector<std::vector<int>> by((size_t) hp.n_expert);
        for (int rr = 0; rr < rows; ++rr) by[(size_t) idx[(size_t) rr]].push_back(rr);
        std::vector<int> used;
        for (int e = 0; e < hp.n_expert; ++e)
            if (!by[(size_t) e].empty()) used.push_back(e);
        const int half = std::max(1, stage_slots / 2);
        const int nbat = (int) ((used.size() + (size_t) half - 1) / (size_t) half);
        struct Batch { std::vector<int> gex, gfi, gnn, ent_t, ent_r; };
        std::vector<Batch> bat((size_t) nbat);
        auto issue = [&](int bi) {                                             // on the copy stream: this batch's tables and experts, then the event the compute waits for
            const int p = bi & 1;
            Batch& B = bat[(size_t) bi];
            const size_t b0 = (size_t) bi * (size_t) half, nb = std::min<size_t>((size_t) half, used.size() - b0);
            for (size_t k = 0; k < nb; ++k) {
                const auto& v = by[(size_t) used[b0 + k]];
                for (size_t s2 = 0; s2 < v.size(); s2 += kExpertTile) {
                    const size_t n = std::min<size_t>(kExpertTile, v.size() - s2);
                    B.gex.push_back((int) k);                                    // the slot of the half: what the kernels index experts by here
                    B.gfi.push_back((int) B.ent_t.size());
                    B.gnn.push_back((int) n);
                    for (size_t j = 0; j < n; ++j) { B.ent_r.push_back(v[s2 + j]); B.ent_t.push_back(v[s2 + j] / nu); }
                }
            }
            CK(cudaEventSynchronize(ev_copied[p]));                              // (the pinned tables of this parity are free: the copy that read them has run)
            CK(cudaStreamWaitEvent(stc, ev_used[p], 0));                         // (and so is the half of the staging: the compute that read it has run)
            int* h = htab[p];
            auto put = [&](int slot, const std::vector<int>& v) {                 // (the pinned block: three group tables of tab_cap_g, then two entry tables of tab_cap_e)
                int* at = h + (slot < 3 ? (size_t) slot * tab_cap_g : 3 * tab_cap_g + (size_t) (slot - 3) * tab_cap_e);
                std::copy(v.begin(), v.end(), at);
                CK(cudaMemcpyAsync(dtab[p][slot], at, v.size() * 4, cudaMemcpyHostToDevice, stc));
            };
            put(0, B.gex); put(1, B.gfi); put(2, B.gnn); put(3, B.ent_t); put(4, B.ent_r);
            for (size_t k = 0; k < nb; ++k) {
                const int e = used[b0 + k];
                CK(cudaMemcpyAsync(stgA + ((size_t) p * (size_t) half + k) * dl.gate_up.expert_bytes, dl.h_gu + (size_t) e * dl.gate_up.expert_bytes, dl.gate_up.expert_bytes, cudaMemcpyHostToDevice, stc));
                CK(cudaMemcpyAsync(stgC + ((size_t) p * (size_t) half + k) * dl.down.expert_bytes, dl.h_dn + (size_t) e * dl.down.expert_bytes, dl.down.expert_bytes, cudaMemcpyHostToDevice, stc));
            }
            CK(cudaEventRecord(ev_copied[p], stc));
        };
        if (nbat > 0) issue(0);
        for (int bi = 0; bi < nbat; ++bi) {
            if (bi + 1 < nbat) issue(bi + 1);                                    // the next batch's copy starts while this one computes
            const int p = bi & 1;
            const Batch& B = bat[(size_t) bi];
            CK(cudaStreamWaitEvent(st, ev_copied[p], 0));
            DevMat bgu = dl.gate_up, bdn = dl.down;
            bgu.d = stgA + (size_t) p * (size_t) half * dl.gate_up.expert_bytes;
            bdn.d = stgC + (size_t) p * (size_t) half * dl.down.expert_bytes;
            exp_ffn(bgu, bdn, (int) B.gex.size(), false, dtab[p][0], dtab[p][1], dtab[p][2], dtab[p][3], dtab[p][4], (int) B.ent_r.size(), dtab[p][4]);
            CK(cudaEventRecord(ev_used[p], st));
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
        const bool prof = opt.profile;
        prof_kind = Tn > 1 ? 1 : 0;
        double tv0 = 0.0;
        if (prof) { CK(cudaStreamSynchronize(st)); tv0 = ms_now(); lap_t = tv0; }
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
        lap_(prof, 0, dl.streamed);                               // (the mixes, controls and norms before the projections)
        int sk = -1, sv = -1;                                     // the quantized-activation slots of the k and v inputs (-1: not quantized)
        const bool qon = q_on(Tn);
        if (qon) {
            prep_any(nq0, E, Tn, 0);
            sk = nk == nq0 ? 0 : 1;
            if (sk == 1) prep_any(nk, E, Tn, 1);
            sv = nv == nq0 ? 0 : nv == nk ? sk : 2;
            if (sv == 2) prep_any(nv, E, Tn, 2);
        }
        mmd(dl.wq, nq0, E, Tn, q, qd, qon ? 0 : -1);
        mmd(dl.wk, nk, E, Tn, kraw, kd, sk);
        mmd(dl.wv, nv, E, Tn, vraw, kd, sv);
        mmd(dl.wrel, nq0, E, Tn, r, hp.n_head * hp.n_rel, qon ? 0 : -1);
        mmd(dl.wsink, nq0, E, Tn, sink, hp.n_head, qon ? 0 : -1);
        mmd(dl.wgate, nq0, E, Tn, gate, qd, qon ? 0 : -1);
        if (!swa) {
            mm(dl.kconv_proj, nk, E, Tn, ktap, hp.n_conv * hp.n_kv);
            mm(dl.vconv_proj, nv, E, Tn, vtap, hp.n_conv * hp.n_kv);
        }
        lap_(prof, 1, dl.streamed);                               // (the q, k, v, relative, sink and gate projections)
        G4_LAUNCH(i2k::k_prep, dim3((unsigned) Tn, (unsigned) (2 * hp.n_head + 2 * hp.n_kv)), b1, st, q, (const float*) kraw, (const float*) vraw, r, (const float*) ktap, (const float*) vtap,
                  (const float*) dl.kconv, (const float*) dl.vconv, (const float*) dl.q_clip, (const float*) dl.rel_clip, (const float*) V.cring, hp.conv_channels(), V.kc, V.vc, V.kq, V.vq, V.ks, V.vs, V.kvq, hp.n_head, hp.n_kv,
                  hp.hd, hp.n_rel, hp.n_conv, pos0, V.cells, !swa, hp.qk_clip, hp.rel_clip, hp.eps, hp.log_alpha, hp.log_floor);
        ring_store(kraw, kd, V.cring, 0, Tn, pos0);
        ring_store(vraw, kd, V.cring, kd, Tn, pos0);
        attention(ip, V, Tn, pos0);
        lap_(prof, 2, dl.streamed);                               // (the head preparation, the cache writes and the attention)
        if (qon) prep_any(att, qd, Tn, 3);
        mmd(dl.wo, att, qd, Tn, proj, E, qon ? 3 : -1);
        const bool push = ip % hp.block == 0;
        conv_res(proj, dl.post_conv, V.cring, 2 * kd, Tn, pos0, push ? nullptr : sa, cur);
        ring_store(proj, E, V.cring, 2 * kd, Tn, pos0);
        if (push) {
            copy_f(push_dst, sa, TE);
            blocks.push_back(push_dst);
        }
        lap_(prof, 3, dl.streamed);                               // (the output projection, its convolution, the residual)
        // the experts
        {
            std::vector<const float*> states = blocks;
            states.push_back(cur);
            mix(states, Tn, dl.ffn_res_query, 1, mixed);
        }
        if (msgs_in) control(mixed, msgs_in + (size_t) 3 * TE, Tn, *lw, 3);
        rms(mixed, dl.ffn_norm, n0, Tn);
        lap_(prof, 4, dl.streamed);                               // (the experts' mix and norm)
        experts(dl, ip, Tn);
        conv_res(moe, dl.ffn_post_conv, V.cring, 2 * kd + E, Tn, pos0, cur, cur);
        ring_store(moe, E, V.cring, 2 * kd + E, Tn, pos0);
        if (prof) {
            lap_(prof, 5, dl.streamed);                           // (the router, the experts, their sum and convolution)
            (dl.streamed ? pfk[prof_kind].str : pfk[prof_kind].res) += ms_now() - tv0;
            ++(dl.streamed ? pfk[prof_kind].nstr : pfk[prof_kind].nres);
        }
        if (trace) {
            CK(cudaStreamSynchronize(st));
            std::vector<float> h(TE);
            CK(cudaMemcpy(h.data(), cur, TE * 4, cudaMemcpyDeviceToHost));
            trace->push_back(h);
        }
    }

    // ------------------------------------------------------------------------------------------------------------------------ a chunk
    void forward_chunk(const int32_t* ids, int Tn, bool want_logits) {
        const double tf0 = opt.profile ? ms_now() : 0.0;
        if (ec.slots && Tn == 1) {                                 // a decoded token: the use counts age by one step
            ec.inc /= opt.cache_decay;
            if (ec.inc > 1e50) { for (double& x : freq) x /= ec.inc; ec.inc = 1.0; }
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
            if (dp_any) prep(xh, E, 0);
            mmd(head, xh, E, 1, logits, hp.n_vocab, dp_any ? 0 : -1);
            if (hp.softcap > 0.f) G4_LAUNCH(g4k::k_softcap, dim3((unsigned) ((hp.n_vocab + 127) / 128)), dim3(g4k::kThreads), st, logits, hp.n_vocab, hp.softcap);
            logits_ready = true;
        } else {
            logits_ready = false;
        }
        CK(cudaStreamSynchronize(st));
        if (opt.profile) { Prof& f = pfk[Tn > 1 ? 1 : 0]; f.total += ms_now() - tf0; f.tokens += Tn; ++f.chunks; }
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
bool GpuEngine::dp_active() const { return p_->dp_any || p_->pf_any; }
void GpuEngine::profile_reset() { p_->pfk[0] = Impl::Prof(); p_->pfk[1] = Impl::Prof(); }
std::string GpuEngine::profile_report() const {
    std::string out;
    static const char* ph[6] = {"norms+mixes", "projections", "prep+attention", "out proj+conv", "ffn mix+norm", "router+experts"};
    char b[900];
    for (int kind = 0; kind < 2; ++kind) {
        const auto& f = p_->pfk[kind];
        if (f.tokens == 0) continue;
        const double n = (double) f.tokens;
        if (kind == 0)
            std::snprintf(b, sizeof b,
                          "profile, per decoded token (%lld tokens, %.1f ms): resident-expert visits %.1f ms (%.1f visits, %.2f ms each); streamed-expert visits %.1f ms (%.1f visits, %.2f ms each), of which waiting for the "
                          "router's choice %.1f ms, the CPU pool %.1f ms; the rest (embedding, output head, sync) %.1f ms\n",
                          f.tokens, f.total / n, f.res / n, (double) f.nres / n, f.nres ? f.res / (double) f.nres : 0.0, f.str / n, (double) f.nstr / n, f.nstr ? f.str / (double) f.nstr : 0.0, f.wait / n, f.cpu / n,
                          (f.total - f.res - f.str) / n);
        else
            std::snprintf(b, sizeof b, "profile, prompt (%lld tokens in %lld chunks, %.1f ms a token, %.0f ms a chunk): resident-expert visits %.0f ms a chunk (%.2f s in all, %.1f ms a visit); streamed-expert visits %.0f ms a chunk (%.1f ms a visit)\n",
                          f.tokens, f.chunks, f.total / n, f.total / (double) f.chunks, f.res / (double) f.chunks, f.res / 1000.0, f.nres ? f.res / (double) f.nres : 0.0, f.str / (double) f.chunks, f.nstr ? f.str / (double) f.nstr : 0.0);
        out += b;
        for (int k = 0; k < 2; ++k) {
            const double vn = (double) (k ? f.nstr : f.nres);
            if (vn <= 0) continue;
            out += k ? "  a streamed visit:" : "  a resident visit:";
            for (int i = 0; i < 6; ++i) { std::snprintf(b, sizeof b, " %s %.3f", ph[i], f.lap[k][i] / vn); out += b; }
            out += " (ms)\n";
            if (kind == 0) {
                out += "  ... of which the host spent queueing it:";
                for (int i = 0; i < 6; ++i) { std::snprintf(b, sizeof b, " %.3f", f.enq[k][i] / vn); out += b; }
                out += " (ms)\n";
            }
        }
    }
    return out;
}
void GpuEngine::cache_stats(int* slots, long long* hits, long long* misses, long long* cpu, long long* shared) const {
    *slots = p_->ec.slots; *hits = p_->ec.hits; *misses = p_->ec.misses;
    if (cpu) *cpu = p_->cpu_evals;
    if (shared) *shared = p_->shared_evals;
}
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
    int bench_pp = 0, bench_tg = 0, bench_warm = 0;
    for (size_t i = 0; i < extra.size(); ++i) {
        const std::string& a = extra[i];
        auto val = [&]() -> std::string { return i + 1 < extra.size() ? extra[++i] : std::string(); };
        if (a == "--prefill" || a == "--chunk") { const std::string v = val(); if (v != "auto") o.chunk = std::max(1, std::atoi(v.c_str())); }
        else if (a == "--host-experts") o.host_experts = std::max(0, std::atoi(val().c_str()));
        else if (a == "--expert-cache-mib") o.cache_mib = std::max(0, std::atoi(val().c_str()));
        else if (a == "--profile") o.profile = true;
        else if (a == "--dp4a") { const std::string v = val(); o.dp4a = v == "off" ? 0 : v == "on" ? 1 : -1; }
        else if (a == "--cpu-experts") { const std::string v = val(); o.cpu_threads = v == "auto" ? -1 : std::max(0, std::atoi(v.c_str())); }
        else if (a == "--cache-decay") o.cache_decay = std::min(1.0, std::max(0.5, std::atof(val().c_str())));
        else if (a == "--swa-slack") o.slack = std::max(0, std::atoi(val().c_str()));
        else if (a == "--vram-reserve-mib") o.reserve_mib = std::max(0, std::atoi(val().c_str()));
        else if (a == "--check") check = true;
        else if (a == "--bench-pp") bench_pp = std::atoi(val().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(val().c_str());
        else if (a == "--bench-warmup") bench_warm = std::max(0, std::atoi(val().c_str()));
        else if (a == "--kv") {                         // the K / V of the global layers (the long ones); the sliding-window layers' rings stay halves
            const std::string v = val();
            if (v == "f16") o.kvq = 0;
            else if (v == "q8_0" || v == "q8") o.kvq = 1;
            else if (v == "q4_0" || v == "q4") o.kvq = 2;
            else std::fprintf(stderr, "strata-inkling2: --kv %s: f16, q8_0 or q4_0 (kept f16)\n", v.c_str());
        }
        else if (a == "--attn-split-keys") o.attn_split_keys = std::max(1, std::atoi(val().c_str()));
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
            { const std::string rep = eng.profile_report(); if (!rep.empty()) std::printf("%s", rep.c_str()); eng.profile_reset(); }
        }
        if (bench_tg > 0) {
            eng.reset();
            eng.process(ids.data(), 16, noop, &done);
            for (int w = 0; w < bench_warm && eng.pos() < ctx - bench_tg - 1; ++w) {      // decoded first, not timed: the expert cache fills up
                const int32_t nx = eng.greedy();
                int64_t d = 0;
                eng.process(&nx, 1, noop, &d);
            }
            eng.profile_reset();
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
            long long ch = 0, cm = 0, cc = 0, csh = 0;
            eng.cache_stats(&cs, &ch, &cm, &cc, &csh);
            { const std::string rep = eng.profile_report(); if (!rep.empty()) std::printf("%s", rep.c_str()); }
            if (cs) std::printf("expert cache: %d slots, %lld hits, %lld misses (%.1f%% hits); of the cold ones, %lld computed on the CPU, %lld copied by PCIe without a place in the cache\n", cs, ch, cm,
                                100.0 * (double) ch / (double) std::max<long long>(1, ch + cm), cc, csh);
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
        const double lim = eng.dp_active() ? 3e-2 : 2e-3;       // (with the integer kernels the activations carry 8-bit noise: --dp4a off for the exact comparison)
        rc = (rel < lim && a == b) ? 0 : 1;
        std::printf("check: %s\n", rc == 0 ? "ok" : eng.dp_active() ? "FAILED (more than 3e-2, or another best token)" : "FAILED (more than 2e-3, or another best token)");
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
