// g4_cuda.cu - the Gemma 4 engine on CUDA: the layers spread over the cards, the weights as the file stores them (read by the kernels in
// g4_kernels.cuh), K / V as halves.  The math is the CPU reference's (g4_ref.hpp); tests/gemma4/g4_emu_test.cpp runs this very file on the
// CPU emulation of CUDA against it.
#include "g4_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <type_traits>

#include "g4_cuda_compat.hpp"
#include "g4_kernels.cuh"
#include "g4_ref.hpp"
#include "g4_serve.hpp"

#ifndef STRATA_G4_VERSION
#define STRATA_G4_VERSION "0.1.0-gemma4"
#endif

namespace g4cuda {

namespace {

#define CK(call)                                                                                                       \
    do {                                                                                                               \
        const cudaError_t e_ = (call);                                                                                 \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-gemma4: CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

template <class F>
void with_type(int t, F&& f) {
    switch (t) {
        case g4::T_F32: f(std::integral_constant<int, g4::T_F32>{}); break;
        case g4::T_F16: f(std::integral_constant<int, g4::T_F16>{}); break;
        case g4::T_BF16: f(std::integral_constant<int, g4::T_BF16>{}); break;
        case g4::T_Q4_0: f(std::integral_constant<int, g4::T_Q4_0>{}); break;
        case g4::T_Q4_1: f(std::integral_constant<int, g4::T_Q4_1>{}); break;
        case g4::T_Q5_0: f(std::integral_constant<int, g4::T_Q5_0>{}); break;
        case g4::T_Q5_1: f(std::integral_constant<int, g4::T_Q5_1>{}); break;
        case g4::T_Q8_0: f(std::integral_constant<int, g4::T_Q8_0>{}); break;
        case g4::T_Q2_K: f(std::integral_constant<int, g4::T_Q2_K>{}); break;
        case g4::T_Q3_K: f(std::integral_constant<int, g4::T_Q3_K>{}); break;
        case g4::T_Q4_K: f(std::integral_constant<int, g4::T_Q4_K>{}); break;
        case g4::T_Q5_K: f(std::integral_constant<int, g4::T_Q5_K>{}); break;
        case g4::T_Q6_K: f(std::integral_constant<int, g4::T_Q6_K>{}); break;
        case g4::T_IQ4_NL: f(std::integral_constant<int, g4::T_IQ4_NL>{}); break;
        case g4::T_IQ4_XS: f(std::integral_constant<int, g4::T_IQ4_XS>{}); break;
        case g4::T_IQ2_XXS: f(std::integral_constant<int, g4::T_IQ2_XXS>{}); break;
        case g4::T_IQ2_XS: f(std::integral_constant<int, g4::T_IQ2_XS>{}); break;
        case g4::T_IQ2_S: f(std::integral_constant<int, g4::T_IQ2_S>{}); break;
        case g4::T_IQ3_XXS: f(std::integral_constant<int, g4::T_IQ3_XXS>{}); break;
        case g4::T_IQ3_S: f(std::integral_constant<int, g4::T_IQ3_S>{}); break;
        case g4::T_MXFP4: f(std::integral_constant<int, g4::T_MXFP4>{}); break;
        default: std::fprintf(stderr, "strata-gemma4: no kernel for the type %d\n", t); std::exit(1);
    }
}

constexpr int kExpertTile = 16;               // (token, expert) pairs of one expert that a block reads the weights for at once, in prompt reading

struct DevMat {
    int type = -1;
    long long n_in = 0, n_out = 0, n_expert = 1;
    size_t row_bytes = 0, expert_bytes = 0, bytes = 0;
    uint8_t* d = nullptr;
    explicit operator bool() const { return d != nullptr; }
};

struct DevLayer {
    DevMat wq, wk, wv, wo, gate, up, down, gate_up, gate_e, up_e, down_e;
    float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *post_attn_norm = nullptr, *ffn_norm = nullptr, *post_ffw_norm = nullptr;
    float *pre2 = nullptr, *post1 = nullptr, *post2 = nullptr, *rscale = nullptr, *dscale = nullptr, *router = nullptr;
    float* blob = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr;
    int ring = 0;
    float out_scale = 1.f;
    int dev = 0;
};

struct Dev {
    int id = 0;
    cudaStream_t st = nullptr;
    std::vector<void*> allocs;
    // activations of a chunk
    float *h = nullptr, *h1 = nullptr, *xn = nullptr, *ye = nullptr, *q = nullptr, *kraw = nullptr, *vraw = nullptr, *ao = nullptr, *proj = nullptr,
          *mlp = nullptr, *moe = nullptr, *g = nullptr, *u = nullptr, *act = nullptr, *gu = nullptr, *eact = nullptr, *ed = nullptr, *topk_w = nullptr;
    int *topk_idx = nullptr, *grp_expert = nullptr, *grp_first = nullptr, *grp_n = nullptr, *ent_tok = nullptr, *ent_row = nullptr;
    int *dec_first = nullptr, *dec_n = nullptr, *dec_tok = nullptr, *dec_row = nullptr;     // the decode step's constant group tables
    float* rope_freqs = nullptr;
    // the head (the last device): the tied output matrix, the logits, the arg-max partials
    DevMat head;
    float *logits = nullptr, *pv = nullptr;
    int *pi = nullptr, *amax = nullptr;
    float* output_norm = nullptr;
    size_t bytes = 0;

    template <class T>
    T* alloc(size_t n) {
        void* p = nullptr;
        CK(cudaMalloc(&p, n * sizeof(T)));
        CK(cudaMemset(p, 0, n * sizeof(T)));
        allocs.push_back(p);
        bytes += n * sizeof(T);
        return static_cast<T*>(p);
    }
};

}  // namespace

struct GpuEngine::Impl {
    const g4::Model& m;
    const g4::Hparams& hp;
    int ctx;
    GpuOptions opt;
    std::vector<Dev> devs;
    std::vector<DevLayer> L;
    std::vector<int> dev_of;                  // layer -> index in devs
    int head_dev = 0;
    int64_t pos = 0;
    int chunk = 512;
    bool logits_ready = false;
    std::vector<float> hbuf;                   // host staging for the embeddings and the hops between devices
    std::vector<float> debug_h;                // test hook: hidden after each layer, last token
    bool debug = false;

    Impl(const g4::Model& mm, int c, GpuOptions o) : m(mm), hp(mm.hp), ctx(c), opt(std::move(o)) { chunk = std::max(1, opt.chunk); }

    ~Impl() {
        for (Dev& d : devs) {
            cudaSetDevice(d.id);
            for (void* p : d.allocs) cudaFree(p);
            if (d.st) cudaStreamDestroy(d.st);
        }
    }

    // ------------------------------------------------------------------------------------------------------------------------ placement
    size_t mat_bytes(const g4::Mat& a) const { return a ? a.bytes : 0; }
    size_t layer_weight_bytes(int il) const {
        const g4::LayerW& w = m.L[(size_t) il];
        size_t b = mat_bytes(w.wq) + mat_bytes(w.wk) + mat_bytes(w.wv) + mat_bytes(w.wo) + mat_bytes(w.ffn_gate) + mat_bytes(w.ffn_up) +
                   mat_bytes(w.ffn_down) + mat_bytes(w.gate_up_exps) + mat_bytes(w.gate_exps) + mat_bytes(w.up_exps) + mat_bytes(w.down_exps);
        b += 4 * ((size_t) 4 * hp.n_embd + 2 * hp.head_dim(il) + (w.has_moe ? ((size_t) 4 * hp.n_embd + hp.n_expert + (size_t) hp.n_expert * hp.n_embd) : 0));
        return b;
    }
    int ring_cells(int il) const {
        if (!hp.swa[(size_t) il]) return ctx;
        return std::min(ctx, hp.window + chunk + std::max(0, opt.slack));
    }
    size_t layer_kv_bytes(int il) const { return (size_t) 2 * 2 * hp.kv_heads(il) * (size_t) ring_cells(il) * hp.head_dim(il); }
    size_t activation_bytes() const {
        const size_t C = (size_t) chunk, E = (size_t) hp.n_embd;
        const size_t qmax = (size_t) hp.n_head * std::max(hp.hd_swa, hp.hd_full);
        size_t kvmax = 0;
        for (int il = 0; il < hp.n_layer; ++il) kvmax = std::max(kvmax, (size_t) hp.kv_dim(il));
        size_t f = C * (E * 8 + 2 * qmax + 2 * kvmax + (size_t) 3 * hp.n_ff);
        if (hp.n_expert > 0) f += C * (size_t) hp.n_used * ((size_t) 2 * hp.n_ff_exp + hp.n_ff_exp + E + 2);
        return f * 4 + (1 << 20);
    }
    size_t head_bytes() const { return m.tok_embd.bytes + (size_t) 4 * hp.n_vocab + (1 << 20); }

    std::string place() {
        int n_dev = 0;
        CK(cudaGetDeviceCount(&n_dev));
        std::vector<int> order = opt.devices;
        if (order.empty()) {
            std::vector<std::pair<size_t, int>> fr;
            for (int d = 0; d < n_dev; ++d) {
                cudaSetDevice(d);
                size_t f = 0, t = 0;
                CK(cudaMemGetInfo(&f, &t));
                fr.push_back({f, d});
            }
            std::stable_sort(fr.begin(), fr.end(), [](auto& a, auto& b) { return a.first > b.first; });
            for (auto& p : fr) order.push_back(p.second);
        }
        if (order.empty()) return "no CUDA device";
        for (int d : order)
            if (d < 0 || d >= n_dev) return "--gpu " + std::to_string(d) + ": only " + std::to_string(n_dev) + " CUDA device(s) visible";
        devs.assign(order.size(), Dev());
        std::vector<long long> budget(order.size());
        for (size_t i = 0; i < order.size(); ++i) {
            devs[i].id = order[i];
            cudaSetDevice(order[i]);
            size_t f = 0, t = 0;
            CK(cudaMemGetInfo(&f, &t));
            budget[i] = (long long) f - ((long long) opt.reserve_mib << 20) - (long long) activation_bytes();
        }
        // The layers fill the devices in order; the output head (the tied embedding, 0.8 GiB for the 26B) goes on the last device that holds a
        // layer, which must have room for it: when it has not, its share is reserved there and the layers are placed again (they move on).
        dev_of.assign((size_t) hp.n_layer, 0);
        const std::vector<long long> free_budget = budget;
        int reserve_on = -1;
        for (size_t attempt = 0; attempt <= devs.size(); ++attempt) {
            budget = free_budget;
            if (reserve_on >= 0) budget[(size_t) reserve_on] -= (long long) head_bytes();
            size_t cur = 0;
            bool placed = true;
            for (int il = 0; il < hp.n_layer && placed; ++il) {
                const long long need = (long long) (layer_weight_bytes(il) + layer_kv_bytes(il));
                while (cur < devs.size() && budget[cur] < need) ++cur;
                if (cur >= devs.size()) { placed = false; break; }
                budget[cur] -= need;
                dev_of[(size_t) il] = (int) cur;
            }
            if (!placed) {
                size_t total = head_bytes(), kv = 0;
                for (int k = 0; k < hp.n_layer; ++k) { total += layer_weight_bytes(k) + layer_kv_bytes(k); kv += layer_kv_bytes(k); }
                long long have = 0;
                for (size_t i = 0; i < devs.size(); ++i) have += std::max<long long>(0, free_budget[i]);
                char buf[420];
                std::snprintf(buf, sizeof buf, "the model needs %.2f GiB of VRAM (weights %.2f GiB, K / V at a context of %d %.2f GiB, the output head %.2f GiB) and the %zu device(s) "
                              "have %.2f GiB free after their reserve and buffers: use a smaller quantization, a shorter --max-context, --swa-slack 0, or another card",
                              (double) total / (1 << 30), (double) (total - head_bytes() - kv) / (1 << 30), ctx, (double) kv / (1 << 30), (double) head_bytes() / (1 << 30),
                              devs.size(), (double) have / (1 << 30));
                return buf;
            }
            const int last_used = dev_of[(size_t) hp.n_layer - 1];
            if (reserve_on == last_used || budget[(size_t) last_used] >= (long long) head_bytes()) { head_dev = last_used; break; }
            reserve_on = last_used;
            head_dev = last_used;
        }
        if (hp.n_expert > kg::kMaxExperts) return "more experts than the router kernel handles (" + std::to_string(kg::kMaxExperts) + ")";
        return "";
    }
    // g4k's limits, named once
    struct kg {
        static constexpr int kMaxExperts = g4k::kMaxExperts;
    };

    // ------------------------------------------------------------------------------------------------------------------------ upload
    DevMat upload(Dev& D, const g4::Mat& a) {
        DevMat d;
        if (!a) return d;
        d.type = a.type;
        d.n_in = a.n_in;
        d.n_out = a.n_out;
        d.n_expert = a.n_expert;
        d.row_bytes = a.row_bytes;
        d.expert_bytes = a.expert_bytes;
        d.bytes = a.bytes;
        d.d = D.alloc<uint8_t>(a.bytes);
        CK(cudaMemcpy(d.d, a.data, a.bytes, cudaMemcpyHostToDevice));
        return d;
    }

    std::string init() {
        std::string err = place();
        if (!err.empty()) return err;
        for (Dev& D : devs) {
            CK(cudaSetDevice(D.id));
            CK(cudaStreamCreate(&D.st));
        }
        L.assign((size_t) hp.n_layer, DevLayer());
        // are the head sizes ones the attention kernel has?
        for (int il = 0; il < hp.n_layer; ++il) {
            const int hd = hp.head_dim(il);
            if (hd % 64 != 0 || hd > g4k::kMaxHead || hd < 64) return "head dimension " + std::to_string(hd) + ": the attention kernel takes multiples of 64 up to " + std::to_string(g4k::kMaxHead);
            if (hp.n_embd > 8192) return "n_embd above 8192 is not supported by the norm kernels";
        }
        for (size_t di = 0; di < devs.size(); ++di) {
            Dev& D = devs[di];
            CK(cudaSetDevice(D.id));
            const size_t C = (size_t) chunk, E = (size_t) hp.n_embd;
            const size_t qmax = (size_t) hp.n_head * std::max(hp.hd_swa, hp.hd_full);
            size_t kvmax = 0;
            for (int il = 0; il < hp.n_layer; ++il) kvmax = std::max(kvmax, (size_t) hp.kv_dim(il));
            D.h = D.alloc<float>(C * E);
            D.h1 = D.alloc<float>(C * E);
            D.xn = D.alloc<float>(C * E);
            D.ye = D.alloc<float>(C * E);
            D.q = D.alloc<float>(C * qmax);
            D.ao = D.alloc<float>(C * qmax);
            D.kraw = D.alloc<float>(C * kvmax);
            D.vraw = D.alloc<float>(C * kvmax);
            D.proj = D.alloc<float>(C * E);
            D.mlp = D.alloc<float>(C * E);
            D.moe = D.alloc<float>(C * E);
            D.g = D.alloc<float>(C * (size_t) hp.n_ff);
            D.u = D.alloc<float>(C * (size_t) hp.n_ff);
            D.act = D.alloc<float>(C * (size_t) hp.n_ff);
            if (hp.n_expert > 0) {
                const size_t R = C * (size_t) hp.n_used;
                D.gu = D.alloc<float>(R * 2 * (size_t) hp.n_ff_exp);
                D.eact = D.alloc<float>(R * (size_t) hp.n_ff_exp);
                D.ed = D.alloc<float>(R * E);
                D.topk_w = D.alloc<float>(R);
                D.topk_idx = D.alloc<int>(R);
                const size_t G = (size_t) hp.n_expert + R / kExpertTile + 8;
                D.grp_expert = D.alloc<int>(G);
                D.grp_first = D.alloc<int>(G);
                D.grp_n = D.alloc<int>(G);
                D.ent_tok = D.alloc<int>(R);
                D.ent_row = D.alloc<int>(R);
                std::vector<int> first((size_t) hp.n_used), one((size_t) hp.n_used, 1), zero((size_t) hp.n_used, 0), rows((size_t) hp.n_used);
                std::iota(first.begin(), first.end(), 0);
                rows = first;
                D.dec_first = D.alloc<int>((size_t) hp.n_used);
                D.dec_n = D.alloc<int>((size_t) hp.n_used);
                D.dec_tok = D.alloc<int>((size_t) hp.n_used);
                D.dec_row = D.alloc<int>((size_t) hp.n_used);
                CK(cudaMemcpy(D.dec_first, first.data(), first.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(D.dec_n, one.data(), one.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(D.dec_tok, zero.data(), zero.size() * 4, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(D.dec_row, rows.data(), rows.size() * 4, cudaMemcpyHostToDevice));
            }
            if (m.rope_freqs) {
                D.rope_freqs = D.alloc<float>((size_t) m.rope_freqs_n);
                CK(cudaMemcpy(D.rope_freqs, m.rope_freqs, (size_t) m.rope_freqs_n * 4, cudaMemcpyHostToDevice));
            }
        }
        for (int il = 0; il < hp.n_layer; ++il) {
            Dev& D = devs[(size_t) dev_of[(size_t) il]];
            CK(cudaSetDevice(D.id));
            const g4::LayerW& w = m.L[(size_t) il];
            DevLayer& dl = L[(size_t) il];
            dl.dev = dev_of[(size_t) il];
            dl.wq = upload(D, w.wq);
            dl.wk = upload(D, w.wk);
            dl.wv = upload(D, w.wv);
            dl.wo = upload(D, w.wo);
            dl.gate = upload(D, w.ffn_gate);
            dl.up = upload(D, w.ffn_up);
            dl.down = upload(D, w.ffn_down);
            dl.gate_up = upload(D, w.gate_up_exps);
            dl.gate_e = upload(D, w.gate_exps);
            dl.up_e = upload(D, w.up_exps);
            dl.down_e = upload(D, w.down_exps);
            // the float vectors in one allocation
            const size_t E = (size_t) hp.n_embd, hd = (size_t) hp.head_dim(il), NE = (size_t) hp.n_expert;
            size_t n = 4 * E + 2 * hd + (w.has_moe ? 4 * E + NE + NE * E : 0);
            dl.blob = D.alloc<float>(n);
            size_t off = 0;
            auto put = [&](const float* src, size_t len) -> float* {
                float* dst = dl.blob + off;
                CK(cudaMemcpy(dst, src, len * 4, cudaMemcpyHostToDevice));
                off += len;
                return dst;
            };
            dl.attn_norm = put(w.attn_norm, E);
            dl.q_norm = put(w.q_norm, hd);
            dl.k_norm = put(w.k_norm, hd);
            dl.post_attn_norm = put(w.post_attn_norm, E);
            dl.ffn_norm = put(w.ffn_norm, E);
            dl.post_ffw_norm = put(w.post_ffw_norm, E);
            if (w.has_moe) {
                dl.pre2 = put(w.pre_ffw_norm_2, E);
                dl.post1 = put(w.post_ffw_norm_1, E);
                dl.post2 = put(w.post_ffw_norm_2, E);
                dl.rscale = put(w.router_scale, E);
                dl.router = put(w.router, NE * E);
                if (w.down_scale) dl.dscale = put(w.down_scale, NE);
            }
            dl.out_scale = w.out_scale;
            dl.ring = ring_cells(il);
            const size_t cells = (size_t) hp.kv_heads(il) * (size_t) dl.ring * hd;
            dl.kc = D.alloc<uint16_t>(cells);
            dl.vc = D.alloc<uint16_t>(cells);
        }
        Dev& H = devs[(size_t) head_dev];
        CK(cudaSetDevice(H.id));
        H.head = upload(H, m.tok_embd);
        H.output_norm = H.alloc<float>((size_t) hp.n_embd);
        CK(cudaMemcpy(H.output_norm, m.output_norm, (size_t) hp.n_embd * 4, cudaMemcpyHostToDevice));
        H.logits = H.alloc<float>((size_t) hp.n_vocab);
        const size_t np = ((size_t) hp.n_vocab + 1023) / 1024;
        H.pv = H.alloc<float>(np);
        H.pi = H.alloc<int>(np);
        H.amax = H.alloc<int>(1);
        for (Dev& D : devs) { CK(cudaSetDevice(D.id)); CK(cudaDeviceSynchronize()); }
        if (opt.verbose) {
            std::fprintf(stderr, "strata-gemma4: %zu CUDA device(s):", devs.size());
            for (size_t i = 0; i < devs.size(); ++i) {
                int first = -1, last = -1, n = 0;
                for (int il = 0; il < hp.n_layer; ++il)
                    if (dev_of[(size_t) il] == (int) i) { if (first < 0) first = il; last = il; ++n; }
                std::fprintf(stderr, " [device %d: %d layers%s%s, %.2f GiB]", devs[i].id, n, n ? (" " + std::to_string(first) + ".." + std::to_string(last)).c_str() : "",
                             (int) i == head_dev ? " + the output head" : "", (double) devs[i].bytes / (1 << 30));
            }
            std::fprintf(stderr, "; context %d, chunk %d\n", ctx, chunk);
        }
        return "";
    }

    // ------------------------------------------------------------------------------------------------------------------------ launches
    static int tile_of(int T) { return T <= 1 ? 1 : T <= 4 ? 4 : T <= 8 ? 8 : 16; }

    /// y[t][y_off + r] = W row r . x[t] for t < T
    void mm(Dev& D, const DevMat& W, const float* x, int ldx, int T, float* y, int ldy, int y_off = 0) {
        const int NT = tile_of(T);
        const dim3 grid((unsigned) ((W.n_out + g4k::kMmWarps - 1) / g4k::kMmWarps), (unsigned) ((T + NT - 1) / NT));
        const dim3 block(g4k::kMmThreads);
        with_type(W.type, [&](auto tag) {
            constexpr int TY = decltype(tag)::value;
            if (NT == 1) G4_LAUNCH(G4_K(g4k::k_mm<TY, 1>), grid, block, D.st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, T, y, ldy, y_off);
            else if (NT == 4) G4_LAUNCH(G4_K(g4k::k_mm<TY, 4>), grid, block, D.st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, T, y, ldy, y_off);
            else if (NT == 8) G4_LAUNCH(G4_K(g4k::k_mm<TY, 8>), grid, block, D.st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, T, y, ldy, y_off);
            else G4_LAUNCH(G4_K(g4k::k_mm<TY, 16>), grid, block, D.st, W.d, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx, T, y, ldy, y_off);
        });
    }

    /// the experts' matrices: n_groups groups, tile 1 (the decode step) or kExpertTile
    void mm_exp(Dev& D, const DevMat& W, const float* x, int ldx, int n_groups, bool decode, const int* grp_expert, const int* grp_first, const int* grp_n,
                const int* ent_in, const int* ent_out, float* y, int ldy, int y_off = 0) {
        const dim3 grid((unsigned) ((W.n_out + g4k::kMmWarps - 1) / g4k::kMmWarps), (unsigned) n_groups);
        const dim3 block(g4k::kMmThreads);
        with_type(W.type, [&](auto tag) {
            constexpr int TY = decltype(tag)::value;
            if (decode) G4_LAUNCH(G4_K(g4k::k_mm_exp<TY, 1>), grid, block, D.st, W.d, (long long) W.expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx,
                                  grp_expert, grp_first, grp_n, ent_in, ent_out, y, ldy, y_off);
            else G4_LAUNCH(G4_K(g4k::k_mm_exp<TY, kExpertTile>), grid, block, D.st, W.d, (long long) W.expert_bytes, (long long) W.row_bytes, (int) W.n_in, (int) W.n_out, x, ldx,
                           grp_expert, grp_first, grp_n, ent_in, ent_out, y, ldy, y_off);
        });
    }

    void attn(Dev& D, int il, int T, int pos0) {
        const DevLayer& dl = L[(size_t) il];
        const int hd = hp.head_dim(il), q_dim = hp.q_dim(il);
        const int window = hp.swa[(size_t) il] ? hp.window : 0;
        const dim3 grid((unsigned) T, (unsigned) hp.n_head), block(g4k::kThreads);
        switch (hd / 64) {
            case 1: G4_LAUNCH(G4_K(g4k::k_attn<1>), grid, block, D.st, D.q, dl.kc, dl.vc, D.ao, hp.n_head, hp.kv_heads(il), pos0, dl.ring, window, q_dim); break;
            case 2: G4_LAUNCH(G4_K(g4k::k_attn<2>), grid, block, D.st, D.q, dl.kc, dl.vc, D.ao, hp.n_head, hp.kv_heads(il), pos0, dl.ring, window, q_dim); break;
            case 4: G4_LAUNCH(G4_K(g4k::k_attn<4>), grid, block, D.st, D.q, dl.kc, dl.vc, D.ao, hp.n_head, hp.kv_heads(il), pos0, dl.ring, window, q_dim); break;
            case 8: G4_LAUNCH(G4_K(g4k::k_attn<8>), grid, block, D.st, D.q, dl.kc, dl.vc, D.ao, hp.n_head, hp.kv_heads(il), pos0, dl.ring, window, q_dim); break;
            default: std::fprintf(stderr, "strata-gemma4: head dimension %d has no attention kernel (64, 128, 256 or 512)\n", hd); std::exit(1);
        }
    }

    /// The experts of a mixture layer, after the router and the norm: the T tokens' (token, expert) pairs, gate+up, the gated activation, down,
    /// and the weighted sum into D.moe.
    void experts(Dev& D, const DevLayer& dl, int T) {
        const int nu = hp.n_used, E = hp.n_embd, F = hp.n_ff_exp;
        const dim3 b1(g4k::kThreads);
        G4_LAUNCH(g4k::k_router, dim3(T), b1, D.st, D.h1, dl.rscale, dl.router, E, hp.n_expert, nu, hp.eps, D.topk_idx, D.topk_w);
        G4_LAUNCH(g4k::k_rms, dim3(T), b1, D.st, D.h1, E, dl.pre2, D.ye, E, E, hp.eps);
        const int rows = T * nu;
        int ng;
        const int *ge, *gf, *gn, *et, *er;
        bool decode = T == 1;
        if (decode) {                              // one token: the router's own choice is the group table, no round trip to the host
            ng = nu;
            ge = D.topk_idx;
            gf = D.dec_first;
            gn = D.dec_n;
            et = D.dec_tok;
            er = D.dec_row;
        } else {                                   // prompt reading: group the pairs by expert on the host (the weights are then read once per 16 pairs)
            std::vector<int> idx((size_t) rows);
            CK(cudaMemcpy(idx.data(), D.topk_idx, (size_t) rows * 4, cudaMemcpyDeviceToHost));
            std::vector<std::vector<int>> by((size_t) hp.n_expert);
            for (int r = 0; r < rows; ++r) by[(size_t) idx[(size_t) r]].push_back(r);
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
            ng = (int) gex.size();
            CK(cudaMemcpy(D.grp_expert, gex.data(), gex.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(D.grp_first, gfi.data(), gfi.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(D.grp_n, gnn.data(), gnn.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(D.ent_tok, ent_t.data(), ent_t.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(D.ent_row, ent_r.data(), ent_r.size() * 4, cudaMemcpyHostToDevice));
            ge = D.grp_expert; gf = D.grp_first; gn = D.grp_n; et = D.ent_tok; er = D.ent_row;
        }
        if (dl.gate_up) {
            mm_exp(D, dl.gate_up, D.ye, E, ng, decode, ge, gf, gn, et, er, D.gu, 2 * F);
            G4_LAUNCH(g4k::k_geglu, dim3((unsigned) ((F + 127) / 128), (unsigned) rows), b1, D.st, D.gu, 2 * F, D.gu + F, 2 * F, D.eact, F, F);
        } else {
            mm_exp(D, dl.gate_e, D.ye, E, ng, decode, ge, gf, gn, et, er, D.gu, 2 * F, 0);
            mm_exp(D, dl.up_e, D.ye, E, ng, decode, ge, gf, gn, et, er, D.gu, 2 * F, F);
            G4_LAUNCH(g4k::k_geglu, dim3((unsigned) ((F + 127) / 128), (unsigned) rows), b1, D.st, D.gu, 2 * F, D.gu + F, 2 * F, D.eact, F, F);
        }
        mm_exp(D, dl.down_e, D.eact, F, ng, decode, ge, gf, gn, er, er, D.ed, E);
        G4_LAUNCH(g4k::k_expert_sum, dim3((unsigned) ((E + 127) / 128), (unsigned) T), b1, D.st, D.ed, D.topk_idx, D.topk_w, dl.dscale, nu, E, D.moe);
    }

    void layer(Dev& D, int il, int T, int pos0) {
        const DevLayer& dl = L[(size_t) il];
        const g4::LayerW& w = m.L[(size_t) il];
        const int E = hp.n_embd, hd = hp.head_dim(il), nkv = hp.kv_heads(il), q_dim = hp.q_dim(il), kv_dim = hp.kv_dim(il);
        const dim3 b1(g4k::kThreads);
        const bool swa = hp.swa[(size_t) il] != 0;
        G4_LAUNCH(g4k::k_rms, dim3(T), b1, D.st, D.h, E, dl.attn_norm, D.xn, E, E, hp.eps);
        mm(D, dl.wq, D.xn, E, T, D.q, q_dim);
        mm(D, dl.wk, D.xn, E, T, D.kraw, kv_dim);
        if (w.has_v) mm(D, dl.wv, D.xn, E, T, D.vraw, kv_dim);
        const float* vsrc = w.has_v ? D.vraw : D.kraw;                         // no attn_v: V is the raw K projection
        G4_LAUNCH(g4k::k_qkv_post, dim3((unsigned) T, (unsigned) (hp.n_head + nkv)), b1, D.st, D.q, (const float*) D.kraw, vsrc, dl.q_norm, dl.k_norm,
                  swa ? (const float*) nullptr : (const float*) D.rope_freqs, dl.kc, dl.vc, hd, hp.n_head, nkv, hp.rot(il), hp.base(il), hp.eps, pos0, dl.ring, q_dim, kv_dim);
        attn(D, il, T, pos0);
        mm(D, dl.wo, D.ao, q_dim, T, D.proj, E);
        G4_LAUNCH(g4k::k_rms_add, dim3(T), b1, D.st, D.proj, dl.post_attn_norm, D.h, D.h1, E, hp.eps);
        G4_LAUNCH(g4k::k_rms, dim3(T), b1, D.st, D.h1, E, dl.ffn_norm, D.xn, E, E, hp.eps);
        mm(D, dl.gate, D.xn, E, T, D.g, hp.n_ff);
        mm(D, dl.up, D.xn, E, T, D.u, hp.n_ff);
        G4_LAUNCH(g4k::k_geglu, dim3((unsigned) ((hp.n_ff + 127) / 128), (unsigned) T), b1, D.st, D.g, hp.n_ff, D.u, hp.n_ff, D.act, hp.n_ff, hp.n_ff);
        mm(D, dl.down, D.act, hp.n_ff, T, D.mlp, E);
        if (w.has_moe) experts(D, dl, T);
        G4_LAUNCH(g4k::k_ffn_combine, dim3(T), b1, D.st, (const float*) D.mlp, w.has_moe ? (const float*) D.moe : (const float*) nullptr, (const float*) D.h1, dl.post1,
                  dl.post2, dl.post_ffw_norm, dl.out_scale, D.h, E, hp.eps);
    }

    // ------------------------------------------------------------------------------------------------------------------------ a chunk
    void forward_chunk(const int32_t* ids, int T, bool want_logits) {
        const int E = hp.n_embd;
        hbuf.resize((size_t) T * E);
        {   // embeddings on the host: the rows of token_embd, scaled by sqrt(n_embd)
            const float sc = std::sqrt((float) E);
            for (int t = 0; t < T; ++t) {
                float* row = hbuf.data() + (size_t) t * E;
                g4::dequant_row(m.tok_embd.type, m.tok_embd.row(ids[t]), E, row);
                for (int i = 0; i < E; ++i) row[i] *= sc;
            }
        }
        int cur = dev_of[0];
        CK(cudaSetDevice(devs[(size_t) cur].id));
        CK(cudaMemcpy(devs[(size_t) cur].h, hbuf.data(), (size_t) T * E * 4, cudaMemcpyHostToDevice));
        if (debug) { debug_h.assign((size_t) (hp.n_layer + 1) * E, 0.f); std::memcpy(debug_h.data(), hbuf.data() + (size_t) (T - 1) * E, (size_t) E * 4); }
        for (int il = 0; il < hp.n_layer; ++il) {
            const int d = dev_of[(size_t) il];
            if (d != cur) {                               // a hop to the next card, through the host
                CK(cudaSetDevice(devs[(size_t) cur].id));
                CK(cudaStreamSynchronize(devs[(size_t) cur].st));
                CK(cudaMemcpy(hbuf.data(), devs[(size_t) cur].h, (size_t) T * E * 4, cudaMemcpyDeviceToHost));
                CK(cudaSetDevice(devs[(size_t) d].id));
                CK(cudaMemcpy(devs[(size_t) d].h, hbuf.data(), (size_t) T * E * 4, cudaMemcpyHostToDevice));
                cur = d;
            }
            CK(cudaSetDevice(devs[(size_t) cur].id));
            layer(devs[(size_t) cur], il, T, (int) pos);
            if (debug) {
                CK(cudaStreamSynchronize(devs[(size_t) cur].st));
                CK(cudaMemcpy(debug_h.data() + (size_t) (il + 1) * E, devs[(size_t) cur].h + (size_t) (T - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToHost));
            }
        }
        if (want_logits) {
            if (cur != head_dev) {                        // the head is on the last card
                CK(cudaSetDevice(devs[(size_t) cur].id));
                CK(cudaStreamSynchronize(devs[(size_t) cur].st));
                CK(cudaMemcpy(hbuf.data(), devs[(size_t) cur].h + (size_t) (T - 1) * E, (size_t) E * 4, cudaMemcpyDeviceToHost));
                CK(cudaSetDevice(devs[(size_t) head_dev].id));
                CK(cudaMemcpy(devs[(size_t) head_dev].h + (size_t) (T - 1) * E, hbuf.data(), (size_t) E * 4, cudaMemcpyHostToDevice));
            }
            Dev& H = devs[(size_t) head_dev];
            CK(cudaSetDevice(H.id));
            G4_LAUNCH(g4k::k_rms, dim3(1), dim3(g4k::kThreads), H.st, (const float*) (H.h + (size_t) (T - 1) * E), E, (const float*) H.output_norm, H.xn, E, E, hp.eps);
            mm(H, H.head, H.xn, E, 1, H.logits, hp.n_vocab);
            if (hp.softcap > 0.f) G4_LAUNCH(g4k::k_softcap, dim3((unsigned) ((hp.n_vocab + 127) / 128)), dim3(g4k::kThreads), H.st, H.logits, hp.n_vocab, hp.softcap);
            logits_ready = true;
        } else {
            logits_ready = false;
        }
        for (Dev& D : devs) { CK(cudaSetDevice(D.id)); CK(cudaStreamSynchronize(D.st)); }
        pos += T;
    }
};

// ============================================================================================================ the public class
GpuEngine::GpuEngine(const g4::Model& m, int ctx, GpuOptions opt) : p_(new Impl(m, ctx, std::move(opt))) {}
GpuEngine::~GpuEngine() = default;
std::string GpuEngine::init() { return p_->init(); }
int GpuEngine::ctx() const { return p_->ctx; }
int GpuEngine::n_vocab() const { return p_->hp.n_vocab; }
int64_t GpuEngine::pos() const { return p_->pos; }
void GpuEngine::reset() { p_->pos = 0; p_->logits_ready = false; }
void GpuEngine::truncate(int64_t q) { p_->pos = std::min(p_->pos, std::max<int64_t>(0, q)); p_->logits_ready = false; }
bool GpuEngine::can_keep(int64_t q) const {
    // Going back to q: the next token (position q) needs the keys of the sliding-window layers from q - window + 1 on.  A ring of `ring` cells
    // holds the last `ring` positions [pos - ring, pos - 1]; a global layer holds all of them.
    if (q > p_->pos) return false;
    if (!p_->hp.n_swa_layers()) return true;
    const int64_t ring = std::min<int64_t>(p_->ctx, (int64_t) p_->hp.window + p_->chunk + std::max(0, p_->opt.slack));
    return p_->pos <= ring || q - p_->hp.window + 1 >= p_->pos - ring;
}
std::vector<float> GpuEngine::debug_hidden(int il) {
    return std::vector<float>(p_->debug_h.begin() + (size_t) (il + 1) * p_->hp.n_embd, p_->debug_h.begin() + (size_t) (il + 2) * p_->hp.n_embd);
}

bool GpuEngine::process(const int32_t* ids, int64_t n, const std::function<bool(int64_t)>& on_piece, int64_t* done) {
    *done = 0;
    for (int64_t i = 0; i < n; i += p_->chunk) {
        const int T = (int) std::min<int64_t>(p_->chunk, n - i);
        if (p_->pos + T > p_->ctx) return false;
        p_->debug = std::getenv("STRATA_G4_DEBUG_HIDDEN") != nullptr;
        p_->forward_chunk(ids + i, T, i + T == n);
        *done = i + T;
        if (!on_piece(i + T)) return i + T >= n;
    }
    return true;
}

void GpuEngine::logits(std::vector<float>& out) {
    Dev& H = p_->devs[(size_t) p_->head_dev];
    out.resize((size_t) p_->hp.n_vocab);
    CK(cudaSetDevice(H.id));
    CK(cudaMemcpy(out.data(), H.logits, out.size() * 4, cudaMemcpyDeviceToHost));
}

int GpuEngine::greedy() {
    Impl& I = *p_;
    Dev& H = I.devs[(size_t) I.head_dev];
    CK(cudaSetDevice(H.id));
    const int np = (I.hp.n_vocab + 1023) / 1024;
    G4_LAUNCH(g4k::k_argmax_part, dim3((unsigned) np), dim3(g4k::kThreads), H.st, (const float*) H.logits, I.hp.n_vocab, H.pv, H.pi);
    G4_LAUNCH(g4k::k_argmax_final, dim3(1), dim3(g4k::kThreads), H.st, (const float*) H.pv, (const int*) H.pi, np, H.amax);
    int r = 0;
    CK(cudaMemcpy(&r, H.amax, 4, cudaMemcpyDeviceToHost));
    return r;
}


// ============================================================================================================ the program
namespace {
std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    for (size_t p = 0; p < s.size();) {
        char* e = nullptr;
        v.push_back((int) std::strtol(s.c_str() + p, &e, 10));
        p = (size_t) (e - s.c_str());
        while (p < s.size() && (s[p] == ',' || s[p] == ' ')) ++p;
        if (e == s.c_str()) break;
    }
    return v;
}
double ms_now() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

int run(const g4::Model& m, int ctx, bool serve, const std::string& tokens, int n_gen, const std::string& dump, const std::string& gpus, const std::set<int32_t>& eos,
        const std::vector<std::string>& extra) {
    GpuOptions o;
    o.devices = gpus.empty() ? std::vector<int>() : parse_list(gpus);
    bool check = false;
    int bench_pp = 0, bench_tg = 0;
    for (size_t i = 0; i < extra.size(); ++i) {
        const std::string& a = extra[i];
        auto val = [&]() -> std::string { return i + 1 < extra.size() ? extra[++i] : std::string(); };
        if (a == "--prefill" || a == "--chunk") { const std::string v = val(); if (v != "auto") o.chunk = std::max(1, std::atoi(v.c_str())); }
        else if (a == "--swa-slack") o.slack = std::max(0, std::atoi(val().c_str()));
        else if (a == "--vram-reserve-mib") o.reserve_mib = std::max(0, std::atoi(val().c_str()));
        else if (a == "--check") check = true;
        else if (a == "--bench-pp") bench_pp = std::atoi(val().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(val().c_str());
        else if (a == "--kv") { const std::string v = val(); if (v != "f16") std::fprintf(stderr, "strata-gemma4: --kv %s: this engine keeps K / V as f16\n", v.c_str()); }
        else std::fprintf(stderr, "strata-gemma4: ignoring %s (not an option of this engine)\n", a.c_str());
    }
    GpuEngine eng(m, ctx, o);
    const std::string err = eng.init();
    if (!err.empty()) { std::fprintf(stderr, "strata-gemma4: %s\n", err.c_str()); return 1; }
    if (serve) {
        g4::Serve<GpuEngine> sv(eng, eos, STRATA_G4_VERSION);
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
            if (!eng.process(ids.data(), std::min<int64_t>(bench_pp, ctx), noop, &done)) { std::fprintf(stderr, "strata-gemma4: the prompt does not fit the context\n"); return 1; }
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
        }
        return 0;
    }
    if (tokens.empty()) { std::fprintf(stderr, "strata-gemma4: --tokens ID,ID,... or --serve\n"); return 2; }
    const std::vector<int32_t> ids = g4::parse_ids(tokens.c_str());
    if (!eng.process(ids.data(), (int64_t) ids.size(), noop, &done)) { std::fprintf(stderr, "strata-gemma4: the prompt does not fit the context\n"); return 1; }
    std::vector<float> lg;
    eng.logits(lg);
    if (!dump.empty()) {
        std::FILE* f = std::fopen(dump.c_str(), "wb");
        if (f) { std::fwrite(lg.data(), 4, lg.size(), f); std::fclose(f); }
    }
    int rc = 0;
    if (check) {   // the same prompt on the CPU reference (K / V rounded to half as here): the engine's own gate, for a machine with a card
        g4::RefOpts ro;
        ro.cache_f16 = true;
        ro.threads = (int) std::max(1u, std::thread::hardware_concurrency());
        g4::RefEngine ref(m, (int) ids.size() + 1, ro);
        std::vector<float> rl;
        const double t0 = ms_now();
        ref.forward(ids.data(), (int) ids.size(), &rl);
        double d2 = 0, s2 = 0, mean = 0;
        for (float x : rl) mean += x;
        mean /= (double) rl.size();
        for (size_t i = 0; i < rl.size(); ++i) { d2 += (double) (lg[i] - rl[i]) * (lg[i] - rl[i]); s2 += (double) (rl[i] - mean) * (rl[i] - mean); }
        const double rel = std::sqrt(d2 / std::max(s2, 1e-30));
        const int a = (int) (std::max_element(lg.begin(), lg.end()) - lg.begin()), b = (int) (std::max_element(rl.begin(), rl.end()) - rl.begin());
        std::vector<int> ia(lg.size()), ib(rl.size());
        std::iota(ia.begin(), ia.end(), 0);
        std::iota(ib.begin(), ib.end(), 0);
        std::partial_sort(ia.begin(), ia.begin() + 5, ia.end(), [&](int x, int y) { return lg[(size_t) x] > lg[(size_t) y]; });
        std::partial_sort(ib.begin(), ib.begin() + 5, ib.end(), [&](int x, int y) { return rl[(size_t) x] > rl[(size_t) y]; });
        int overlap = 0;
        for (int i = 0; i < 5; ++i) for (int j = 0; j < 5; ++j) overlap += ia[(size_t) i] == ib[(size_t) j];
        std::printf("check: %zu tokens, GPU against the CPU reference (%.0f ms): logits differ by %.2e (rms / spread), best token %d / %d, top-5 overlap %d of 5\n",
                    ids.size(), ms_now() - t0, rel, a, b, overlap);
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

}  // namespace g4cuda
