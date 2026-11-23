// i2_model.hpp - Inkling 2 Mini (GGUF architecture `inkling2_loop`, also `inkling2`): the hyper-parameters read from a GGUF and its tensors bound by
// name with their shapes checked.  The same as llama.cpp's llama_model_inkling2(_loop)::load_arch_hparams / load_arch_tensors
// (src/models/inkling2.cpp, inkling2-loop.cpp), for the text model: the vision projector and the MTP / DSpark drafter are not read.
//
// The model, in the words of its layers (docs/INKLING2.md has the whole forward pass):
//   24 physical layers; layers [loop.start, loop.end) = [6, 18) run `loop.count` = 3 times per token (the same weights, a K / V cache and a
//   convolution state of their own for every visit: 48 visits).  Attention: no rope, a learned relative-position bias per head, a sink,
//   an output gate, QK clipping, short convolutions on K, V and the outputs.  Block attention residuals: every 6th layer starts a new
//   residual stream and the earlier ones are mixed back by a small attention over the depth.  MoE: 128 experts, 8 per token, sigmoid router
//   with a selection bias, the SITU activation; no shared expert.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "../gemma4/g4_quant.hpp"

namespace i2 {

using g4::dequant_row;

struct Hparams {
    int n_layer = 0;                        // physical layers
    int n_embd = 0, n_head = 0, n_kv = 0, hd = 0;
    int n_vocab = 0, n_vocab_padded = 0;
    int n_expert = 0, n_used = 0, n_ff_exp = 0;
    int window = 0, ctx_train = 0;
    int n_conv = 4, block = 6;
    int n_rel = 16, rel_extent = 1024;
    float eps = 1e-6f, qk_clip = 22.627417f, rel_clip = 8.f, log_floor = 8192.f, log_alpha = 0.1f;
    float situ_beta = 4.f, situ_lin = 25.f, route_scale = 8.f, softcap = 30.f;
    // the loop
    int loop_start = 0, loop_end = 0, passes = 1, gate_rank = 0;
    bool varying = false;
    int eos = 200006;
    std::vector<uint8_t> swa;               // per physical layer: a sliding-window (local) layer
    int visits() const { return n_layer + (passes - 1) * (loop_end - loop_start); }
    /// the physical layer of visit v (0 .. visits() - 1)
    int phys(int v) const { return v < n_layer ? v : loop_start + (v - n_layer) % (loop_end - loop_start); }
    int q_dim() const { return n_head * hd; }
    int kv_dim() const { return n_kv * hd; }
    int conv_channels() const { return 2 * kv_dim() + 2 * n_embd; }
    bool has_loop() const { return loop_end > loop_start; }
};

/// A weight matrix as the file stores it: `n_out` rows of `n_in` values in `type` blocks (GGUF order [n_in, n_out(, n_expert)]).
struct Mat {
    int type = -1;
    long long n_in = 0, n_out = 0, n_expert = 1;
    const uint8_t* data = nullptr;       // into the mapped file
    size_t row_bytes = 0, expert_bytes = 0, bytes = 0;
    explicit operator bool() const { return data != nullptr; }
    const uint8_t* row(long long r, long long e = 0) const { return data + (size_t) e * expert_bytes + (size_t) r * row_bytes; }
};

struct LoopW {
    const float* msg_query = nullptr;               // [4][n_embd]: the q, k, v and ffn messages' queries
    const float *down[4] = {}, *gain[4] = {}, *forget[4] = {};   // [rank][2 n_embd], [n_embd][rank], [n_embd][rank]
};

struct LayerW {
    Mat wq, wk, wv, wo, wgate, wrel, wsink, kconv_proj, vconv_proj;
    Mat gate_up, down;                               // experts: gate rows then up rows per expert; down
    const float *attn_norm = nullptr, *q_inp_norm = nullptr, *k_inp_norm = nullptr, *v_inp_norm = nullptr;
    const float *q_clip = nullptr, *rel_clip = nullptr, *rel_bias = nullptr;     // [n_head][hd], [n_head][n_rel], [extent][n_rel]
    const float *kconv = nullptr, *vconv = nullptr, *post_conv = nullptr, *ffn_post_conv = nullptr;   // [channels][n_conv]
    const float *attn_res_query = nullptr, *ffn_res_query = nullptr;             // [1 or 3][n_embd], [n_embd]
    const float *ffn_norm = nullptr, *router = nullptr, *exp_bias = nullptr;       // router [n_expert][n_embd]
    const float *gu_scale = nullptr, *dn_scale = nullptr;                          // per-expert scalars of the gate+up and down products (NVFP4 files; null: 1)
};

struct Model {
    Hparams hp;
    std::vector<LayerW> L;
    std::vector<LoopW> loop;                         // [loop_end - loop_start]
    Mat tok_embd, output;
    const float *tok_norm = nullptr, *output_norm = nullptr, *output_res_query = nullptr;
    std::unique_ptr<strata::GgufFile> file;
    std::vector<std::vector<float>> owned;           // tensors converted to F32 at load (anything not stored as F32)
    std::string name;
};

namespace detail {
inline long long meta_int(const strata::GgufFile& f, const std::string& k, long long def) {
    const strata::MetaValue* v = f.get(k);
    return v && v->is_num() ? (long long) v->num() : def;
}
inline double meta_f(const strata::GgufFile& f, const std::string& k, double def) {
    const strata::MetaValue* v = f.get(k);
    return v && v->is_num() ? v->num() : def;
}
}  // namespace detail

/// Hyper-parameters from the metadata; "" when fine, else the reason the file is not an Inkling 2 this engine runs.
inline std::string read_hparams(const strata::GgufFile& f, Hparams& hp) {
    using detail::meta_f;
    using detail::meta_int;
    const strata::MetaValue* arch = f.get("general.architecture");
    if (!arch || (arch->s != "inkling2_loop" && arch->s != "inkling2"))
        return "general.architecture is " + (arch ? "\"" + arch->s + "\"" : std::string("missing")) + ", expected \"inkling2_loop\" (Inkling 2 Mini)";
    const std::string p = arch->s + ".";
    // Two spellings of the same model: ggml-org's (arch inkling2_loop: block_count 24 physical layers, loop.start / end / count) and Unsloth's (arch inkling2: block_count
    // 48 = the layers as visited, num_loops and loop.core_start / core_end, tensors for the 24 physical layers, the loop's messages as two global tensors)
    const bool unsloth = arch->s == "inkling2" && f.get(p + "num_loops") != nullptr;
    const bool loop = arch->s == "inkling2_loop" || unsloth;
    hp.n_layer = (int) meta_int(f, p + "block_count", 0);
    int n_meta = hp.n_layer;                                  // entries of the per-layer arrays
    if (unsloth) {
        hp.loop_start = (int) meta_int(f, p + "loop.core_start", 0);
        hp.loop_end = (int) meta_int(f, p + "loop.core_end", 0);
        hp.passes = (int) meta_int(f, p + "num_loops", 0);
        const int core = hp.loop_end - hp.loop_start;
        n_meta = hp.n_layer;
        hp.n_layer = n_meta - (hp.passes - 1) * core;
        if (core <= 0 || hp.passes < 2 || hp.n_layer <= hp.loop_end) return "invalid loop core / num_loops in the Unsloth spelling (block_count counts the layers as visited)";
    }
    hp.n_embd = (int) meta_int(f, p + "embedding_length", 0);
    hp.hd = (int) meta_int(f, p + "attention.key_length", 0);
    hp.n_expert = (int) meta_int(f, p + "expert_count", 0);
    hp.n_used = (int) meta_int(f, p + "expert_used_count", 0);
    hp.window = (int) meta_int(f, p + "attention.sliding_window", 0);
    hp.ctx_train = (int) meta_int(f, p + "context_length", 0);
    hp.n_conv = (int) meta_int(f, p + "shortconv.l_cache", 0);
    hp.block = (int) meta_int(f, p + "attn_res.block_size", 0);
    hp.n_rel = (int) meta_int(f, p + "attention.relative_length", 0);
    hp.rel_extent = (int) meta_int(f, p + "attention.relative_extent", 0);
    hp.n_vocab_padded = (int) meta_int(f, p + "vocab_size", 0);
    hp.n_vocab = (int) meta_int(f, p + "vocab_size_unpadded", 0);
    hp.eps = (float) meta_f(f, p + "attention.layer_norm_rms_epsilon", 1e-6);
    hp.qk_clip = (float) meta_f(f, p + "attention.qk_clip", 0);
    hp.rel_clip = (float) meta_f(f, p + "attention.relative_clip", 0);
    hp.log_floor = (float) meta_f(f, p + "attention.log_scale_floor", 0);
    hp.log_alpha = (float) meta_f(f, p + "attention.log_scale_alpha", -1);
    hp.situ_beta = (float) meta_f(f, p + "activation.situ_beta", 0);
    hp.situ_lin = (float) meta_f(f, p + "activation.situ_linear_beta", 0);
    hp.route_scale = (float) meta_f(f, p + "expert_weights_scale", 1.0);
    hp.softcap = (float) meta_f(f, p + "final_logit_softcapping", 0.0);
    hp.eos = (int) meta_int(f, "tokenizer.ggml.eos_token_id", 200006);
    if (hp.n_layer <= 0 || hp.n_embd <= 0 || hp.hd <= 0) return "missing or zero inkling2 geometry (block_count, embedding_length, attention.key_length)";
    if (hp.n_conv < 2 || hp.n_conv > 64 || hp.block <= 0 || hp.n_rel <= 0 || hp.rel_extent <= 0 || hp.window <= 0) return "missing shortconv / attn_res / relative / sliding_window parameters";
    if (!(hp.qk_clip > 0) || !(hp.rel_clip > 0) || !(hp.log_floor > 0) || !(hp.log_alpha >= 0) || !(hp.situ_beta > 0) || !(hp.situ_lin > 0))
        return "missing or non-positive clipping / scaling / SITU parameters";
    if (hp.n_expert <= 0 || hp.n_used <= 0 || hp.n_used > hp.n_expert) return "missing expert_count / expert_used_count";
    if (hp.n_vocab <= 0 || hp.n_vocab > hp.n_vocab_padded) return "missing vocab_size / vocab_size_unpadded";
    const strata::MetaValue* tok = f.get("tokenizer.ggml.tokens");
    if (!tok || (int) tok->count != hp.n_vocab) return "the tokenizer's size is not vocab_size_unpadded";
    // per-layer arrays (or one value for all)
    auto per_layer = [&](const std::string& k, std::vector<long long>& out) -> bool {
        const strata::MetaValue* v = f.get(p + k);
        if (!v) return false;
        out.assign((size_t) hp.n_layer, 0);
        for (int i = 0; i < n_meta; ++i) {                    // (every entry is read: the engine needs them all alike)
            long long x;
            if (v->type == strata::MetaType::ARRAY) {
                if ((int) v->count != n_meta || (int) v->items.size() != n_meta) return false;
                x = (long long) v->items[(size_t) i].u;
            } else {
                x = (long long) v->u;
            }
            if (i < hp.n_layer) out[(size_t) i] = x;                   // (the visited copies past the physical layers: read, not kept)
        }
        return true;
    };
    std::vector<long long> heads, kv, ffe;
    if (!per_layer("attention.head_count", heads) || !per_layer("attention.head_count_kv", kv)) return "attention.head_count / head_count_kv missing or not one per layer";
    for (int i = 1; i < hp.n_layer; ++i)
        if (heads[(size_t) i] != heads[0] || kv[(size_t) i] != kv[0]) return "the engine takes one head count and one KV head count for every layer";
    hp.n_head = (int) heads[0];
    hp.n_kv = (int) kv[0];
    if (hp.n_kv <= 0 || hp.n_head % hp.n_kv != 0) return "head_count is not a multiple of head_count_kv";
    if (meta_int(f, p + "attention.value_length", hp.hd) != hp.hd || meta_int(f, p + "attention.key_length_swa", hp.hd) != hp.hd ||
        meta_int(f, p + "attention.value_length_swa", hp.hd) != hp.hd)
        return "the engine takes one head size for every layer and for K and V";
    if (!per_layer("expert_feed_forward_length", ffe)) return "expert_feed_forward_length is missing";
    hp.n_ff_exp = (int) ffe[0];
    for (long long v : ffe)
        if (v != ffe[0]) return "the engine takes one expert width for every layer";
    const strata::MetaValue* pat = f.get(p + "attention.sliding_window_pattern");
    if (!pat || pat->type != strata::MetaType::ARRAY || (int) pat->count != n_meta || (int) pat->items.size() != n_meta)
        return "attention.sliding_window_pattern is missing or does not have one entry per layer";
    hp.swa.assign((size_t) hp.n_layer, 0);
    const int core_len = hp.loop_end - hp.loop_start;
    for (int i = 0; i < hp.n_layer; ++i) {
        const int v = (unsloth && i >= hp.loop_end) ? i + (hp.passes - 1) * core_len : i;      // the first visit of physical layer i
        hp.swa[(size_t) i] = pat->items[(size_t) v].u ? 1 : 0;
    }
    if (unsloth) {
        hp.gate_rank = (int) meta_int(f, p + "loop.gate_rank", 0);
        hp.varying = false;
        if (hp.gate_rank <= 0 || hp.loop_start <= 0 || hp.loop_start % hp.block || hp.loop_end % hp.block) return "invalid loop gate rank / core in the Unsloth spelling";
    } else if (loop) {
        hp.loop_start = (int) meta_int(f, p + "loop.start", 0);
        hp.loop_end = (int) meta_int(f, p + "loop.end", 0);
        hp.passes = (int) meta_int(f, p + "loop.count", 0);
        hp.gate_rank = (int) meta_int(f, p + "loop.gate_rank", 0);
        hp.varying = meta_int(f, p + "loop.varying", 0) != 0;
        if (hp.passes < (hp.varying ? 1 : 2) || hp.gate_rank <= 0 || hp.loop_start <= 0 || hp.loop_start >= hp.loop_end || hp.loop_end > hp.n_layer ||
            hp.loop_start % hp.block || hp.loop_end % hp.block)
            return "invalid loop core [" + std::to_string(hp.loop_start) + ", " + std::to_string(hp.loop_end) + ") / passes / gate rank";
    }
    return "";
}

/// Bind and check the tensors.
inline std::string bind_tensors(Model& m) {
    const strata::GgufFile& f = *m.file;
    const Hparams& hp = m.hp;
    auto find = [&](const std::string& n) -> const strata::TensorInfo* { return f.find(n); };
    std::string err;
    auto mat = [&](const std::string& n, long long n_in, long long n_out, long long n_exp, Mat& out, bool required) -> std::string {
        const strata::TensorInfo* t = find(n);
        if (!t) return required ? "missing tensor " + n : "";
        if (g4::block_elems((int) t->type) == 0)
            return "tensor " + n + " is " + t->type_name() + ", a type the engine does not read (supported: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K..Q6_K IQ2..IQ4 MXFP4 NVFP4)";
        const size_t nd = t->shape.size();
        if (nd > 3 || (long long) t->shape[0] != n_in || (nd > 1 ? (long long) t->shape[1] : 1) != n_out || (nd > 2 ? (long long) t->shape[2] : 1) != n_exp)
            return "tensor " + n + " has another shape than [" + std::to_string(n_in) + ", " + std::to_string(n_out) + (n_exp > 1 ? ", " + std::to_string(n_exp) : "") + "]";
        const int ty = (int) t->type;
        if (n_in % g4::block_elems(ty) != 0) return "tensor " + n + ": " + std::to_string(n_in) + " values per row is not a multiple of the " + g4::type_name(ty) + " block";
        out.type = ty;
        out.n_in = n_in;
        out.n_out = n_out;
        out.n_expert = n_exp;
        out.row_bytes = (size_t) g4::row_bytes(ty, n_in);
        out.expert_bytes = out.row_bytes * (size_t) n_out;
        out.bytes = out.expert_bytes * (size_t) n_exp;
        out.data = f.tensor_data(*t);
        if (t->offset + out.bytes + f.data_start() > f.file_size()) return "tensor " + n + " runs past the end of the file";
        return "";
    };
    auto vec = [&](const std::string& n, long long len, const float*& out, bool required) -> std::string {
        const strata::TensorInfo* t = find(n);
        if (!t) return required ? "missing tensor " + n : "";
        if (t->elements() != (uint64_t) len) return "tensor " + n + " has " + std::to_string(t->elements()) + " values, expected " + std::to_string(len);
        if (t->type == (uint32_t) g4::T_F32) {
            out = reinterpret_cast<const float*>(f.tensor_data(*t));
            return "";
        }
        const int be = g4::block_elems((int) t->type);       // another type (F16 / BF16 / quantized): converted once
        if (be == 0) return "tensor " + n + " has an unsupported type " + t->type_name();
        if (len % be != 0) return "tensor " + n + ": its length is not a multiple of its block";
        m.owned.emplace_back((size_t) len);
        g4::dequant_row((int) t->type, f.tensor_data(*t), len, m.owned.back().data());
        out = m.owned.back().data();
        return "";
    };
    const long long E = hp.n_embd, H = hp.n_head, KV = hp.n_kv, hd = hp.hd, qd = H * hd, kd = KV * hd, R = hp.n_rel, C = hp.n_conv;
    if (!(err = mat("token_embd.weight", E, hp.n_vocab_padded, 1, m.tok_embd, true)).empty()) return err;
    if (!(err = mat("output.weight", E, hp.n_vocab_padded, 1, m.output, true)).empty()) return err;
    if (!(err = vec("token_embd_norm.weight", E, m.tok_norm, true)).empty()) return err;
    if (!(err = vec("output_norm.weight", E, m.output_norm, true)).empty()) return err;
    if (!(err = vec("output_res_query.weight", E, m.output_res_query, true)).empty()) return err;
    m.L.assign((size_t) hp.n_layer, LayerW());
    for (int il = 0; il < hp.n_layer; ++il) {
        LayerW& w = m.L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        const bool swa = hp.swa[(size_t) il] != 0;
        const long long extent = swa ? hp.window : hp.rel_extent;
        if (!(err = mat(b + "attn_q.weight", E, qd, 1, w.wq, true)).empty()) return err;
        if (!(err = mat(b + "attn_k.weight", E, kd, 1, w.wk, true)).empty()) return err;
        if (!(err = mat(b + "attn_v.weight", E, kd, 1, w.wv, true)).empty()) return err;
        if (!(err = mat(b + "attn_output.weight", qd, E, 1, w.wo, true)).empty()) return err;
        if (!(err = mat(b + "attn_gate.weight", E, qd, 1, w.wgate, true)).empty()) return err;
        if (!(err = vec(b + "attn_q_clip.weight", hd * H, w.q_clip, true)).empty()) return err;
        if (!(err = mat(b + "attn_sink_proj.weight", E, H, 1, w.wsink, true)).empty()) return err;
        if (!(err = mat(b + "attn_rel.weight", E, R * H, 1, w.wrel, true)).empty()) return err;
        if (!(err = vec(b + "attn_rel_bias.weight", R * extent, w.rel_bias, true)).empty()) return err;
        if (!(err = vec(b + "attn_rel_clip.weight", R * H, w.rel_clip, true)).empty()) return err;
        if (!(err = vec(b + "attn_k_conv.weight", C * kd, w.kconv, true)).empty()) return err;
        if (!(err = vec(b + "attn_v_conv.weight", C * kd, w.vconv, true)).empty()) return err;
        if (!(err = vec(b + "attn_post_conv.weight", C * E, w.post_conv, true)).empty()) return err;
        if (!(err = vec(b + "ffn_post_conv.weight", C * E, w.ffn_post_conv, true)).empty()) return err;
        if (swa) {
            if (!(err = vec(b + "attn_norm.weight", E, w.attn_norm, true)).empty()) return err;
        } else {
            if (!(err = vec(b + "attn_q_inp_norm.weight", E, w.q_inp_norm, true)).empty()) return err;
            if (!(err = vec(b + "attn_k_inp_norm.weight", E, w.k_inp_norm, true)).empty()) return err;
            if (!(err = vec(b + "attn_v_inp_norm.weight", E, w.v_inp_norm, true)).empty()) return err;
            if (!(err = mat(b + "attn_k_conv_proj.weight", E, C * KV, 1, w.kconv_proj, true)).empty()) return err;
            if (!(err = mat(b + "attn_v_conv_proj.weight", E, C * KV, 1, w.vconv_proj, true)).empty()) return err;
        }
        if (il > 0 && !(err = vec(b + "attn_res_query.weight", E * (swa ? 1 : 3), w.attn_res_query, true)).empty()) return err;
        if (!(err = vec(b + "ffn_res_query.weight", E, w.ffn_res_query, true)).empty()) return err;
        if (!(err = vec(b + "ffn_norm.weight", E, w.ffn_norm, true)).empty()) return err;
        if (!(err = vec(b + "ffn_gate_inp.weight", E * hp.n_expert, w.router, true)).empty()) return err;
        if (!(err = vec(b + "exp_probs_b.bias", hp.n_expert, w.exp_bias, false)).empty()) return err;
        if (!(err = mat(b + "ffn_gate_up_exps.weight", E, 2LL * hp.n_ff_exp, hp.n_expert, w.gate_up, true)).empty()) return err;
        if (!(err = mat(b + "ffn_down_exps.weight", hp.n_ff_exp, E, hp.n_expert, w.down, true)).empty()) return err;
        if (!(err = vec(b + "ffn_gate_up_exps.scale", hp.n_expert, w.gu_scale, false)).empty()) return err;
        if (!(err = vec(b + "ffn_down_exps.scale", hp.n_expert, w.dn_scale, false)).empty()) return err;
    }
    if (hp.has_loop()) {
        static const char* kind[4] = {"q", "k", "v", "ffn"};
        static const char* kind_unsloth[4] = {"attn", "k", "v", "mlp"};          // the Unsloth names of the q, k, v and ffn controllers
        m.loop.assign((size_t) (hp.loop_end - hp.loop_start), LoopW());
        const int core = hp.loop_end - hp.loop_start;
        const float *gq = nullptr, *gkv = nullptr;                              // Unsloth: the messages' queries of the whole loop, [2 core][n_embd] each
        const bool per_block = find("blk." + std::to_string(hp.loop_start) + ".loop_msg_query.weight") != nullptr;
        if (!per_block) {
            if (!(err = vec("loop_res_query.weight", 2LL * core * E, gq, true)).empty()) return err;
            if (!(err = vec("loop_kv_res_query.weight", 2LL * core * E, gkv, true)).empty()) return err;
        }
        for (int il = hp.loop_start; il < hp.loop_end; ++il) {
            LoopW& l = m.loop[(size_t) (il - hp.loop_start)];
            const std::string b = "blk." + std::to_string(il) + ".";
            if (per_block) {
                if (!(err = vec(b + "loop_msg_query.weight", 4 * E, l.msg_query, true)).empty()) return err;
            } else {                                                            // core layer i mixes its q, k, v and ffn messages with q[2i], kv[2i], kv[2i+1], q[2i+1]
                const int i = il - hp.loop_start;
                m.owned.emplace_back((size_t) (4 * E));
                float* d = m.owned.back().data();
                std::copy(gq + (size_t) (2 * i) * E, gq + (size_t) (2 * i + 1) * E, d);
                std::copy(gkv + (size_t) (2 * i) * E, gkv + (size_t) (2 * i + 1) * E, d + E);
                std::copy(gkv + (size_t) (2 * i + 1) * E, gkv + (size_t) (2 * i + 2) * E, d + 2 * E);
                std::copy(gq + (size_t) (2 * i + 1) * E, gq + (size_t) (2 * i + 2) * E, d + 3 * E);
                l.msg_query = d;
            }
            for (int c = 0; c < 4; ++c) {
                const std::string k = per_block ? b + "loop_" + kind[c] + "_ctrl_" : b + "loop_" + kind_unsloth[c] + "_";
                if (!(err = vec(k + "down.weight", 2 * E * hp.gate_rank, l.down[c], true)).empty()) return err;
                if (!(err = vec(k + "gain.weight", E * hp.gate_rank, l.gain[c], true)).empty()) return err;
                if (!(err = vec(k + "forget.weight", E * hp.gate_rank, l.forget[c], true)).empty()) return err;
            }
        }
    }
    return "";
}

/// Open a GGUF and bind it: "" or the reason it cannot run.
inline std::string load_model(const std::string& path, Model& m) {
    try {
        m.file = std::make_unique<strata::GgufFile>(path);
    } catch (const std::exception& e) {
        return std::string("cannot read ") + path + ": " + e.what();
    }
    std::string err = read_hparams(*m.file, m.hp);
    if (!err.empty()) return err;
    const strata::MetaValue* nm = m.file->get("general.name");
    m.name = nm ? nm->s : "inkling2";
    return bind_tensors(m);
}

}  // namespace i2
