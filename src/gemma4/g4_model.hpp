// g4_model.hpp - Gemma 4 (arch `gemma4`: the 26B-A4B mixture of experts, and the dense sizes it shares code with): the hyper-parameters read from
// a GGUF, and its tensors bound by name with their shapes checked.  The same as llama.cpp's llama_model_gemma4::load_arch_hparams /
// load_arch_tensors (src/models/gemma4.cpp), for the text model: the vision / audio towers are not read, per-layer embeddings (E2B / E4B)
// and the KV sharing between layers are refused with a message.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "g4_quant.hpp"

namespace g4 {

struct Hparams {
    int n_layer = 0, n_embd = 0, n_head = 0, n_vocab = 0, n_ff = 0, n_expert = 0, n_used = 0, n_ff_exp = 0, window = 0, ctx_train = 0;
    int hd_swa = 0, hd_full = 0, rot_swa = 0, rot_full = 0;
    float eps = 1e-6f, softcap = 0.f, rope_base = 1e6f, rope_base_swa = 1e4f;
    std::vector<uint8_t> swa;        // per layer: a sliding-window layer
    std::vector<int> n_kv;           // per layer: KV heads
    int eos = 106;
    bool has_experts() const { return n_expert > 0; }
    int head_dim(int il) const { return swa[(size_t) il] ? hd_swa : hd_full; }
    int rot(int il) const { return swa[(size_t) il] ? rot_swa : rot_full; }
    float base(int il) const { return swa[(size_t) il] ? rope_base_swa : rope_base; }
    int kv_heads(int il) const { return n_kv[(size_t) il]; }
    int q_dim(int il) const { return n_head * head_dim(il); }
    int kv_dim(int il) const { return n_kv[(size_t) il] * head_dim(il); }
    int n_swa_layers() const { int n = 0; for (uint8_t s : swa) n += s; return n; }
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

struct LayerW {
    Mat wq, wk, wv, wo, ffn_gate, ffn_up, ffn_down;
    Mat gate_up_exps, gate_exps, up_exps, down_exps;      // gate_up_exps: gate rows then up rows per expert; or separate gate_exps and up_exps
    const float *attn_norm = nullptr, *q_norm = nullptr, *k_norm = nullptr, *post_attn_norm = nullptr, *ffn_norm = nullptr, *post_ffw_norm = nullptr;
    const float *post_ffw_norm_1 = nullptr, *pre_ffw_norm_2 = nullptr, *post_ffw_norm_2 = nullptr, *router_scale = nullptr, *down_scale = nullptr;
    const float *router = nullptr;                        // [n_expert][n_embd] as floats (a router stored in another type is dequantized by the loader)
    float out_scale = 1.f;
    bool has_v = false, has_moe = false;
};

struct Model {
    Hparams hp;
    std::vector<LayerW> L;
    Mat tok_embd;
    const float* output_norm = nullptr;
    const float* rope_freqs = nullptr;                    // [hd_full / 2] frequency divisors of the global layers' rope (1e30: not rotated)
    int rope_freqs_n = 0;
    std::unique_ptr<strata::GgufFile> file;
    std::vector<std::vector<float>> owned;                // tensors converted to F32 at load (anything not stored as F32)
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

/// Hyper-parameters from the metadata; "" when fine, else the reason the file is not a Gemma 4 this engine runs.
inline std::string read_hparams(const strata::GgufFile& f, Hparams& hp) {
    using detail::meta_f;
    using detail::meta_int;
    const strata::MetaValue* arch = f.get("general.architecture");
    if (!arch || arch->s != "gemma4") return "general.architecture is " + (arch ? "\"" + arch->s + "\"" : std::string("missing")) + ", expected \"gemma4\"";
    const std::string p = "gemma4.";
    hp.n_layer = (int) meta_int(f, p + "block_count", 0);
    hp.n_embd = (int) meta_int(f, p + "embedding_length", 0);
    hp.n_ff = (int) meta_int(f, p + "feed_forward_length", 0);
    hp.n_head = (int) meta_int(f, p + "attention.head_count", 0);
    hp.n_expert = (int) meta_int(f, p + "expert_count", 0);
    hp.n_used = (int) meta_int(f, p + "expert_used_count", 0);
    hp.n_ff_exp = (int) meta_int(f, p + "expert_feed_forward_length", 0);
    hp.window = (int) meta_int(f, p + "attention.sliding_window", 0);
    hp.ctx_train = (int) meta_int(f, p + "context_length", 0);
    hp.hd_full = (int) meta_int(f, p + "attention.key_length", 0);
    hp.hd_swa = (int) meta_int(f, p + "attention.key_length_swa", hp.hd_full);
    hp.rot_full = (int) meta_int(f, p + "rope.dimension_count", hp.hd_full);
    hp.rot_swa = (int) meta_int(f, p + "rope.dimension_count_swa", hp.hd_swa);
    hp.eps = (float) meta_f(f, p + "attention.layer_norm_rms_epsilon", 1e-6);
    hp.softcap = (float) meta_f(f, p + "final_logit_softcapping", 0.0);
    hp.rope_base = (float) meta_f(f, p + "rope.freq_base", 1e6);
    hp.rope_base_swa = (float) meta_f(f, p + "rope.freq_base_swa", 1e4);
    hp.eos = (int) meta_int(f, "tokenizer.ggml.eos_token_id", 106);
    const strata::MetaValue* tok = f.get("tokenizer.ggml.tokens");
    hp.n_vocab = tok ? (int) tok->count : 0;
    if (hp.n_layer <= 0 || hp.n_embd <= 0 || hp.n_head <= 0 || hp.hd_full <= 0) return "missing or zero gemma4 geometry (block_count, embedding_length, head_count, key_length)";
    if (meta_int(f, p + "embedding_length_per_layer_input", 0) != 0)
        return "this is a Gemma 4 with per-layer embeddings (E2B / E4B); the engine runs the 26B-A4B and the dense sizes without them";
    if (meta_int(f, p + "attention.shared_kv_layers", 0) != 0) return "this Gemma 4 shares K / V between layers (E2B / E4B); not supported";
    if (meta_int(f, p + "attention.value_length", hp.hd_full) != hp.hd_full || meta_int(f, p + "attention.value_length_swa", hp.hd_swa) != hp.hd_swa)
        return "Gemma 4 needs key and value heads of one size";
    const strata::MetaValue* pat = f.get(p + "attention.sliding_window_pattern");
    const strata::MetaValue* kvh = f.get(p + "attention.head_count_kv");
    if (!pat || pat->type != strata::MetaType::ARRAY || (int) pat->count != hp.n_layer || (int) pat->items.size() != hp.n_layer)
        return "attention.sliding_window_pattern is missing or does not have one entry per layer";
    if (!kvh) return "attention.head_count_kv is missing";
    hp.swa.assign((size_t) hp.n_layer, 0);
    hp.n_kv.assign((size_t) hp.n_layer, 0);
    for (int i = 0; i < hp.n_layer; ++i) {
        hp.swa[(size_t) i] = pat->items[(size_t) i].u ? 1 : 0;
        if (kvh->type == strata::MetaType::ARRAY) {
            if ((int) kvh->count != hp.n_layer || (int) kvh->items.size() != hp.n_layer) return "attention.head_count_kv does not have one entry per layer";
            hp.n_kv[(size_t) i] = (int) kvh->items[(size_t) i].u;
        } else {
            hp.n_kv[(size_t) i] = (int) kvh->u;
        }
        if (hp.n_kv[(size_t) i] <= 0 || hp.n_head % hp.n_kv[(size_t) i] != 0) return "head_count is not a multiple of head_count_kv in layer " + std::to_string(i);
    }
    if (hp.window <= 0 && hp.n_swa_layers() > 0) return "sliding-window layers but no attention.sliding_window";
    if (hp.n_expert > 0 && (hp.n_used <= 0 || hp.n_used > hp.n_expert || hp.n_ff_exp <= 0)) return "mixture of experts without a valid expert_used_count / expert_feed_forward_length";
    if (hp.rot_full > hp.hd_full || hp.rot_swa > hp.hd_swa || (hp.rot_full & 1) || (hp.rot_swa & 1)) return "rope.dimension_count is odd or larger than the head";
    return "";
}

/// Bind and check the tensors of a Gemma 4.
inline std::string bind_tensors(Model& m) {
    const strata::GgufFile& f = *m.file;
    const Hparams& hp = m.hp;
    auto find = [&](const std::string& n) -> const strata::TensorInfo* { return f.find(n); };
    std::string err;
    auto mat = [&](const std::string& n, long long n_in, long long n_out, long long n_exp, Mat& out, bool required) -> std::string {
        const strata::TensorInfo* t = find(n);
        if (!t) return required ? "missing tensor " + n : "";
        if (g4::block_elems((int) t->type) == 0)
            return "tensor " + n + " is " + t->type_name() + ", a type the Gemma 4 engine does not read (supported: F32 F16 BF16 Q4_0 Q4_1 Q5_0 Q5_1 Q8_0 Q2_K Q3_K Q4_K Q5_K Q6_K IQ4_NL IQ4_XS)";
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
    const long long E = hp.n_embd;
    if (!(err = mat("token_embd.weight", E, hp.n_vocab, 1, m.tok_embd, true)).empty()) return err;
    if (!(err = vec("output_norm.weight", E, m.output_norm, true)).empty()) return err;
    if (find("output.weight") != nullptr) return "this file has its own output.weight; the engine ties the output to token_embd, as Gemma 4 does";
    bool any_global = false;
    for (int i = 0; i < hp.n_layer; ++i) any_global |= !hp.swa[(size_t) i];
    if (const strata::TensorInfo* rf = find("rope_freqs.weight")) {
        m.rope_freqs_n = (int) rf->elements();
        if (m.rope_freqs_n != hp.hd_full / 2) return "rope_freqs.weight has " + std::to_string(m.rope_freqs_n) + " values, expected " + std::to_string(hp.hd_full / 2);
        if (!(err = vec("rope_freqs.weight", m.rope_freqs_n, m.rope_freqs, true)).empty()) return err;
    } else if (any_global) {
        return "missing tensor rope_freqs.weight (the global layers' proportional rope)";
    }
    m.L.assign((size_t) hp.n_layer, LayerW());
    for (int il = 0; il < hp.n_layer; ++il) {
        LayerW& w = m.L[(size_t) il];
        const std::string b = "blk." + std::to_string(il) + ".";
        const long long hd = hp.head_dim(il), qd = hp.q_dim(il), kd = hp.kv_dim(il);
        if (!(err = vec(b + "attn_norm.weight", E, w.attn_norm, true)).empty()) return err;
        if (!(err = mat(b + "attn_q.weight", E, qd, 1, w.wq, true)).empty()) return err;
        if (!(err = mat(b + "attn_k.weight", E, kd, 1, w.wk, true)).empty()) return err;
        if (!(err = mat(b + "attn_v.weight", E, kd, 1, w.wv, false)).empty()) return err;
        w.has_v = (bool) w.wv;
        if (!(err = mat(b + "attn_output.weight", qd, E, 1, w.wo, true)).empty()) return err;
        if (!(err = vec(b + "attn_q_norm.weight", hd, w.q_norm, true)).empty()) return err;
        if (!(err = vec(b + "attn_k_norm.weight", hd, w.k_norm, true)).empty()) return err;
        if (!(err = vec(b + "post_attention_norm.weight", E, w.post_attn_norm, true)).empty()) return err;
        if (!(err = vec(b + "ffn_norm.weight", E, w.ffn_norm, true)).empty()) return err;
        if (!(err = mat(b + "ffn_gate.weight", E, hp.n_ff, 1, w.ffn_gate, true)).empty()) return err;
        if (!(err = mat(b + "ffn_up.weight", E, hp.n_ff, 1, w.ffn_up, true)).empty()) return err;
        if (!(err = mat(b + "ffn_down.weight", hp.n_ff, E, 1, w.ffn_down, true)).empty()) return err;
        if (!(err = vec(b + "post_ffw_norm.weight", E, w.post_ffw_norm, true)).empty()) return err;
        if (hp.n_expert > 0 && find(b + "ffn_gate_inp.weight") != nullptr) {
            w.has_moe = true;
            if (!(err = vec(b + "ffn_gate_inp.weight", E * hp.n_expert, w.router, true)).empty()) return err;
            if (!(err = vec(b + "ffn_gate_inp.scale", E, w.router_scale, true)).empty()) return err;
            if (!(err = vec(b + "pre_ffw_norm_2.weight", E, w.pre_ffw_norm_2, true)).empty()) return err;
            if (!(err = vec(b + "post_ffw_norm_1.weight", E, w.post_ffw_norm_1, true)).empty()) return err;
            if (!(err = vec(b + "post_ffw_norm_2.weight", E, w.post_ffw_norm_2, true)).empty()) return err;
            if (find(b + "ffn_gate_up_exps.weight")) {
                if (!(err = mat(b + "ffn_gate_up_exps.weight", E, 2LL * hp.n_ff_exp, hp.n_expert, w.gate_up_exps, true)).empty()) return err;
            } else {
                if (!(err = mat(b + "ffn_gate_exps.weight", E, hp.n_ff_exp, hp.n_expert, w.gate_exps, true)).empty()) return err;
                if (!(err = mat(b + "ffn_up_exps.weight", E, hp.n_ff_exp, hp.n_expert, w.up_exps, true)).empty()) return err;
            }
            if (!(err = mat(b + "ffn_down_exps.weight", hp.n_ff_exp, E, hp.n_expert, w.down_exps, true)).empty()) return err;
            if (!(err = vec(b + "ffn_down_exps.scale", hp.n_expert, w.down_scale, false)).empty()) return err;
        }
        if (find(b + "layer_output_scale.weight")) {
            const float* sp = nullptr;
            if (!(err = vec(b + "layer_output_scale.weight", 1, sp, true)).empty()) return err;
            w.out_scale = *sp;
        }
        if (!hp.swa[(size_t) il] && m.rope_freqs == nullptr) return "global layer " + std::to_string(il) + " without rope_freqs";
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
    m.name = nm ? nm->s : "gemma4";
    return bind_tensors(m);
}

}  // namespace g4
