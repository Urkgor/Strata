#!/usr/bin/env python3
"""tools/gemma4/make_tiny.py - a small random Gemma 4 GGUF for checks: the real model's layout and tokenizer, tiny dimensions.

    tools/gemma4/make_tiny.py --tokenizer-from gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --out tiny-f16.gguf [--quantize llama-quantize --out-q tiny.gguf]

Same tensor names and shapes (per layer: sliding-window layers with their own head size and KV heads, global layers with a bigger head,
no attn_v there (V = K), a dense GELU MLP beside the 8-of-N experts with fused gate+up, the router's scale, the per-expert down scale,
a layer scalar, the shared rope_freqs, a tied output), the real vocabulary and chat template (copied from the GGUF), and random weights.
llama.cpp runs it (tools/gemma4/llama_ref.cpp), so it is the common ground of the engine's checks.  --quantize then gives every
weight a type with llama-quantize, a different mix per layer so that each kernel is met: Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q4_K, Q5_K, Q6_K.
"""
import argparse
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/tmp/claude-0/b/gg9/gguf-py")      # overridden by --gguf-py
import gguf  # noqa: E402

CFG = dict(n_layer=6, n_embd=256, n_head=4, kv_swa=2, kv_full=1, hd_swa=64, hd_full=128, n_ff=192, n_ff_exp=64, n_expert=8, n_used=2,
           window=8, pattern=[True, True, False, True, True, False], softcap=30.0, eps=1e-6)


def read_tokenizer(path):
    r = gguf.GGUFReader(path)
    out = {}
    for k, f in r.fields.items():
        if not k.startswith("tokenizer."):
            continue
        if k == "tokenizer.ggml.tokens" or k == "tokenizer.ggml.merges":
            out[k] = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        elif k == "tokenizer.chat_template" or k == "tokenizer.ggml.model":
            out[k] = bytes(f.parts[f.data[0]]).decode("utf-8")
        elif k in ("tokenizer.ggml.scores", "tokenizer.ggml.token_type"):
            out[k] = np.concatenate([np.asarray(f.parts[i]) for i in f.data])
        else:
            out[k] = f.contents()
    return out


def dequantize_copy(src, dst):
    """A copy of a GGUF with every tensor as F32: the same values the quantized file stands for, so llama.cpp's CPU runs it without
    quantizing the activations (its dot products against quantized weights work on 8-bit activations) and an engine that keeps floats
    must agree with it to rounding."""
    r = gguf.GGUFReader(src)
    arch = bytes(r.fields["general.architecture"].parts[r.fields["general.architecture"].data[0]]).decode()
    w = gguf.GGUFWriter(dst, arch)
    for k, f in r.fields.items():
        if k.startswith("GGUF.") or k == "general.architecture":
            continue
        vt = f.types[0]
        if vt == gguf.GGUFValueType.ARRAY:
            sub = f.types[1]
            if sub == gguf.GGUFValueType.STRING:
                val = [bytes(f.parts[i]).decode("utf-8") for i in f.data]
            else:
                val = np.concatenate([np.asarray(f.parts[i]) for i in f.data]).tolist()
            w.add_key_value(k, val, vt, sub)
        elif vt == gguf.GGUFValueType.STRING:
            w.add_key_value(k, bytes(f.parts[f.data[0]]).decode("utf-8"), vt)
        else:
            w.add_key_value(k, f.contents(), vt)
    for t in r.tensors:
        data = t.data
        if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            data = gguf.quants.dequantize(t.data, t.tensor_type)
        w.add_tensor(t.name, np.ascontiguousarray(np.asarray(data, dtype=np.float32)))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print("wrote", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dequantize", nargs=2, metavar=("SRC", "DST"), help="write DST: SRC with every tensor as F32 (no other option needed)")
    ap.add_argument("--tokenizer-from")
    ap.add_argument("--out")
    ap.add_argument("--quantize")
    ap.add_argument("--out-q")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--iq", action="store_true", help="--quantize with the code-book types (IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS) where a row is a multiple of 256")
    ap.add_argument("--vocab", type=int, help="keep only the first N tokens and no merges: a small model for the engine's emulation tests (llama.cpp cannot load it)")
    ap.add_argument("--gguf-py")
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    if a.dequantize:
        dequantize_copy(*a.dequantize)
        return
    if not a.tokenizer_from or not a.out:
        ap.error("--tokenizer-from and --out are required")
    c = CFG
    tk = read_tokenizer(a.tokenizer_from)
    if a.vocab:
        tk["tokenizer.ggml.tokens"] = tk["tokenizer.ggml.tokens"][:a.vocab]
        tk["tokenizer.ggml.scores"] = tk["tokenizer.ggml.scores"][:a.vocab]
        tk["tokenizer.ggml.token_type"] = tk["tokenizer.ggml.token_type"][:a.vocab]
        tk["tokenizer.ggml.merges"] = []
    n_vocab = len(tk["tokenizer.ggml.tokens"])
    rng = np.random.default_rng(a.seed)
    w = gguf.GGUFWriter(a.out, "gemma4")
    w.add_name("tiny-gemma4")
    w.add_block_count(c["n_layer"])
    w.add_context_length(4096)
    w.add_embedding_length(c["n_embd"])
    w.add_feed_forward_length(c["n_ff"])
    w.add_head_count(c["n_head"])
    w.add_head_count_kv([c["kv_swa"] if s else c["kv_full"] for s in c["pattern"]])
    w.add_float32("gemma4.rope.freq_base", 1000000.0)
    w.add_float32("gemma4.rope.freq_base_swa", 10000.0)
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_expert_count(c["n_expert"])
    w.add_expert_used_count(c["n_used"])
    w.add_key_length(c["hd_full"])
    w.add_value_length(c["hd_full"])
    w.add_float32("gemma4.final_logit_softcapping", c["softcap"])
    w.add_sliding_window(c["window"])
    w.add_uint32("gemma4.attention.shared_kv_layers", 0)
    w.add_uint32("gemma4.embedding_length_per_layer_input", 0)
    w.add_array("gemma4.attention.sliding_window_pattern", c["pattern"])
    w.add_uint32("gemma4.attention.key_length_swa", c["hd_swa"])
    w.add_uint32("gemma4.attention.value_length_swa", c["hd_swa"])
    w.add_expert_feed_forward_length(c["n_ff_exp"])
    w.add_rope_dimension_count(c["hd_full"])
    w.add_uint32("gemma4.rope.dimension_count_swa", c["hd_swa"])
    # the vocabulary
    w.add_tokenizer_model("gemma4")
    w.add_token_list(tk["tokenizer.ggml.tokens"])
    w.add_token_scores(tk["tokenizer.ggml.scores"].tolist())
    w.add_token_types(tk["tokenizer.ggml.token_type"].tolist())
    if tk["tokenizer.ggml.merges"]:
        w.add_token_merges(tk["tokenizer.ggml.merges"])
    for key, fn in (("bos", w.add_bos_token_id), ("eos", w.add_eos_token_id), ("unknown", w.add_unk_token_id), ("padding", w.add_pad_token_id),
                    ("mask", w.add_mask_token_id)):
        v = tk.get("tokenizer.ggml.%s_token_id" % key)
        if v is not None:
            fn(int(v))
    w.add_add_bos_token(bool(tk.get("tokenizer.ggml.add_bos_token", True)))
    w.add_add_space_prefix(bool(tk.get("tokenizer.ggml.add_space_prefix", False)))
    if "tokenizer.chat_template" in tk:
        w.add_chat_template(tk["tokenizer.chat_template"])

    E, V = c["n_embd"], n_vocab

    def mat(rows, cols, std):          # a [rows(out), cols(in)] weight: ggml's [cols, rows]; stored f16 (quantize.py converts)
        return (rng.standard_normal((rows, cols)) * std).astype(np.float16)

    def vec(n, base=1.0, jitter=0.1):
        return (base + jitter * rng.standard_normal(n)).astype(np.float32)

    def f32(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float32))

    def f16(name, arr):
        w.add_tensor(name, np.ascontiguousarray(arr, dtype=np.float16))

    f16("token_embd.weight", (rng.standard_normal((V, E)) * 0.5).astype(np.float16))
    f32("output_norm.weight", vec(E))
    hd_full = c["hd_full"]
    values = [1.0] * (hd_full // 4 // 2) + [1e30] * (hd_full // 2 - hd_full // 4 // 2)    # the converter's proportional rope: the first quarter rotates
    f32("rope_freqs.weight", np.array(values, dtype=np.float32))
    for il in range(c["n_layer"]):
        swa = c["pattern"][il]
        hd = c["hd_swa"] if swa else c["hd_full"]
        nkv = c["kv_swa"] if swa else c["kv_full"]
        p = "blk.%d." % il
        f32(p + "attn_norm.weight", vec(E))
        f16(p + "attn_q.weight", mat(c["n_head"] * hd, E, 0.06))
        f16(p + "attn_k.weight", mat(nkv * hd, E, 0.06))
        if swa:                                              # the global layers have no attn_v: V = K
            f16(p + "attn_v.weight", mat(nkv * hd, E, 0.06))
        f16(p + "attn_output.weight", mat(E, c["n_head"] * hd, 0.06))
        f32(p + "attn_q_norm.weight", vec(hd))
        f32(p + "attn_k_norm.weight", vec(hd))
        f32(p + "post_attention_norm.weight", vec(E))
        f32(p + "ffn_norm.weight", vec(E))
        f16(p + "ffn_gate.weight", mat(c["n_ff"], E, 0.06))
        f16(p + "ffn_up.weight", mat(c["n_ff"], E, 0.06))
        f16(p + "ffn_down.weight", mat(E, c["n_ff"], 0.06))
        f32(p + "post_ffw_norm.weight", vec(E))
        f32(p + "post_ffw_norm_1.weight", vec(E))
        f32(p + "post_ffw_norm_2.weight", vec(E))
        f32(p + "pre_ffw_norm_2.weight", vec(E))
        f32(p + "ffn_gate_inp.weight", (rng.standard_normal((c["n_expert"], E)) * 0.7))
        f32(p + "ffn_gate_inp.scale", vec(E))
        f16(p + "ffn_gate_up_exps.weight", (rng.standard_normal((c["n_expert"], 2 * c["n_ff_exp"], E)) * 0.06))
        f16(p + "ffn_down_exps.weight", (rng.standard_normal((c["n_expert"], E, c["n_ff_exp"])) * 0.06))
        f32(p + "ffn_down_exps.scale", vec(c["n_expert"], 1.0, 0.3))
        f32(p + "layer_output_scale.weight", np.array([0.8 + 0.2 * rng.random()], dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print("wrote", a.out)
    if a.quantize:
        # a type per tensor: a mix by layer so that every kernel meets real data; the experts as in the Unsloth Q4_K_M file
        # (gate+up Q4_K, down Q5_1; the last layer's down Q8_0), the rest of layer 0 Q8_0
        mixes = {
            0: dict(attn_q="q8_0", attn_k="q8_0", attn_v="q8_0", attn_output="q8_0", ffn_gate="q8_0", ffn_up="q8_0", ffn_down="q8_0"),
            1: dict(attn_q="q6_k", attn_k="q5_k", attn_v="q4_0", attn_output="q4_1", ffn_gate="q5_0", ffn_up="q4_k", ffn_down="q5_0"),
            2: dict(attn_q="q4_k", attn_k="q8_0", attn_output="q5_k", ffn_gate="q4_1", ffn_up="q6_k", ffn_down="q4_0"),
            3: dict(attn_q="q5_0", attn_k="q5_1", attn_v="q6_k", attn_output="q8_0", ffn_gate="q4_k", ffn_up="q5_1", ffn_down="q8_0"),
        }
        if a.iq:     # the rows that are a multiple of 256 take the IQ types; the others (ffn_down: 192, experts' down: 64) what they can
            mixes = {
                0: dict(attn_q="iq2_xs", attn_k="iq2_s", attn_v="iq3_xxs", attn_output="q8_0", ffn_gate="iq3_s", ffn_up="iq4_xs", ffn_down="q8_0"),
                1: dict(attn_q="iq4_nl", attn_k="iq3_s", attn_v="iq2_xs", attn_output="q8_0", ffn_gate="iq2_s", ffn_up="iq3_xxs", ffn_down="q8_0"),
                2: dict(attn_q="iq3_xxs", attn_k="iq4_xs", attn_output="iq3_s", ffn_gate="iq4_nl", ffn_up="iq2_xs", ffn_down="q8_0"),
                3: dict(attn_q="iq3_s", attn_k="iq2_s", attn_v="iq4_nl", attn_output="q8_0", ffn_gate="iq3_xxs", ffn_up="iq3_s", ffn_down="q8_0"),
            }
        args = [a.quantize]
        if a.iq:     # the IQ2 / IQ3 types need an importance matrix: a flat one (every weight equally important), in llama.cpp's legacy .dat format
            import struct
            ents = []
            for il in range(c["n_layer"]):
                swa = c["pattern"][il]
                for nm, n_in, n_exp in (("attn_q", E, 1), ("attn_k", E, 1), ("attn_v", E, 1), ("attn_output", c["n_head"] * (c["hd_swa"] if swa else c["hd_full"]), 1),
                                        ("ffn_gate", E, 1), ("ffn_up", E, 1), ("ffn_down", c["n_ff"], 1), ("ffn_gate_up_exps", E, c["n_expert"]),
                                        ("ffn_down_exps", c["n_ff_exp"], c["n_expert"])):
                    if nm == "attn_v" and not swa:
                        continue
                    ents.append(("blk.%d.%s.weight" % (il, nm), n_in * n_exp))
            with open(a.out_q + ".imatrix.dat", "wb") as f:
                f.write(struct.pack("<i", len(ents)))
                for nm, n in ents:
                    b = nm.encode()
                    f.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", 1, n) + np.ones(n, dtype="<f4").tobytes())
                f.write(struct.pack("<i", 1) + struct.pack("<i", 4) + b"none")
            args += ["--imatrix", a.out_q + ".imatrix.dat"]
        for il, m in mixes.items():
            for t, ty in m.items():
                args += ["--tensor-type", "blk\\.%d\\.%s\\.weight=%s" % (il, t, ty)]
        for il in range(c["n_layer"]):
            args += ["--tensor-type", "blk\\.%d\\.ffn_gate_up_exps\\.weight=%s" % (il, ("iq3_s", "iq3_xxs", "iq2_s", "iq2_xs", "iq4_xs", "iq4_nl")[il % 6] if a.iq else "q4_k"),
                     "--tensor-type", "blk\\.%d\\.ffn_down_exps\\.weight=%s" % (il, "q8_0" if il == c["n_layer"] - 1 else "q5_1")]
        args += [a.out, a.out_q, "q8_0"]
        print(" ".join(args[:6]), "...")
        subprocess.run(args, check=True)


if __name__ == "__main__":
    main()
