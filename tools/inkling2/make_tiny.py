#!/usr/bin/env python3
"""tools/inkling2/make_tiny.py - a small random Inkling 2 (inkling2_loop) GGUF for checks: the real model's layout, tiny dimensions.

    tools/inkling2/make_tiny.py --out tiny-f32.gguf [--type f32|f16|q8_0|q8_0rt] [--seed N] [--gguf-py DIR]

The tensor names, shapes and metadata are the ones llama.cpp's conversion writes (conversion/inkling2.py, src/models/inkling2.cpp): 12 layers in
blocks of 3 (a global layer ends each block), layers 3 .. 8 looped 3 times (24 visits), 128-wide... no: 64-wide hidden, 4 heads over 2 KV heads of 32,
a window of 6, 8 experts with 2 used, a vocabulary of 300 byte-level tokens padded to 320.  Random weights, scaled to keep every stage finite.
llama.cpp runs it (tools/inkling2/llama_ref.cpp), so it is the common ground of the engine's checks.
"""
import argparse
import sys

import numpy as np

CFG = dict(n_layer=12, block=3, loop_start=3, loop_end=9, passes=3, rank=8, n_embd=64, n_head=4, n_kv=2, hd=32, n_rel=8, extent=16, window=6,
           n_conv=4, n_expert=8, n_used=2, n_ff_exp=32, vocab=300, vocab_padded=320, eps=1e-6, qk_clip=22.627417, rel_clip=8.0, log_floor=16.0,
           log_alpha=0.1, situ_beta=4.0, situ_lin=25.0, route_scale=8.0, softcap=30.0)


def byte_tokens():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = dict(zip(bs, [chr(c) for c in cs]))
    return [m[b] for b in range(256)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--type", default="f32", choices=["f32", "f16", "q8_0", "q8_0rt"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gguf-py", default="/tmp/claude-0/ll2/src/gguf-py")
    ap.add_argument("--passes", type=int, default=None)
    ap.add_argument("--mixed", action="store_true", help="experts in F16, those of the last 2 layers in Q8_0 (the real IQ4_NL file mixes types per layer)")
    ap.add_argument("--pre", default="gpt-2", help="tokenizer.ggml.pre (inkling2: the o200k-style pre-tokenizer of the real model)")
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import gguf

    c = dict(CFG)
    if a.passes:
        c["passes"] = a.passes
    rng = np.random.default_rng(a.seed)
    E, H, KV, hd, R, C = c["n_embd"], c["n_head"], c["n_kv"], c["hd"], c["n_rel"], c["n_conv"]
    qd, kd, NE, F = H * hd, KV * hd, c["n_expert"], c["n_ff_exp"]
    arch = "inkling2_loop"
    w = gguf.GGUFWriter(a.out, arch)
    swa = [(i % c["block"]) != c["block"] - 1 for i in range(c["n_layer"])]
    w.add_name("tiny-inkling2")
    w.add_block_count(c["n_layer"])
    w.add_context_length(4096)
    w.add_embedding_length(E)
    w.add_head_count([H] * c["n_layer"])
    w.add_head_count_kv([KV] * c["n_layer"])
    w.add_key_length(hd)
    w.add_value_length(hd)
    w.add_key_length_swa(hd)
    w.add_value_length_swa(hd)
    w.add_layer_norm_rms_eps(c["eps"])
    w.add_rope_dimension_count(0)
    w.add_vocab_size(c["vocab_padded"])
    w.add_vocab_size_unpadded(c["vocab"])
    w.add_expert_count(NE)
    w.add_expert_used_count(c["n_used"])
    w.add_expert_feed_forward_length(F)
    w.add_sliding_window(c["window"])
    w.add_sliding_window_pattern(swa)
    w.add_shortconv_l_cache(C)
    w.add_expert_weights_scale(c["route_scale"])
    w.add_activation_situ_beta(c["situ_beta"])
    w.add_activation_situ_linear_beta(c["situ_lin"])
    w.add_attn_res_block_size(c["block"])
    w.add_final_logit_softcapping(c["softcap"])
    w.add_relative_length(R)
    w.add_relative_extent(c["extent"])
    w.add_qk_clip(c["qk_clip"])
    w.add_relative_clip(c["rel_clip"])
    w.add_log_scale_floor(int(c["log_floor"]))
    w.add_log_scale_alpha(c["log_alpha"])
    w.add_loop_start(c["loop_start"])
    w.add_loop_end(c["loop_end"])
    w.add_loop_count(c["passes"])
    w.add_loop_gate_rank(c["rank"])
    w.add_loop_varying(False)
    # a byte-level BPE vocabulary: the 256 byte tokens, a few merges, specials
    toks = byte_tokens()
    merges = []
    for x, y in (("a", "b"), ("t", "h"), ("th", "e"), ("Ġ", "t")):
        merges.append(f"{x} {y}")
        toks.append(x + y)
    while len(toks) < c["vocab"] - 2:
        toks.append(f"<|unused_{len(toks)}|>")
    toks += ["<|message_user|>", "<|end_message|>"]
    toks = toks[: c["vocab"]]
    types = [gguf.TokenType.NORMAL] * c["vocab"]
    types[-1] = types[-2] = gguf.TokenType.CONTROL
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre(a.pre)
    w.add_token_list(toks)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_eos_token_id(c["vocab"] - 1)
    w.add_add_bos_token(False)


    def rnd(*shape, s=0.05):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    def put(name, arr, big=True):
        """big: a weight matrix that takes the chosen type (the rest is F32)"""
        if a.mixed and "_exps" in name:
            if int(name.split(".")[1]) >= c["n_layer"] - 2:
                w.add_tensor(name, gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
            else:
                w.add_tensor(name, arr.astype(np.float16))
            return
        if big and a.type == "q8_0rt":      # the Q8_0 weights written back as F32: llama.cpp then computes with the values the engine reads from the Q8_0 file
            q = gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0)
            w.add_tensor(name, gguf.quants.dequantize(q, gguf.GGMLQuantizationType.Q8_0).astype(np.float32))
        elif big and a.type == "q8_0":
            w.add_tensor(name, gguf.quants.quantize(arr, gguf.GGMLQuantizationType.Q8_0), raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        elif big and a.type == "f16":
            w.add_tensor(name, arr.astype(np.float16))
        else:
            w.add_tensor(name, arr.astype(np.float32))

    put("token_embd.weight", rnd(c["vocab_padded"], E, s=0.3))
    put("token_embd_norm.weight", 1 + rnd(E), False)
    put("output.weight", rnd(c["vocab_padded"], E, s=0.3))
    put("output_norm.weight", 1 + rnd(E), False)
    put("output_res_query.weight", rnd(E, s=0.5), False)
    for il in range(c["n_layer"]):
        b = f"blk.{il}."
        local = swa[il]
        extent = c["window"] if local else c["extent"]
        put(b + "attn_q.weight", rnd(qd, E))
        put(b + "attn_k.weight", rnd(kd, E))
        put(b + "attn_v.weight", rnd(kd, E))
        put(b + "attn_output.weight", rnd(E, qd))
        put(b + "attn_gate.weight", rnd(qd, E))
        put(b + "attn_q_clip.weight", 1 + rnd(H, hd), False)
        put(b + "attn_sink_proj.weight", rnd(H, E))
        put(b + "attn_rel.weight", rnd(R * H, E))
        put(b + "attn_rel_bias.weight", rnd(extent, R, s=0.3), False)
        put(b + "attn_rel_clip.weight", 1 + rnd(H, R), False)
        put(b + "attn_k_conv.weight", rnd(kd, C, s=0.4), False)
        put(b + "attn_v_conv.weight", rnd(kd, C, s=0.4), False)
        put(b + "attn_post_conv.weight", rnd(E, C, s=0.4), False)
        put(b + "ffn_post_conv.weight", rnd(E, C, s=0.4), False)
        if local:
            put(b + "attn_norm.weight", 1 + rnd(E), False)
        else:
            for n in ("q", "k", "v"):
                put(b + f"attn_{n}_inp_norm.weight", 1 + rnd(E), False)
            put(b + "attn_k_conv_proj.weight", rnd(C * KV, E))
            put(b + "attn_v_conv_proj.weight", rnd(C * KV, E))
        if il > 0:
            put(b + "attn_res_query.weight", rnd(1 if local else 3, E, s=0.5), False)
        put(b + "ffn_res_query.weight", rnd(E, s=0.5), False)
        put(b + "ffn_norm.weight", 1 + rnd(E), False)
        put(b + "ffn_gate_inp.weight", rnd(NE, E, s=0.3), False)
        put(b + "exp_probs_b.bias", rnd(NE, s=0.1), False)
        put(b + "ffn_gate_up_exps.weight", rnd(NE, 2 * F, E))
        put(b + "ffn_down_exps.weight", rnd(NE, E, F))
    for il in range(c["loop_start"], c["loop_end"]):
        b = f"blk.{il}."
        put(b + "loop_msg_query.weight", rnd(4, E, s=0.5), False)
        for k in ("q", "k", "v", "ffn"):
            put(b + f"loop_{k}_ctrl_down.weight", rnd(c["rank"], 2 * E, s=0.2), False)
            put(b + f"loop_{k}_ctrl_gain.weight", rnd(E, c["rank"], s=0.4), False)
            put(b + f"loop_{k}_ctrl_forget.weight", rnd(E, c["rank"], s=0.4), False)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
