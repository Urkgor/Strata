#!/usr/bin/env python3
"""A small random-weight Qwen3.5-MoE model, as a GGUF and as PyTorch reference outputs, for testing strata-q35.

Qwen3.6-35B-A3B is `model_type qwen3_5_moe`: Gated DeltaNet layers (3 of 4) and gated attention (1 of 4), a
plain residual, 256 experts with top-8 and a shared expert.  This builds the same architecture at 350 thousand
parameters, so the whole chain can be checked on any PC in seconds:

    transformers (reference)  ->  llama.cpp's convert_hf_to_gguf.py  ->  strata-q35  ->  the same tokens

    python tools/q35_tiny_model.py OUTDIR --llama-cpp /path/to/llama.cpp [--preset tiny|real-shape]

`--preset real-shape` keeps Qwen3.6-35B-A3B's own geometry - hidden size 2048, head dimension 256 with a quarter of it
rotated (M-RoPE sections 11/11/10), 16 query and 2 key/value heads, Gated DeltaNet with 16 key and 32 value heads of
128, 512-wide experts with top-8 and a 512-wide shared expert - but only 2 layers (one of each kind) and 32 experts:
about 0.3 billion parameters, 1.2 GB as F32.  Its matrices are multiples of 256, so llama.cpp's K-quants (Q4_K,
Q5_K, Q6_K, IQ4_XS) apply to it: `llama-quantize` it and compare logits to see them on the real shapes.

needs `torch`, `transformers` (5.x, with qwen3_5_moe), `tokenizers` and `numpy`; llama.cpp's checkout is only used
for its converter (the commit in third_party/ggml/VERSION.txt).  OUTDIR gets `hf/` (the model), `tiny-f32.gguf` and
`ref.json` (with --mtp also `tiny-mtp-f32.gguf`, the same model plus a multi-token-prediction block fitted to guess; three prompts of 5, 40 and 200 token ids with their greedy continuations of 24 tokens and the 8 best
logits after the prompt).  tools/test_q35.py takes OUTDIR.
"""
import argparse
import json
import os
import runpy
import sys

SEQ_LENS = (5, 40, 200)
N_GEN = 24

PRESETS = {
    "tiny": dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                 linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
                 num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
                 layer_types=["linear_attention"] * 3 + ["full_attention"], mrope_section=[2, 1, 1],
                 init=("fixed", 0.35)),
    "real-shape": dict(hidden_size=2048, num_hidden_layers=2, num_attention_heads=16, num_key_value_heads=2, head_dim=256,
                       linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128,
                       linear_value_head_dim=128, num_experts=32, num_experts_per_tok=8, moe_intermediate_size=512,
                       shared_expert_intermediate_size=512, layer_types=["linear_attention", "full_attention"],
                       mrope_section=[11, 11, 10], init=("fanin", 2.0)),
}


def make_model(out, preset="tiny"):
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    os.makedirs(out, exist_ok=True)
    # a byte-level BPE: the 256 bytes, a few merges and the three special tokens Qwen's chat format needs
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["hello world, this is a tiny corpus for a tiny model.",
              "the quick brown fox jumps over the lazy dog 0123456789",
              "Strata runs Qwen on a normal PC.", "def f(x): return x * 2  # python"] * 20
    trainer = trainers.BpeTrainer(vocab_size=300, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>' + '\n'}}"
                          "{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}")
    fast.save_pretrained(out)
    vocab = len(fast)

    torch.manual_seed(1234)
    pr = dict(PRESETS[preset])
    kind, amount = pr.pop("init")
    mrope = pr.pop("mrope_section")
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=vocab, linear_conv_kernel_dim=4, max_position_embeddings=512, rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": mrope, "mrope_interleaved": True},
        eos_token_id=fast.eos_token_id, pad_token_id=fast.convert_tokens_to_ids("<|endoftext|>"), **pr)
    cfg.architectures = ["Qwen3_5MoeForCausalLM"]
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():   # peaked logits: the greedy path must not hang on near-ties
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, amount if kind == "fixed" else amount / p.shape[-1] ** 0.5)
            elif "norm" in name:
                if p.mean() > 0.5:
                    p.fill_(1.0)
                else:
                    p.normal_(0, 0.1)
    model.save_pretrained(out, safe_serialization=True)
    return vocab


def convert(llama_cpp, hf_dir, gguf, mtp=False):
    """llama.cpp's converter, with the BPE pre-tokenizer named by hand: the tiny tokenizer has no known hash.
    `mtp`: keep the checkpoint's MTP block (the nextn layer) in the GGUF."""
    sys.path.insert(0, llama_cpp)
    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    import conversion.base as base
    base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
    argv = sys.argv
    sys.argv = ["convert_hf_to_gguf.py", hf_dir, "--outfile", gguf, "--outtype", "f32"] + ([] if mtp else ["--no-mtp"])
    try:
        runpy.run_path(os.path.join(llama_cpp, "convert_hf_to_gguf.py"), run_name="__main__")
    finally:
        sys.argv = argv


def add_mtp(hf_dir, out_dir, prompts, n_gen=300, steps=1500):
    """The same model with a multi-token-prediction block, as Qwen3.6's checkpoint carries it: `mtp.fc` (the hidden state and
    the next token's embedding in, one hidden size out), the two pre-norms, one more decoder layer (a copy of the last,
    full-attention one) and the final norm.

    The block is made to guess something: its decoder layer is switched off (attention output and every down projection
    zero, so the layer adds nothing to the residual) and `fc` is fitted - by gradient descent, on the model's own greedy
    continuations of `prompts` - to give, from (hidden state at p, token p+1), the token the model puts at p+2.  A linear
    map cannot do that well; it gets a fair share of the guesses right, which is what a test of the accepted-draft paths
    needs (a random block would almost never be right).  Returns the share of right guesses on the data it was fitted to."""
    import shutil
    import torch
    from safetensors.torch import load_file, save_file
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(hf_dir):
        if name != "model.safetensors" and name != "config.json":
            shutil.copy(os.path.join(hf_dir, name), os.path.join(out_dir, name))
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    n_layer = cfg["num_hidden_layers"]
    last = f"model.layers.{n_layer - 1}."
    w = load_file(os.path.join(hf_dir, "model.safetensors"))
    hidden = cfg["hidden_size"]

    # the data: (hidden state after the final norm at p, token p+1) -> the token at p+2, over the model's own answers
    model = Qwen3_5MoeForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    feats_h, feats_x, targets = [], [], []
    with torch.no_grad():
        for ids in prompts:
            seq = model.generate(torch.tensor([ids]), max_new_tokens=n_gen, do_sample=False, use_cache=True,
                                 eos_token_id=None, pad_token_id=0)
            h = model.model(seq).last_hidden_state[0]
            for p in range(len(ids) - 1, seq.shape[1] - 2):
                feats_h.append(h[p])
                feats_x.append(seq[0, p + 1])
                targets.append(seq[0, p + 2])
        H = torch.stack(feats_h)
        X = model.get_input_embeddings()(torch.stack(feats_x))
        Y = torch.stack(targets)
    norm = model.model.norm                                   # (1 + w) RMSNorm, the model's own
    pre = type(norm)(hidden, eps=cfg["rms_norm_eps"])         # zero weights: a plain RMS norm
    g = torch.Generator().manual_seed(99)
    fc = (torch.randn(hidden, 2 * hidden, generator=g) / (2 * hidden) ** 0.5).requires_grad_(True)
    opt = torch.optim.Adam([fc], lr=0.02)
    head = model.lm_head.weight.detach()
    cat = torch.cat([pre(X), pre(H)], dim=-1).detach()
    for _ in range(steps):
        logits = norm(cat @ fc.T) @ head.T
        loss = torch.nn.functional.cross_entropy(logits, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        right = float(((norm(cat @ fc.T) @ head.T).argmax(-1) == Y).float().mean())

    for k in [k for k in w if k.startswith(last)]:
        w["mtp.layers.0." + k[len(last):]] = w[k].clone()
    for k in [k for k in w if k.startswith("mtp.layers.0.") and (k.endswith("o_proj.weight") or k.endswith("down_proj.weight"))]:
        w[k] = torch.zeros_like(w[k])                         # the layer adds nothing to the residual
    w["mtp.fc.weight"] = fc.detach().clone()
    w["mtp.pre_fc_norm_embedding.weight"] = torch.zeros(hidden)
    w["mtp.pre_fc_norm_hidden.weight"] = torch.zeros(hidden)
    w["mtp.norm.weight"] = w["model.norm.weight"].clone()
    save_file({k: v.contiguous() for k, v in w.items()}, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    cfg["mtp_num_hidden_layers"] = 1
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    return right


def reference(hf_dir, out_json):
    import torch
    from transformers.models.qwen3_5_moe import Qwen3_5MoeForCausalLM

    model = Qwen3_5MoeForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    g = torch.Generator().manual_seed(7)
    out = []
    for n in SEQ_LENS:
        ids = torch.randint(3, 300, (n,), generator=g).tolist()
        x = torch.tensor([ids])
        with torch.no_grad():
            top = torch.topk(model(x).logits[0, -1], 8)
            gen = model.generate(x, max_new_tokens=N_GEN, do_sample=False, use_cache=True, eos_token_id=None,
                                 pad_token_id=0)[0, n:].tolist()
            logits = model(torch.tensor([ids + gen])).logits[0]
            margin = min(float(torch.topk(logits[i], 2).values.diff().abs()) for i in range(n - 1, n - 1 + len(gen)))
        out.append({"ids": ids, "gen": gen, "top_ids": top.indices.tolist(),
                    "top_logits": [float(v) for v in top.values], "min_margin": margin})
    json.dump(out, open(out_json, "w"))
    return out



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir")
    ap.add_argument("--llama-cpp", required=True, help="a llama.cpp checkout (only its converter is used)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    ap.add_argument("--mtp", action="store_true", help="also write tiny-mtp-f32.gguf: the model with an MTP block that was fitted to guess (for --spec mtp)")
    a = ap.parse_args()
    hf = os.path.join(a.outdir, "hf")
    vocab = make_model(hf, a.preset)
    print(f"model: vocabulary {vocab}")
    gguf = os.path.join(a.outdir, "tiny-f32.gguf")
    convert(a.llama_cpp, hf, gguf)
    ref = reference(hf, os.path.join(a.outdir, "ref.json"))
    if a.mtp:
        right = add_mtp(hf, os.path.join(a.outdir, "hf-mtp"), [r["ids"] for r in ref])
        convert(a.llama_cpp, os.path.join(a.outdir, "hf-mtp"), os.path.join(a.outdir, "tiny-mtp-f32.gguf"), mtp=True)
        print(f"with an MTP block: {os.path.join(a.outdir, 'tiny-mtp-f32.gguf')} (its guess is right {100 * right:.0f}% of the time on the data it was fitted to)")
    print("reference:", [(len(r["ids"]), round(r["min_margin"], 3)) for r in ref], "(prompt length, smallest argmax margin)")
    print("done:", gguf)


if __name__ == "__main__":
    main()
