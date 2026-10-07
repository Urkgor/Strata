#!/usr/bin/env python3
"""A tiny random-weight Qwen3.5-MoE model, as a GGUF and as PyTorch reference outputs, for testing strata-q35.

Qwen3.6-35B-A3B is `model_type qwen3_5_moe`: Gated DeltaNet layers (3 of 4) and gated attention (1 of 4), a
plain residual, 256 experts with top-8 and a shared expert.  This builds the same architecture at 350 thousand
parameters, so the whole chain can be checked on any PC in seconds:

    transformers (reference)  ->  llama.cpp's convert_hf_to_gguf.py  ->  strata-q35  ->  the same tokens

    python tools/q35_tiny_model.py OUTDIR --llama-cpp /path/to/llama.cpp

needs `torch`, `transformers` (5.x, with qwen3_5_moe), `tokenizers` and `numpy`; llama.cpp's checkout is only used
for its converter (the commit in third_party/ggml/VERSION.txt).  OUTDIR gets `hf/` (the model), `tiny-f32.gguf` and
`ref.json` (three prompts of 5, 40 and 200 token ids with their greedy continuations of 24 tokens and the 8 best
logits after the prompt).  tools/test_q35.py takes OUTDIR.
"""
import argparse
import json
import os
import runpy
import sys

SEQ_LENS = (5, 40, 200)
N_GEN = 24


def make_model(out):
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
    cfg = Qwen3_5MoeTextConfig(
        vocab_size=vocab, hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, num_experts=8, num_experts_per_tok=2,
        moe_intermediate_size=32, shared_expert_intermediate_size=32, max_position_embeddings=512,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        eos_token_id=fast.eos_token_id, pad_token_id=fast.convert_tokens_to_ids("<|endoftext|>"))
    cfg.architectures = ["Qwen3_5MoeForCausalLM"]
    model = Qwen3_5MoeForCausalLM(cfg).eval()
    with torch.no_grad():   # peaked logits: the greedy path must not hang on near-ties
        for name, p in model.named_parameters():
            if p.ndim >= 2:
                p.normal_(0, 0.35)
            elif "norm" in name:
                if p.mean() > 0.5:
                    p.fill_(1.0)
                else:
                    p.normal_(0, 0.1)
    model.save_pretrained(out, safe_serialization=True)
    return vocab


def convert(llama_cpp, hf_dir, gguf):
    """llama.cpp's converter, with the BPE pre-tokenizer named by hand: the tiny tokenizer has no known hash."""
    sys.path.insert(0, llama_cpp)
    sys.path.insert(0, os.path.join(llama_cpp, "gguf-py"))
    import conversion.base as base
    base.TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen35"
    argv = sys.argv
    sys.argv = ["convert_hf_to_gguf.py", hf_dir, "--outfile", gguf, "--outtype", "f32", "--no-mtp"]
    try:
        runpy.run_path(os.path.join(llama_cpp, "convert_hf_to_gguf.py"), run_name="__main__")
    finally:
        sys.argv = argv


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
    a = ap.parse_args()
    hf = os.path.join(a.outdir, "hf")
    vocab = make_model(hf)
    print(f"model: vocabulary {vocab}")
    gguf = os.path.join(a.outdir, "tiny-f32.gguf")
    convert(a.llama_cpp, hf, gguf)
    ref = reference(hf, os.path.join(a.outdir, "ref.json"))
    print("reference:", [(len(r["ids"]), round(r["min_margin"], 3)) for r in ref], "(prompt length, smallest argmax margin)")
    print("done:", gguf)


if __name__ == "__main__":
    main()
