#!/usr/bin/env python3
"""tools/gemma4/tokenizer_parity.py - Gemma4Tokenizer (tools/strata_tokenizer.py) against llama.cpp's tokenizer, id for id.

    tools/gemma4/tokenizer_parity.py --gguf gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --llama-ref ./llama_ref [--fuzz 400]

llama_ref (tools/gemma4/llama_ref.cpp) is built against a llama.cpp checkout and loads the vocabulary only, so the header of the
GGUF is enough (the first 64 MB: `curl -r 0-67108863`, then extend the file to its size, see README.md).  The corpus is fixed text
that hits the rules (newline runs, special tokens, spaces, byte fallback, scripts, emoji) plus seeded random strings; every string is
tokenized with parse_special on and off.  Exit status 1 on the first disagreement count above zero.
"""
import argparse
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import strata_tokenizer as ST  # noqa: E402

CORPUS = [
    "", " ", "a", "Hello, world!", "  leading and trailing  ", "a\n\n\nb", "\n", "\n\n", "\n\n\n\n\n\n\n\n\n", "x\ny\n", "\t\tindented",
    "def f(x):\n    return x * 2  # comment\n", "int main() {\n  printf(\"hi\\n\");\n}\n", "{\"a\": [1, 2, {\"b\": null}], \"c\": \"d\"}",
    "你好，世界。", "こんにちは世界", "مرحبا بالعالم", "Привет, мир!",
    "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "é́ combining", "\x00\x01\x7f control", " nbsp ", "  ", "\r\n\r\n",
    "1234567890", "MixedCASE_and-dashes", "  two  spaces   three", "trailing space ", " ", "▁▁ literal metaspace",
    "<bos>hello", "<|turn>user\nHi<turn|>\n<|turn>model\n", "<|turn>system\nYou are helpful.<turn|>\n", "<|channel>thought\nhmm<channel|>ok",
    "<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:3}<tool_call|>", "<|tool>declaration:f{}<tool|>", "<|think|>", "<pad><unk><mask>",
    "<|tool_response>response:f{value:1}<tool_response|>", "<|image|><|audio|><|video|>", "<unused0>", "<start_of_turn>", "<eos>", "<turn|",
    "<|turn", "<|turn>>", "<<|turn>", "<|turn><|turn>", "a<|turn>b<turn|>c", "<0x41>", "<0xE4><0xBD>",
    "The quick brown fox jumps over the lazy dog. " * 20, "x" * 3000, ("你好 " * 400), "word " * 600 + "\n" + "tail",
]


def fuzz(n, seed):
    rng = random.Random(seed)
    pool = list("abcdefghijklmnopqrstuvwxyzABCXYZ0123456789      \n\n\t.,;:!?'\"()[]{}<>|-_/\\#*+=~`") + \
        ["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|\"|>", "▁", "é", "中", "Ж", "\U0001f600", "‍", "ab", "the ", "ing ", "tion"]
    out = []
    for _ in range(n):
        out.append("".join(rng.choice(pool) for _ in range(rng.randint(1, 60))))
    return out


def run_ref(ref, gguf, texts, special):
    inp = "\n".join(t.encode("utf-8").hex() for t in texts) + "\n"
    cmd = [ref, "tok", gguf] + ([] if special else ["--no-special"])
    out = subprocess.run(cmd, input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
    return [[int(x) for x in line.split(",")] if line else [] for line in out[:len(texts)]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-ref", required=True)
    ap.add_argument("--fuzz", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--files", nargs="*", default=[], help="text files cut into pieces of random length (real prose and code)")
    a = ap.parse_args()
    tk = ST.Tokenizer.from_gguf(a.gguf)
    assert isinstance(tk, ST.Gemma4Tokenizer), type(tk)
    texts = CORPUS + fuzz(a.fuzz, a.seed)
    rng = random.Random(a.seed)
    for path in a.files:
        data = open(path, encoding="utf-8", errors="replace").read()
        pos = 0
        while pos < len(data):
            n = rng.choice([40, 200, 1000, 4000])
            texts.append(data[pos:pos + n])
            pos += n
    bad = 0
    for special in (True, False):
        want = run_ref(a.llama_ref, a.gguf, texts, special)
        for t, w in zip(texts, want):
            got = tk.encode(t, parse_special=special)
            if got != w:
                bad += 1
                if bad <= 8:
                    print("MISMATCH special=%s %r\n   mine %s\n   ref  %s" % (special, t[:60], got[:24], w[:24]))
    # decoding: ids of ordinary text come back as the text
    rt = 0
    for t in CORPUS:
        if "<" in t or "\r" in t or "\u2581" in t:
            continue
        if tk.decode(tk.encode(t)) != t:
            rt += 1
            print("ROUND TRIP FAILED", repr(t[:50]))
    print("%d strings x 2 modes: %d disagreements with llama.cpp; round trip failures %d" % (len(texts), bad, rt))
    return 1 if bad or rt else 0


if __name__ == "__main__":
    sys.exit(main())
