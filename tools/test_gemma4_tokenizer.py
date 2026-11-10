"""Gemma4Tokenizer on a small synthetic vocabulary (no GGUF): merges by rank, newline runs, special tokens, byte fallback,
decoding, and PromptEncoder's resume points.  The comparison with llama.cpp on the real 262,144-token vocabulary is
tools/gemma4/tokenizer_parity.py (it needs the GGUF header and a llama.cpp build)."""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as ST  # noqa: E402

S = "▁"


def make():
    toks = ["<pad>", "<eos>", "<bos>", "<|turn>", "<turn|>", "<|tool_response>", "<|channel>", "\n", "\n\n", S, "a", "b", "c", "d", "e",
            S + "a", "ab", "abc", S + "ab", "bc", "é", "<0xC3>", "<0xA9>", "<0x41>", "x"]
    types = [3, 1, 3, 3, 3, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 1]
    # the rank is the position: the metaspace joins `a` first, then `a b`, `ab c`, and `b c` is the last resort
    merges = [S + " a", S + "a b", "a b", "ab c", "b c"]
    return ST.Gemma4Tokenizer(toks, merges, types, {"tokenizer.ggml.bos_token_id": 2, "tokenizer.ggml.eos_token_id": 4}, add_bos=True), toks


class Gemma4TokenizerTest(unittest.TestCase):
    def setUp(self):
        self.tk, self.toks = make()
        self.id = {t: i for i, t in enumerate(self.toks)}

    def test_merge_order_is_by_rank(self):
        # "abc": a b merges first (rank 2), then ab c (rank 3); "b c" (rank 4) never gets its turn
        self.assertEqual(self.tk.encode("abc"), [self.id["abc"]])
        self.assertEqual(self.tk.encode("bc"), [self.id["bc"]])
        self.assertEqual(self.tk.encode("ab"), [self.id["ab"]])

    def test_spaces_are_metaspace(self):
        self.assertEqual(self.tk.encode(" ab"), [self.id[S + "ab"]])
        self.assertEqual(self.tk.encode("a b"), [self.id["a"], self.id[S], self.id["b"]])
        self.assertEqual(self.tk.decode(self.tk.encode("a b")), "a b")
        self.assertEqual(self.tk.decode(self.tk.encode(" ab")), " ab")

    def test_newlines_split_words_and_a_run_that_is_a_token_is_one(self):
        self.assertEqual(self.tk.encode("a\nb"), [self.id["a"], self.id["\n"], self.id["b"]])
        self.assertEqual(self.tk.encode("\n\n"), [self.id["\n\n"]])
        self.assertEqual(self.tk.encode("a\n\n\nb"), [self.id["a"], self.id["\n"], self.id["\n"], self.id["\n"], self.id["b"]])
        # three newlines: not a token; BPE over the characters, no merge: three one-newline tokens
        self.assertEqual(self.tk.encode("\n\n\n"), [self.id["\n"]] * 3)

    def test_special_tokens_and_parse_special(self):
        t = "<|turn>a<turn|>"
        self.assertEqual(self.tk.encode(t, parse_special=True), [self.id["<|turn>"], self.id["a"], self.id["<turn|>"]])
        plain = self.tk.encode(t, parse_special=False)          # control tokens are text then
        self.assertNotIn(self.id["<|turn>"], plain)
        self.assertNotIn(self.id["<turn|>"], plain)
        # a user-defined token is matched either way ... except the end-of-generation ones, which are control
        self.assertEqual(self.tk.encode("<|channel>", parse_special=False), [self.id["<|channel>"]])
        self.assertNotIn(self.id["<|tool_response>"], self.tk.encode("<|tool_response>", parse_special=False))
        self.assertEqual(self.tk.encode("<|tool_response>", parse_special=True), [self.id["<|tool_response>"]])
        self.assertEqual(self.tk.encode("<eos>", parse_special=True), [self.id["<eos>"]])

    def test_byte_fallback(self):
        # "é" is a token; "A" has no token of its own here: its byte token <0x41> stands for it
        self.assertEqual(self.tk.encode("é"), [self.id["é"]])
        self.assertEqual(self.tk.encode("A"), [self.id["<0x41>"]])
        self.assertEqual(self.tk.decode([self.id["<0xC3>"], self.id["<0xA9>"]]), "é")     # two bytes make one character

    def test_add_special_puts_bos_first(self):
        self.assertEqual(self.tk.encode("a", add_special=True)[0], self.id["<bos>"])
        self.assertEqual(self.tk.encode("a")[0], self.id["a"])

    def test_plain_spans_keep_a_literal_as_text(self):
        t = "<|turn>x"
        self.assertEqual(self.tk.encode(t, parse_special=True, plain=[(0, 8)])[0] != self.id["<|turn>"], True)

    def test_prompt_encoder_resumes_exactly(self):
        pe = ST.PromptEncoder(self.tk)
        rng = random.Random(3)
        pool = ["a", "b", "c", " ", "\n", "<|turn>", "<turn|>", "<|channel>", "ab", "x", "é"]
        base = "".join(rng.choice(pool) for _ in range(300))
        first = pe.encode(base)
        self.assertEqual(first, self.tk.encode(base, parse_special=True))
        longer = base + "".join(rng.choice(pool) for _ in range(120))
        self.assertEqual(pe.encode(longer), self.tk.encode(longer, parse_special=True))
        self.assertGreater(pe.last_reused, 0)

    def test_from_pack_picks_the_class(self):
        tk = ST.from_pack(self.toks, ["a b"], [1] * len(self.toks), {"model": "gemma4"})
        self.assertIsInstance(tk, ST.Gemma4Tokenizer)


if __name__ == "__main__":
    unittest.main()
