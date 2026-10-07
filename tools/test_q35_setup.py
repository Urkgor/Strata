"""Tests for tools/q35_setup.py: what it reads from a GGUF, what it writes for the server, what it refuses.  The GGUFs
here are metadata-only files written with tools/gguf_writer.py (a small vocabulary, no weights); no engine, no GPU,
no downloads.

    python -m unittest tools.test_q35_setup
"""
from __future__ import annotations

import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import numpy as np  # noqa: E402
import q35_setup  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402


def byte_vocab() -> list[str]:
    """The 256 byte-level characters of GPT-2's alphabet, then three specials: a real, tiny BPE vocabulary."""
    import strata_tokenizer
    chars = [strata_tokenizer.BYTE_TO_UNICODE[b] for b in range(256)]
    return chars + ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def write_gguf(path: Path, arch: str = "qwen35moe", sampling: bool = True, template: str = "{{ messages }}") -> Path:
    w = GGUFWriter()
    w.add("general.architecture", arch)
    w.add("general.name", "Qwen3.6-35B-A3B")
    for key, v in (("block_count", 40), ("embedding_length", 2048), ("expert_count", 256), ("expert_used_count", 8),
                   ("context_length", 262144)):
        w.add(f"{arch}.{key}", v, "u32")
    tokens = byte_vocab()
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "qwen35")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.merges", [], "array:string")
    w.add("tokenizer.ggml.token_type", [1] * 256 + [3] * 3, "array:i32")
    w.add("tokenizer.ggml.eos_token_id", 258, "u32")
    w.add("tokenizer.chat_template", template)
    if sampling:
        w.add("general.sampling.temp", 0.7, "f32")
        w.add("general.sampling.top_p", 0.8, "f32")
        w.add("general.sampling.top_k", 20, "u32")
        w.add("general.sampling.min_p", 0.0, "f32")
    w.add_f32("token_embd.weight", np.zeros((4, 8), dtype=np.float32))
    return w.write(path)


class Q35Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.gguf = write_gguf(self.dir / "model.gguf")

    def tearDown(self):
        self.tmp.cleanup()

    def run_setup(self, *extra, gguf=None):
        out = self.dir / "pack"
        argv = ["--gguf", str(gguf or self.gguf), "--engine", str(self.dir / "strata-q35"), "--out", str(out), *extra]
        return q35_setup.main(argv), out

    def test_the_files_of_a_split_model_are_found(self):
        first = self.dir / "Qwen3.6-35B-A3B-Q4_K_M-00001-of-00003.gguf"
        names = [p.name for p in q35_setup.shards_of(first)]
        self.assertEqual(names, [f"Qwen3.6-35B-A3B-Q4_K_M-0000{i}-of-00003.gguf" for i in (1, 2, 3)])
        self.assertEqual(q35_setup.shards_of(self.gguf), [self.gguf])

    def test_the_facts_come_from_the_file(self):
        f = q35_setup.read_facts(self.gguf)
        self.assertEqual((f["arch"], f["layers"], f["experts"], f["experts_used"], f["embd"], f["context_train"]),
                         ("qwen35moe", 40, 256, 8, 2048, 262144))
        self.assertEqual(f["bytes"], self.gguf.stat().st_size)
        self.assertEqual(f["missing_shards"], [])
        self.assertEqual(f["quantization"], {"F32": 1})

    def test_the_models_own_sampling_goes_into_the_server_config(self):
        code, out = self.run_setup()
        self.assertEqual(code, 0)
        cfg = json.loads((out / "config.json").read_text())
        self.assertEqual(set(cfg["sampling"]), {"temperature", "top_p", "top_k", "min_p"})
        self.assertAlmostEqual(cfg["sampling"]["temperature"], 0.7, places=6)   # the GGUF stores f32
        self.assertAlmostEqual(cfg["sampling"]["top_p"], 0.8, places=6)
        self.assertEqual((cfg["sampling"]["top_k"], cfg["sampling"]["min_p"]), (20, 0.0))
        # the server refuses a sampling block it cannot use: every key and range must be one it knows
        sys.path.insert(0, str(ROOT / "serve"))
        import server
        self.assertEqual(set(server.sampling_defaults_from_config(cfg)), {"temperature", "top_p", "top_k", "min_p"})

    def test_a_top_k_the_server_cannot_take_is_clamped_or_left_out(self):
        self.assertEqual(q35_setup.sampling_block({"sampling": {"top_k": 100}}), {"top_k": 64})
        self.assertEqual(q35_setup.sampling_block({"sampling": {"top_k": 0, "temperature": 1.0}}), {"temperature": 1.0})
        self.assertEqual(q35_setup.sampling_block({}), {})
        no = write_gguf(self.dir / "plain.gguf", sampling=False)
        self.assertEqual(q35_setup.read_facts(no)["sampling"], {})

    def test_what_is_written(self):
        code, out = self.run_setup("--max-context", "65536", "--n-cpu-moe", "12", "--kv", "q8_0", "--threads", "16",
                                   "--numa", "distribute", "--model-name", "q36")
        self.assertEqual(code, 0)
        cfg = json.loads((out / "config.json").read_text())
        self.assertEqual(cfg["exe"], str(self.dir / "strata-q35"))
        self.assertEqual(cfg["args"], ["--native", str(self.gguf), "--max-context", "65536", "--n-cpu-moe", "12",
                                       "--kv", "q8_0", "--threads", "16", "--numa", "distribute",
                                       "--expert-cache", "auto", "--cache-profile", str(out / "hot-experts.bin")])
        self.assertEqual((cfg["model_name"], cfg["tokenizer"]), ("q36", str(out / "tokenizer")))
        tok = out / "tokenizer"
        for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"):
            self.assertTrue((tok / name).exists(), name)
        self.assertEqual(len(json.loads((tok / "vocab.json").read_text())), 259)
        self.assertEqual((tok / "chat_template.jinja").read_text(), "{{ messages }}")
        script = out / "run-q35.sh"
        self.assertTrue(script.stat().st_mode & stat.S_IXUSR)
        text = script.read_text()
        self.assertIn("serve/server.py --engine strata --config", text)
        self.assertIn("--port 8095", text)
        self.assertTrue(text.startswith("#!/bin/sh\n"))

    def test_cpu_moe_and_n_cpu_moe_are_alternatives(self):
        with self.assertRaises(SystemExit):
            q35_setup.parse(["--gguf", "x", "--cpu-moe", "--n-cpu-moe", "3"])
        a = q35_setup.parse(["--gguf", "x", "--cpu-moe", "--gpu-layers", "all", "--fit", "off", "--no-expert-cache",
                             "--engine-arg=--verbose"])
        self.assertEqual(q35_setup.engine_args(a, Path("m.gguf")),
                         ["--native", "m.gguf", "--max-context", "32768", "--cpu-moe", "--gpu-layers", "all", "--fit",
                          "off", "--verbose"])

    def test_speed_options_are_passed_on(self):
        a = q35_setup.parse(["--gguf", "x", "--ubatch", "2048", "--batch", "4096", "--no-repack", "--pin-experts", "--spec", "lookup",
                             "--main-gpu", "1", "--split-mode", "none", "--no-expert-cache"])
        self.assertEqual(q35_setup.engine_args(a, Path("m.gguf")),
                         ["--native", "m.gguf", "--max-context", "32768", "--ubatch", "2048", "--batch", "4096", "--no-repack",
                          "--pin-experts", "--spec", "lookup", "--main-gpu", "1", "--split-mode", "none"])

    def test_the_expert_cache_is_on_by_default_and_takes_a_size(self):
        a = q35_setup.parse(["--gguf", "x"])
        self.assertEqual(q35_setup.engine_args(a, Path("m.gguf")),
                         ["--native", "m.gguf", "--max-context", "32768", "--expert-cache", "auto"])
        a = q35_setup.parse(["--gguf", "x", "--expert-cache", "6000"])
        self.assertEqual(q35_setup.engine_args(a, Path("m.gguf"), Path("/o"))[-4:],
                         ["--expert-cache", "6000", "--cache-profile", "/o/hot-experts.bin"])
        for bad in ("lots", "0", "-5"):
            with self.assertRaises(SystemExit):
                q35_setup.parse(["--gguf", "x", "--expert-cache", bad])

    def test_another_architecture_is_refused_with_the_right_pointer(self):
        flash = write_gguf(self.dir / "flash.gguf", arch="qwen4exp")
        code, out = self.run_setup(gguf=flash)
        self.assertEqual(code, 1)
        self.assertFalse((out / "config.json").exists())
        llama = write_gguf(self.dir / "llama.gguf", arch="llama")
        self.assertEqual(self.run_setup(gguf=llama)[0], 1)
        self.assertEqual(self.run_setup("--force", gguf=llama)[0], 0)      # on request

    def test_a_split_model_with_a_missing_shard_is_refused(self):
        first = write_gguf(self.dir / "m-00001-of-00002.gguf")
        code, out = self.run_setup(gguf=first)
        self.assertEqual(code, 1)

    def test_a_missing_file_is_an_error(self):
        self.assertEqual(self.run_setup(gguf=self.dir / "nope.gguf")[0], 1)


class Q35Pin(unittest.TestCase):
    """One llama.cpp commit for the whole repository: the engine's API calls were written against it."""

    def test_the_commit_is_the_same_everywhere(self):
        import re
        sha = (ROOT / "third_party/ggml/VERSION.txt").read_text().split()[0]
        cmake = (ROOT / "q35/CMakeLists.txt").read_text()
        self.assertEqual(re.search(r'STRATA_Q35_LLAMACPP_TAG "([0-9a-f]{40})"', cmake).group(1), sha)
        top = (ROOT / "CMakeLists.txt").read_bytes().decode("utf-8-sig")
        self.assertEqual(re.search(r"GIT_TAG ([0-9a-f]{40})", top).group(1), sha)

    def test_the_build_script_reads_the_pin_from_the_same_file(self):
        script = (ROOT / "tools/build_q35.sh").read_text()
        self.assertIn("third_party/ggml/VERSION.txt", script)
        self.assertIn("q35/", script)


if __name__ == "__main__":
    unittest.main()
