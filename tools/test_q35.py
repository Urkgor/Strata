"""strata-q35 against a reference implementation: the protocol and the conversation cache, on a tiny model.

The model is the 350-thousand-parameter Qwen3.5-MoE that tools/q35_tiny_model.py makes (random weights, the real
architecture: Gated DeltaNet + gated attention + routed and shared experts).  Its greedy tokens and logits come from
PyTorch (transformers' own qwen3_5_moe), so every check here is "the same tokens as the reference", not "some tokens".
No GPU, no downloads; about a minute on a laptop.

    python tools/q35_tiny_model.py /tmp/q35-tiny --llama-cpp /path/to/llama.cpp     # once
    STRATA_Q35_ENGINE=build-q35/strata-q35 STRATA_Q35_MODEL_DIR=/tmp/q35-tiny python -m unittest tools.test_q35

Without those two variables every test is skipped.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

ENGINE = os.environ.get("STRATA_Q35_ENGINE", "")
MODEL_DIR = os.environ.get("STRATA_Q35_MODEL_DIR", "")
READY = bool(ENGINE) and Path(ENGINE).exists() and bool(MODEL_DIR) and (Path(MODEL_DIR) / "ref.json").exists()
GGUF = str(Path(MODEL_DIR) / "tiny-f32.gguf") if MODEL_DIR else ""
REF = json.loads((Path(MODEL_DIR) / "ref.json").read_text()) if READY else []
N_GEN = 24


class Engine:
    """One `strata-q35 --serve` process, spoken to the way serve/server.py speaks to `strata --serve`."""

    def __init__(self, *args: str, context: int = 4096):
        self.proc = subprocess.Popen([ENGINE, "--serve", "--native", GGUF, "--max-context", str(context), "--threads", "2",
                                      *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self.info: dict = {}
        while True:
            line = self.read(60)
            if line.startswith("INFO "):
                self.info = dict(kv.partition("=")[::2] for kv in line.split()[1:])
            if line.startswith("READY"):
                self.ready = line.split()
                break

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def read(self, timeout: float = 120) -> str:
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise EOFError("the engine ended")
        return line

    def send(self, text: str):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def gen(self, ids, max_new=N_GEN, keys="", stop_after=None):
        """-> dict(tokens, resume, pp, done: DONE fields, err)."""
        self.send(f"GEN {max_new}{(' ' + keys) if keys else ''} {','.join(map(str, ids))}")
        out = {"tokens": [], "resume": None, "pp": [], "done": None, "err": None}
        stopped = False
        while True:
            line = self.read()
            if line.startswith("T "):
                out["tokens"].append(int(line[2:]))
                if stop_after and len(out["tokens"]) >= stop_after and not stopped:
                    self.send("STOP")
                    stopped = True
            elif line.startswith("PP "):
                out["pp"].append(tuple(float(x) for x in line.split()[1:]))
            elif line.startswith("RESUME "):
                out["resume"] = int(line.split()[1])
            elif line.startswith("DONE"):
                out["done"] = line.split()
                return out
            elif line.startswith("ERR"):
                out["err"] = line[4:]
                return out

    def ask(self, text: str, prefix: str) -> str:
        self.send(text)
        while True:
            line = self.read()
            if line.startswith(prefix) or line.startswith("ERR") or line.startswith("SERR"):
                return line

    def close(self):
        try:
            self.send("QUIT")
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
            self.proc.wait()
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except OSError:
                pass


def cold(ids, max_new=N_GEN, *args, keys=""):
    """What a fresh engine answers: the standard every cached path must reproduce."""
    e = Engine(*args)
    try:
        return e.gen(ids, max_new, keys)
    finally:
        e.close()


@unittest.skipUnless(READY, "set STRATA_Q35_ENGINE and STRATA_Q35_MODEL_DIR (see the module docstring)")
class Q35Engine(unittest.TestCase):
    def test_ready_and_info(self):
        e = Engine()
        try:
            self.assertEqual(e.ready[:2], ["READY", "4096"])
            self.assertIn("stop", e.ready)            # the server only sends STOP to an engine that says so
            self.assertEqual(e.info["arch"], "qwen35moe")
            self.assertEqual(e.info["context"], "4096")
            self.assertTrue(e.info["engine"].startswith("q35-"))
        finally:
            e.close()

    def test_quit_ends_the_process(self):
        e = Engine()
        e.gen([5, 6, 7, 8], 4)
        e.send("QUIT")
        self.assertEqual(e.proc.wait(timeout=20), 0)   # a reader thread blocked in std::cin used to hold exit() for ever
        e.close()

    def test_greedy_tokens_are_pytorchs(self):
        e = Engine()
        try:
            for r in REF:
                got = e.gen(r["ids"], len(r["gen"]))
                self.assertEqual(got["tokens"], r["gen"], f"prompt of {len(r['ids'])} tokens")
                f = got["done"]
                self.assertEqual((f[1], f[2], f[5]), (str(len(r["gen"])), str(len(r["ids"])), "length"))
                self.assertGreater(len(got["pp"]), 0)
        finally:
            e.close()

    def test_logits_match_pytorch(self):
        e = Engine()
        try:
            r = REF[1]
            line = e.ask("LOGITS " + ",".join(map(str, r["ids"])), "LOGITS")
            f = line.split()
            self.assertEqual(f[1], "8")
            ids = [int(x.split(":")[0]) for x in f[2:]]
            vals = [float(x.split(":")[1]) for x in f[2:]]
            self.assertEqual(ids, r["top_ids"])
            for a, b in zip(vals, r["top_logits"]):
                self.assertAlmostEqual(a, b, delta=5e-3)   # F32 everywhere: ggml's and PyTorch's sums differ by rounding
        finally:
            e.close()

    def test_a_conversation_continues_from_memory(self):
        p, extra = REF[1]["ids"], [11, 12, 13, 14, 15]
        e = Engine()
        try:
            first = e.gen(p)
            self.assertEqual(first["resume"], 0)
            turn2 = p + first["tokens"] + extra
            got = e.gen(turn2)
            # memory holds the prompt and the 23 tokens it was fed (the 24th was sampled, never fed)
            self.assertEqual(got["resume"], len(p) + N_GEN - 1)
            self.assertEqual(got["done"][8], str(len(p) + N_GEN - 1))      # DONE's <reused>
            self.assertEqual(got["tokens"], cold(turn2)["tokens"])
        finally:
            e.close()

    def test_the_same_prompt_again_resumes_one_token_before_its_end(self):
        p = REF[2]["ids"]
        e = Engine()
        try:
            first = e.gen(p)
            again = e.gen(p)                    # a recurrent state cannot step back: it comes from the checkpoint
            self.assertEqual(again["resume"], len(p) - 1)
            self.assertEqual(again["tokens"], first["tokens"])
            self.assertEqual(first["tokens"], REF[2]["gen"])
        finally:
            e.close()

    def test_an_edit_in_the_middle_restores_the_nearest_checkpoint(self):
        p = REF[2]["ids"]                      # 200 tokens, read in chunks of 16 with a checkpoint at every chunk
        args = ("--batch", "16", "--ubatch", "16", "--ckpt-every", "16", "--ckpt-slots", "4")
        edited = p[:180] + [20 + i for i in range(30)]
        e = Engine(*args)
        try:
            e.gen(p)
            got = e.gen(edited)
            self.assertEqual(got["resume"], 176)    # the newest of the last four checkpoints at or before token 180
            self.assertEqual(got["tokens"], cold(edited, N_GEN, *args)["tokens"])
            early = p[:100] + [30 + i for i in range(20)]
            e.gen(p)                                 # checkpoints are again 160, 176, 192 and 199
            got = e.gen(early)                       # none at or before 100: everything is read again
            self.assertEqual(got["resume"], 0)
            self.assertEqual(got["tokens"], cold(early, N_GEN, *args)["tokens"])
        finally:
            e.close()

    def test_stop_ends_the_request_and_the_engine_goes_on(self):
        p = REF[0]["ids"]
        e = Engine()
        try:
            got = e.gen(p, 3000, stop_after=3)
            self.assertEqual(got["done"][5], "cancel")
            self.assertGreaterEqual(len(got["tokens"]), 3)
            self.assertLess(len(got["tokens"]), 3000)
            again = e.gen(p)                         # still in step, and still right
            self.assertEqual(again["tokens"], REF[0]["gen"])
        finally:
            e.close()

    def test_save_and_restore_a_conversation(self):
        p, extra = REF[1]["ids"], [41, 42, 43]
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "chat.bin")
            e = Engine()
            try:
                first = e.gen(p)
                saved = e.ask(f"SAVE {path}", "SAVED").split()
                self.assertEqual(saved[0], "SAVED")
                self.assertEqual(int(saved[1]), len(p) + N_GEN - 1)
                self.assertGreater(int(saved[2]), 0)
                e.gen(REF[0]["ids"], 4)             # something else in memory
                back = e.ask(f"RESTORE {path}", "RESTORED").split()
                self.assertEqual(back[0], "RESTORED")
                self.assertEqual(int(back[1]), len(p) + N_GEN - 1)
                turn2 = p + first["tokens"] + extra
                got = e.gen(turn2)
                self.assertEqual(got["resume"], len(p) + N_GEN - 1)
                self.assertEqual(got["tokens"], cold(turn2)["tokens"])
                missing = e.ask(f"RESTORE {path}.nope", "RESTORED")
                self.assertTrue(missing.startswith("SERR invalid 0 "), missing)
            finally:
                e.close()

    def test_sampling_keys(self):
        p = REF[0]["ids"]
        e = Engine()
        try:
            a = e.gen(p, 16, "temperature=0.9 top_k=20 top_p=0.95 min_p=0.02 seed=11")["tokens"]
            b = e.gen(p, 16, "temperature=0.9 top_k=20 top_p=0.95 min_p=0.02 seed=11")["tokens"]
            c = e.gen(p, 16, "temperature=0.9 top_k=20 top_p=0.95 min_p=0.02 seed=12")["tokens"]
            self.assertEqual(len(a), 16)
            self.assertEqual(a, b)                   # the same seed, the same answer
            self.assertNotEqual(a, c)
            greedy = e.gen(p, 16)["tokens"]
            self.assertEqual(greedy, REF[0]["gen"][:16])
            penal = e.gen(p, 16, "penalty_repeat=1.5 penalty_last_n=32")["tokens"]
            self.assertEqual(len(penal), 16)
            self.assertNotEqual(penal, greedy)       # the repeat penalty changes a path that repeats
        finally:
            e.close()

    def test_unusable_requests_are_refused_and_the_engine_stays_in_step(self):
        e = Engine(context=256)
        try:
            self.assertIn("exceeds the context", e.gen([5] * 250, 20)["err"])
            self.assertIn("outside the vocabulary", e.gen([5, 99999], 4)["err"])
            self.assertIn("bad request", e.gen([5, 6], 0)["err"])
            self.assertTrue(e.ask("HELLO", "ERR").startswith("ERR expected"))
            self.assertIn("vision", e.ask("GENI 4 /nonexistent 1,2", "ERR"))
            self.assertTrue(e.ask("VRAM 512", "ERR").startswith("ERR VRAM"))
            self.assertEqual(e.gen(REF[0]["ids"])["tokens"], REF[0]["gen"])
        finally:
            e.close()

    def test_a_prompt_is_read_in_reported_chunks(self):
        p = REF[2]["ids"]
        e = Engine("--batch", "64", "--ubatch", "64")
        try:
            got = e.gen(p, 4)
            reads = [int(x[0]) for x in got["pp"]]
            self.assertEqual(reads, sorted(reads))
            self.assertEqual(reads[-1], len(p))
            self.assertTrue(all(int(x[1]) == len(p) for x in got["pp"]))
            self.assertGreaterEqual(len(reads), 3)   # 199 tokens in chunks of 64, then the last one alone
        finally:
            e.close()

    def test_generating_text_without_serve(self):
        r = subprocess.run([ENGINE, "--native", GGUF, "-p", "hello world", "-n", "8", "--threads", "2"],
                           capture_output=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr.decode(errors="replace")[-400:])
        self.assertIn(b"answer 8 tokens", r.stderr)
        r = subprocess.run([ENGINE, "--native", GGUF, "--info"], capture_output=True, timeout=120)
        self.assertEqual(r.returncode, 0)
        self.assertIn(b"qwen35moe", r.stdout)
        r = subprocess.run([ENGINE, "--native", GGUF], capture_output=True, timeout=120)
        self.assertNotEqual(r.returncode, 0)         # no prompt, no --serve: says so

    def test_a_missing_model_is_an_error(self):
        r = subprocess.run([ENGINE, "--native", str(Path(MODEL_DIR) / "no-such.gguf")], capture_output=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)


CACHE = ["--cpu-moe", "--no-repack", "--expert-cache", "auto", "--cache-in-ram", "--cache-check", "off"]


@unittest.skipUnless(READY, "set STRATA_Q35_ENGINE and STRATA_Q35_MODEL_DIR (see the module docstring)")
class Q35ExpertCache(unittest.TestCase):
    """The hybrid expert cache (docs/Q35.md), with the cache in RAM: the graph, the tables, the copies from the GGUF file
    and the policy all run as they do with a card, so what is checked here is that the cache never changes an answer."""

    def test_the_tokens_do_not_depend_on_what_is_cached(self):
        for slots in (1, 3, 5, 8):
            for tokens in (1, 64):
                e = Engine(*CACHE, "--cache-slots", str(slots), "--cache-tokens", str(tokens))
                try:
                    for r in REF:
                        got = e.gen(r["ids"], len(r["gen"]))
                        self.assertEqual(got["tokens"], r["gen"], f"{slots} slots, batches of up to {tokens}")
                        hits, lookups = int(got["done"][9]), int(got["done"][10])
                        self.assertLessEqual(hits, lookups)
                finally:
                    e.close()

    def test_logits_are_the_same_to_the_last_digit(self):
        ids = ",".join(map(str, REF[1]["ids"]))
        plain = Engine("--cpu-moe", "--no-repack")
        cached = Engine(*CACHE, "--cache-slots", "3", "--cache-tokens", "64")
        try:
            want = plain.ask("LOGITS " + ids, "LOGITS")
            for _ in range(3):                                   # the first call fills the cache, the others use it
                self.assertEqual(cached.ask("LOGITS " + ids, "LOGITS"), want)
        finally:
            plain.close()
            cached.close()

    def test_the_cache_learns_what_the_model_asks_for(self):
        e = Engine(*CACHE, "--cache-slots", "4", "--cache-tokens", "64")
        try:
            rates = []
            for _ in range(4):
                done = e.gen(REF[0]["ids"], N_GEN)["done"]
                rates.append(int(done[9]) / max(1, int(done[10])))
            self.assertGreater(rates[-1], rates[0])
            self.assertGreater(rates[-1], 0.5)                   # 4 of the 8 experts of a layer, the 4 it uses
            self.assertEqual(e.info["expert_slots"], "16")       # 4 layers x 4 slots
        finally:
            e.close()

    def test_a_profile_warms_the_next_start(self):
        with tempfile.TemporaryDirectory() as d:
            profile = str(Path(d) / "hot.bin")
            args = (*CACHE, "--cache-slots", "3", "--cache-tokens", "64", "--cache-profile", profile)
            e = Engine(*args)
            try:
                first = e.gen(REF[0]["ids"], N_GEN)["done"]
                for _ in range(3):
                    e.gen(REF[0]["ids"], N_GEN)
            finally:
                e.close()
            self.assertTrue(Path(profile).exists())
            e = Engine(*args)
            try:
                warm = e.gen(REF[0]["ids"], N_GEN)["done"]
            finally:
                e.close()
            self.assertGreater(int(warm[9]) / max(1, int(warm[10])), int(first[9]) / max(1, int(first[10])))

    def test_a_profile_of_another_model_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            profile = Path(d) / "hot.bin"
            profile.write_bytes(b"not a profile at all")
            e = Engine(*CACHE, "--cache-slots", "3", "--cache-profile", str(profile))
            try:
                self.assertEqual(e.gen(REF[0]["ids"], N_GEN)["tokens"], REF[0]["gen"])
            finally:
                e.close()

    def test_sim_measures_without_a_cache(self):
        r = subprocess.run([ENGINE, "--native", GGUF, "-p", "the quick brown fox jumps over the lazy dog", "-n", "12", "--threads", "2",
                            "--expert-cache", "sim", "--sim-cache", "25,50,100"], capture_output=True, timeout=120)
        err = r.stderr.decode(errors="replace")
        self.assertEqual(r.returncode, 0, err[-400:])
        self.assertIn("a cache of  25%", err)
        self.assertIn("would have served", err)
        rates = [float(x.rstrip("%")) for x in err.split("would have served")[1].split("of the expert uses")[0].split()]
        self.assertEqual(len(rates), 3)
        self.assertEqual(rates[2], 100.0)                        # every expert cached: every use is a hit (after the first fills)
        self.assertLessEqual(rates[0], rates[1])

    def test_bench_runs_with_and_without_the_cache(self):
        r = subprocess.run([ENGINE, "--native", GGUF, "--bench", "--bench-pp", "24", "--bench-tg", "6", "--bench-reps", "2",
                            "--bench-compare", "--threads", "2", *CACHE, "--cache-slots", "3"], capture_output=True, timeout=240)
        out = r.stdout.decode(errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr.decode(errors="replace")[-400:])
        self.assertIn("bench[cache] 2/2", out)
        self.assertIn("bench[usual] 2/2", out)
        self.assertIn("% hits", out)
        self.assertIn("the cache makes generation", out)

    def test_bad_cache_options_are_refused(self):
        for args in (["--expert-cache", "lots"], ["--expert-cache", "off", "--cache-slots", "2"], ["--sim-cache", "0"]):
            r = subprocess.run([ENGINE, "--native", GGUF, "-p", "hi", *args], capture_output=True, timeout=60)
            self.assertEqual(r.returncode, 2, args)

    def test_a_check_that_passes_leaves_the_cache_on(self):
        r = subprocess.run([ENGINE, "--native", GGUF, "-p", "hello world", "-n", "4", "--threads", "2", "--cpu-moe", "--no-repack",
                            "--expert-cache", "auto", "--cache-in-ram", "--cache-slots", "3"], capture_output=True, timeout=120)
        err = r.stderr.decode(errors="replace")
        self.assertEqual(r.returncode, 0, err[-400:])
        self.assertIn("expert cache check passed", err)
        self.assertIn("expert cache: 12 slots in 4 layers", err)


if __name__ == "__main__":
    unittest.main()
