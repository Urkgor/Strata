"""serve/calibrate.py without a GPU: the option handling, the search (a fake engine whose speed depends on its options) and the
quality guard (an option that changes the greedy text is refused), and the config written back."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from serve import calibrate as cal  # noqa: E402


class Tok:
    def encode(self, s, parse_special=False):
        return [ord(c) % 1000 for c in s]


class FakeEngine:
    """Prompt rate and decode rate from the options; --small-scores slows the prompt; --pcie-share writes other text."""
    def __init__(self, args):
        self.args = args
        if "--mtp-resident" in args:
            raise RuntimeError("out of memory on the card")
        self.last = {}

    def generate(self, ids, max_new, sampling, cancel):
        a = self.args
        pp = 500.0 * (0.6 if "--small-scores" in a else 1.0)
        tg = 50.0 * (1.2 if "--seg-graphs" in a else 1.0)
        mtp = a[a.index("--mtp") + 1] if "--mtp" in a else "0"
        if mtp != "0" and (not sampling or "--mtp-sampled" in a):
            tg *= {"1": 1.3, "2": 1.5, "3": 1.4}[mtp]
        if "--cpu-threads" in a:
            tg *= 1.0 + 0.1 * int(a[a.index("--cpu-threads") + 1])
        out = [7, 8, 9] * (max_new // 3)
        if not sampling and "--pcie-share" in a:
            out = [7, 8, 1] * (max_new // 3)       # a different greedy text
        self.last = {"prompt_ms": 1000.0 * len(ids) / pp, "decode_ms": 1000.0 * len(out) / tg, "generated": len(out)}
        yield from out

    def close(self):
        pass


class CalibrateTest(unittest.TestCase):
    def test_option_helpers(self):
        args = ["--native", "m.gguf", "--mtp", "2", "--seg-graphs", "--max-context", "4096", "--cpu-threads", "8"]
        self.assertEqual(cal.strip_managed(args), ["--native", "m.gguf", "--max-context", "4096"])
        self.assertEqual(cal.current_managed(args), ["--mtp", "2", "--seg-graphs", "--cpu-threads", "8"])
        o = cal.set_option(["--mtp", "2"], "--mtp", "3")
        self.assertEqual(o, ["--mtp", "3"])
        self.assertEqual(cal.set_option(o, "--seg-graphs", True), ["--mtp", "3", "--seg-graphs"])
        self.assertEqual(cal.set_option(["--seg-graphs"], "--seg-graphs", None), [])
        self.assertEqual(cal.get_option(["--mtp", "2"], "--mtp"), "2")
        self.assertIsNone(cal.get_option([], "--mtp"))

    def run_cal(self, args, profile):
        cfg = {"args": args}
        logs = []
        c = cal.Calibrator(cfg, Tok(), FakeEngine, profile, log=logs.append, cores=6)
        return c, c.run(), logs

    def test_search_keeps_what_is_faster_and_refuses_what_changes_the_text(self):
        prof = dict(cal.PROFILES["agent"], context=3000)
        c, best, logs = self.run_cal(["--native", "m.gguf", "--mtp", "2"], prof)
        self.assertTrue(best.ok)
        self.assertIn("--seg-graphs", best.options)                  # faster: kept
        self.assertNotIn("--pcie-share", best.options)               # other greedy text: refused
        self.assertTrue(any("refused" in t.note for t in c.trials if "--pcie-share" in t.options))
        self.assertNotIn("--small-scores", best.options)             # slower prompt: not kept
        self.assertNotIn("--mtp-resident", best.options)             # failed to start: not kept
        self.assertEqual(cal.get_option(best.options, "--mtp"), "2")
        self.assertIn("--mtp-sampled", best.options)                 # a sampled profile: the drafts need it
        self.assertEqual(cal.get_option(best.options, "--cpu-threads"), "6")
        self.assertTrue(any("best:" in l for l in logs))

    def test_greedy_profile_needs_no_mtp_sampled(self):
        prof = dict(cal.PROFILES["chat"], temperature=0.0)
        _, best, _ = self.run_cal(["--native", "m.gguf", "--mtp", "1"], prof)
        self.assertNotIn("--mtp-sampled", best.options)
        self.assertEqual(cal.get_option(best.options, "--mtp"), "2")

    def test_write_config(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "c.json")
        with open(p, "w") as f:
            json.dump({"exe": "x", "args": ["--native", "m", "--mtp", "2", "--small-scores"], "port": 1}, f)
        t = cal.Trial("x", ["--mtp", "3", "--seg-graphs"], ok=True)
        bak = cal.write_config(p, {}, t)
        self.assertTrue(os.path.exists(bak))
        got = json.load(open(p))
        self.assertEqual(got["args"], ["--native", "m", "--mtp", "3", "--seg-graphs"])
        self.assertEqual(got["port"], 1)


if __name__ == "__main__":
    unittest.main()
