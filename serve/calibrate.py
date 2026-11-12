"""`server.py --calibrate`: find the engine options that run fastest on this machine, for this kind of use, without changing
what the model writes.

The engine is started once per trial with the config's own arguments plus the options under test, and runs the same work
each time:

1. a quality check: a short prompt answered greedily (temperature 0); its tokens must be the reference trial's, token for
   token.  Every option tried here is meant to give the same bits (graphs, PCIe copies, where the embeddings live, MTP drafts,
   thread count): a trial that writes something else is refused, whatever its speed;
2. the measure: a long prompt read from scratch (its prompt rate), then an answer of `gen` tokens with the profile's sampling
   (its decode rate, MTP drafts included);
3. a follow-up turn on the same conversation (the conversation cache, a short read, a second decode figure at depth).

The score is the time one turn of the profile would take: new prompt tokens at the measured prompt rate plus generated tokens
at the decode rate (lower is better).  The search goes option by option (coordinate descent): an option is kept when the
turn gets faster by more than `min_gain`.  Nothing is written unless `--calibrate-write`: then the config's "args" take the
best options and the old file is kept beside it (.bak-calibrate).

Every trial restarts the engine (minutes on a large model in the hybrid mode): `--calibrate-quick` tries the options that
matter most only.  The options and the profiles are the tables below.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# The options calibration manages: name -> takes a value.  They are taken out of the config's args before a trial and put back
# as the trial says.
MANAGED = {
    "--seg-graphs": False,
    "--embd-host": False,
    "--pcie-share": False,
    "--small-scores": False,
    "--mtp": True,
    "--mtp-sampled": False,
    "--mtp-resident": False,
    "--draft-vocab": True,
    "--cpu-threads": True,
}

# What a turn looks like: the prompt it reads (new tokens at a depth of `context`), what it writes, with what sampling.
PROFILES = {
    # a coding agent (DSH, Claude Code-like): long contexts, sizable new tool results each turn, medium answers
    "agent": {"context": 16000, "new_prompt": 2000, "gen": 300, "temperature": 0.6},
    # a chat: short turns, longer answers
    "chat": {"context": 2000, "new_prompt": 150, "gen": 500, "temperature": 0.7},
    # long documents: big reads, short answers
    "long": {"context": 30000, "new_prompt": 8000, "gen": 150, "temperature": 0.6},
}

IM_START, IM_END = "<|im_start|>", "<|im_end|>"


@dataclass
class Trial:
    label: str
    options: list[str]
    ok: bool = False
    note: str = ""
    prompt_tok_s: float = 0.0
    decode_tok_s: float = 0.0
    decode2_tok_s: float = 0.0
    drafts: str = ""
    turn_s: float = float("inf")
    load_s: float = 0.0
    greedy: list[int] = field(default_factory=list)


def strip_managed(args: list[str]) -> list[str]:
    """The args without the managed options (and their values)."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in MANAGED:
            i += 2 if MANAGED[a] and i + 1 < len(args) else 1
            continue
        out.append(a)
        i += 1
    return out


def current_managed(args: list[str]) -> list[str]:
    """The managed options the args hold now, in order (the starting point of the search)."""
    out, i = [], 0
    while i < len(args):
        a = args[i]
        if a in MANAGED:
            if MANAGED[a] and i + 1 < len(args):
                out += [a, args[i + 1]]
                i += 2
                continue
            out.append(a)
        i += 1
    return out


def set_option(opts: list[str], name: str, value) -> list[str]:
    """opts with `name` set: None / False removes it, True adds the flag, anything else is its value."""
    base = strip_managed_one(opts, name)
    if value is None or value is False:
        return base
    if value is True:
        return base + [name]
    return base + [name, str(value)]


def strip_managed_one(opts: list[str], name: str) -> list[str]:
    out, i = [], 0
    while i < len(opts):
        if opts[i] == name:
            i += 2 if MANAGED.get(name) and i + 1 < len(opts) else 1
            continue
        out.append(opts[i])
        i += 1
    return out


def get_option(opts: list[str], name: str):
    if name not in opts:
        return None
    i = opts.index(name)
    return opts[i + 1] if MANAGED.get(name) and i + 1 < len(opts) else True


def physical_cores() -> int:
    """Physical cores (Linux: the distinct (package, core) pairs of /proc/cpuinfo), else the logical count."""
    try:
        seen, phys, core = set(), None, None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core = line.split(":")[1].strip()
            elif not line.strip():
                if core is not None:
                    seen.add((phys, core))
                phys = core = None
        if core is not None:
            seen.add((phys, core))
        if seen:
            return len(seen)
    except OSError:
        pass
    return os.cpu_count() or 1


def steps(quick: bool, has_mtp: bool, sampled: bool, cores: int) -> list[tuple[str, list]]:
    """The search, option by option: (name, values to try besides the current one)."""
    s: list[tuple[str, list]] = [("--seg-graphs", [True, None]), ("--embd-host", [True, None]), ("--pcie-share", [True, None])]
    if has_mtp:
        s.append(("--mtp", ["0", "1", "2", "3"]))
        if sampled:
            s.append(("--mtp-sampled", [True, None]))
        if not quick:
            s.append(("--mtp-resident", [True, None]))
            s.append(("--draft-vocab", ["32768", None]))
    if not quick:
        s.append(("--small-scores", [True, None]))
        threads = sorted({max(1, cores - 2), max(1, cores - 1), cores})
        s.append(("--cpu-threads", [str(t) for t in threads] + [None]))
    return s


def workload_text(root: Path, n_chars: int) -> str:
    """A prompt body like a coding agent's: the repository's own sources and docs, as many characters as asked."""
    parts, total = [], 0
    for rel in ("src/qwen35/q36_kernels.cuh", "src/qwen35/q36_attn.cuh", "README-RHEL7.md", "serve/calibrate.py",
                "src/qwen35/q36_sample.cuh", "include/strata/qwen35/sampling.hpp", "src/qwen35/q36_ckpt.hpp"):
        p = root / rel
        try:
            t = p.read_text(encoding="utf-8")
        except OSError:
            continue
        parts.append(f"### {rel}\n```\n{t}\n```\n")
        total += len(parts[-1])
        if total >= n_chars:
            break
    body = "".join(parts)
    while len(body) < n_chars and body:
        body += body
    return body[:n_chars]


def chat(user: str, history: str = "") -> str:
    return f"{history}{IM_START}user\n{user}{IM_END}\n{IM_START}assistant\n"


class Calibrator:
    def __init__(self, cfg: dict, tok, start_engine: Callable[[list[str]], object], profile: dict, *,
                 quick: bool = False, min_gain: float = 0.03, log=print, root: Path | None = None, cores: int | None = None):
        self.cfg, self.tok, self.start_engine, self.p = cfg, tok, start_engine, profile
        self.quick, self.min_gain, self.log = quick, min_gain, log
        self.root = root or Path(__file__).resolve().parent.parent
        self.cores = cores or physical_cores()
        self.trials: list[Trial] = []
        self.ref_greedy: list[int] | None = None
        self._prompts()

    def _prompts(self):
        """The token ids of the three requests (built once: every trial reads the same ones)."""
        enc = lambda s: self.tok.encode(s, parse_special=True)
        self.q_ids = enc(chat("Write a Python function that returns the n-th Fibonacci number, with a docstring and two tests."))
        # the long prompt: the body cut to `context` tokens (characters at ~3.5 per token, then the tokens themselves)
        ctx = int(self.p["context"])
        body = workload_text(self.root, int(ctx * 4.2))
        ask = ("Here are files of a project.  Explain in detail, function by function, what this code does, what could be faster, "
               "and write the improved versions.\n\n")
        ids = enc(chat(ask + body))
        if len(ids) > ctx:
            head = enc(f"{IM_START}user\n{ask}")
            tail = enc(f"{IM_END}\n{IM_START}assistant\n")
            body_ids = self.tok.encode(body)[: max(1, ctx - len(head) - len(tail))]
            ids = head + body_ids + tail
        self.long_ids = ids
        n_new = int(self.p["new_prompt"])
        follow = workload_text(self.root, int(n_new * 4.2) + 2000)[-int(n_new * 4.2):]
        self.follow_text = (f"{IM_END}\n{IM_START}user\nAnd these are the test results; go on with the same detail.\n\n"
                            f"{follow}{IM_END}\n{IM_START}assistant\n")

    def sampling(self) -> dict:
        t = float(self.p.get("temperature") or 0.0)
        return {"temperature": t, "top_p": 0.95, "top_k": 20, "seed": 1234} if t > 0 else {}

    def _run(self, eng, ids: list[int], max_new: int, sampling: dict) -> tuple[list[int], dict]:
        out = []
        for t in eng.generate(ids, max_new, sampling, threading.Event()):
            if t is not None:
                out.append(t)
        return out, dict(getattr(eng, "last", {}) or {})

    def trial(self, label: str, options: list[str]) -> Trial:
        tr = Trial(label, list(options))
        args = strip_managed(list(self.cfg["args"])) + list(options)
        t0 = time.time()
        eng = None
        try:
            eng = self.start_engine(args)
            tr.load_s = time.time() - t0
            # 1. quality: greedy, short
            tr.greedy, _ = self._run(eng, self.q_ids, 48, {})
            if self.ref_greedy is not None and tr.greedy != self.ref_greedy:
                n = next((i for i, (a, b) in enumerate(zip(tr.greedy, self.ref_greedy)) if a != b), min(len(tr.greedy), len(self.ref_greedy)))
                tr.note = f"refused: the greedy text differs from the reference from token {n} (quality)"
                return tr
            # 2. the long prompt, from scratch, and an answer
            gen = int(self.p["gen"])
            ans, last = self._run(eng, self.long_ids, gen, self.sampling())
            pm, dm, g = float(last.get("prompt_ms") or 0), float(last.get("decode_ms") or 0), int(last.get("generated") or len(ans))
            if pm <= 0 or dm <= 0 or g < 16:
                tr.note = f"no figures (prompt {pm:.0f} ms, {g} tokens in {dm:.0f} ms)"
                return tr
            tr.prompt_tok_s = len(self.long_ids) / (pm / 1000.0)
            tr.decode_tok_s = g / (dm / 1000.0)
            off, acc = last.get("drafts_offered"), last.get("drafts_accepted")
            if off:
                tr.drafts = f"{acc}/{off}"
            # 3. a follow-up turn: the conversation so far + new text (the cache reads only the new part)
            hist = self.long_ids + ans + self.tok.encode(self.follow_text, parse_special=True)
            ans2, last2 = self._run(eng, hist, gen, self.sampling())
            dm2, g2 = float(last2.get("decode_ms") or 0), int(last2.get("generated") or len(ans2))
            tr.decode2_tok_s = g2 / (dm2 / 1000.0) if dm2 > 0 and g2 >= 16 else tr.decode_tok_s
            tg = (tr.decode_tok_s + tr.decode2_tok_s) / 2
            tr.turn_s = float(self.p["new_prompt"]) / tr.prompt_tok_s + float(self.p["gen"]) / tg
            tr.ok = True
            return tr
        except Exception as e:   # (an option the engine cannot run here: out of memory, refused, ...)
            tr.note = f"failed: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
            return tr
        finally:
            if eng is not None:
                try:
                    eng.close()
                except Exception:
                    pass

    def _say(self, tr: Trial):
        if tr.ok:
            self.log(f"[calibrate] {tr.label:<34} turn {tr.turn_s:6.2f} s   prompt {tr.prompt_tok_s:7.1f} tok/s   decode "
                     f"{tr.decode_tok_s:5.1f} / {tr.decode2_tok_s:5.1f} tok/s" + (f"   drafts {tr.drafts}" if tr.drafts else "")
                     + f"   (load {tr.load_s:.0f} s)")
        else:
            self.log(f"[calibrate] {tr.label:<34} {tr.note}")

    def run(self) -> Trial:
        start = current_managed(list(self.cfg["args"]))
        has_mtp = get_option(start, "--mtp") != "0"   # (--mtp 0 in the config: a GGUF without the MTP block, likely)
        sampled = float(self.p.get("temperature") or 0) > 0
        plan = steps(self.quick, has_mtp, sampled, self.cores)
        self.log(f"[calibrate] profile: a turn reads {self.p['new_prompt']} new tokens at a depth of about {len(self.long_ids)}, "
                 f"writes {self.p['gen']} (temperature {self.p.get('temperature')}); {len(plan)} options to try, one engine start each")
        best = self.trial("as configured: " + (" ".join(start) or "(none)"), start)
        self._say(best)
        self.trials.append(best)
        if not best.ok:
            self.log("[calibrate] the configured engine did not run the work: nothing to compare against")
            return best
        self.ref_greedy = best.greedy
        for name, values in plan:
            # (the value the step starts from is measured already - except the draft count of a sampled profile when the drafts
            # were off for sampled requests: each count is then tried with them on)
            tried = {get_option(best.options, name)}
            if name == "--mtp" and sampled and get_option(best.options, "--mtp-sampled") is None:
                tried = set()
            for v in values:
                if v in tried:
                    continue
                tried.add(v)
                opts = set_option(best.options, name, v)
                if name == "--mtp" and v == "0":
                    opts = set_option(set_option(set_option(opts, "--mtp-sampled", None), "--mtp-resident", None), "--draft-vocab", None)
                if name == "--mtp" and v not in (None, "0") and sampled and get_option(opts, "--mtp-sampled") is None:
                    opts = set_option(opts, "--mtp-sampled", True)   # (drafts for a sampled profile need it)
                if name in ("--mtp-sampled", "--mtp-resident", "--draft-vocab") and get_option(opts, "--mtp") in (None, "0"):
                    continue
                label = f"{name} {v if v not in (True, None) else ('on' if v else 'off')}"
                tr = self.trial(label, opts)
                self._say(tr)
                self.trials.append(tr)
                if tr.ok and tr.turn_s < best.turn_s * (1.0 - self.min_gain):
                    best = tr
        self.log(f"[calibrate] best: {' '.join(best.options) or '(no option)'}: a turn in {best.turn_s:.2f} s (prompt "
                 f"{best.prompt_tok_s:.0f} tok/s, decode {best.decode_tok_s:.1f} tok/s), against {self.trials[0].turn_s:.2f} s as configured")
        return best


def write_config(path: str, cfg: dict, best: Trial) -> str:
    """The config's "args" with the best options; the old file kept as .bak-calibrate.  Returns the backup's path."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw["args"] = strip_managed(list(raw.get("args") or [])) + list(best.options)
    bak = path + ".bak-calibrate"
    shutil.copyfile(path, bak)
    Path(path).write_text(json.dumps(raw, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return bak
