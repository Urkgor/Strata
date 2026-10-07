#!/usr/bin/env python3
"""Sets up the Strata server for a Qwen3.6-35B-A3B GGUF that you already have, on the strata-q35 engine.

`setup.py` installs the original engine (Qwen3.8-Flash-Next) from ready-made releases.  Qwen3.6 runs on strata-q35,
which is built from source (docs/Q35.md, docs/RHEL7.md), so this is its small installer: it reads the GGUF, takes the
tokenizer and the chat template out of it with tools/strata_tokenizer.py, and writes

    <out>/tokenizer/       vocabulary, merges, the model's own chat template
    <out>/config.json      what `serve/server.py --engine strata --config` reads
    <out>/run-q35.sh       starts the server (the web app and the OpenAI / Anthropic APIs)

    python tools/q35_setup.py --gguf /models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf --engine build-q35/strata-q35

Where the model runs is strata-q35's own business (the experts that do not fit the card go to RAM, fitted to the free
VRAM); --cpu-moe, --n-cpu-moe, --kv and the rest are passed on as the engine's arguments.  Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

FAMILY = ("qwen35moe", "qwen35", "qwen3next")
# the GGUF keys llama.cpp's converter writes from the model's generation_config.json -> the server's sampling block
SAMPLING_KEYS = {"general.sampling.temp": "temperature", "general.sampling.top_p": "top_p",
                 "general.sampling.top_k": "top_k", "general.sampling.min_p": "min_p"}


def shards_of(first: Path) -> list[Path]:
    """`name-00001-of-00003.gguf` -> all three; a single file -> itself."""
    import re
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    n = int(m.group(2))
    stem = first.name[:m.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def read_facts(gguf: Path) -> dict:
    """What the file says about itself (the first shard carries the metadata)."""
    from gguf_reader import GGUFFile
    g = GGUFFile(gguf)
    arch = g.metadata.get("general.architecture", "")
    f = {"arch": arch, "name": g.metadata.get("general.name", ""), "path": str(gguf)}
    for key, field in (("block_count", "layers"), ("expert_count", "experts"), ("expert_used_count", "experts_used"),
                       ("context_length", "context_train"), ("embedding_length", "embd")):
        v = g.metadata.get(f"{arch}.{key}")
        if isinstance(v, int):
            f[field] = v
    f["bytes"] = sum(p.stat().st_size for p in shards_of(gguf) if p.exists())
    f["missing_shards"] = [p.name for p in shards_of(gguf) if not p.exists()]
    f["quantization"] = g.by_type() if g.tensors else {}
    f["sampling"] = {SAMPLING_KEYS[k]: g.metadata[k] for k in SAMPLING_KEYS
                     if isinstance(g.metadata.get(k), (int, float)) and not isinstance(g.metadata.get(k), bool)}
    return f


def sampling_block(facts: dict) -> dict:
    """The model's own recommended sampling, when its GGUF carries it, in the server's limits (top_k 1..64)."""
    s = dict(facts.get("sampling") or {})
    if "top_k" in s:
        s["top_k"] = max(1, min(64, int(s["top_k"]))) if int(s["top_k"]) > 0 else None
    return {k: v for k, v in s.items() if v is not None}


def engine_args(a: argparse.Namespace, gguf: Path, out: Path | None = None) -> list[str]:
    args = ["--native", str(gguf), "--max-context", str(a.max_context)]
    if a.cpu_moe:
        args.append("--cpu-moe")
    elif a.n_cpu_moe:
        args += ["--n-cpu-moe", str(a.n_cpu_moe)]
    if a.gpu_layers is not None:
        args += ["--gpu-layers", str(a.gpu_layers)]
    if a.kv:
        args += ["--kv", a.kv]
    if a.ubatch:
        args += ["--ubatch", str(a.ubatch)]
    if a.batch:
        args += ["--batch", str(a.batch)]
    if a.no_repack:
        args.append("--no-repack")
    if a.pin_experts:
        args.append("--pin-experts")
    if a.spec != "off":
        args += ["--spec", a.spec]
    if a.main_gpu is not None:
        args += ["--main-gpu", str(a.main_gpu)]
    if a.split_mode:
        args += ["--split-mode", a.split_mode]
    if a.threads:
        args += ["--threads", str(a.threads)]
    if a.numa:
        args += ["--numa", a.numa]
    if a.fit:
        args += ["--fit", a.fit]
    if a.expert_cache != "off":
        # the hot experts on the card, the CPU computes only the rest (docs/Q35.md); the engine says why if it cannot
        args += ["--expert-cache", a.expert_cache]
        if out is not None:
            args += ["--cache-profile", str(out / "hot-experts.bin")]   # what the model asked for: a warm cache at the next start
    args += list(a.engine_arg or [])
    return args


def make_config(a: argparse.Namespace, gguf: Path, out: Path, facts: dict) -> dict:
    cfg = {"exe": str(Path(a.engine).resolve()), "args": engine_args(a, gguf, out), "cwd": str(ROOT),
           "tokenizer": str(out / "tokenizer"), "model_name": a.model_name, "log": str(out / "engine.log")}
    sampling = sampling_block(facts)
    if sampling:
        cfg["sampling"] = sampling
    if a.host:
        cfg["host"] = a.host
    return cfg


def write_run_script(out: Path, port: int) -> Path:
    venv = ROOT / ".venv" / "bin" / "python"
    script = out / "run-q35.sh"
    text = ("#!/bin/sh\n"
            "# Starts the Strata server on strata-q35 (the web app, /v1/chat/completions, /v1/messages).\n"
            f'cd {shlex.quote(str(ROOT))} || exit 1\n'
            f'PY={shlex.quote(str(venv))}\n'
            '[ -x "$PY" ] || PY=python3\n'
            f'exec "$PY" serve/server.py --engine strata --config {shlex.quote(str(out / "config.json"))} '
            f'--port {port} "$@"\n')
    with open(script, "w", encoding="utf-8", newline="\n") as f:        # (Path.write_text has no newline= before 3.10)
        f.write(text)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model (the first file of a split model)")
    ap.add_argument("--engine", default=str(ROOT / "build-q35" / "strata-q35"), help="the strata-q35 binary")
    ap.add_argument("--out", default=str(ROOT / "pack-q35"), help="where the tokenizer, config and run script go")
    ap.add_argument("--model-name", default="qwen3.6-35b-a3b", help="the name the API reports")
    ap.add_argument("--max-context", type=int, default=32768, help="tokens of context (default 32768)")
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--host", default=None, help="the address to listen on (default: this PC only; use --api-key)")
    ap.add_argument("--cpu-moe", action="store_true", help="all routed experts in RAM, the rest on the card")
    ap.add_argument("--n-cpu-moe", type=int, default=0, metavar="N", help="the experts of the first N layers in RAM")
    ap.add_argument("--gpu-layers", default=None, metavar="N|all")
    ap.add_argument("--kv", default=None, help="K/V cache type: f16 (the engine's default), q8_0, q4_0")
    ap.add_argument("--ubatch", type=int, default=0, metavar="N",
                    help="prompt micro-batch (engine default 512). With the experts in RAM the card is handed them once per micro-batch: "
                         "2048 or 4096 read long prompts several times faster")
    ap.add_argument("--batch", type=int, default=0, metavar="N", help="prompt batch (engine default 2048; at least --ubatch)")
    ap.add_argument("--no-repack", action="store_true",
                    help="keep the CPU's weights as in the file, so the card can compute them for long prompts (otherwise the CPU does)")
    ap.add_argument("--pin-experts", action="store_true", help="page-lock the experts in RAM: faster PCIe copies for long prompts (experimental)")
    ap.add_argument("--spec", default="off", choices=["off", "lookup"], help="speculation from repeats in the context (docs/Q35.md)")
    ap.add_argument("--main-gpu", type=int, default=None, metavar="N", help="the card that holds the model when there are several")
    ap.add_argument("--split-mode", default=None, choices=["none", "layer", "row"], help="several cards: none (one card), layer, row")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--numa", default=None, choices=["distribute", "isolate", "numactl"])
    ap.add_argument("--fit", default=None, choices=["on", "off"])
    ap.add_argument("--expert-cache", default="auto", metavar="auto|MIB|off",
                    help="the experts the model asks for most are copied to the card and computed there, the CPU only does the "
                         "others (default auto: the card's free memory; off: llama.cpp's placement alone)")
    ap.add_argument("--no-expert-cache", dest="expert_cache", action="store_const", const="off", help="same as --expert-cache off")
    ap.add_argument("--engine-arg", action="append", metavar="ARG", help="another strata-q35 argument (repeatable)")
    ap.add_argument("--force", action="store_true", help="write the files even for another architecture")
    a = ap.parse_args(argv)
    if a.cpu_moe and a.n_cpu_moe:
        ap.error("--cpu-moe and --n-cpu-moe are alternatives")
    if a.expert_cache not in ("auto", "off", "sim") and not (a.expert_cache.isdigit() and int(a.expert_cache) > 0):
        ap.error("--expert-cache takes auto, off, sim or a size in MiB")
    return a


def main(argv=None) -> int:
    a = parse(argv)
    gguf = Path(a.gguf).expanduser().resolve()
    if not gguf.exists():
        print(f"q35_setup: {gguf} does not exist", file=sys.stderr)
        return 1
    facts = read_facts(gguf)
    if facts["arch"] not in FAMILY and not a.force:
        hint = " (Qwen3.8-Flash-Next runs on the `strata` engine: ./setup.sh)" if facts["arch"] == "qwen4exp" else ""
        print(f"q35_setup: {gguf.name} is a '{facts['arch']}' model; strata-q35 runs {', '.join(FAMILY)}{hint}", file=sys.stderr)
        return 1
    if facts["missing_shards"]:
        print("q35_setup: shards missing next to the first file: " + ", ".join(facts["missing_shards"]), file=sys.stderr)
        return 1
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    import strata_tokenizer
    strata_tokenizer.extract(gguf, out)
    cfg = make_config(a, gguf, out, facts)
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    script = write_run_script(out, a.port)
    gib = facts["bytes"] / 2**30
    print(f"model      {facts['name'] or gguf.name}: {facts['arch']}, {facts.get('layers', '?')} layers, "
          f"{facts.get('experts', '?')} experts ({facts.get('experts_used', '?')} per token), {gib:.1f} GiB")
    if facts.get("quantization"):
        print("types      " + ", ".join(f"{k} x{v}" for k, v in sorted(facts['quantization'].items(), key=lambda kv: -kv[1])[:6]))
    if cfg.get("sampling"):
        print("sampling   " + ", ".join(f"{k}={v}" for k, v in cfg["sampling"].items()) + "  (the model's own, from its GGUF)")
    exe = Path(cfg["exe"])
    print(f"engine     {exe}" + ("" if exe.exists() else "   <- not built yet: tools/build_q35.sh, docs/RHEL7.md"))
    print(f"config     {out / 'config.json'}")
    print(f"start      {script}   (then open http://127.0.0.1:{a.port}/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
