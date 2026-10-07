# AGENTS.md

Strata runs the Qwen3.8-Flash-Next mixture-of-experts model (and its Coder, Swift 1.5 and Unsloth variants) on a
normal PC: one NVIDIA or AMD graphics card plus system RAM, on Windows or Linux. It has a C++/CUDA/HIP engine
(`src/`, `include/`), a Python server with an OpenAI- and Anthropic-compatible API and a web app (`serve/`), and a
one-click installer (`setup.py`, started by `START-HERE.bat` / `setup.sh`).

A second engine, `strata-q35` (`q35/`), runs Qwen3.6-35B-A3B (llama.cpp's `qwen35moe` graph, the same `--serve`
protocol, so the server runs on it unchanged): [docs/Q35.md](docs/Q35.md), and for RHEL / CentOS 7
[docs/RHEL7.md](docs/RHEL7.md). It is llama.cpp's graph plus a patch of ours (`q35/patches/`, applied by the build): the
expert cache and the code that drives it are `q35/expert_cache.h`, `q35/hybrid_experts.cpp`, `q35/strata_q35.cpp`. Its tests:
`python -m unittest tools.test_q35_setup` (no engine needed), `build-q35/test-expert-cache` (the cache's policy, no engine) and
`tools/test_q35.py` (needs the engine and the tiny model from `tools/q35_tiny_model.py`; the cache runs there with `--cache-in-ram`).
When the patch changes, regenerate it from the patched llama.cpp checkout (`git add -N src/llama-hybrid.h && git diff`).

The Python server (`serve/`) and `tools/strata_tokenizer.py` use **only the standard library**: chat templates are
rendered by `serve/jinja_lite.py`, the tokenizer's Unicode classes come from `tools/unicode_classes.py` (made by
`tools/gen_unicode_classes.py`). Do not import `jinja2` or `regex` there. `python -m unittest serve.test_jinja_lite
serve.test_tokenizer_classes` hold both to the real packages when they are installed.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the model by RAM, run setup non-interactively,
start and verify the server, and connect the user's apps. Never expose the server beyond `127.0.0.1` without
`--api-key`. As an alternative to shell commands, Strata's MCP server ([docs/MCP_SERVER.md](docs/MCP_SERVER.md))
offers the same steps as tools.

## Working on the code

- How the engine works, every measured number, the API and all settings: [docs/DETAILS.md](docs/DETAILS.md) and
  the [paper](docs/paper/Strata-Paper.pdf).
- AMD (HIP) build and validation: [docs/AMD_HIP.md](docs/AMD_HIP.md); multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md).
- Setup's own tests run without a GPU or downloads: `python tools/test_setup_<name>.py` (for example
  `tools/test_setup_amd.py`, `tools/test_setup_choices.py`).
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a
  measurement.
