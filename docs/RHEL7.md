# strata-q35 on RHEL / CentOS 7

How to build and run [`strata-q35`](Q35.md) (Qwen3.6-35B-A3B) on RHEL 7 or CentOS 7. Back to the [README](../README.md).

> **What was and was not checked.** The engine's sources were read for what RHEL 7 lacks, and everything was built and
> tested on Ubuntu 24.04 with GCC 13. **Nothing was built or run on RHEL 7**: the repositories below could not be reached
> from the session that wrote this page. The steps are the usual ones for that system; the build prints what it finds and
> `tools/check_glibc_symbols.sh` tells you whether the result can run there. Where a statement comes from memory and not
> from a check, it says so.

## What RHEL 7 is

glibc 2.17, kernel 3.10, GCC 4.8.5, CMake 2.8 (3.17 as `cmake3` from EPEL), Python 3.6. The engine needs a **C++17
compiler (GCC 9 or newer)** and **CMake 3.14 or newer (3.18 for the CUDA backend)**; both are one package away (below),
and neither changes the system's own compiler or C library.

What the sources use from the system, read in the engine and in llama.cpp at the pinned commit (`mmap`, `posix_fadvise`,
`posix_madvise`, pthreads, `std::filesystem`): nothing newer than glibc 2.17 or Linux 3.10. `io_uring`, `memfd_create`,
`getrandom` and the like do not appear. The one Linux-only extra, `O_DIRECT` (`--load direct`), is used only on request.
**The default `--load auto` is `mmap`.**

## 1. The compiler and CMake

Red Hat's Software Collections give GCC 11 beside the system's:

```
# RHEL 7:    sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms
# CentOS 7:  sudo yum install centos-release-scl         (CentOS 7 ended in 2024: its repositories moved to vault.centos.org)
sudo yum install devtoolset-11-gcc-c++ devtoolset-11-binutils
sudo yum install cmake3 git patch      # EPEL's cmake3 is 3.17: enough for a CPU build. For CUDA (3.18+): python3 -m pip install --user cmake
```

`devtoolset-11` also brings the newer binutils that AVX-512 and VNNI code needs. `tools/build_q35.sh` enables
`/opt/rh/devtoolset-11/enable` by itself when `g++` is older than 9; by hand: `scl enable devtoolset-11 bash`.

The build **links libstdc++ and libgcc statically** (`STRATA_Q35_STATIC_LIBSTDCXX`, on by default), so the binary does not
depend on the toolset's runtime libraries.

## 2. CUDA (NVIDIA cards)

- NVIDIA's toolkit has to support RHEL 7 **and** your card. From memory, not checked here: CUDA 11.8 supports RHEL 7 and
  cards up to Ada (RTX 40, `sm_89`); some 12.x releases still did and later ones dropped it; the RTX 50 series
  (Blackwell, `sm_120`) needs CUDA 12.8 or newer. Check NVIDIA's CUDA installation guide for the support table of the
  toolkit you want.
- If no toolkit that supports your card runs on RHEL 7, the engine still builds and runs on the CPU
  (`tools/build_q35.sh --cuda off`); or use a newer system for the card.
- `nvcc` must use the same GCC as everything else: the build passes `-DCMAKE_CUDA_HOST_COMPILER=<devtoolset's g++>`.
  CUDA 11.8 supports host GCC up to 11.
- `--arch` takes the compute capabilities (`"70;80;86;89"`); without it the build asks `nvidia-smi` (a driver too old to
  answer falls back to llama.cpp's defaults).

## 3. Build

```
git clone <this repository> && cd Strata
tools/build_q35.sh --cuda on --arch "80;86" --jobs 8       # or --cuda off
tools/check_glibc_symbols.sh build-q35/strata-q35           # RHEL 7: nothing above GLIBC 2.17
```

The script fetches llama.cpp (one commit, about 30 MB) into `third_party/llama.cpp-src`: with `git fetch --depth 1`, else
with `curl` and the tarball. **Offline or behind a proxy:** clone `https://github.com/ggml-org/llama.cpp` elsewhere, check out the
commit in `third_party/ggml/VERSION.txt`, copy it over and pass `--llama-cpp DIR`.

The build applies the engine's patch to those sources (the [expert cache](Q35.md#the-expert-cache-what-makes-it-more-than-llamacpp); it needs `patch` or `git`).

The first build takes 5 to 20 minutes (llama.cpp and its CUDA kernels); later ones rebuild only what changed.
`--portable` builds for an AVX2 baseline instead of the build machine's CPU, so one binary serves several servers of the
same family; without it the binary uses everything the build machine has (AVX-512 included) and may crash with
"illegal instruction" on an older CPU.

**A binary built on another distribution does not run on RHEL 7**: it asks for a newer C library (the build here on
Ubuntu 24.04 asks for GLIBC 2.38; `tools/check_glibc_symbols.sh` shows it). Build on RHEL 7, or in a CentOS 7 container.

## 4. Run

```
build-q35/strata-q35 --native /models/Qwen3.6-35B-A3B-Q4_K_M.gguf --chat -p "Hello" -n 200       # a quick check
build-q35/strata-q35 --native /models/model.gguf --cpu-moe --threads 24 --numa distribute --serve   # the engine itself
```

For a server with several sockets, `--numa distribute` (threads on all nodes) or `isolate` (only the node the process
started on, with `numactl --cpunodebind=0 --membind=0`) usually matters more than any other option; try both and measure.
`--threads` defaults to the physical cores; more than that rarely helps the experts' memory-bound work. `--load mlock` pins
the model in RAM (raise `ulimit -l`, or run it from a systemd unit with `LimitMEMLOCK=infinity`).

Without a graphics card, or with a small one, the model's weights live in RAM: **the file's size must fit in RAM** (with
`--load mmap`, the default, a model larger than RAM runs from the disk's page cache, much slower).

## 5. The server (web app, OpenAI and Anthropic APIs)

The engine alone needs nothing but its binary. The Strata server around it (`serve/server.py`) needs **Python 3.8 or newer
and no package at all**: it renders the model's chat template with `serve/jinja_lite.py` and splits text for the tokenizer with
`re` (`tools/unicode_classes.py`), so there is no Jinja2 and no `regex` to install, and no `pip`.

RHEL 7's own Python is 3.6, too old; Red Hat's Software Collections have 3.8 (`rh-python38`; the package name is from memory,
not checked here). The server's whole test suite (506 tests) and a real chat through `strata-q35` were run on Python 3.8.20
with no third-party package; on 3.6 nothing was run, and the server's code does not fit it (it uses `from __future__ import annotations`).

```
sudo yum install rh-python38                               # the collection's name: from memory
scl enable rh-python38 bash                                # python3 is now 3.8
python3 tools/q35_setup.py --gguf /models/model.gguf --engine build-q35/strata-q35 --max-context 65536 --cpu-moe
pack-q35/run-q35.sh --host 127.0.0.1                      # http://127.0.0.1:8095/
```

As a service, a systemd unit with `ExecStart=/path/Strata/pack-q35/run-q35.sh`, `WorkingDirectory=/path/Strata` and
`Restart=on-failure` is enough; the server restarts the engine itself when it dies. Before the server listens on anything
but `127.0.0.1`, set an API key (`STRATA_API_KEY`).

## If it does not work

| You see | Probably |
| --- | --- |
| `needs GCC 9 or newer` | the devtoolset is not installed or not enabled: `scl enable devtoolset-11 bash`, then again |
| `version GLIBC_2.xx not found` when starting | the binary was built on a newer system; build on RHEL 7 |
| `illegal instruction` | built with `GGML_NATIVE` on a newer CPU than this one: rebuild with `--portable` |
| `CUDA error: no kernel image is available` | `--arch` missed your card's compute capability: rebuild with `--arch "<yours>"` |
| `could not create the context` | not enough memory for `--max-context`: lower it, use `--kv q8_0`, or `--cpu-moe` |
| the model loads, but writing is slow | the experts do not fit the card: `--expert-cache auto` (with `--cpu-moe` or on its own), `--threads` = physical cores; `numactl` on several sockets; [measure what the cache gives](Q35.md#measuring-what-the-cache-gives-you) |
| `CUDA error: an illegal memory access ...` while loading | the expert cache met a kernel of your card it cannot use: add `--expert-cache off` (`q35_setup.py --no-expert-cache`) and report it with `--verbose`; the current build tries the cache in a second process first and leaves it off by itself if that dies |
| `no expert cache: ...` | the line says why (the whole model fits the card, no card in this build, per-expert scales); `the expert cache was switched off` means its first-tokens check failed on your card: please report it with `--verbose` |
