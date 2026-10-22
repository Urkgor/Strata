# Strata on RHEL / CentOS 7

How to build and run Strata's CUDA engine on RHEL 7 or CentOS 7 (glibc 2.17, kernel 3.10). Back to the [README](../README.md). The commands alone, in French, from build to server: [README-RHEL7.md](../README-RHEL7.md).

The ready-made engines and the Linux app are built on glibc 2.35 and cannot start on this system, so on RHEL 7 the
engine is **always compiled on the machine** (setup does that by itself, [below](#1-with-setup)). Nothing in the engine's
source had to change for that: the build needs a newer compiler, a newer CMake and a CUDA 12 toolkit, and all three can sit
beside the system's own.

> **What was and was not checked.** Built and tested in a CentOS 7.9 container (`quay.io/centos/centos:centos7`: glibc 2.17,
> Red Hat's devtoolset-11 with GCC 11.2.1 and binutils 2.36, CUDA 12.4.131 from NVIDIA's RHEL 7 repository, CMake 3.28.4 from
> pip, Python 3.6.8), on a 4-core PC with a Linux 6.18 kernel and no GPU:
>
> - `tools/build_rhel7.sh --arch 61 --portable` compiles `strata` and `libstrata_kq.so` for a GTX 10 card (sm_61) in about 5
>   minutes, with no change to the C++ or CUDA sources. It was run twice: with CMake 3.28.4 and Ninja from pip, and with **only
>   Kitware's CMake 3.20.2** (unpacked from its tarball), `make`, and no pip and no ninja on the `PATH`; both build, and both binaries pass the glibc check below (nvcc was given `-std=c++20` in each). `tools/check_glibc_symbols.sh` finds nothing above GLIBC 2.16, GLIBCXX 3.4.19 and CXXABI
>   1.3.7 in them, which is what RHEL 7's own libraries provide. `strata --help` starts.
> - The same source also compiles with nvcc 12.0 and GCC 11 on Ubuntu 24.04 (12.0 is the oldest CUDA 12).
> - 51 of the 53 CPU-side tests (`ctest`, `STRATA_BUILD_TESTS=ON` without CUDA) pass in the container; the other two read the
>   model's `pack/full/experts.bin`, which is not there.
> - `setup.sh` run in the container with a stand-in `nvidia-smi` that lists a GTX 1080 Ti: it fetched its Python, installed the
>   pip packages, found the card, chose the CUDA 12 engine and compiled it into `engine-cuda12/` (the steps under
>   [with setup](#1-with-setup)); that `strata` also asks for GLIBC 2.16 at most.
>
> **Not checked:** a run on a real RHEL 7 kernel (a container uses the host's, here 6.18), and any run on a GPU. Whether the
> engine loads a model and how fast it is on a GTX 1080 Ti is not measured; the nearest numbers are the Tesla P40's in
> [OLDER_GPUS.md](OLDER_GPUS.md#tesla-p40-sm_61-what-was-measured-and-changed). Where a statement below comes from memory or
> from reading the source and not from a check, it says so.

## What RHEL 7 lacks, and what to put beside it

| Needs | RHEL 7 has | Use | Why |
| --- | --- | --- | --- |
| a C++20 compiler | GCC 4.8.5 | **devtoolset-11** (GCC 11), from the Software Collections | nvcc 12 takes C++20 with a host GCC 10 or newer |
| CMake 3.20 or newer | 2.8; `cmake3` from EPEL is 3.17 | Kitware's binary tarball (no root, no pip: `cmake-3.20.2-linux-x86_64.tar.gz` from github.com/Kitware/CMake/releases, asks for glibc 2.17), or pip's `cmake` (setup installs it that way) | the project says 3.20; before 3.25.2 CMake does not know nvcc's `-std=c++20`, so `CMakeLists.txt` supplies it (it does nothing on a newer CMake) |
| CUDA 12.x | none | **12.0 to 12.4** from NVIDIA's RHEL 7 repository | C++20 in `.cu` files; CUDA 13 dropped Pascal and Volta; the repository's newest is 12.4 (read from its metadata) |
| Python 3.10 or newer, for `setup.sh` | 3.6 | the standalone Python 3.12 that `setup.sh` downloads | the server and setup use `:=` and newer library calls |
| a driver | whatever is installed | R525 or newer for a CUDA 12 engine | NVIDIA's rule for CUDA 12 (from [OLDER_GPUS.md](OLDER_GPUS.md#nvidia-the-cuda-12-engine); not tested on RHEL 7). R580 is the last branch with Pascal (NVIDIA's announcement) |

glibc 2.17 itself is enough: `tools/check_glibc_symbols.sh` shows what a binary asks the system for.

## 1. With setup

```
# the compiler (CentOS 7: sudo yum install centos-release-scl first; RHEL 7: sudo subscription-manager repos --enable rhel-server-rhscl-7-rpms, from memory)
sudo yum install devtoolset-11-gcc-c++ devtoolset-11-binutils git
# CUDA 12.4: add NVIDIA's repository (below), then
sudo yum install cuda-nvcc-12-4 cuda-cudart-devel-12-4 cuda-driver-devel-12-4 libcublas-devel-12-4 cuda-cccl-12-4
./setup.sh
```

NVIDIA's repository file, `/etc/yum.repos.d/cuda-rhel7.repo` (NVIDIA publishes its signing key on the same site; the lines
below leave the signature check off to stay short):

```
[cuda-rhel7]
name=NVIDIA CUDA (RHEL 7)
baseurl=https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64
enabled=1
gpgcheck=0
```

CentOS 7 reached its end of life in June 2024: its repositories moved to `vault.centos.org`. If `yum` cannot reach
`mirror.centos.org`, point the repository files at `https://vault.centos.org` (`docker/rhel7/Dockerfile` has the two `sed` lines).

What `setup.sh` does on this system, each step run in the container above:

1. No Python 3.10: it downloads the standalone CPython 3.12.15 (pinned, SHA-256 checked) into `.python/`. That Python asks for
   glibc 2.17 at most.
2. The pip packages: `requirements.txt` pins numpy 2.5.3 and pillow 12.3.0, which have no wheel for glibc 2.17. Setup
   tries the pins; when pip refuses, it installs the same packages unpinned with `--prefer-binary`, which on glibc 2.17 gives
   numpy 2.2.6 and pillow 12.2.0. The build tools (`cmake 4.4.3`, `ninja 1.13.2`) have wheels for glibc 2.17 and install as pinned;
   that CMake configured the engine in the container.
3. A C library older than 2.28 means the ready-made engine is not an option: setup says so and compiles
   (`--build` is implied).
4. The g++ on `PATH` is older than GCC 10: setup puts `/opt/rh/devtoolset-11/root/usr/bin` (or 12, 10; `gcc-toolset-N` on RHEL 8)
   first on `PATH` for the build. `CC`, `CXX` and `CUDAHOSTCXX` still override this, and `STRATA_NVCC=<nvcc>` picks one CUDA toolkit.
5. A Pascal or Volta card runs the [CUDA 12 engine](OLDER_GPUS.md#nvidia-the-cuda-12-engine), compiled into `engine-cuda12/`.

## 2. The engine only: `tools/build_rhel7.sh`

```
tools/build_rhel7.sh --arch 61        # GTX 10 series; the card is read from nvidia-smi without --arch
tools/build_rhel7.sh --help
```

It does the compiling part of setup in plain shell, with no Python: it switches devtoolset on, finds the newest CUDA 12 toolkit in
`/usr/local`, takes the first `cmake` that is 3.20 or newer (or the one `--cmake PATH` names; `pip --user` only when there is none), picks the CPU's instruction-set floor from
`/proc/cpuinfo` (a Xeon without AVX2 gets the [older-CPU build](INSTALL.md#older-cpus-experimental)), builds `strata` and
`libstrata_kq.so` in `build-rhel7/`, and ends with the glibc check. `--portable` builds for an AVX2 baseline instead of this CPU's
own instruction set, so the binary can be copied to a similar machine. The result runs with the NVIDIA driver's `libcuda.so.1`
and the toolkit's `libcudart.so.12`, `libcublas.so.12` and `libcublasLt.so.12` (found through the binary's run path to the
toolkit it was built with; copy them next to it for another machine).

### No pip, no root

`setup.sh` needs Python and pip (it makes a virtual environment), so on a machine where pip cannot run use this script. It needs
only the compiler, the CUDA toolkit and a CMake 3.20 or newer; `make` does the building when there is no `ninja`:

```
tar xzf cmake-3.20.2-linux-x86_64.tar.gz -C ~            # from github.com/Kitware/CMake/releases; no root
scl enable devtoolset-11 bash                            # or let the script switch it on: it does when /opt/rh/devtoolset-11 exists
tools/build_rhel7.sh --cmake ~/cmake-3.20.2-linux-x86_64/bin/cmake --arch 61
```

The script runs pip only when it finds no suitable CMake at all, and says how to do without it if pip fails.
llama.cpp (the `ggml` the engine builds on) is fetched by CMake with `git` at configure time; with no network pass `--ggml DIR`
(a checkout at exactly the commit in `third_party/ggml/VERSION.txt`, `3cf03257f219afbe7334045ff7c6a06ac68c627d`: any other llama.cpp breaks the `mmq_*.cu` files).

## 3. In Docker

`docker/rhel7/Dockerfile` is the system the checks above ran in: CentOS 7.9, devtoolset-11, CUDA 12.4, CMake and ninja from pip.

```
docker build -t strata-rhel7 docker/rhel7
docker run --rm -v "$PWD":/src -w /src strata-rhel7 tools/build_rhel7.sh --arch 61 --build-dir /src/build-rhel7
```

Behind a proxy that re-signs HTTPS with the company's certificate, put the certificate (PEM, `.crt`) in `docker/rhel7/extra-ca/`
and pass `--build-arg https_proxy=...` to `docker build`. The image takes about 3.6 GB. The engine built in it is for the
system it is copied to, not for the container: run it where the card and the driver are.

## At run time

Read from the source, not run on a 3.10 kernel: the engine asks the system for nothing newer than Linux 3.10 and glibc 2.17
(`mmap` with `MAP_HUGETLB`, `madvise`, `posix_fadvise`, Linux AIO for `O_DIRECT` reads, `move_pages`; the 2 MB page flag is defined
by the engine when the headers lack it), and it reads the memory and CPU limits of cgroup v1, the version RHEL 7 uses, as well as v2.
RHEL 7 turns transparent huge pages on (`always`); the engine already handles that setting
([TROUBLESHOOTING.md](TROUBLESHOOTING.md): `STRATA_NO_ARENA_THP=1` turns its request off).

## When something stops

| What you see | Meaning and fix |
| --- | --- |
| `no instance of function template "mul_mat_q_case" (or "strata_w4a4_mul_mat_q_case") matches the argument list`, in `mmq_*.cu` | the llama.cpp given with `--ggml` is not the commit the engine is written for (`third_party/ggml/VERSION.txt`, `3cf03257f219afbe7334045ff7c6a06ac68c627d`); reproduced here with a newer llama.cpp. Use that commit: `cd DIR && git fetch origin 3cf03257f219afbe7334045ff7c6a06ac68c627d && git checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d`, or unpack `https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.tar.gz` (a tarball has no git: the script cannot check it). Without `--ggml` CMake fetches the right one. `tools/build_rhel7.sh` now refuses a git checkout at another commit |
| `... Error 2` (`CMakeFiles/Makefile:1633: ... strata.dir/rule`) | only make's wrapper line; the cause is above it. `tools/build_rhel7.sh` keeps the output in `build-rhel7/build.log`, repeats the build on one job after a failure and prints the first `error:` lines. By hand: `cmake --build build-rhel7 -j1 2>&1 \| grep -n -m5 -E 'error:\|Killed\|undefined reference'`. `Killed` means too little RAM for the jobs: `--jobs 1` |
| `Target ... requires the language dialect "CUDA20"` | this CMakeLists.txt is older than the fix, or the CMake is older than 3.20: use 3.20 or newer (`tools/build_rhel7.sh --cmake DIR/bin/cmake` after unpacking Kitware's tarball) |
| `CMake 3.20 or higher is required` | `cmake3` of EPEL is 3.17: same fix; pip is not needed |
| `the C++ compiler (/usr/bin/g++) is GCC 4` | no devtoolset installed: `sudo yum install devtoolset-11-gcc-c++ devtoolset-11-binutils` |
| `unsupported GNU version`, or errors in `<bits/...>` or `<concepts>` | nvcc 12.0 to 12.3 refuse a host GCC newer than 12; devtoolset-11 is what to use |
| `no kernel image is available for execution on the device` at start | the engine has no code for the card: build with `--arch` of the card (`nvidia-smi --query-gpu=compute_cap --format=csv`) |
| `no such instruction`, `unknown opcode` from the assembler | the system's binutils 2.27 is in use (from memory): run the build inside `scl enable devtoolset-11 bash`, or let setup / `tools/build_rhel7.sh` put devtoolset first on `PATH` |
| `version 'GLIBC_2.xx' not found` | a binary built on another system; build on RHEL 7 or in `docker/rhel7` and check it with `tools/check_glibc_symbols.sh` |
| `version 'GLIBCXX_3.4.xx' not found` | the same, for libstdc++: devtoolset's build links the newer parts in and needs 3.4.19; a build with another GCC does not |
| `libcudart.so.12: cannot open shared object file` | the CUDA libraries are not on the run path: `export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH` or copy them next to `strata` |
| `CUDA driver version is insufficient for CUDA runtime version` | the driver is older than R525 |
| `python3: ... invalid syntax` at `:=` | the system's Python 3.6 started a Strata script; use the `.venv` that `setup.sh` made (or `.python/python/bin/python3`) |
