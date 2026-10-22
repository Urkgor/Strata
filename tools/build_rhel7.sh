#!/bin/sh
# Build Strata's CUDA engine on RHEL / CentOS 7 (docs/RHEL7.md).  Run it from a checkout, on the machine whose card
# will run the model, or in docker/rhel7:
#
#     tools/build_rhel7.sh                      # the card's architecture is read from nvidia-smi
#     tools/build_rhel7.sh --arch 61            # GTX 10 series; "61;75" for several cards
#     tools/build_rhel7.sh --help
#
# What RHEL 7 lacks, and what this does about it:
#   * GCC 4.8.5 (the engine is C++20): Red Hat's devtoolset-11 (or 12 / 10), switched on here when g++ is older than 10.
#   * CMake 2.8 / 3.17: the build needs 3.20 or newer.  Use one you have (--cmake), or Kitware's binary tarball (no root, no
#     pip); pip's cmake is tried only when none is found.
#   * CUDA: NVIDIA's RHEL 7 repository stops at 12.4.  Cards older than RTX 20 (GTX 10, Tesla P100/P40) need CUDA 12.x
#     anyway (CUDA 13 dropped them), so this is the "CUDA 12 engine" of docs/OLDER_GPUS.md.
#   * the prebuilt engines: they ask for glibc 2.35, RHEL 7 has 2.17.  An engine built here asks for nothing above it.
# The result is $BUILD_DIR/strata (and libstrata_kq.so beside it); the last step checks them with
# tools/check_glibc_symbols.sh.
set -eu
cd "$(dirname "$0")/.."

ARCH=""; CMAKE_ARG=""; CUDA_DIR=""; BUILD_DIR="build-rhel7"; JOBS=""; PORTABLE=""; GGML_DIR=""; FLOOR="${STRATA_ISA_FLOOR:-}"; TARGETS="strata"
usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; cat <<'EOF'

options:
  --arch LIST        CUDA architectures without the dot, ';' or ',' between: 61 (GTX 10), 60 (P100), 70 (V100), 75, 86 ...
                     Default: the cards nvidia-smi lists.  Below 75 the experimental Pascal/Volta build is switched on.
  --cuda-home DIR    the CUDA toolkit (default: $CUDA_HOME, the newest /usr/local/cuda-12.*, /usr/local/cuda, nvcc on PATH)
  --cmake PATH       the cmake to use (default: the first cmake on PATH that is 3.20 or newer; pip is the last resort)
  --build-dir DIR    where to build (default build-rhel7)
  --jobs N           parallel compiles (default: half the CPUs; every compile of the engine can take 2-3 GB of RAM)
  --portable         an AVX2 baseline instead of this CPU's own instruction set: one binary for several machines
  --isa-floor avx|none   a CPU without AVX2 (Sandy/Ivy Bridge: avx; Nehalem/Westmere: none).  Default: read from /proc/cpuinfo
  --ggml DIR         a llama.cpp checkout at the commit in third_party/ggml/VERSION.txt (default: fetched by CMake with git)
  --no-kq            leave out libstrata_kq.so, the prompt kernels for Unsloth's UD-Q4/Q5/Q6_K_XL files (default: built)
EOF
}
KQ=1
while [ $# -gt 0 ]; do
  case "$1" in
    --arch) ARCH="$2"; shift 2 ;;
    --cuda-home) CUDA_DIR="$2"; shift 2 ;;
    --cmake) CMAKE_ARG="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --portable) PORTABLE=ON; shift ;;
    --isa-floor) FLOOR="$2"; shift 2 ;;
    --ggml) GGML_DIR="$2"; shift 2 ;;
    --no-kq) KQ=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "build_rhel7: unknown option $1 (--help lists them)" >&2; exit 2 ;;
  esac
done
die() { echo "build_rhel7: $*" >&2; exit 1; }
note() { echo "build_rhel7: $*"; }
# version a >= b ?
ver_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]; }

# ---- 1. the C++ compiler: C++20 means GCC 10 or newer
gcc_major() { "$1" -dumpversion 2>/dev/null | cut -d. -f1; }
CXX_BIN=${CXX:-g++}
if ! command -v "$CXX_BIN" >/dev/null 2>&1 || [ "$(gcc_major "$CXX_BIN" || echo 0)" -lt 10 ]; then
  for v in 11 12 10; do
    if [ -f "/opt/rh/devtoolset-$v/enable" ]; then
      set +u; . "/opt/rh/devtoolset-$v/enable"; set -u      # Red Hat's script reads variables that may be unset
      CXX_BIN=g++; note "devtoolset-$v switched on"; break
    fi
  done
fi
command -v "$CXX_BIN" >/dev/null 2>&1 || die "no C++ compiler.  RHEL 7: sudo yum install devtoolset-11-gcc-c++ devtoolset-11-binutils (CentOS 7: after yum install centos-release-scl)"
GCC_V=$(gcc_major "$CXX_BIN" || echo 0)
[ "$GCC_V" -ge 10 ] || die "$CXX_BIN is GCC $GCC_V; the engine is C++20 and needs GCC 10 or newer.  RHEL 7: sudo yum install devtoolset-11-gcc-c++ devtoolset-11-binutils, or point CXX at one"
CXX_PATH=$(command -v "$CXX_BIN")
CC_BIN=${CC:-gcc}; command -v "$CC_BIN" >/dev/null 2>&1 || CC_BIN=$(dirname "$CXX_PATH")/gcc
note "C++ compiler: $CXX_PATH (GCC $GCC_V)"

# ---- 2. the CUDA toolkit
NVCC=""
if [ -n "$CUDA_DIR" ]; then NVCC="$CUDA_DIR/bin/nvcc"
elif [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then NVCC="$CUDA_HOME/bin/nvcc"
else
  newest=$(ls -d /usr/local/cuda-12.* 2>/dev/null | sort -V | tail -n 1 || true)
  if [ -n "$newest" ] && [ -x "$newest/bin/nvcc" ]; then NVCC="$newest/bin/nvcc"
  elif [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc
  elif command -v nvcc >/dev/null 2>&1; then NVCC=$(command -v nvcc); fi
fi
[ -n "$NVCC" ] && [ -x "$NVCC" ] || die "nvcc not found.  Install NVIDIA's CUDA toolkit for RHEL 7 (12.0 to 12.4: sudo yum install cuda-nvcc-12-4 cuda-cudart-devel-12-4 cuda-driver-devel-12-4 libcublas-devel-12-4 cuda-cccl-12-4 from developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64) or pass --cuda-home"
CUDA_ROOT=$(cd "$(dirname "$NVCC")/.." && pwd -P)
CUDA_V=$("$NVCC" --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -n 1)
[ -n "$CUDA_V" ] || die "could not read the version of $NVCC"
CUDA_MAJOR=${CUDA_V%%.*}
[ "$CUDA_MAJOR" -ge 12 ] || die "CUDA $CUDA_V is too old: the engine is C++20, nvcc takes that from 12.0.  NVIDIA's RHEL 7 repository has 12.0 to 12.4"
note "CUDA toolkit: $CUDA_ROOT (nvcc $CUDA_V)"

# ---- 3. the cards
if [ -z "$ARCH" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d ' .\r' | sort -un | tr '\n' ';' | sed 's/;$//' || true)
  fi
  [ -n "$ARCH" ] || die "no --arch given and nvidia-smi lists no card.  Pass --arch (GTX 1080 Ti and the other GTX 10 cards: 61)"
  note "architectures from nvidia-smi: $ARCH"
fi
ARCH=$(printf '%s' "$ARCH" | tr ',' ';' | tr -d ' ')
case "$ARCH" in *[!0-9\;]*|"") die "--arch wants numbers like 61 or 61;75, not '$ARCH'" ;; esac
LOWEST=$(printf '%s' "$ARCH" | tr ';' '\n' | sort -n | head -n 1)
HIGHEST=$(printf '%s' "$ARCH" | tr ';' '\n' | sort -n | tail -n 1)
SM60=OFF
if [ "$LOWEST" -lt 75 ]; then
  [ "$LOWEST" -ge 60 ] || die "sm_$LOWEST: the engine needs compute capability 6.0 (Pascal) or newer"
  [ "$CUDA_MAJOR" -lt 13 ] || die "sm_$LOWEST needs a CUDA 12 toolkit (CUDA 13 dropped Pascal and Volta); this one is $CUDA_V"
  SM60=ON
  note "sm_$LOWEST is below the supported RTX 20 floor: the experimental Pascal/Volta build (docs/OLDER_GPUS.md)"
fi

# ---- 4. CMake 3.20 or newer (the project says 3.20; it adds nvcc's C++20 flag itself when CMake does not know it), ninja when there is one
CMAKE_MIN=3.20
cmake_ok() { v=$("$1" --version 2>/dev/null | sed -n '1s/.*version //p'); [ -n "$v" ] && ver_ge "$v" "$CMAKE_MIN"; }
CMAKE_BIN=""
if [ -n "$CMAKE_ARG" ]; then
  cmake_ok "$CMAKE_ARG" || die "--cmake $CMAKE_ARG: not a CMake $CMAKE_MIN or newer"
  CMAKE_BIN=$CMAKE_ARG
else
  for c in cmake cmake3 "$HOME/.local/bin/cmake"; do
    if command -v "$c" >/dev/null 2>&1 && cmake_ok "$c"; then CMAKE_BIN=$(command -v "$c"); break; fi
  done
fi
if [ -z "$CMAKE_BIN" ]; then
  PIP_PY=""
  for p in python3 python3.12 python3.11 python3.10 python3.9 python3.8 python3.6; do
    if command -v "$p" >/dev/null 2>&1; then PIP_PY=$p; break; fi
  done
  nocmake="CMake $CMAKE_MIN or newer is needed.  Without pip or root: unpack Kitware's binary (cmake-3.20.2-linux-x86_64.tar.gz from github.com/Kitware/CMake/releases; it asks for glibc 2.17) anywhere and pass --cmake DIR/bin/cmake.  RHEL 7's cmake3 package is 3.17, too old"
  [ -n "$PIP_PY" ] || die "no CMake $CMAKE_MIN or newer found.  $nocmake"
  note "no CMake $CMAKE_MIN or newer: trying pip (--user)"
  # RHEL 7's pip 9 does not know the manylinux2014 wheels CMake is shipped in; pip 21.3 is the last that runs on its Python 3.6
  "$PIP_PY" -m pip install --user --quiet --upgrade pip >/dev/null 2>&1 || true
  "$PIP_PY" -m pip install --user --quiet "cmake>=$CMAKE_MIN" ninja || die "pip could not install CMake.  $nocmake"
  # where pip puts a program: the user base (--user), or the interpreter's own scripts folder when it was installed already
  for d in "$("$PIP_PY" -m site --user-base 2>/dev/null)/bin" "$("$PIP_PY" -c 'import sysconfig; print(sysconfig.get_path("scripts"))' 2>/dev/null)"; do
    if [ -x "$d/cmake" ] && cmake_ok "$d/cmake"; then PATH="$d:$PATH"; export PATH; CMAKE_BIN="$d/cmake"; break; fi
  done
  [ -n "$CMAKE_BIN" ] || die "pip left no CMake $CMAKE_MIN or newer where it could be found.  $nocmake"
fi
GEN="Unix Makefiles"; command -v ninja >/dev/null 2>&1 && GEN=Ninja
note "CMake: $CMAKE_BIN ($("$CMAKE_BIN" --version | sed -n '1s/.*version //p')), generator $GEN"

# ---- 5. the CPU: AVX2 is the engine's floor; older CPUs get the experimental build
if [ -z "$FLOOR" ] && [ -r /proc/cpuinfo ]; then
  flags=$(sed -n 's/^flags[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo | head -n 1)
  case " $flags " in
    *" avx2 "*) FLOOR="" ;;
    *" avx "*) FLOOR=avx ;;
    *" sse4_2 "*) FLOOR=none ;;
    *) die "this CPU has no AVX and no SSE4.2: the engine does not run on it" ;;
  esac
fi
case "$FLOOR" in ""|avx|none) ;; *) die "--isa-floor is avx or none, not '$FLOOR'" ;; esac
[ -z "$FLOOR" ] || note "this CPU has no AVX2: building for the '$FLOOR' floor (experimental, docs/DETAILS.md)"
[ -z "$FLOOR" ] || [ -z "$PORTABLE" ] || die "--portable (AVX2) and --isa-floor exclude each other"

# ---- 6. configure and build
[ -n "$JOBS" ] || { n=$(nproc 2>/dev/null || echo 4); JOBS=$((n / 2)); [ "$JOBS" -ge 2 ] || JOBS=2; }
set -- -S . -B "$BUILD_DIR" -G "$GEN" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_Q6K_EXPERTS=ON -DSTRATA_MMQ_KQUANTS=ON \
  "-DCMAKE_CUDA_ARCHITECTURES=$ARCH" "-DCMAKE_CUDA_COMPILER=$NVCC" "-DCUDAToolkit_ROOT=$CUDA_ROOT" \
  "-DCMAKE_C_COMPILER=$(command -v "$CC_BIN")" "-DCMAKE_CXX_COMPILER=$CXX_PATH" "-DCMAKE_CUDA_HOST_COMPILER=$CXX_PATH" \
  "-DSTRATA_EXPERIMENTAL_SM60=$SM60"
[ -z "$PORTABLE" ] || set -- "$@" -DSTRATA_PORTABLE=ON
[ -z "$FLOOR" ] || set -- "$@" "-DSTRATA_ISA_FLOOR=$FLOOR"
[ -z "$GGML_DIR" ] || set -- "$@" "-DSTRATA_GGML_DIR=$GGML_DIR"
[ "$KQ" = 1 ] && TARGETS="strata strata_kq"
note "configuring $BUILD_DIR (sm_$(printf '%s' "$ARCH" | sed 's/;/, sm_/g'))"
"$CMAKE_BIN" "$@"
note "building $TARGETS with $JOBS jobs (a few minutes for one architecture, 10-30 for several)"
"$CMAKE_BIN" --build "$BUILD_DIR" --target $TARGETS -j "$JOBS"

# ---- 7. will it run on RHEL 7?
OUT=""
for f in "$BUILD_DIR/strata" "$BUILD_DIR/libstrata_kq.so"; do [ -f "$f" ] && OUT="$OUT $f"; done
[ -n "$OUT" ] || die "the build finished without $BUILD_DIR/strata"
# shellcheck disable=SC2086
sh tools/check_glibc_symbols.sh $OUT || die "the engine asks for a newer C library than RHEL 7 has (above); build it on RHEL 7 or in docker/rhel7"
note "done: $BUILD_DIR/strata  (CUDA libraries it loads: $CUDA_ROOT/lib64 or targets/x86_64-linux/lib; the NVIDIA driver provides libcuda.so.1)"
note "check: $BUILD_DIR/strata --help; the driver needs to be R525 or newer for a CUDA 12 engine"
