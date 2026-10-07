#!/bin/sh
# Builds strata-q35, the Qwen3.6-35B-A3B engine (q35/, docs/Q35.md).  Written for RHEL / CentOS 7 and fine anywhere else:
#
#   * a C++17 compiler (GCC 9 or newer): on RHEL 7 it enables devtoolset-11 (or 10, 9) from /opt/rh when the stock
#     GCC 4.8 is what `gcc` is
#   * CMake 3.14 or newer: cmake3 (EPEL), cmake, or the one `pip install cmake` gives
#   * CUDA: nvcc on PATH or in /usr/local/cuda*/bin turns the NVIDIA backend on; --cuda off builds for the CPU only
#
#     tools/build_q35.sh                       # CUDA when nvcc is there, else CPU; build-q35/strata-q35
#     tools/build_q35.sh --cuda on --arch "80;86"   --jobs 8
#     tools/build_q35.sh --portable            # AVX2 baseline instead of this CPU: the binary runs on other PCs
#     tools/build_q35.sh --llama-cpp ~/llama.cpp    # a checkout you already have (offline build)
#
# The first build fetches llama.cpp (the commit in third_party/ggml/VERSION.txt, about 30 MB: one commit, not the history)
# into third_party/llama.cpp-src and builds it: 5 to 20 minutes.  With git too old for that, it takes the tarball with curl.
set -eu
cd "$(dirname "$0")/.."

BUILD_DIR=${BUILD_DIR:-build-q35}
CUDA=auto
ARCH=""
JOBS=""
PORTABLE=OFF
FIT=ON
LLAMA_DIR=""
CLEAN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --cuda) CUDA="$2"; shift 2 ;;
    --arch) ARCH="$2"; shift 2 ;;
    --jobs|-j) JOBS="$2"; shift 2 ;;
    --portable) PORTABLE=ON; shift ;;
    --no-fit) FIT=OFF; shift ;;
    --llama-cpp) LLAMA_DIR="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "build_q35: unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
done
case "$CUDA" in on|off|auto) ;; *) echo "build_q35: --cuda takes on, off or auto" >&2; exit 2 ;; esac

gcc_major() { "$1" -dumpversion 2>/dev/null | cut -d. -f1; }

# ---- the compiler
CXX_BIN=${CXX:-g++}
major=$(gcc_major "$CXX_BIN" || echo 0)
if [ "${major:-0}" -lt 9 ]; then
  for t in devtoolset-11 devtoolset-10 devtoolset-9; do
    if [ -f "/opt/rh/$t/enable" ]; then
      echo "build_q35: g++ is version ${major:-?}; using $t"
      # shellcheck disable=SC1090
      . "/opt/rh/$t/enable"
      CXX_BIN=g++
      major=$(gcc_major "$CXX_BIN" || echo 0)
      break
    fi
  done
fi
if [ "${major:-0}" -lt 9 ]; then
  echo "build_q35: needs GCC 9 or newer (g++ is ${major:-missing})." >&2
  echo "  RHEL / CentOS 7:  sudo yum install centos-release-scl && sudo yum install devtoolset-11-gcc-c++   (docs/RHEL7.md)" >&2
  exit 1
fi

# ---- CMake
CMAKE_BIN=""
for c in cmake3 cmake; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" --version | head -1 | sed 's/[^0-9.]*//; s/-.*//')
    maj=${v%%.*}; rest=${v#*.}; min=${rest%%.*}
    if [ "$maj" -gt 3 ] || { [ "$maj" -eq 3 ] && [ "$min" -ge 14 ]; }; then CMAKE_BIN=$c; break; fi
  fi
done
if [ -z "$CMAKE_BIN" ]; then
  echo "build_q35: needs CMake 3.14 or newer.  sudo yum install cmake3   (EPEL),  or  python3 -m pip install --user cmake" >&2
  exit 1
fi

# ---- CUDA
if [ "$CUDA" = auto ]; then
  CUDA=off
  if command -v nvcc >/dev/null 2>&1; then CUDA=on
  else
    for d in /usr/local/cuda /usr/local/cuda-12* /usr/local/cuda-11*; do
      if [ -x "$d/bin/nvcc" ]; then PATH="$d/bin:$PATH"; export PATH; CUDA=on; break; fi
    done
  fi
fi
CUDA_FLAGS=""
if [ "$CUDA" = on ]; then
  command -v nvcc >/dev/null 2>&1 || { echo "build_q35: --cuda on, but nvcc is not on PATH (try PATH=/usr/local/cuda/bin:\$PATH)" >&2; exit 1; }
  # nvcc must use the same GCC as everything else (devtoolset's, on RHEL 7)
  CUDA_FLAGS="-DSTRATA_Q35_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=$(command -v "$CXX_BIN")"
  if [ -z "$ARCH" ] && command -v nvidia-smi >/dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -u | tr -d '. ' | tr '\n' ';' | sed 's/;$//')
    case "$cc" in ''|*[!0-9\;]*) ;; *) ARCH="$cc" ;; esac
  fi
  [ -n "$ARCH" ] && CUDA_FLAGS="$CUDA_FLAGS -DCMAKE_CUDA_ARCHITECTURES=$ARCH"
  echo "build_q35: CUDA on ($(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p'))${ARCH:+, architectures $ARCH}"
else
  echo "build_q35: CUDA off (CPU only)"
fi

# ---- llama.cpp's source: the commit this engine was written against (its API moves from week to week)
SHA=$(awk '{print $1; exit}' third_party/ggml/VERSION.txt)
if [ -z "$LLAMA_DIR" ]; then
  LLAMA_DIR=third_party/llama.cpp-src
  have=$(cat "$LLAMA_DIR/.strata-commit" 2>/dev/null || true)
  if [ "$have" != "$SHA" ]; then
    echo "build_q35: fetching llama.cpp $SHA"
    rm -rf "$LLAMA_DIR"; mkdir -p "$LLAMA_DIR"
    if command -v git >/dev/null 2>&1 && (cd "$LLAMA_DIR" && git init -q . && git remote add origin https://github.com/ggml-org/llama.cpp.git \
         && git fetch -q --depth 1 origin "$SHA" && git checkout -q FETCH_HEAD) >/dev/null 2>&1; then
      :
    else
      echo "build_q35: git could not fetch it; trying the tarball"
      rm -rf "$LLAMA_DIR"; mkdir -p "$LLAMA_DIR"
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL "https://github.com/ggml-org/llama.cpp/archive/$SHA.tar.gz" | tar xz --strip-components=1 -C "$LLAMA_DIR"
      else
        wget -qO- "https://github.com/ggml-org/llama.cpp/archive/$SHA.tar.gz" | tar xz --strip-components=1 -C "$LLAMA_DIR"
      fi
    fi
    [ -f "$LLAMA_DIR/CMakeLists.txt" ] || { echo "build_q35: could not get llama.cpp; clone it yourself and pass --llama-cpp DIR (docs/RHEL7.md)" >&2; exit 1; }
    echo "$SHA" > "$LLAMA_DIR/.strata-commit"
  fi
else
  [ -f "$LLAMA_DIR/CMakeLists.txt" ] || { echo "build_q35: $LLAMA_DIR is not a llama.cpp checkout" >&2; exit 1; }
  [ "$(cd "$LLAMA_DIR" && git rev-parse HEAD 2>/dev/null || true)" = "$SHA" ] ||
    echo "build_q35: WARNING: $LLAMA_DIR is not at $SHA; the engine was written against that commit and may not build" >&2
fi

[ "$CLEAN" = 1 ] && rm -rf "$BUILD_DIR"
[ -n "$JOBS" ] || JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)

# shellcheck disable=SC2086
"$CMAKE_BIN" -S q35 -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DSTRATA_Q35_PORTABLE=$PORTABLE -DSTRATA_Q35_FIT=$FIT \
  -DSTRATA_Q35_LLAMACPP_DIR="$(cd "$LLAMA_DIR" && pwd)" $CUDA_FLAGS
"$CMAKE_BIN" --build "$BUILD_DIR" -j "$JOBS" --target strata-q35

BIN="$BUILD_DIR/strata-q35"
echo "build_q35: built $BIN"
"$BIN" --version
sh tools/check_glibc_symbols.sh "$BIN" || true
