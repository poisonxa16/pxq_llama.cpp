#!/bin/bash
# buildv60.sh — sm_60 (Tesla P100) build of the PXQ4 torch extension.
#
# The stock sm_70 vLLM image CANNOT be used here: its torch 2.10+cu128 ships no sm_60 kernel
# image (arch list sm_70..sm_120), so nothing built against it can even allocate on a P100.
# The last official torch with Pascal support is 2.7.1+cu126, so this build needs its own
# venv holding it:
#
#   python3 -m venv "$SM60_ROOT/venv"
#   "$SM60_ROOT/venv/bin/pip" install torch==2.7.1 numpy \
#       --index-url https://download.pytorch.org/whl/cu126
#
# and its own builder image ($SM60_IMAGE): nvidia/cuda:12.8.1-devel-ubuntu24.04 plus
# python3, python3-venv, cmake and ninja.
#
# TORCH_CUDA_ARCH_LIST=6.0 is REQUIRED: torch cmake overrides CMAKE_CUDA_ARCHITECTURES with
# its no-GPU default list (5.0;8.0;8.6;...) when the env var is absent, silently producing a
# library with no sm_60 cubin at all. Verify with cuobjdump --list-elf: exactly one member,
# pxq4_kernel.sm_60.cubin.
#
# Gates (all must pass on a P100 before any number is quoted):
#   gpu_gate_v6.py <lib> --stress     on-device parity: fused-mt == monolithic, 400-launch stress
#   gate_linear.py <lib>              run with DEFAULT env: PXQ4_MMV_SLICE_MAX=8 reroutes
#                                     M=9..16 onto dequant+cuBLAS, which is not bit-exact to the
#                                     gate's sliced-mmv reference and FAILS it by design
#   xarch_dump.py on P100 (sm_60 lib) and V100 (shipping sm_70 lib) with CPU-seeded inputs,
#                                     then bitwise-compare the two dumps (2026-08-19: 72/72 equal)
#
# P100 dispatch note: the split-vs-mono crossover is 2*SM = 112 on a 56-SM P100, which puts
# tp4_gate_up (136 panels) on mono at M=1. PXQ4_MMV_SPLIT_MAX_BLOCKS=300 forces it back onto
# split: measured 116.9us -> 89.9us on P100. Harmless elsewhere; consider it for any P100 serve.
#
#   SM60_ROOT   working root: holds venv/, build60/ and the resulting .so  (default ./pxq4-sm60)
#   SM60_IMAGE  builder image with the CUDA toolkit                        (default pxa-sm60-dev)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=${PXQ4_SRC:-$HERE/../src}
SM60_ROOT=${SM60_ROOT:-$PWD/pxq4-sm60}
SM60_IMAGE=${SM60_IMAGE:-pxa-sm60-dev}
OUT=$SM60_ROOT
BUILD=$SM60_ROOT/build60
mkdir -p "$BUILD"
docker run --rm \
  -v "$SM60_ROOT":"$SM60_ROOT" -v "$SRC":"$SRC":ro \
  -w "$BUILD" -e SM60_ROOT="$SM60_ROOT" \
  "$SM60_IMAGE" bash -lc '
  set -euo pipefail
  PY=$SM60_ROOT/venv/bin/python
  PREFIX=$($PY -c "import torch;print(torch.utils.cmake_prefix_path)")
  ABI=$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
  export TORCH_CUDA_ARCH_LIST=6.0
  cmake -S '"$SRC"' -B . -DCMAKE_PREFIX_PATH="$PREFIX" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI" > cm.log 2>&1
  cmake --build . -j $(nproc) 2>&1 | tail -3
  cp libpxq4_sm70.so '"$OUT"'/libpxq4_sm60.so
  cuobjdump --list-elf '"$OUT"'/libpxq4_sm60.so
'
echo BUILD_DONE
