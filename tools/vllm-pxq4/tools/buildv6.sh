#!/bin/bash
# buildv6.sh — sm_70 build of the PXQ4 torch extension into an existing install prefix.
#
#   VLLM_IMAGE  sm_70-capable vLLM container image to build against   (REQUIRED)
#   PXQ4_ROOT   install prefix; src/, build/ and site/ live under it  (default ./pxq4-out)
set -euo pipefail
VLLM_IMAGE=${VLLM_IMAGE:?set VLLM_IMAGE to an sm_70-capable vLLM container image}
PXQ4_ROOT=${PXQ4_ROOT:-$PWD/pxq4-out}
SRC=${PXQ4_SRC:-$PXQ4_ROOT/src}
OUT=${PXQ4_OUT:-$PXQ4_ROOT/site/pxq4_vllm/_lib}
BUILD=${PXQ4_BUILD:-$PXQ4_ROOT/build}
mkdir -p "$BUILD" "$OUT"
docker run --rm \
  -v "$PXQ4_ROOT":"$PXQ4_ROOT" \
  -w "$BUILD" \
  -e SRC="$SRC" -e OUT="$OUT" \
  --entrypoint /bin/bash \
  "$VLLM_IMAGE" -lc '
    set -euo pipefail
    PY=/opt/vllm-venv/bin/python
    PREFIX="$($PY -c "import torch;print(torch.utils.cmake_prefix_path)")"
    ABI="$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")"
    cmake -S "$SRC" -B . \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI" > cm.log 2>&1
    cmake --build . -j "$(nproc)" -- CUDA_FLAGS_EXTRA=1 2>&1 | tail -5
    cp -v libpxq4_sm70.so "$OUT/"
  '
echo BUILD_DONE
