#!/usr/bin/env bash
# build.sh — build libpxq4_sm70.so against a vLLM image's torch, WITHOUT touching any
# running service.
#
# CONSTRAINTS THIS SCRIPT EXISTS TO RESPECT:
#   * a vLLM container serving traffic is never stopped, restarted, or written to. At most
#     this script reads the image name off it.
#   * the image itself is treated as read-only: nothing is installed into it, and no build
#     tree lives on its overlay. Deployments where the image root filesystem is full or
#     read-only are the normal case, not the exception.
#   * every byte written goes to a mounted host directory you choose.
#
# Usage:  bash build.sh
#
#   PXQ4_IMAGE          vLLM container image to build against
#   PXQ4_REF_CONTAINER  alternatively, a RUNNING vLLM container to read the image name from
#   PXQ4_SRC            source directory                 (default: this script's directory)
#   PXQ4_OUT            where libpxq4_sm70.so is placed  (default: $PXQ4_ROOT/site/pxq4_vllm/_lib)
#   PXQ4_BUILD          cmake build tree                 (default: $PXQ4_ROOT/build)
#   PXQ4_ROOT           install prefix                   (default: ./pxq4-out next to src/)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PXQ4_ROOT="${PXQ4_ROOT:-$HERE/../pxq4-out}"
SRC="${PXQ4_SRC:-$HERE}"
OUT="${PXQ4_OUT:-$PXQ4_ROOT/site/pxq4_vllm/_lib}"
BUILD="${PXQ4_BUILD:-$PXQ4_ROOT/build}"

# Build against the SAME image the server runs, so the torch/nvcc/gcc it links against are
# byte-identical to the ones that will dlopen the result. PXQ4_IMAGE names it directly;
# PXQ4_REF_CONTAINER derives it from a running container. --rm, and no GPU is requested:
# nvcc does not need a device to compile for sm_70.
IMAGE="${PXQ4_IMAGE:-}"
REF="${PXQ4_REF_CONTAINER:-}"
if [ -z "$IMAGE" ] && [ -n "$REF" ]; then
  IMAGE="$(docker inspect -f '{{.Config.Image}}' "$REF" 2>/dev/null || true)"
fi
if [ -z "$IMAGE" ]; then
  echo "ERROR: could not resolve the vLLM image." >&2
  echo "  set PXQ4_IMAGE=<image>            (e.g. PXQ4_IMAGE=vllm/vllm-openai:v0.x)" >&2
  echo "  or  PXQ4_REF_CONTAINER=<name>     (a RUNNING vLLM container to copy the image from)" >&2
  exit 1
fi
echo "building against image: $IMAGE"

mkdir -p "$BUILD" "$OUT"

# Mount the common ancestor of SRC/OUT/BUILD rather than a hardcoded path, so the script
# works wherever the tree actually lives.
MOUNT_ROOT="${PXQ4_MOUNT_ROOT:-$(printf '%s\n%s\n%s\n' "$SRC" "$OUT" "$BUILD" \
  | sed 's|/[^/]*$||' | sort | awk 'NR==1{p=$0} {while (index($0,p)!=1) sub(/\/[^/]*$/,"",p)} END{print p}')}"
[ -z "$MOUNT_ROOT" ] && MOUNT_ROOT=/
echo "mounting: $MOUNT_ROOT"

docker run --rm \
  -v "$MOUNT_ROOT":"$MOUNT_ROOT" \
  -w "$BUILD" \
  -e SRC="$SRC" -e OUT="$OUT" \
  --entrypoint /bin/bash \
  "$IMAGE" -lc '
    set -euo pipefail
    PY=/opt/vllm-venv/bin/python
    PREFIX="$($PY -c "import torch;print(torch.utils.cmake_prefix_path)")"
    ABI="$($PY -c "import torch;print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")"
    echo "torch $($PY -c "import torch;print(torch.__version__)")  cxx11abi=$ABI"
    cmake -S "$SRC" -B . \
      -DCMAKE_PREFIX_PATH="$PREFIX" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=70 \
      -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$ABI"
    cmake --build . -j "$(nproc)"
    cp -v libpxq4_sm70.so "$OUT/"
  '

echo
echo "smoke test (still no GPU work — just proves the ops register):"
docker run --rm -v "$MOUNT_ROOT":"$MOUNT_ROOT" --entrypoint /bin/bash "$IMAGE" -lc "
  /opt/vllm-venv/bin/python - <<'PY'
import torch
torch.ops.load_library('$OUT/libpxq4_sm70.so')
print('pxq4 version        :', torch.ops.pxq4.version())
print('pxq4 mmv_max_m      :', torch.ops.pxq4.mmv_max_m())
print('pxq4 mmv_supported  :', {K: torch.ops.pxq4.mmv_supported(K) for K in (4352, 5120, 6144, 17408)})
print('pxq4 builtin tables :')
print(torch.ops.pxq4.builtin_tables())
PY
"
