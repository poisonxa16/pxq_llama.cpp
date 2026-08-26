#!/usr/bin/env bash
# Build the PXA vLLM serving images from THIS tree. No third-party image is involved.
#
#   ./scripts/build-images.sh sm60     Tesla P100 / Pascal  -> pxa-vllm:sm60
#   ./scripts/build-images.sh sm70     Tesla V100 / Volta   -> pxa-vllm:sm70
#   ./scripts/build-images.sh both
#
# The two variants differ ONLY in the torch they build against, and that difference is
# forced by hardware, not preference:
#
#   sm60  torch 2.7.1 + VLLM_SKIP_C_STABLE=1 + VLLM_TORCH27_COMPAT
#         2.7.1 is the LAST torch whose wheels ship sm_60 cubins. A 2.10-based image
#         passes every extension gate and still dies on a P100 with "no kernel image is
#         available" from torch's OWN kernels.
#
#   sm70  torch 2.10 (the tree's own pins), libtorch_stable BUILT, no compat shim
#         Volta never needed the downgrade. Taking it anyway is what produced the V100
#         boot chain: a missing fused RMSNorm op, an absent aot_compile, and PEP 604
#         annotations 2.7.1's dynamo cannot trace. None of those exist here.
#
# Run this from the repo root (the build context must be the repo).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f Dockerfile ] || { echo "run from the repo root (no Dockerfile here)" >&2; exit 2; }

build_sm60() {
  echo "=== building pxa-vllm:sm60  (Pascal: torch 2.7.1, arch 6.0;7.0)"
  docker build -t pxa-vllm:sm60 \
    --build-arg VARIANT=sm60 \
    --build-arg TORCH_SPEC="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1" \
    --build-arg TORCH_INDEX="https://download.pytorch.org/whl/cu126" \
    --build-arg TORCH_ASSERT="2.7.1+cu126" \
    --build-arg ARCH_LIST="6.0;7.0" \
    --build-arg SKIP_C_STABLE="1" \
    --build-arg EXTRA_CMAKE="-DVLLM_TORCH27_COMPAT=ON" \
    .
}

build_sm70() {
  echo "=== building pxa-vllm:sm70  (Volta: torch 2.10, arch 7.0, libtorch_stable ON)"
  docker build -t pxa-vllm:sm70 \
    --build-arg VARIANT=sm70 \
    --build-arg TORCH_SPEC="torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0" \
    --build-arg TORCH_INDEX="https://download.pytorch.org/whl/cu128" \
    --build-arg TORCH_ASSERT="2.10.0" \
    --build-arg ARCH_LIST="7.0" \
    --build-arg SKIP_C_STABLE="0" \
    --build-arg EXTRA_CMAKE="" \
    .
}

case "${1:-both}" in
  sm60) build_sm60 ;;
  sm70) build_sm70 ;;
  both) build_sm60; build_sm70 ;;
  *) echo "usage: $0 [sm60|sm70|both]" >&2; exit 2 ;;
esac

echo
echo "=== built:"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}  {{.CreatedSince}}' | grep '^  pxa-vllm:' || true
cat <<'NOTE'

NEXT: neither image is verified by building it. Run the smoke gate against the card class
it targets before trusting either one - a healthy /health is not evidence, a RAW
non-chat-templated 1-token completion is. scripts/fat-image-smoke.sh does the sm60 half.
NOTE
