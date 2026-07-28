#!/bin/bash
# Build one 35B quant arm from the bf16 source with the codec-fix binary.
# usage: build_arm.sh <tag> <outdir> [extra env/args via env vars]
#   ARM_BACKBONE  -> PXA_PXQ_BACKBONE value (empty = default rev2)
#   ARM_CUSTOMQ   -> --custom-q value (empty = none)
#   ARM_TIER      -> tier positional (default PXQ4)
#   ARM_SRC       -> source gguf (default the fusion4 bf16)
set -o pipefail
TAG=$1; OUTDIR=${2:-<local-path>}
BUILD=<local-path>
SRC=${ARM_SRC:-<local-path>}
TIER=${ARM_TIER:-PXQ4}
mkdir -p $OUTDIR
OUT=$OUTDIR/F4-$TAG.gguf
[ -s $OUT ] && { echo "[arm $TAG] already exists: $OUT"; exit 0; }
ENVARGS=""
[ -n "$ARM_BACKBONE" ] && ENVARGS="-e PXA_PXQ_BACKBONE=$ARM_BACKBONE"
CQARGS=()
[ -n "$ARM_CUSTOMQ" ] && CQARGS=(--custom-q "$ARM_CUSTOMQ")
echo "[arm $TAG] quantize $TIER backbone='${ARM_BACKBONE:-rev2-default}' customq='${ARM_CUSTOMQ:-none}'"
docker run --rm --name pxq-arm-$TAG --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=${ARM_GPU:-6} \
  -e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src $ENVARGS \
  -v $BUILD:/build:ro -v $(dirname $SRC):/src:ro -v $OUTDIR:/out \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /build/bin/llama-quantize --allow-requantize "${CQARGS[@]}" \
  /src/$(basename $SRC) /out/$(basename $OUT) $TIER > $OUTDIR/quant-$TAG.log 2>&1
RC=$?
echo "[arm $TAG] rc=$RC size=$(stat -c %s $OUT 2>/dev/null)"
[ $RC -ne 0 ] && { tail -25 $OUTDIR/quant-$TAG.log; exit 2; }
