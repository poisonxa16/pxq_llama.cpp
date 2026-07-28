#!/bin/bash
# Paired-protocol perplexity for one arm. usage: ppl_arm.sh <model> <tag> <gpus> <ts> [chunks]
set -o pipefail
MODEL=$1; TAG=$2; GPUS=$3; TS=$4; CHUNKS=${5:-1000}
BUILD=${PPL_BUILD:-<local-path>}
OUTDIR=<local-path>
mkdir -p $OUTDIR
LOG=$OUTDIR/ppl-$TAG.log
[ -s $OUTDIR/nll-$TAG.txt ] && { echo "[ppl $TAG] exists"; exit 0; }
docker run --rm --name pxq-ppl-$TAG --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUS \
  -e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src \
  -v $BUILD:/build:ro -v $(dirname $MODEL):/mdir:ro -v <local-path>:/corp:ro \
  nvidia/cuda:12.8.1-devel-ubuntu24.04 \
  /build/bin/llama-perplexity -m /mdir/$(basename $MODEL) -f /corp/ppl-eval-half.txt \
  --chunks $CHUNKS -c 512 -b 512 -ub 512 --ppl-output-type 1 --seed 1 \
  -ngl 99 -sm layer -ts $TS -fa on > $LOG 2>&1
RC=$?
# per-chunk NLL series: --ppl-output-type 1 lines are "<idx>  <cum_ppl>  <chunk_nll>  <..>";
# field 3 is the per-chunk NLL (verified against the p5-sweep nll-PXQ6.txt series).
grep -E "^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+[[:space:]]+[0-9.]+" $LOG | awk "{print \$3}" > $OUTDIR/nll-$TAG.txt
echo "[ppl $TAG] rc=$RC lines=$(wc -l < $OUTDIR/nll-$TAG.txt) final=$(grep -E 'Final estimate' $LOG | tail -1)"
