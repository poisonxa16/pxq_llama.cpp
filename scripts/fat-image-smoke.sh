#!/usr/bin/env bash
# fat-image-smoke.sh — the EXECUTION half of the fat-image gate.
#
# The in-Dockerfile FAT GATE proves the right bytes exist (sm_60+sm_70 cubins in
# _C/_moe_C, sm_70 in flash_attn_v100, sm_60 in torch's own libtorch_cuda.so).
# It cannot prove a P100 can dispatch into them: docker build cannot attach GPUs,
# and 2026-08-24 produced an image that passed every byte check and died on a
# P100 inside torch's own kernels ("no kernel image is available for execution
# on the device" — torch 2.10/cu128 wheels ship no sm_60).
#
# SM70 FUSED LONG-PREFILL OP: OFF for this image. sm70_gemma_long_prefill_fused_add_rms_norm
# lives ONLY in the _C_stable_libtorch target (csrc/libtorch_stable/), which cannot build
# against torch 2.7.1 (no stable-ABI headers pre-2.8; VLLM_SKIP_C_STABLE=1 in the fat build).
# It is the ONLY stable-target op without a regular-_C twin (audited 2026-08-24). The env
# falls back to the unfused path — correct, slightly slower gemma-arch long prefill on sm_70.
# Follow-up queued: port that one kernel into the regular _C target and drop this line.
#
# CAPTURE MODE: stage 2 serves with cudagraph_mode=FULL_DECODE_ONLY — the shipping
# capture mode for this stack (ENGINE-VERDICT 2026-08-24): on the breakable-cudagraph
# stack, ANY captured-prefill config corrupts raw prompts short enough to fit a
# captured graph (<=8 tok; reproduced by this script's first run against the new fat
# image: ptok=1/5 garbage, 391 fine). That is a serving-config defect, not an image
# defect; the smoke tests the config we actually ship. Do NOT quietly drop the mode
# from this file: with it absent, a PASSING battery does not prove short raw prompts.
#
# RULE: no pxa-vllm:sm60-sm70 tag moves, and no Dockerfile commit claims
# verification, until BOTH stages of this script pass on a real P100 (and, for
# the sm_70 side, on a real V100 with GPU_IDX/ATTN_BACKEND overridden).
#
# Usage:
#   ./scripts/fat-image-smoke.sh IMAGE GPU_IDX [GPU_IDX2]
#   e.g. ./scripts/fat-image-smoke.sh pxa-vllm:sm60-sm70 0 6        # P100 pair
#        ATTN_BACKEND=FLASH_ATTN_V100 MODEL=/c/models/qwen38-27b-unc-vllm-p2cf \
#          ./scripts/fat-image-smoke.sh pxa-vllm:sm60-sm70 2 4      # V100 pair
set -uo pipefail
IMAGE=${1:?image}
G1=${2:?gpu index}
G2=${3:-}
CARDS=$G1${G2:+,$G2}
TP=$([ -n "$G2" ] && echo 2 || echo 1)
MODEL=${MODEL:-/c/models/qwen38-27b-unc-vllm-p1f}
ATTN_BACKEND=${ATTN_BACKEND:-PASCAL_SDPA}
PORT=${PORT:-8433}
# Plugin site + sidecar lib are ARCH-SPECIFIC. Defaults = the sm_60 pair. For a
# V100 run pass SITE=/c/moe-branch/site and PXQ4_LIB= (empty: the site's bundled
# _lib sm_70 v10 .so loads). NEVER point a V100 at libpxq4_sm60.so, and do not
# use libpxq4_sm6070.so (the do-not-adopt op_ver-7 artifact, WORKLOG 2026-08-24).
SITE=${SITE:-/c/pxq4-sm60/site}
# GMU: 0.92 default; on a 2x V100 pair pass GMU=0.85 to leave VRAM headroom
# deliberately holds 1.5 GiB/card (Alina-keeper protection) and 0.92 cannot fit.
GMU=${GMU:-0.92}
PXQ4_LIB=${PXQ4_LIB-/c/pxq4-sm60/libpxq4_sm60.so}
NAME=fat-smoke-$$

echo "== stage 1: bare torch op on GPU(s) $CARDS (catches missing-arch torch in 5s)"
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$CARDS" --entrypoint python "$IMAGE" \
  -c "import torch; x=torch.ones(8, device=\"cuda\"); assert float(x.sum())==8.0; print(\"stage1 OK:\", torch.__version__, torch.cuda.get_device_name(0))" \
  || { echo "STAGE 1 FAILED: torch cannot execute on this GPU — image is dead here regardless of extension cubins"; exit 1; }

echo "== stage 2: full serve + correctness battery"
docker rm -f "$NAME" >/dev/null 2>&1
docker run -d --name "$NAME" --runtime=nvidia \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" \
  -e TORCHDYNAMO_DISABLE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
  -e VLLM_SM70_GDN_DECODE_FLASHQLA=0 -e VLLM_SM70_FUSED_SIGMOID_MIXED_QKV=0 \
  -e PXQ4_MMV_SLICE_MAX=8 -e PXQ4_MMV_SPLIT_MAX_BLOCKS=300 \
  -e HOME=/tmp -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  -e PYTHONPATH="$SITE" ${PXQ4_LIB:+-e PXQ4_LIB="$PXQ4_LIB"} \
  -e VLLM_SM70_GEMMA_LONG_PREFILL_FUSED=0 \
  -v <local-path>:/c -p 127.0.0.1:$PORT:$PORT --shm-size=16g --ipc=host \
  "$IMAGE" python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --quantization pxq4 \
    --attention-backend "$ATTN_BACKEND" --tensor-parallel-size $TP --dtype float16 \
    --host 0.0.0.0 --port $PORT --served-model-name m --trust-remote-code \
    --gpu-memory-utilization $GMU --max-model-len 16384 --max-num-seqs 4 \
    --disable-custom-all-reduce \
    --compilation-config '{"custom_ops": ["none"], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4]}' \
  || { echo "docker run failed"; exit 1; }

code=000
for i in $(seq 1 75); do
  st=$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)
  [ "$st" = "exited" ] && { echo "STAGE 2 FAILED: server died:"; docker logs "$NAME" 2>&1 | grep -aE "Error|Traceback|no kernel image" | grep -vE "min_frames|max_frames" | head -8; docker logs "$NAME" > /tmp/fat-smoke-death-$$.log 2>&1; echo "full log: /tmp/fat-smoke-death-$$.log (container kept for autopsy: $NAME)"; exit 1; }
  code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
  [ "$code" = "200" ] && break
  sleep 20
done
[ "$code" = "200" ] || { echo "STAGE 2 FAILED: never healthy"; exit 1; }

fail=0
one=$(curl -s -m 90 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":6,"temperature":0}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null)
echo "ptok=5 -> ${one:-<none>}"
case "$one" in *Paris*) ;; *) echo "STAGE 2 FAILED: expected Paris"; fail=1;; esac

raw1=$(curl -s -m 90 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"Hi","max_tokens":12,"temperature":0}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["text"])' 2>/dev/null)
echo "ptok=1 -> ${raw1:-<none>}"
case "$raw1" in ""|*"!!!!"*) echo "STAGE 2 FAILED: 1-token prompt empty or garbage"; fail=1;; esac

ans=$(curl -s -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"What is 17*23? Reply with only the number."}],"max_tokens":200,"temperature":0}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[-10:])' 2>/dev/null)
echo "17*23 -> ${ans:-<none>}"
case "$ans" in *391*) ;; *) echo "STAGE 2 FAILED: expected 391"; fail=1;; esac

docker rm -f "$NAME" >/dev/null 2>&1
[ "$fail" = "0" ] && echo "SMOKE PASS: $IMAGE serves correctly on GPU(s) $CARDS" || echo "SMOKE FAIL"
exit $fail
