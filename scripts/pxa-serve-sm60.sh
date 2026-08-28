#!/usr/bin/env bash
# PXA Network — PXQ4 sm_60 serving launcher (NVIDIA Tesla P100 class).
#
# Boots a PXQ4-quantized model on vLLM inside a container, waits for /health,
# then counts the fp16 mmv hook, whose absence is otherwise silent.
#
#   MODEL=/path/to/your-pxq4-model scripts/pxa-serve-sm60.sh
#   scripts/pxa-serve-sm60.sh --help      # every parameter and its default
#
# ---------------------------------------------------------------------------
# THE MEASURED CONFIGURATION
#
# Measured on a 2x Tesla P100-PCIE-16GB pair (sm_60, TP=2) on 2026-08-27. Every
# arm was correctness-gated AND byte-gated (20/20 greedy outputs identical to an
# NCCL reference boot) before its number was recorded. tok/s:
#
#   gate config (old site, v7 lib, CAR off, custom_ops none)  13.31 single
#   this config                                               24.0-26.4 single
#                                                             70-72 aggregate @8
#                                                             ~218 prefill
#
# WHY EACH VALUE, so none of them get "tidied" away later:
#
#  TORCHDYNAMO_DISABLE=1   LOAD-BEARING. Without it profile_run compiles the
#      LANGUAGE model through Inductor and dies with GPUTooOldForTriton on a
#      capability-6.0 card. This is the single flag that decides whether the
#      server starts at all. Do not remove it to "let dynamo try".
#  CAR left ON            The dominant speed lever: 13.3 -> 24.0, ~1.8x. It is
#      NOT passed --disable-custom-all-reduce. Byte-gated 20/20 against an NCCL
#      reference on this exact lib+site+config, so it is safe HERE. It remains
#      unsafe for MoE models on Pascal -- that is a different model class.
#  custom_ops default     Deliberately NOT "none". "none" is a correctness
#      workaround for a different config: with defaults at MNS=4/ladder[1,2,4]
#      a raw 1-token prompt returns "!!!!" while Paris and 391 still pass. It
#      clears at MNS=8/ladder[1,2,4,8], which is why this config is clean.
#  MNS=8, ladder[1,2,4,8] Both required for the above. MNS=16 is UNVERIFIED on
#      this arch -- its only test ran on a broken image.
#  GMU 0.90               0.94 reintroduces the "!!!!" raw-prompt failure.
#  SPLIT_MAX_BLOCKS=300   NOT 150. 150 is bimodal: four boots gave 24.58,
#      24.61, 19.75 and 9.27. 300 gave 24.01/23.97/23.97 across three. A single
#      good 150 sample looks like a win and is not.
#  PXQ4_MMV_SLICE_MAX=8   Confirmed a dead knob at serving level (14.43 vs
#      14.24 at 16). Kept only to pin it against re-litigation.
#  PXQ4_LIB explicit      The union sidecar tree bundles an sm70-only .so built
#      against the torch 2.10 ABI. Unset, the loader can reach it on this
#      image and die on `undefined symbol: ...incref_pyobject...` mid-load.
#  PASCAL_SDPA            Pascal has no tensor cores; FLASH_ATTN_V100 is sm_70.
# ---------------------------------------------------------------------------
set -uo pipefail

usage() {
  cat <<'USAGE'
PXA Network - PXQ4 sm_60 (Tesla P100 class) vLLM serving launcher.

  MODEL=/path/to/pxq4-model scripts/pxa-serve-sm60.sh
  MODEL=... CARDS=0,1 PORT=8199 scripts/pxa-serve-sm60.sh

Parameters (environment variables). Only MODEL has no usable default.

  MODEL         PXQ4 model directory, or a name the image can resolve. REQUIRED
  CARDS         comma-separated GPU indices; TP size = how many you list  [0,1]
  PORT          port to serve on                                         [8199]
  BIND          host interface the port is published on             [127.0.0.1]
  NAME          container name                                 [pxa-pxq4-sm60]
  IMAGE         vLLM image carrying the PXQ4 backend            [pxa-vllm:sm60]
  PKG           PXQ4 sidecar package                           [<repo>/pxa/pxq4]
  SITE          plugin tree placed on PYTHONPATH          [$PKG/sidecar/site-sm60]
  LIB           kernel library pinned into PXQ4_LIB
                                          [$PKG/kernels/libpxq4_sm60_v10.so]
  ALIAS         --served-model-name                          [basename $MODEL]
  GMU           --gpu-memory-utilization                                 [0.90]
  MML           --max-model-len                                          [8192]
  MNS           --max-num-seqs                                              [8]
  LADDER        cudagraph capture sizes                              [1,2,4,8]
  SPLIT_MAX_BLOCKS  PXQ4_MMV_SPLIT_MAX_BLOCKS                             [300]
  EXTRA_ARGS    appended verbatim to the vLLM command line              [empty]
  BOOT_TIMEOUT  seconds to wait for /health                              [1500]
  PXA_RESERVED_CARDS  comma-separated GPU indices this launcher must never
                touch, for operators who keep cards for other workloads. The
                run is refused if CARDS intersects it. Unset by default.
  PXA_REQUIRE_HOST  optional guard: refuse to run unless `hostname` equals
                this value. Unset by default, i.e. no host check at all.
USAGE
}

case "${1:-}" in
  -h|--help) usage; exit 0 ;;
esac

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# Optional guard for operators who pin a launcher to one machine. Off by default.
if [ -n "${PXA_REQUIRE_HOST:-}" ] && [ "$(hostname)" != "$PXA_REQUIRE_HOST" ]; then
  echo "refusing: PXA_REQUIRE_HOST=$PXA_REQUIRE_HOST but hostname is $(hostname)" >&2
  exit 1
fi

MODEL=${MODEL:-}
if [ -z "$MODEL" ]; then
  echo "MODEL is not set. Point it at a PXQ4 model directory, e.g." >&2
  echo "  MODEL=/path/to/your-pxq4-model $0" >&2
  echo "Run '$0 --help' for the full parameter list." >&2
  exit 2
fi

NAME=${NAME:-pxa-pxq4-sm60}
PORT=${PORT:-8199}
BIND=${BIND:-127.0.0.1}
CARDS=${CARDS:-0,1}
IMAGE=${IMAGE:-pxa-vllm:sm60}
PKG=${PKG:-$REPO/pxa/pxq4}
SITE=${SITE:-$PKG/sidecar/site-sm60}
LIB=${LIB:-$PKG/kernels/libpxq4_sm60_v10.so}
ALIAS=${ALIAS:-$(basename "$MODEL")}
GMU=${GMU:-0.90}; MML=${MML:-8192}; MNS=${MNS:-8}
LADDER=${LADDER:-1,2,4,8}
SPLIT_MAX_BLOCKS=${SPLIT_MAX_BLOCKS:-300}
EXTRA_ARGS=${EXTRA_ARGS:-}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-1500}
TP=$(awk -F, '{print NF}' <<<"$CARDS")

# Opt-in reservation list, for hosts where some cards belong to another workload.
if [ -n "${PXA_RESERVED_CARDS:-}" ]; then
  for c in ${CARDS//,/ }; do
    for r in ${PXA_RESERVED_CARDS//,/ }; do
      [ "$c" = "$r" ] && { echo "REFUSING: card $c is in PXA_RESERVED_CARDS." >&2; exit 1; }
    done
  done
fi

[ -d "$SITE" ] || { echo "sidecar site tree not found: $SITE (override with SITE=)" >&2; exit 2; }
[ -f "$LIB"  ] || { echo "kernel library not found: $LIB (override with LIB=)" >&2; exit 2; }

# Mount whatever holds the model so the container can read it. A MODEL that is
# not a local path is passed through untouched, for images that resolve names.
MOUNTS=(-v "$PKG":"$PKG")
if [ -e "$MODEL" ]; then
  MODEL=$(cd "$(dirname "$MODEL")" && printf '%s/%s' "$(pwd)" "$(basename "$MODEL")")
  MODEL_DIR=$([ -d "$MODEL" ] && echo "$MODEL" || dirname "$MODEL")
  MOUNTS+=(-v "$MODEL_DIR":"$MODEL_DIR":ro)
fi

echo "PXA Network — PXQ4 sm_60"
echo "  model $MODEL  (served as $ALIAS)"
echo "  cards $CARDS (TP=$TP)  $BIND:$PORT"
echo "  image $IMAGE"
echo "  lib   $(basename "$LIB")   site $(basename "$SITE")"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e TORCHDYNAMO_DISABLE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
  -e VLLM_SM70_GDN_DECODE_FLASHQLA=0 -e VLLM_SM70_FUSED_SIGMOID_MIXED_QKV=0 \
  -e VLLM_SM70_GEMMA_LONG_PREFILL_FUSED=0 \
  -e PXQ4_MMV_SLICE_MAX=8 -e PXQ4_MMV_SPLIT_MAX_BLOCKS="$SPLIT_MAX_BLOCKS" \
  -e PYTHONPATH="$SITE" -e PXQ4_LIB="$LIB" \
  -e HOME=/tmp -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
  -p "${BIND}:${PORT}:${PORT}" \
  "${MOUNTS[@]}" \
  --shm-size=16g --ipc=host \
  "$IMAGE" /bin/bash -c "exec python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --quantization pxq4 \
    --attention-backend PASCAL_SDPA --tensor-parallel-size $TP --dtype float16 \
    --host 0.0.0.0 --port $PORT --served-model-name $ALIAS --trust-remote-code \
    --gpu-memory-utilization $GMU --max-model-len $MML --max-num-seqs $MNS \
    --compilation-config '{\"cudagraph_mode\": \"FULL_DECODE_ONLY\", \"cudagraph_capture_sizes\": [$LADDER]}' \
    $EXTRA_ARGS" \
  >/dev/null || { echo "docker run failed" >&2; exit 1; }

echo -n "booting (~6 min: graph capture on Pascal is slow)"
STEP=15
for i in $(seq 1 $(( BOOT_TIMEOUT / STEP ))); do
  code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo; echo "HEALTHY after $((i*STEP))s"; break; }
  case "$(docker ps -a --filter "name=^/${NAME}$" --format '{{.Status}}')" in
    Exited*) echo; echo "DIED. First real error:" >&2
             docker logs "$NAME" 2>&1 | grep -aiE "GPUTooOldForTriton|next_power_of_2|undefined symbol|out of memory|ValueError" | head -5 >&2
             exit 1 ;;
  esac
  echo -n "."; sleep $STEP
done
[ "${code:-000}" = "200" ] || { echo "never came up; container kept as $NAME" >&2; exit 1; }

# The fp16 mmv hook is worth ~12% here and its absence is silent, so COUNT it.
# Note the string is "fp16", not "f16" -- grepping the wrong one reports a
# working hook as absent.
ARMED=$(docker logs "$NAME" 2>&1 | grep -ac 'fp16 mmv fast path armed')
echo "  fp16 mmv hook armed on $ARMED layers"
[ "$ARMED" -lt 50 ] && echo "  WARNING: hook did not arm; expect ~12% less decode. Check PYTHONPATH/PXQ4_LIB." >&2
echo "  Triton disable events (expect 0 with the dynamo flag set): $(docker logs "$NAME" 2>&1 | grep -ac 'Disabling Triton')"
