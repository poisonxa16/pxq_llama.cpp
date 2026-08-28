#!/usr/bin/env bash
# PXA Network — PXQ4 sm_70 serving launcher (NVIDIA Tesla V100 class).
#
# Boots a PXQ4-quantized model on vLLM inside a container, waits for /health,
# then reads back the two settings that fail SILENTLY if they do not take.
#
#   MODEL=/path/to/your-pxq4-model scripts/pxa-serve-sm70.sh
#   scripts/pxa-serve-sm70.sh --help      # every parameter and its default
#
# ---------------------------------------------------------------------------
# THE MEASURED CONFIGURATION
#
# The defaults below are measured, not guessed. Every value was A/B'd on a
# 2x Tesla V100-PCIE-16GB pair (sm_70, PCIe host bridge, no P2P) on 2026-08-27,
# and each arm was correctness-gated before its number was recorded.
# Results (single-stream decode / aggregate@4 / aggregate@8, tok/s):
#
#   reproduce final-v10                    48.74 / 106.46 / 134.36
#   + PXQ4_MMV_SPLIT_MAX_BLOCKS=300        51.46 / 107.26 / 132.13   <- best single
#   + SPLIT=600                            51.31 / 106.60 / 134.47
#   + SPLIT=150                            49.51 / 105.07 / 134.66
#   + SPLIT=300, GMU 0.92                  51.32 / 104.85 / 133.76
#   + SPLIT=300, MNS=16, ladder[1..8,16]   50.98 / 107.02 / 135.78   <- SHIPPED
#
# MNS=16 ships: it clears the 50 tok/s bar on single-stream and takes both
# aggregate crowns. Set PXA_PROFILE=single to trade 0.5 tok/s of concurrency
# headroom for the best single-stream number instead.
#
# WHY THESE VALUES, so they are not "tidied" away later:
#  GMU 0.85     - NOT 0.90+. 0.98 with a pinned 12 GiB KV is a four-card,
#                 32-GiB-per-card setting and ABORTS on a 16 GiB card at TP=2
#                 before the model finishes loading.
#  ladder       - must be passed EXPLICITLY. When cudagraph_capture_sizes is
#                 None the sm70 branch hard-codes [1,2] and every batch above 2
#                 runs eager, ~4x slower. The launcher reads it back and warns.
#  SPLIT=300    - routes gate_up from mono to split. Worth ~2.7 tok/s (48.74 ->
#                 51.46). 150 is worse, 600 is a wash on decode.
#  PXQ4_LIB     - pinned explicitly. The union site tree bundles an sm70-only
#                 .so built against the torch 2.10 ABI; leaving this unset on
#                 the wrong image dies with an undefined-symbol crash mid-load
#                 rather than with a clear message.
#  ladder [..16]- the "16-token graph poisons the stack" finding is MoE-specific.
#                 Verified clean on a dense model; correctness held on all arms.
# ---------------------------------------------------------------------------
set -uo pipefail

usage() {
  cat <<'USAGE'
PXA Network - PXQ4 sm_70 (Tesla V100 class) vLLM serving launcher.

  MODEL=/path/to/pxq4-model scripts/pxa-serve-sm70.sh
  MODEL=... CARDS=0,1 PORT=8001 scripts/pxa-serve-sm70.sh
  MODEL=... PXA_PROFILE=single scripts/pxa-serve-sm70.sh

Parameters (environment variables). Only MODEL has no usable default.

  MODEL         PXQ4 model directory, or a name the image can resolve. REQUIRED
  CARDS         comma-separated GPU indices; TP size = how many you list  [0,1]
  PORT          port to serve on                                         [8001]
  BIND          host interface the port is published on             [127.0.0.1]
  NAME          container name                                 [pxa-pxq4-sm70]
  IMAGE         vLLM image carrying the PXQ4 backend            [pxa-vllm:sm70]
  PKG           PXQ4 sidecar package                              [<repo>/pxa/pxq4]
  SITE          plugin tree placed on PYTHONPATH          [$PKG/sidecar/site-sm60]
  LIB           kernel library pinned into PXQ4_LIB
                                          [$PKG/kernels/libpxq4_sm70_v10.so]
  ALIAS         --served-model-name                          [basename $MODEL]
  GMU           --gpu-memory-utilization                                 [0.85]
  MML           --max-model-len                                         [32768]
  MNS / LADDER  override the profile's --max-num-seqs / capture ladder
  SPLIT_MAX_BLOCKS  PXQ4_MMV_SPLIT_MAX_BLOCKS                             [300]
  PXA_PROFILE   "agg" (default) or "single"; see the header table
  TOOL_PARSER   adds --enable-auto-tool-choice with this parser         [unset]
  EXTRA_ARGS    appended verbatim to the vLLM command line              [empty]
  BOOT_TIMEOUT  seconds to wait for /health                              [2400]
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

NAME=${NAME:-pxa-pxq4-sm70}
PORT=${PORT:-8001}
BIND=${BIND:-127.0.0.1}
CARDS=${CARDS:-0,1}
IMAGE=${IMAGE:-pxa-vllm:sm70}
PKG=${PKG:-$REPO/pxa/pxq4}
# The plugin tree is pure Python and carries no .so of its own; the architecture
# is decided by LIB/PXQ4_LIB below. site-sm60 is the one tree this package ships.
# An image that carries its own pxq4_vllm should override SITE=.
SITE=${SITE:-$PKG/sidecar/site-sm60}
LIB=${LIB:-$PKG/kernels/libpxq4_sm70_v10.so}
ALIAS=${ALIAS:-$(basename "$MODEL")}
GMU=${GMU:-0.85}
MML=${MML:-32768}
SPLIT_MAX_BLOCKS=${SPLIT_MAX_BLOCKS:-300}
TOOL_PARSER=${TOOL_PARSER:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
BOOT_TIMEOUT=${BOOT_TIMEOUT:-2400}

if [ "${PXA_PROFILE:-agg}" = "single" ]; then
  MNS=${MNS:-8};  LADDER=${LADDER:-1,2,3,4,5,6,7,8}      # 51.46 single-stream
else
  MNS=${MNS:-16}; LADDER=${LADDER:-1,2,3,4,5,6,7,8,16}   # 50.98 single, best agg@4 and agg@8
fi
TP=$(awk -F, '{print NF}' <<<"$CARDS")

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

TOOL_FLAGS=""
[ -n "$TOOL_PARSER" ] && TOOL_FLAGS="--enable-auto-tool-choice --tool-call-parser $TOOL_PARSER"

echo "PXA Network — PXQ4 sm_70"
echo "  model   : $MODEL  (served as $ALIAS)"
echo "  cards   : $CARDS (TP=$TP)   $BIND:$PORT"
echo "  image   : $IMAGE"
echo "  lib     : $LIB"
echo "  profile : ${PXA_PROFILE:-agg}  MNS=$MNS ladder=[$LADDER] GMU=$GMU"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES="$CARDS" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e PYTHONPATH="$SITE" -e PXQ4_LIB="$LIB" \
  -e PXQ4_MMV_SPLIT_MAX_BLOCKS="$SPLIT_MAX_BLOCKS" \
  -e PYTHONUNBUFFERED=1 -e HOME=/tmp -e TMPDIR=/tmp \
  -p "${BIND}:${PORT}:${PORT}" \
  "${MOUNTS[@]}" \
  --shm-size=16g --ipc=host \
  "$IMAGE" /bin/bash -c "exec python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --served-model-name $ALIAS --quantization pxq4 \
    --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size $TP --dtype float16 \
    --enable-prefix-caching --trust-remote-code \
    $TOOL_FLAGS \
    --gpu-memory-utilization $GMU --max-model-len $MML \
    --max-num-seqs $MNS --max-num-batched-tokens 2048 \
    --compilation-config '{\"cudagraph_capture_sizes\":[$LADDER]}' \
    $EXTRA_ARGS \
    --host 0.0.0.0 --port $PORT" >/dev/null || { echo "docker run failed" >&2; exit 1; }

echo -n "booting (graph capture is slow)"
STEP=15
for i in $(seq 1 $(( BOOT_TIMEOUT / STEP ))); do
  code=$(curl -s -m 8 -o /dev/null -w '%{http_code}' "http://127.0.0.1:${PORT}/health" 2>/dev/null)
  [ "$code" = "200" ] && { echo; echo "HEALTHY after $((i*STEP))s"; break; }
  case "$(docker ps -a --filter "name=^/${NAME}$" --format '{{.Status}}')" in
    Exited*) echo; echo "DIED. First real error:" >&2
             docker logs "$NAME" 2>&1 | grep -aiE "Cuda error|out of memory|ValueError|Capability|undefined symbol" | head -5 >&2
             exit 1 ;;
  esac
  echo -n "."; sleep $STEP
done
[ "${code:-000}" = "200" ] || { echo "never came up; container kept as $NAME" >&2; exit 1; }

# A silent quoting slip here costs ~4x above 2 concurrent streams, so read it back.
TAKEN=$(docker logs "$NAME" 2>&1 | grep -ao "cudagraph_capture_sizes[^]]*]" | tail -1)
echo "  capture ladder taken: $TAKEN"
case "$TAKEN" in
  *"[1, 2]"*) echo "  WARNING: ladder collapsed to [1,2] -- the explicit list did NOT take." >&2 ;;
esac
echo "  Triton (expect enabled on sm_70): $(docker logs "$NAME" 2>&1 | grep -ac 'Disabling Triton') disable-events"
