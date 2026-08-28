#!/usr/bin/env bash
# PXA Network — Qwen3.8-Flash-Next (qwen4exp) PXQU serving launcher.
#
# Serves a PXQU-quantized Flash-Next GGUF with llama-server across several GPUs.
# Runs the binary directly by default; set IMAGE= to run it inside a container.
#
#   MODEL=/path/to/Qwen3.8-Flash-Next-PXQU-ub1024.gguf scripts/pxa-serve-flashnext.sh
#   scripts/pxa-serve-flashnext.sh --help     # every parameter and its default
#
# ---------------------------------------------------------------------------
# WHAT WAS MEASURED, 2026-08-28. Not a guess.
#
# HOW MUCH GPU YOU NEED: PXQU is 46702 MiB of GPU-resident weights (the PXQ4
# build of the same model is 65032 MiB, which is why that one needs six cards).
# The reference layout is four Tesla P100-PCIE-16GB plus one GP102 (GeForce GTX
# 1080 Ti, sm_61) as a fifth card. Four P100s alone CANNOT hold this at a large
# context: card 1 came up FORTY-EIGHT MiB short at c150016, and a PXQU layer is
# ~973 MiB, so there is nothing to shed. Two different -ts values produced
# byte-identical allocations, proving it is layer granularity and not tuning.
#
# ON THE GP102: sm_61 JITs from the compute_60 PTX in libggml.so, so no special
# build is needed. GP102 fp16 is 1/64 rate, but k_pxa_gemv_f16_wide reads half2
# and does its MATH in fp32, so the f16 hc_* tensors placed there are not
# penalised. Keep the HEAD on a P100 (it is the last device in PCI order) - its
# compute buffer is 980 MiB against ~570 elsewhere. If that card is shared with
# another workload, budget for both: the reference run took ~5.0 GiB on top of
# a resident ~4.15 GiB, out of 11264 MiB total.
#
# -ts PARTITIONS BYTES, NOT LAYERS (llama.cpp:4071-4100), and llama.cpp folds a
# per-device compute allowance into the walk - so CHANGING -ub REPACKS THE
# LAYERS. Any -ts you use is capacity-proportional for one -ub at one context;
# re-derive it if you change either. TS is empty by default, which lets
# llama.cpp split automatically. The measured five-card reference value, for
# ub1024 at 160k context on the layout above, was:
#
#     TS=110,268,94,268,261
#
# CONTEXT CEILING: 163840 (160k) is the shipping value. 262144 LOADS but OOMs at
# decode - card 0 was left 51 MiB. Decode needs transient buffers beyond the
# reported compute buffer, so every card needs ~1200 MiB spare.
#
# THE MODEL MUST LIVE ON SOLID STATE. per_layer_token_embd is CPU-resident and
# mmap'd, so every decode token faults PLE pages off whatever disk holds the
# file. Off a parity-RAID spindle array that measured ~17 MB/s, which wedges the
# process in uninterruptible D state. The launcher warns if it can tell the
# model is on rotational media.
# ---------------------------------------------------------------------------
set -uo pipefail

usage() {
  cat <<'USAGE'
PXA Network - Qwen3.8-Flash-Next (qwen4exp) PXQU llama-server launcher.

  MODEL=/path/to/model.gguf scripts/pxa-serve-flashnext.sh
  MODEL=... CARDS=0,1,2,3,4 TS=110,268,94,268,261 scripts/pxa-serve-flashnext.sh
  MODEL=... IMAGE=pxq-llama:cuda scripts/pxa-serve-flashnext.sh   # containerised

Parameters (environment variables). Only MODEL has no usable default.

  MODEL         PXQU Flash-Next .gguf file.                          REQUIRED
  CARDS         comma-separated GPU indices                       [all visible]
  TS            llama.cpp --tensor-split, comma-separated    [empty = automatic]
  UB            --ubatch-size. Changing this REPACKS the -ts layout     [1024]
  NCTX          --ctx-size                                            [163840]
  THREADS       --threads                                                 [16]
  PORT          port to serve on                                        [8080]
  BIND          host interface the port is published on            [127.0.0.1]
  ALIAS         --alias, the served model name         [qwen3.8-flash-next]
  LLAMA_SERVER  llama-server binary            [<repo>/build/bin/llama-server,
                                                  else llama-server on PATH]
  IMAGE         if set, run LLAMA_SERVER inside this container image instead
                of on the host. Unset by default.                     [unset]
  NAME          container name, when IMAGE is set             [pxa-flashnext]
  EXTRA_ARGS    appended verbatim to the llama-server command line     [empty]
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
  echo "MODEL is not set. Point it at a PXQU Flash-Next .gguf, e.g." >&2
  echo "  MODEL=/path/to/Qwen3.8-Flash-Next-PXQU-ub1024.gguf $0" >&2
  echo "Run '$0 --help' for the full parameter list." >&2
  exit 2
fi
[ -f "$MODEL" ] || { echo "model missing: $MODEL" >&2; exit 2; }
MODEL=$(cd "$(dirname "$MODEL")" && printf '%s/%s' "$(pwd)" "$(basename "$MODEL")")

NAME=${NAME:-pxa-flashnext}
PORT=${PORT:-8080}
BIND=${BIND:-127.0.0.1}
CARDS=${CARDS:-}
TS=${TS:-}
UB=${UB:-1024}
NCTX=${NCTX:-163840}
THREADS=${THREADS:-16}
ALIAS=${ALIAS:-qwen3.8-flash-next}
IMAGE=${IMAGE:-}
EXTRA_ARGS=${EXTRA_ARGS:-}
LLAMA_SERVER=${LLAMA_SERVER:-}
if [ -z "$LLAMA_SERVER" ]; then
  if [ -x "$REPO/build/bin/llama-server" ]; then LLAMA_SERVER=$REPO/build/bin/llama-server
  else LLAMA_SERVER=$(command -v llama-server 2>/dev/null); fi
fi
[ -n "$LLAMA_SERVER" ] || { echo "llama-server not found. Build the repo, or set LLAMA_SERVER=." >&2; exit 2; }

# per_layer_token_embd stays on the CPU and is paged in from disk on every
# decode token, so rotational media stalls the whole server. Best effort only:
# if the backing device cannot be identified, say nothing.
model_on_rotational() {
  local src base rot
  src=$(df --output=source -- "$1" 2>/dev/null | tail -n1) || return 1
  case "$src" in /dev/*) ;; *) return 1 ;; esac
  base=$(lsblk -no PKNAME "$src" 2>/dev/null | head -n1)
  [ -n "$base" ] || base=$(basename "$src")
  rot=$(cat "/sys/block/${base}/queue/rotational" 2>/dev/null) || return 1
  [ "$rot" = "1" ]
}
if model_on_rotational "$MODEL"; then
  echo "WARNING: the model is on rotational media. per_layer_token_embd is CPU-resident" >&2
  echo "         and mmap'd, so decode will fault PLE pages off a spindle. Use SSD/NVMe." >&2
fi

ARGS=(-m "$MODEL" -ngl 99
      -ot 'per_layer_token_embd\.weight=CPU'
      -c "$NCTX" -ub "$UB" -t "$THREADS"
      --host 0.0.0.0 --port "$PORT" --alias "$ALIAS")
[ -n "$TS" ] && ARGS+=(-ts "$TS")
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

echo "PXA Network — Flash-Next PXQU"
echo "  model $MODEL"
echo "  cards ${CARDS:-<all visible>}   ts ${TS:-<auto>}   ub $UB   ctx $NCTX   port $PORT"

if [ -z "$IMAGE" ]; then
  [ -n "$CARDS" ] && export CUDA_VISIBLE_DEVICES="$CARDS"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  echo "  running $LLAMA_SERVER on the host"
  exec "$LLAMA_SERVER" "${ARGS[@]}"
fi

# Containerised path.
#
# `docker rm -f` RETURNS BEFORE THE NAME IS RELEASED. Firing docker run straight
# after it loses the race with "container name is already in use", which then
# leaves NOTHING running at all - the old container is already gone. Wait it out.
docker rm -f "$NAME" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  docker ps -a --format '{{.Names}}' | grep -qx "$NAME" || break
  sleep 1
done

BIN_DIR=$(dirname "$LLAMA_SERVER")
BUILD_ROOT=$(dirname "$BIN_DIR")
MODEL_DIR=$(dirname "$MODEL")
# A CMake build tree leaves libllama/libggml beside the binary or one level up,
# depending on the generator, so offer the loader all three.
docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
  -e NVIDIA_VISIBLE_DEVICES="${CARDS:-all}" -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e LD_LIBRARY_PATH="$BIN_DIR:$BUILD_ROOT/src:$BUILD_ROOT/ggml/src" \
  -v "$MODEL_DIR":"$MODEL_DIR":ro -v "$BUILD_ROOT":"$BUILD_ROOT":ro -w "$BUILD_ROOT" \
  -p "${BIND}:${PORT}:${PORT}" \
  "$IMAGE" "$LLAMA_SERVER" "${ARGS[@]}" \
  >/dev/null || { echo "docker run failed" >&2; exit 1; }
echo "started $NAME; watch: docker logs -f $NAME"
