#!/usr/bin/env bash
# Serve a PXQ4 model under vLLM. You say which cards; this works out the rest.
#
# Anything after the flags below is passed through to vLLM untouched, so you are never
# boxed in by this wrapper.
set -euo pipefail

IMAGE=${IMAGE:-pxq4-vllm:local}
NAME=${NAME:-pxq4-vllm}
PORT=${PORT:-8000}
CARDS=""
MODEL=""
PASSTHRU=()

usage() {
    cat <<'EOF'
usage: pxq4-serve.sh --cards <list> --model <path> [--port N] [--name N] [vllm args...]

  --cards   GPU indices, e.g. 0,1        (required)
  --model   path to a PXQ4 model dir     (required)
  --port    host port                    (default 8000)
  --name    container name               (default pxq4-vllm)

  IMAGE=<img>   override the image       (default pxq4-vllm:local)

Cards must be the same architecture -- one kernel library cannot serve both Pascal and
Volta. Tensor-parallel size defaults to the number of cards given.

example:
  ./docker/pxq4-serve.sh --cards 0,1 --model /models/coder-35b-pxq4
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --cards) CARDS="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --port)  PORT="$2";  shift 2 ;;
        --name)  NAME="$2";  shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

[ -n "$CARDS" ] || { echo "error: --cards is required" >&2; usage >&2; exit 2; }
[ -n "$MODEL" ] || { echo "error: --model is required" >&2; usage >&2; exit 2; }
[ -d "$MODEL" ] || { echo "error: model directory not found: $MODEL" >&2; exit 2; }

TP=$(echo "$CARDS" | tr ',' '\n' | grep -c .)

# docker rm returns before the name is actually released, so a plain rm+run races and
# fails with "name already in use". Wait for the name to disappear.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "removing existing container $NAME"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    for _ in $(seq 1 30); do
        docker ps -a --format '{{.Names}}' | grep -qx "$NAME" || break
        sleep 1
    done
fi

MODEL_ABS=$(cd "$MODEL" && pwd)
MODEL_MNT=/models/$(basename "$MODEL_ABS")

echo "image  : $IMAGE"
echo "cards  : $CARDS  (tensor-parallel-size $TP)"
echo "model  : $MODEL_ABS -> $MODEL_MNT"
echo "port   : $PORT"

exec docker run -d --name "$NAME" --runtime=nvidia --restart unless-stopped \
    --gpus "\"device=${CARDS}\"" \
    -v "$MODEL_ABS":"$MODEL_MNT":ro \
    -p "${PORT}:8000" \
    --ipc=host \
    "$IMAGE" \
        --model "$MODEL_MNT" \
        --quantization pxq4 \
        --tensor-parallel-size "$TP" \
        --host 0.0.0.0 --port 8000 \
        "${PASSTHRU[@]}"
