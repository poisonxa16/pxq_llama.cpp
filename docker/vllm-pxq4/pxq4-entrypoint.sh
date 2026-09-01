#!/usr/bin/env bash
# Pick the right PXQ4 kernel for the GPUs that are actually visible, then hand off to
# vLLM. The user should not have to know that sm_60 and sm_70 need different .so files.
set -euo pipefail

PXQ4_ROOT=${PXQ4_ROOT:-/opt/pxq4}

# The Dockerfile already puts the plugin on PYTHONPATH (so it still works if someone
# overrides the entrypoint). Only prepend if it is genuinely absent, or the path shows up
# twice and every log line about it reads like a bug.
case ":${PYTHONPATH:-}:" in
    *":${PXQ4_ROOT}/site:"*) : ;;
    *) export PYTHONPATH="${PXQ4_ROOT}/site${PYTHONPATH:+:$PYTHONPATH}" ;;
esac

pick_kernel() {
    # compute_cap comes back like "6.0" / "7.0", one line per visible GPU.
    local caps
    caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
           | tr -d ' ' | sort -u) || true

    if [ -z "$caps" ]; then
        echo "pxq4: cannot read compute capability (no nvidia-smi, or no GPU visible)." >&2
        echo "pxq4: set PXQ4_LIB explicitly to one of:" >&2
        ls "${PXQ4_ROOT}/kernels" >&2
        exit 2
    fi

    # A mixed-architecture set cannot be served by one kernel library. Say so plainly
    # rather than silently picking one and producing garbage on half the cards.
    if [ "$(echo "$caps" | wc -l)" -gt 1 ]; then
        echo "pxq4: visible GPUs report more than one compute capability:" >&2
        echo "$caps" | sed 's/^/pxq4:   sm_/' >&2
        echo "pxq4: restrict the container to one architecture with" >&2
        echo "pxq4:   NVIDIA_VISIBLE_DEVICES / --gpus '\"device=0,1\"'" >&2
        echo "pxq4: or pin PXQ4_LIB yourself." >&2
        exit 2
    fi

    case "$caps" in
        6.0|6.1) echo "${PXQ4_ROOT}/kernels/libpxq4_sm60_v10.so" ;;
        7.0|7.2) echo "${PXQ4_ROOT}/kernels/libpxq4_sm70_v10.so" ;;
        *)
            echo "pxq4: no PXQ4 kernel for compute capability ${caps}." >&2
            echo "pxq4: PXQ4 kernels are built for Pascal (6.0/6.1) and Volta (7.0)." >&2
            echo "pxq4: on newer cards use a standard vLLM quantization instead." >&2
            exit 2
            ;;
    esac
}

if [ -z "${PXQ4_LIB:-}" ]; then
    PXQ4_LIB=$(pick_kernel)
fi
[ -f "$PXQ4_LIB" ] || { echo "pxq4: kernel library not found: $PXQ4_LIB" >&2; exit 2; }
export PXQ4_LIB

echo "pxq4: kernel  $PXQ4_LIB" >&2
echo "pxq4: plugin  ${PXQ4_ROOT}/site" >&2

# Bare `bash`/`sh` means the caller wants a shell, not a server -- honour that so the
# image stays debuggable.
case "${1:-}" in
    bash|sh|/bin/bash|/bin/sh) exec "$@" ;;
esac

# Default to the OpenAI-compatible server, but let the caller pass a full command.
if [ "$#" -eq 0 ]; then
    echo "pxq4: no arguments; pass vLLM server flags, e.g. --model /models/foo --quantization pxq4" >&2
    exit 2
fi

exec python3 -m vllm.entrypoints.openai.api_server "$@"
