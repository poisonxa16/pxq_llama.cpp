#!/bin/bash
# Decode/prefill speed harness — the exact procedure behind the per-card table:
#   V100 16GB + PXQ6  = 99.1 t/s decode, ~1920-1960 t/s prefill @ ub2048
#   P100 16GB + PXQ3  = 55.8 t/s decode
#   2xP100    + PXQ6  = 55.7 t/s decode
#   1080Ti 11GB + PXQ2 = 71.4 t/s decode
# Method: llama-server, model fully GPU-resident, 200-token generations, median of 3 runs,
# speed read from the server's own timings.predicted_per_second (no client-side clocking).
# We deliberately do NOT use llama-bench here: the published numbers are end-to-end server
# numbers (the thing you actually get), and the fork's PXQ env-gated kernels are wired
# through the server path. llama-bench (target exists: `cmake --build build --target llama-bench`)
# gives comparable-but-not-identical figures.
#
# Usage: MODEL=/path/PXA-Fusion2-35B-PXQ3.gguf [PORT=8080] [UB=512] ./speed-bench.sh
set -eu
BUILD=${BUILD:-./build}
MODEL=${MODEL:?path to a PXQ gguf}
PORT=${PORT:-8080}
UB=${UB:-512}   # decode is ub-insensitive; prefill was published at UB=2048 (V100)

export LD_LIBRARY_PATH="$BUILD/bin:$BUILD/src:$BUILD/ggml/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1
export PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1

"$BUILD/bin/llama-server" -m "$MODEL" -c 8192 -np 1 -ngl 99 -sm layer -fa on \
  -ctk f16 -ctv f16 -b "$UB" -ub "$UB" --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port "$PORT" &
SRV=$!; trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf "http://127.0.0.1:$PORT/health" | grep -q ok; do sleep 2; done

echo "run,prompt_tps,gen_tps"
for i in 1 2 3; do
  curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d '{
    "messages":[{"role":"user","content":"Write a detailed 300-word essay about GPUs."}],
    "max_tokens":200,"temperature":1.0}' |
  python3 -c 'import json,sys; t=json.load(sys.stdin).get("timings",{}); print(f"'"$i"',{t.get("prompt_per_second"):.1f},{t.get("predicted_per_second"):.1f}")'
done
# Take the MEDIAN gen_tps of the 3 runs. First run includes warmup; median absorbs it.
# Report alongside: GPU model, driver (nvidia-smi), CUDA version, UB, and the tier.
