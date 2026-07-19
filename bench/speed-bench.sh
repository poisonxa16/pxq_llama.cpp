#!/bin/bash
# Decode/prefill speed harness — the exact procedure behind the per-card table:
#   V100 16GB + PXQU-16 (q8_0 head) = 101.3 t/s decode, ~1800-1900 t/s prefill @ ub2048
#   P100 16GB + PXQU-16 (q8_0 head) = 62.4 t/s decode
#   P100 16GB + PXQ3  = 55.8 t/s decode
#   2xP100    + PXQ4  = 55.7 t/s decode  (the 4-bit flagship, formerly PXQ6, is 18.7 GB: it does NOT fit one 16 GB card)
#   1080Ti 11GB + PXQ2 = 71.4 t/s decode (prefill on 11 GB: use UB=768; ub2048 cannot allocate;
#                         opt-in PXA_PXQ_INT8_PREFILL=1 prefill 251 -> 709 t/s, decode untouched)
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
UB=${UB:-512}   # decode is ub-insensitive; prefill published at UB=2048 (16 GB cards) / UB=768 (1080 Ti 11 GB)

export LD_LIBRARY_PATH="$BUILD/bin:$BUILD/src:$BUILD/ggml/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1
export PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1
export PXA_FUSE_DELTANET=3   # 2026-07-19: bit-exact DeltaNet decode fusion, +3-4% decode P100/V100
export PXA_G2_ADDFUSE=1      # 2026-07-19 late: bit-exact residual-add fusion, +1.9% V100 / +1.2% P100

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
