#!/bin/bash
# qwen4exp: original weights -> BF16 GGUF -> PXQ4 -> six Teslas, n-gram table on CPU.
#
# Source is on <local-path>; output goes to <local-path> -- a DIFFERENT SPINDLE
# (verified via stat -c %d: 2309 vs 2311). Reading and writing hundreds of GiB
# on one spindle cost 9x throughput earlier in this build.
set -euo pipefail

if [ "$(hostname)" != "the box" ]; then echo "WRONG HOST: $(hostname)"; exit 1; fi

SRC=<local-path>
REPO=<local-path>
OUT=<local-path>
BF16=$OUT/Qwen3.8-Flash-Next-BF16.gguf
PXQ4=$OUT/Qwen3.8-Flash-Next-PXQ4.gguf
REF=<local-path>
LOGD=$OUT/logs
mkdir -p "$OUT" "$LOGD"

[ "$(stat -c %d <local-path>)" != "$(stat -c %d <local-path>)" ] || { echo "disk5 and disk7 are the SAME device"; exit 1; }

step() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# ---------------------------------------------------------------- 1. convert
if [ ! -s "$BF16" ]; then
  step "convert HF -> BF16 GGUF"
  df -h <local-path> | tail -1
  # capped so a runaway cannot take the host down (188 GB RAM, zero swap)
  # pxa-memcap.sh <name> <limit> <command...>. 64G is far above the converter's
  # working set (one tensor at a time) but stops a runaway allocation dead --
  # that is what this cap is for. memory.max counts page cache, which is
  # reclaimable, so a big streaming read/write reclaims rather than OOMs.
  /usr/local/bin/pxa-memcap.sh qwen4exp-convert 64G \
      python3 -u "$REPO/convert_qwen4exp.py" "$SRC" \
      --outfile "$BF16" 2>&1 | tee "$LOGD/convert.log"
else
  step "BF16 already present, skipping convert"
fi
ls -l "$BF16"

# ------------------------------------------------------- 2. header vs reference
step "header diff vs shipped reference"
python3 -u "$REPO/convert_qwen4exp.py" "$SRC" --outfile /dev/null \
    --kv-only "$REF" 2>&1 | tail -8

# ------------------------------------------------ 3. behaviour gate on BF16
# The PLE conv read-back bug (task #91) degrades output after ~30 tokens with
# conv ON, in the shipped model too. Gate the CONVERSION with conv OFF so a
# failure here means the conversion is wrong, not that known bug.
step "behaviour gate: BF16 on GPU, PLE conv disabled"
PXA_QWEN4EXP_NO_PLE_CONV=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 \
  "$REPO/build-cuda/bin/llama-cli" -m "$BF16" \
  -ot 'per_layer_token_embd\.weight=CPU' -ngl 99 -ts 7,16,16,16,16,16 \
  -c 2048 -n 64 --temp 0 -no-cnv \
  -p "The capital of France is" 2>&1 | tee "$LOGD/gate-bf16.log" | tail -20

# --------------------------------------------------------------- 4. quantize
if [ ! -s "$PXQ4" ]; then
  step "quantize BF16 -> PXQ4"
  # the row-gather guard refuses panel codecs on per_layer_token_embd and
  # falls back to Q4_K; it is deliberately not overridable by --custom-q.
  "$REPO/build-cuda/bin/llama-quantize" "$BF16" "$PXQ4" PXQ4 12 \
      2>&1 | tee "$LOGD/quantize.log" | tail -30
fi
ls -l "$PXQ4"

step "confirm per_layer_token_embd did NOT get a panel codec"
python3 - <<'PY'
import sys, glob
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader
r = GGUFReader(glob.glob("<local-path>")[0])
for t in r.tensors:
    if "per_layer_token_embd" in t.name:
        print("  %s -> %s  (must NOT be PXQ*/MXFP4)" % (t.name, t.tensor_type.name))
PY

step "DONE -- artifacts in $OUT"
