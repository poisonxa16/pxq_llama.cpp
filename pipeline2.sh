#!/bin/bash
# Continuation after the BF16 GGUF exists.
#
# Why this is separate from pipeline.sh: that script gated behaviour on the
# BF16 file across the six Teslas. That can never pass -- BF16 non-PLE weights
# are ~258 GiB against ~87 GiB of usable VRAM. The BF16 file gets a STRUCTURAL
# check instead; behaviour is gated on the PXQ4 artifact, which actually fits.
set -euo pipefail
if [ "$(hostname)" != "the box" ]; then echo "WRONG HOST: $(hostname)"; exit 1; fi

REPO=<local-path>
OUT=<local-path>
BF16=$OUT/Qwen3.8-Flash-Next-BF16.gguf
PXQ4=$OUT/Qwen3.8-Flash-Next-PXQ4.gguf
REF=<local-path>
LOGD=$OUT/logs
mkdir -p "$LOGD"
step() { echo; echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

step "BF16 structural check"
python3 - "$BF16" "$REF" <<'PY'
import sys
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader
mine, ref = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
mt = {t.name: t for t in mine.tensors}
rt = {t.name: t for t in ref.tensors}
print(f"  tensors: mine={len(mt)} reference-shard={len(rt)}")
missing = [n for n in rt if n not in mt]
print(f"  present in reference shard but missing here: {len(missing)}")
for n in missing[:10]:
    print("     ", n)
bad = 0
for n, t in rt.items():
    if n in mt and tuple(mt[n].shape) != tuple(t.shape):
        print(f"  SHAPE {n}: mine={tuple(mt[n].shape)} ref={tuple(t.shape)}"); bad += 1
print(f"  shape mismatches: {bad}")
ple = mt.get("per_layer_token_embd.weight")
if ple is not None:
    print(f"  per_layer_token_embd: shape={tuple(ple.shape)} type={ple.tensor_type.name}")
sys.exit(1 if (bad or missing) else 0)
PY

step "quantize BF16 -> PXQ4"
if [ ! -s "$PXQ4" ]; then
  # The row-gather guard refuses panel codecs on per_layer_token_embd and falls
  # back to Q4_K; deliberately not overridable by --custom-q.
  /usr/local/bin/pxa-memcap.sh qwen4exp-quant 64G \
    "$REPO/build-cuda/bin/llama-quantize" "$BF16" "$PXQ4" PXQ4 12 \
      2>&1 | tee "$LOGD/quantize.log" | tail -40
fi
ls -l "$PXQ4"

step "confirm the row-gather table did NOT get a panel codec"
python3 - "$PXQ4" <<'PY'
import sys
sys.path.insert(0, "<local-path>")
from gguf import GGUFReader
r = GGUFReader(sys.argv[1])
bad = 0
for t in r.tensors:
    if "per_layer_token_embd" in t.name or t.shape[-1] >= 1_000_000:
        ty = t.tensor_type.name
        flag = "REFUSED-OK" if not (ty.startswith("PXQ") or ty == "MXFP4") else "PANEL CODEC ON A GATHER TABLE"
        print(f"  {t.name} -> {ty}   [{flag}]")
        if flag != "REFUSED-OK": bad += 1
sys.exit(1 if bad else 0)
PY

# MODEL/NCTX/NPRED/PROMPT are environment variables for run-flashnext.sh, not
# arguments -- anything positional gets appended to the llama-cli command line.
# NCTX=2048 keeps indexer_top_k=2048 selecting every token, so dense == sparse
# and the gate tests the conversion rather than the unbuilt QSA sparsity.
GATE_PROMPT="Name the capital city of each country, one per line.
France:"

step "behaviour gate: PXQ4 on six Teslas, n-gram table on CPU, PLE conv OFF"
# conv OFF isolates the conversion from the known read-back bug (task #91).
PXA_QWEN4EXP_NO_PLE_CONV=1 MODEL="$PXQ4" NCTX=2048 NPRED=80 PROMPT="$GATE_PROMPT" \
  "$REPO/run-flashnext.sh" --temp 0 -no-cnv 2>&1 \
  | tee "$LOGD/gate-pxq4-noconv.log" | tail -40

step "same, PLE conv ON (expected to degrade after ~30 tokens -- task #91)"
MODEL="$PXQ4" NCTX=2048 NPRED=80 PROMPT="$GATE_PROMPT" \
  "$REPO/run-flashnext.sh" --temp 0 -no-cnv 2>&1 \
  | tee "$LOGD/gate-pxq4-conv.log" | tail -40

step "DONE"
