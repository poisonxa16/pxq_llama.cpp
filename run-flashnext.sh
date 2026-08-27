#!/bin/bash
# Qwen3.8-Flash-Next (qwen4exp) across the six Teslas, n-gram table on CPU.
#
# CARD MAP, in PCI_BUS_ID order (CUDA_VISIBLE_DEVICES orders FASTEST-FIRST otherwise, which
# would silently renumber everything):
#   0 P100  <- also carries a production granite seat, ~8.4 GiB already resident
#   1 P100
#   2 V100
#   3 GTX 1080 Ti  <- PRODUCTION VLM + embeddings. NEVER included.
#   4 V100
#   5 P100
#   6 P100
#
# Why per_layer_token_embd goes to the CPU: it is 160 x 320,001,536 in IQ4_NL, about 28.8 GiB,
# and it is a pure GET_ROWS table - a gather per token per head, no GEMM. Keeping it in host
# RAM costs a PCIe gather and frees nearly a third of the file from VRAM. The remaining ~38 GiB
# of weights sit comfortably in the ~87 GiB the six cards have free.
#
# The -ot pattern is ANCHORED. A loose 'ple' regex would also match blk.1's ple_key,
# ple_conv1d and three F32 ple_norm_* tensors, which must stay on the GPU.
set -u
[ "$(hostname)" != "the box" ] && { echo "WRONG HOST: $(hostname)"; exit 1; }

ENGINE=${ENGINE:-<local-path>}
MODEL=${MODEL:-<local-path>}
NCTX=${NCTX:-4096}
NPRED=${NPRED:-64}
PROMPT=${PROMPT:-"Explain in two sentences what makes a mixture-of-experts model efficient."}

[ -x "$ENGINE" ] || { echo "engine not built yet: $ENGINE"; exit 1; }
[ -f "$MODEL" ]  || { echo "model shard 1 missing: $MODEL"; exit 1; }

# refuse to run if the protected card would be visible
for d in ${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6}; do
  [ "$d" = "3" ] && { echo "REFUSING: card 3 is the production 1080 Ti"; exit 1; }
done

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6}

# card 0 shares with a production seat, so it takes a smaller slice
TS=${TS:-7,16,16,16,16,16}

exec "$ENGINE" \
  -m "$MODEL" \
  -ngl 99 \
  -ts "$TS" \
  -ot 'per_layer_token_embd\.weight=CPU' \
  -c "$NCTX" \
  -n "$NPRED" \
  -t 16 \
  --no-warmup \
  -p "$PROMPT" \
  "$@"
