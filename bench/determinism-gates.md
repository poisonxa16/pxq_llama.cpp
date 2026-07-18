# PXQ determinism & correctness gates (G1/G2/G3)

Every PXQ kernel and quantizer change ships only after passing a fixed gate battery. These are
the gates behind the phrase "bit-exact fast kernels" in the README — publishing them because
kernel speed claims without a determinism story are vibes.

## Quantizer gates

- **Q-G1 — byte-parity:** `pxa-bench/pxq6_ref.cpp` (a standalone build of the *production*
  converter) must BYTE-MATCH the golden numpy implementation (`pxa-bench/pxqu_golden.py`) on
  both the quantized bytes AND the dequantized values, for every tier (PXQ2/PXQ3/PXQ6).
- **Q-G2 — wrel reproduction:** the C quantizer's relative weight error must equal the numpy
  harness (`pxa-bench/pxqu_wrel.py`) on a frozen 36-slice rng-seed-42 protocol to ±1e-4, per
  tier. This pins quant quality to a reference before any ppl run.

## Kernel gates (the `PXA_PXQ6_*` fast paths)

Each env-gated fast kernel is proven **memcmp bit-exact** against the baseline kernel on real
model tensors before it defaults on — same FMA order, same accumulation chains:

| gate | kernel | proof |
|---|---|---|
| K1 `PXA_PXQ6_KSPLIT` | decode gate/up K-split + persistent workspace reducer | memcmp bit-exact, all formats |
| K2 `PXA_PXQ6_PAIRLUT` / `PXA_PXQ6_VECX` | byte-pair LUT / float4 activation loads | memcmp bit-exact |
| K3 `PXA_PXQ6_GUFUSE` / `PXA_PXQ6_SCATFUSE` | fused up+gate GEMM + GLU epilogue / fused MoE scatter | memcmp bit-exact vs the unfused pipeline |
| K4 `PXA_PXQ6_RAGTAIL` | ragged-tail FMA skip | memcmp bit-exact (skipped work was store-masked) |
| K5 `PXA_PXQ6_PIPE` | sm_60 2-stage register prefetch | memcmp bit-exact (identical arithmetic DAG) |

End-to-end: temp-0 generation SHA over a fixed prompt battery is **identical** with all fast
paths on vs all off, on sm_60 (P100), sm_61 (1080 Ti) and sm_70 (V100).

## G3 — the non-bit-exact paths (declared, gated differently)

- **CPU↔CUDA dequant:** top-20 logprob parity + identical temp-0 generation at `-ngl 0` vs
  `-ngl 99` (bit-exactness across backends is not claimed — parity of outcomes is).
- **`PXA_PXQ6_WMMA` (experimental V100 tensor-core prefill):** deterministic but NOT bit-exact
  (~1e-6 output deltas by design). It stays default-OFF and is ppl-regated separately before
  any future enable. If you benchmark with it on, say so.

## Reproducing

The gate harnesses live in `pxa-bench/`: `pxq6_ref.cpp`, `pxq6_test.cu` (device memcmp
battery), `pxqu_golden.py`, `pxqu_wrel.py`, `pxqu_ref.cpp`. Build notes are at the top of each
file. Run them against any PXQ GGUF tier; a failure of any bit-exact gate is a release-blocking
bug — report it.
