# Fair battle — pxq_llama vs upstream ik_llama.cpp (2026-07-19)

Best-vs-best per card: upstream at its own documented best (pinned 2026-07-18 HEAD,
`GGML_CUDA_F16=ON` build per its docs, its best-fitting IQ_K quant, `-fa on`, f16 KV), pxq_llama at
its documented best (`docs/LEVERS.md` recommended env; `PXA_PXQ_INT8_PREFILL=1` on sm_61). Same 35B
MoE architecture, same card, same protocol both sides.

**Protocol:** single `/completion`, cold 5,801-token prompt, `n_predict=200`, `temperature=0`,
`seed=42`, `cache_prompt=false`, median of 3, numbers from server `timings`. Harness:
`bench/speed-bench.sh`-compatible; runs below are verbatim CSV from the harness.

## Headline (best config per side, per metric)

| card | quant (upstream vs pxq) | prefill t/s | decode t/s |
|---|---|---|---|
| Tesla V100 16 GB | IQ3_KS (14.2 GB) vs PXQU-16+q8head (14.1 GB) | 1,155 → **1,269 (+10%)** | 84.5 → **95.5 (+13%)** |
| Tesla P100 16 GB | IQ3_KS (14.2 GB) vs PXQU-16+q8head (14.1 GB) | 513 → **817 (+59%)** | 44.7 → **58.1 (+30%)** |
| GTX 1080 Ti 11 GB | IQ2_KS (10.1 GB) vs PXQ2 (10.7 GB) | **703** → 639 (−9%) | 52.2 → **64.6 (+24%)** |

Notes:
- V100: `ub2048` OOMs on **both** sides (16 GB card, ~14 GB model) — `ub512` is the best fitting
  config for both. P100: prefill best at `ub2048` for both, decode best at `ub512` for both;
  each side gets its best per metric.
- 1080 Ti: upstream genuinely wins cold prefill by ~9% there (its IQ2_KS MMQ path is strong on
  dp4a cards); pxq_llama wins decode by 24%. Before the int8 tile (`PXA_PXQ_INT8_PREFILL=1`,
  this release) the PXQ prefill on that card was 251 t/s.

## Same-quant control (identical gguf on both builds)

pxq_llama running upstream's own IQ_K ggufs — only the arch-level fusions differ
(`PXA_FUSE_DELTANET=3 PXA_G2_ADDFUSE=1`):

| card | quant | upstream decode | pxq_llama decode | Δ | output |
|---|---|---|---|---|---|
| V100 | IQ3_KS @ub512 | 84.5 | 87.2 | +3.2% | **bit-identical** (same temp-0 sha) |
| P100 | IQ3_KS @ub2048 | 44.0 | 45.2 | +2.7% | coherent, sha-stable runs |
| 1080 Ti | IQ2_KS @ub768 | 52.2 | 53.9 | +3.3% | coherent, sha-stable runs |

Prefill on the same quant is within ±5% both ways (536 vs 513 P100; 1,125 vs 1,155 V100;
686 vs 703 1080 Ti) — the prefill headline above comes from the PXQ tier's fused paths, not from
penalizing the baseline.

## Raw harness rows

```
vik_V100_IQ3KS_ub512,OK,1154.8,84.5,prefills=[1133.9,1154.8,1157.6],decodes=[84.53,84.44,84.69],sha_stable=True
vik_V100_IQ3KS_ub2048,REQUEST_FAILED (CUDA out of memory, launch_fattn)
vik_P100_IQ3KS_ub512,OK,303.1,44.7,prefills=[303.1,302.7,303.1],decodes=[44.70,44.84,44.73],sha_stable=True
vik_P100_IQ3KS_ub2048,OK,512.6,44.0,prefills=[511.0,512.6,512.8],decodes=[44.02,44.11,43.98],sha_stable=True
vik_1080Ti_IQ2KS_ub768,OK,702.8,52.2,prefills=[704.3,702.1,702.8],decodes=[52.15,52.21,51.84],sha_stable=True
pxq_V100_U16q8_ub512,OK,1268.6,95.5,prefills=[1251.3,1268.6,1270.2],decodes=[95.13,95.49,95.51]
pxq_V100_U16q8_ub2048,OOM (16 GB card, 14.1 GB model — compute buffer does not fit)
pxq_P100_U16q8_ub512,OK,604.9,58.1,prefills=[602.8,606.1,604.9],decodes=[58.04,58.10,58.08]
pxq_P100_U16q8_ub2048,OK,816.8,56.7,prefills=[807.3,816.8,818.1],decodes=[56.66,56.72,56.94]
pxq_1080Ti_PXQ2_ub768,OK,639.0,64.6,prefills=[641.3,636.1,639.0],decodes=[64.50,64.64,64.65]
samequant_V100_IQ3KS_ub512,OK,1124.5,87.2,decodes=[87.23,87.18,87.38],sha=matches upstream run
samequant_P100_IQ3KS_ub2048,OK,536.0,45.2,decodes=[45.08,45.26,45.16]
samequant_1080Ti_IQ2KS_ub768,OK,685.5,53.9,decodes=[53.62,54.08,53.94]
```
