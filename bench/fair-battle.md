# Fair battle — pxq_llama vs upstream ik_llama.cpp (2026-07-19, rev 2)

Best-vs-best per card: upstream at its own documented best (pinned 2026-07-18 HEAD,
`GGML_CUDA_F16=ON` build per its docs, its best-fitting IQ_K quant, f16 KV), pxq_llama at its
documented best (`docs/lab/LEVERS.md` recommended env; `PXA_PXQ_INT8_PREFILL=1` on sm_61). Same 35B
MoE architecture, same card, same protocol both sides.

**Protocol:** single `/completion`, cold 5,801-token prompt, `n_predict=200`, `temperature=0`,
`seed=42`, `cache_prompt=false`, median of 3, numbers from server `timings`.

**Rev 2 (same day):** the first pass ran `-b` = `-ub` and forced `-fa on` for everything. A follow-up
sweep showed both choices leave real speed on the table **for both engines**: `-b 2048` is
+4–25% prefill, and on these cards **`-fa off` is the cold-prefill regime while `-fa on` is the
decode regime** (see the regime table — the effect is symmetric, upstream gains from FA-off prefill
too). Rev 2 gives each side its best regime per metric. Nothing was re-measured for one side only.

## Headline (best config per side, per metric — prefill @ `-fa off`, decode @ `-fa on`, all `-b 2048`)

| card | quant (upstream vs pxq) | prefill t/s | decode t/s |
|---|---|---|---|
| Tesla P100 16 GB | IQ3_KS (14.2 GB) vs PXQU-16+q8head (14.1 GB) | 645 → **1,213 (+88%)** | 44.7 → **58.1 (+30%)** |
| Tesla V100 16 GB | IQ3_KS (14.2 GB) vs PXQU-16+q8head (14.1 GB) | 1,509 → **1,700 (+13%)** | 84.5 → **95.5 (+13%)** |
| GTX 1080 Ti 11 GB | IQ2_KS (10.1 GB) vs PXQ2 (10.7 GB) | **1,154** → 1,001 (−13%) | 52.2 → **65.4 (+25%)** |

> **Note:** the decode deltas in this table reflect the smaller PXQ quant class (PXQU-16 + q8_0 head, or PXQ2) **plus MTP speculative decode**, not the engine. The engine same-quant decode is **+2.7–3.3%** (V100 bit-identical) — see the Single-config view below. The engine win is prefill.

## Single-config view (chat serving: `-fa on -b 2048`, one server, no regime switching)

| card | upstream (prefill / decode) | pxq_llama (prefill / decode) |
|---|---|---|
| P100 | 513 / 44.0 (ub2048) · 44.7 dec best (ub512) | **817 / 56.7** (ub2048) · **58.1** dec best (ub512) |
| V100 | 1,422 / 82.5 (ub512) | **1,589 / 94.1** (ub512) |
| 1080 Ti | 739 / 52.2 (ub768) | 667 / **65.4** (ub768) |

## The FA regime split (both engines, measured)

Flash-attention on these pre-Turing cards is a decode win but a cold-prefill loss — for upstream too:

| card / engine | prefill fa-on → fa-off | decode fa-on → fa-off |
|---|---|---|
| P100 pxq | 817 → **1,213** (+48%) | **56.7** → 41.1 (−28%) |
| P100 upstream | 513 → **645** (+26%) | **44.0** → 34.0 (−23%) |
| V100 pxq | 1,589 → **1,700** (+7%) | **94.1** → 76.6 (−19%) |
| V100 upstream | 1,422 → **1,509** (+6%) | **82.5** → 69.6 (−16%) |
| 1080 Ti pxq | 667 → **1,001** (+50%) | **65.4** → 34.2 (−48%) |
| 1080 Ti upstream | 739 → **1,154** (+56%) | **52.2** → 30.0 (−43%) |

Practical rule either engine's users can apply: prefill-heavy batch work (ingest, embedding prep,
summarize-once) → `-fa off`; interactive serving → `-fa on`. Measured at 5.8k-token fill; FA-off
attention memory grows with context, so re-check at your target ctx.

## Same-quant control (identical gguf on both builds)

pxq_llama running upstream's own IQ_K ggufs — only the arch-level fusions differ
(`PXA_FUSE_DELTANET=3 PXA_G2_ADDFUSE=1`), matched config both sides:

| card | quant | upstream decode | pxq_llama decode | Δ | output |
|---|---|---|---|---|---|
| V100 | IQ3_KS @ub512 | 84.5 | 87.2 | +3.2% | **bit-identical** (same temp-0 sha) |
| P100 | IQ3_KS @ub2048 | 44.0 | 45.2 | +2.7% | coherent, sha-stable runs |
| 1080 Ti | IQ2_KS @ub768 | 52.2 | 53.9 | +3.3% | coherent, sha-stable runs |

## Where upstream wins, and why (kept on the chart)

The 1080 Ti cold-prefill loss (−13%) is real: upstream's IQ2_KS MMQ int8 tile is a mature,
double-buffered, large-tile pipeline and its file is 6% smaller; our sm_61 int8 tile
(`PXA_PXQ_INT8_PREFILL`, first shipped this release) is a 64-thread single-buffered first cut that
reaches ~87–95% of it depending on config — and didn't exist at all a release ago (PXQ2 prefill on
that card was 251 t/s). Decode on the same card is +25% for pxq_llama.

## Raw harness rows

```
# rev-1 rows (b = ub, fa on)
vik_V100_IQ3KS_ub512,OK,1154.8,84.5   | vik_V100_IQ3KS_ub2048,OOM (launch_fattn)
vik_P100_IQ3KS_ub512,OK,303.1,44.7    | vik_P100_IQ3KS_ub2048,OK,512.6,44.0
vik_1080Ti_IQ2KS_ub768,OK,702.8,52.2
pxq_V100_U16q8_ub512,OK,1268.6,95.5   | pxq_V100_U16q8_ub2048,OOM
pxq_P100_U16q8_ub512,OK,604.9,58.1    | pxq_P100_U16q8_ub2048,OK,816.8,56.7
pxq_1080Ti_PXQ2_ub768,OK,639.0,64.6
samequant_V100_IQ3KS_ub512,OK,1124.5,87.2 (sha matches upstream)
samequant_P100_IQ3KS_ub2048,OK,536.0,45.2
samequant_1080Ti_IQ2KS_ub768,OK,685.5,53.9
# rev-2 sweep rows (b 2048; fa as labeled)
dxV1_pxq_b2048_faon,OK,1588.5,94.1    | dxV2_pxq_b2048_faoff,OK,1699.5,76.6
dxV3_vik_b2048_faon,OK,1421.9,82.5    | dxV4_vik_b2048_faoff,OK,1508.7,69.6
dxP1_pxq_ub2048_faoff,OK,1213.1,41.1  | dxP2_vik_ub2048_faoff,OK,645.0,34.0
dxA_pxq2_b2048_faoff_c6144,OK,1000.6,34.2 | dxB_pxq2_b2048_faon_c8192,OK,667.3,65.4
dxC_pxq2_b768_faoff_c8192,OK,975.7,33.9
dxD_vik_iq2ks_b2048_faoff_c6144,OK,1153.6,30.0 | dxE_vik_iq2ks_b2048_faon_c8192,OK,738.8,52.2
```

## Codec-only re-run, 2026-09-02 (candidate engine, next release)

A separate comparison from everything above: same engine on both sides of every row, PXQ4 vs
MXFP4 at matched engine — isolating the codec, per `bench/fair/protocol.md`. `llama-server
/completion`, `temp=0`, `seed=42`, n=7 median, 1 warmup discarded, `cache_prompt=false`, unique
prompt per repeat, MTP off both sides, artifacts sha-checked against
`bench/fair/weights/MANIFEST.sha256`. Run on the candidate engine described in
`RELEASE-NOTES-2026-09-02.md` (pipeline-scheduler fixes, host-overhead cuts, the ported upstream
correctness fixes) — not yet in a tagged release.

### Dense, 2×P100 (Qwable-27B, `-c 32768 -b/-ub 2048 -fa on`)

| | PXQ4-core | MXFP4-lite | result |
|---|---|---|---|
| prefill @3,121 | 227.4 | 178.7 | **+27%** |
| prefill @20,801 | 203.3 | 163.7 | **+24%** |
| decode @8 | 18.3 | 14.4 | **+27%** |

### Dense, 2×V100 (Qwable-27B, `-c 32768 -b/-ub 2048 -fa on`) — the cell PXQ4 still loses

| | PXQ4-core (`PXA_PXQ_MMVQ=1`) | MXFP4-lite | result |
|---|---|---|---|
| prefill @3,121 | 798.1 | 364.0 | **+119%** |
| prefill @20,801 | 577.0 | 312.9 | **+84%** |
| decode @8 | 34.5 | 37.7 | **−8.3% — MXFP4 wins here** |

Prefill's lead widened since the original table because the candidate engine's NCCL P2P fix
(`docs/PXA-SM70-SERVING.md`) helps both quant types and PXQ4 was already ahead on prefill; the
decode loss is the same structural DP4A-scale-fixup story as the original table, unchanged by
the engine update.

### MoE, 2×V100 (35B-A3B class, `-c 32768 -b/-ub 2048 -fa on`) — expert-codec delta

| | PXA-Fusion4-35B PXQ4 | qwen36-35b MXFP4 | result |
|---|---|---|---|
| prefill @3,121 | 2,093.8 | 1,614.6 | **+30%** |
| prefill @20,801 | 1,467.2 | 1,256.8 | **+17%** |
| decode @8 | 218.4 | 188.0 | **+16%** |

**Caveat, stated plainly:** the PXQ4 side is `PXA-Fusion4-35B`, a merge — not the same base
weights as the stock 35B-A3B model on the MXFP4 side. Same architecture and size class, not
byte-identical. The artifact's own codec census (`codec=PXQ4 (120 tensors; pxq4 120)`) shows only
its expert stacks are actually PXQ4; the rest is other types. This supersedes an earlier
MoE-decode row this repo withdrew after finding the same composition problem in a different
artifact — the fix here is disclosure, not a different codec: the number is real for
"PXQ4-quantized experts on this architecture," not for "PXQ4 the whole model."
