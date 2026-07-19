# PXA levers — every shipping `PXA_*` environment variable

The definitive reference for what each knob does, its default, its **measured** effect (with the
config the number came from), and its correctness gate class. Two gate classes appear below:

- **bit-exact** — flag ON produces byte-identical logits/output to flag OFF (proven by temp-0
  output sha identity and/or memcmp kernel gates; see `bench/determinism-gates.md`).
- **G3-class** — deterministic but *not* bit-identical to the reference path (different but
  equally valid arithmetic order/precision); gated by temp-0 coherence + semantic equivalence +
  top-k logit spot-checks before shipping.

Bench protocol for every decode number unless stated otherwise: `llama-server`, model fully
GPU-resident, 200-token temp-0 generations, median of ≥3, speed read from the server's
`timings.predicted_per_second` (`bench/speed-bench.sh`); paired interleaved A/B with a ≤1%
baseline-spread guard. "U16-q8out" = `fusion2-35b-U16-q8head.gguf` (PXQU-16 + q8_0 output head),
the artifact behind the published P100/V100 decode rows.

## 1. Format enables (required to run PXQ files)

| var | default | what it does | notes |
|---|---|---|---|
| `PXA_PXQ6` | off | enables the fused CUDA kernel family for the 4-bit tier (PXQ4, formerly PXQ6) and PXQ4-HQ; `=0` keeps the dequant→cuBLAS fallback | set it for any PXQ model; the fallback is correct but ~2× slower decode |
| `PXA_PXQ2` | off | enables the 2-bit (LM4) kernel family | set for PXQ2 / mixed PXQU files |
| `PXA_PXQ3` | off | enables the 3-bit (LM8 bit-plane) kernel family | set for PXQ3 / mixed PXQU files |
| `PXA_PXQ5` | off | enables the legacy PXQ5 kernel family | legacy tier; only for PXQ5 files |
| `PXA_PXQ4` | off | enables the legacy MXFP4-repack (PXQ4-LEGACY, type id 250) fused kernels | legacy tier |

Set all of `PXA_PXQ6/2/3` for a PXQ-UNIVERSAL (mixed-tier) file.

## 2. Recommended-ON performance levers (all bit-exact)

| var | default | what it does | measured | verdict |
|---|---|---|---|---|
| `PXA_PXQ6_KSPLIT` | off | K1: splits the gate/up decode GEMV over K-segments with a fixed-order workspace reducer (more blocks in flight on small-R launches) | part of the published kernel set (all published decode numbers use it) | **ON** |
| `PXA_PXQ6_VECX` | off | K2b: float4 activation loads in the decode mmv inner loop | part of the published kernel set | **ON** |
| `PXA_PXQ6_GUFUSE` | off | K3a: fuses the up+gate GEMV pair + GLU epilogue into one kernel | part of the published kernel set | **ON** |
| `PXA_PXQ6_SCATFUSE` | off | K3b: fuses the MoE scatter/accumulate epilogue | part of the published kernel set | **ON** |
| `PXA_PXQ6_RAGTAIL` | off | K4: skips store-masked FMA work on ragged tail tiles | part of the published kernel set | **ON** |
| `PXA_FUSE_DELTANET` | 0 | `=3` fuses the DeltaNet (linear-attention) decode glue-kernel chain | **+3.7% decode P100 U16** (57.2→60.2→62.4 with q8 head, published config) | **ON (=3)** |
| `PXA_G2_ADDFUSE` | off | G2-F4: re-enables ADD+FUSED_RMS_NORM pair fusion (ne0≥256) + fuses the residual add into the MUL_MULTI_ADD epilogue (experts first, residual last — bitwise-commutative-identical) | **+1.9% decode V100** (100.1→102.0, U16-q8out quiet-window) / **+1.2% P100** (62.25→63.0, same protocol) | **ON** |

The published per-card numbers in `bench/README.md` were measured with section-2 rows 1–6 ON
(`ADDFUSE` landed after; its gain stacks on top — see the cookbook).

## 3. Available but NOT in the recommended env (measured no-gain on the published configs)

| var | default | what it does | measured | verdict |
|---|---|---|---|---|
| `PXA_PXQ6_PAIRLUT` | off | K2a: 256-entry float2 byte-pair LUT for code expansion (bit-exact) | **+0.1% P100 U16-q8out** (62.3 vs 62.3) — an earlier +4.8% reading came from a 2×P100 `-ts 1,1` 4-bit-flagship config and does not transfer | OFF (harmless; config-specific) |
| `PXA_PXQ6_PIPE` | off | K5: sm_60 2-stage register prefetch in the decode mmv (bit-exact) | no measured gain on published configs | OFF |
| `PXA_PXQ6_KSPLIT_GEN` | off | K1 variant: K-split on the generic (non-fused) mmv path; G3-class | superseded by the fused path | OFF |
| `PXA_PXQ5_FAST` | off | legacy PXQ5 fast-path variant | legacy | OFF |
| `PXA_MMVQ_MOE_NWARPS` | 1 | forces 2/4 warps in the routed MoE GEMV (mmvq) | "prove faster + shadow-clean before defaulting" — no win recorded | leave unset |

## 4. Opt-in levers with trade-offs (read before enabling)

| var | default | what it does | measured | gate class |
|---|---|---|---|---|
| `PXA_PXQ_INT8_PREFILL` | off | `=1`: routes PXQ prefill GEMMs through an int8 dp4a MMQ-style tile on **sm_61 only** (codes→s8 via the snapped book, per-16/per-8 sub-scales folded into a per-tile fp32 rescale, activations q8-per-32). `=2` lifts the arch gate (TEST — sm_60 dp4a is emulated, never ship there) | **1080 Ti PXQ2 cold 5.8k-token prefill 251→709 t/s (+182%)**, 95% of the native-MMQ ceiling; decode byte-untouched (66.4/66.5) | **G3-class** (temp-0 64-tok continuation sha-identical in our gates; top-1 logits identical every spot-check; tail top-5 order can shift at p≈0.015). Flag OFF = byte-identical dispatch |
| `PXA_PXQ6_WMMA` | 0 | experimental V100 tensor-core prefill path (fp16 fragments; `=1` fp32-accum, `=2` fp16-accum twin); auto-guarded to the 4-bit tier, cc 7.0 only | **+0.97% e2e prefill** after the 256-thread launch fix — not worth it | G3-class; experimental, keep OFF |
| `PXA_VOLTA_CUBLAS_NE11` | 0 | on sm_70, routes dense quantized GEMMs with `ne11 ≥ N` to fp16 cuBLAS (tensor cores) instead of DP4A MMQ | +6.5% prefill in one internal 35B config (`=64`); decode untouched (tiny ne11 stays MMQ) | G3-class; tune per model or leave unset |
| `PXA_VOLTA_CUBLAS_ID_NE11` | 0 | same idea for the routed (mul_mat_id) expert GEMMs on sm_70 | measured a LOSS on the configs tried — MXFP4 experts already ride fast MMQ | leave unset |
| `PXA_P100_FP16_GEMM` | on | sm_60 dense-GEMM prefill path: fp16 dequant + GemmEx-16F (GP100 has full-rate fp16) | `=0` rolls back to fp32 SGEMM (the old, slower path) | G3-class; ON is the shipped default |
| `PXA_MXFP4_DEQ_V2` | on | fast coalesced smem-table MXFP4→f16 dequant kernel | 150→397 GB/s dequant, bit-identical output; `=0` rolls back | bit-exact |
| `PXA_PXQ6_FORCE_PREFILL` | off | TEST ONLY: bypasses the sm_60/70 prefill arch gate so correctness A/Bs can run on other archs | correctness testing only | never in production |

## 5. Documented dead ends (kept for reproducibility — measured no-gain or loss, default OFF)

| var | what it was | measured outcome |
|---|---|---|
| `PXA_G2_REDFUSE` | G2-F1: absorb the gateup ksplit-reduce + GLU into the down-mmv prologue | **−0.8% decode V100** — the 8× workspace re-read + GLU recompute costs more than the reduce it removes. KILL |
| `PXA_G2_NORMFUSE` | G2-F3: fused rms-norm emits a q8_1 sidecar so the mmvq chain skips `quantize_q8_1` | no measurable gain over ADDFUSE alone (P100 63.0→62.95); bit-exact, kept OFF |
| `PXA_G2_QUANTFOLD` | G2-F2: the DeltaNet out-gate kernel emits the q8_1 sidecar for `linear_attn_out` | same — no gain over ADDFUSE on the measured configs; bit-exact, kept OFF |
| `PXA_PASCAL_DMMV` | alternate Pascal DMMV dispatch experiment | measured loss; documented dead end |
| `PXA_CUDA_GRAPH_V2` (+`PXA_CUDA_GRAPH_LOG`) | keyed whole-token CUDA-graph replay cache (replay provably fires every token, byte-identical output) | **−2 to −4% decode V100** — decode is ~90% GPU-busy; launch overhead was a tracer artifact. Instrumentation honesty, not a speed lever |
| `PXA_CUDA_GRAPH_BATCH` / `_MOE` / `_LRU` / `_REARM` / `_MAX_NY` / `PXA_CUDA_GRAPHS_PASCAL` | earlier opt-in CUDA-graph capture experiments for small multi-token batches | no shipped gain; diagnostic lineage of GRAPH_V2 |

## 6. Speculative-decode / MTP / server levers (engine features, model-dependent)

| var | default | what it does |
|---|---|---|
| `PXA_MTP_LAZY_WARMUP` | off | `=1`: skips the per-prompt-batch MTP companion warmup and stops flagging every prompt token as an MTP output (large prefill win on MTP models; temp-0 bit-identical; eager mode byte-unchanged when unset) |
| `PXA_MTP_ADAPTIVE` (+`_K`) | off | adaptive draft-length cap for MTP speculation (per-slot acceptance feedback) |
| `PXA_NGRAM_RESET_STREAK` | 3 | streak threshold for the ngram-speculator map reset; `0` = never reset on acceptance (helps varied-writer models keep the map warm) |
| `PXA_NP_SPEC_GATE` | off | opt-in gate for speculation under np>1 (shelved feature; leave unset) |
| `PXA_SPEC_RELAXED` (+`_PMIN`) | off | relaxed draft-acceptance experiment (G3-class; not recommended) |
| `PXA_SHARED_MTP_BATCH_COMMIT` | on | batches MTP commit work across slots; `=0` restores fully-serial behavior (rollback knob) |
| `PXA_MOE_FASTTG_MAX_NY` | 8 | max verify-batch Ny that stays on the per-token fast-TG path; `=1` routes Ny>1 MTP verify batches to the expert-grouped batched path (weights read once per traversal) |
| `PXA_MOE_GROUPED` / `_VERIFY` / `PXA_MOE_BATCHED_VERIFY` | off | A1 expert-grouped batched-MoE verify kernels + shadow-verify harness (G3-class; incompatible with graph capture) |
| `PXA_PROMPT_INTERLEAVE` | on | co-decodes resident slots while another slot prefills; `=0` reverts to serialize-behind-decode (ops kill-switch) |
| `PXA_HEALTH_STALL_MS` | 60000 | `/health` reports stalled if a queued probe can't be served within the deadline (`0` = off) — keeps health honest instead of parking an HTTP worker |
| `PXA_WEDGE_EXIT_MS` | 0 | in-server watchdog: one `llama_decode` stuck longer than this (×3 checks) → process exit so a supervisor can restart (`0` = off) |

## 7. Multi-GPU / partition levers

| var | default | what it does |
|---|---|---|
| `PXA_EXPERT_SHARD` | unset | comma list of device indices: shards expert tensors of the listed home devices across the group (consumed by the CUDA MoE up/gate shard branch). Unset = bit-identical stock placement |
| `PXA_REPLICATE_RECURRENT` | off | replicates recurrent (DeltaNet) state full-head per device instead of head-splitting — trades memory for no cross-device reduce |
| `PXA_REDUCE_CAPTURE` | off | allows the cross-device reduce inside graph capture once the per-device events exist |

## 8. Quantizer inputs (build-time, `llama-quantize`)

| var | what it does |
|---|---|
| `PXA_PXQ6_BOOK` / `PXA_PXQ2_BOOK` / `PXA_PXQ3_BOOK` / `PXA_PXQ5_BOOK` | override the frozen codebook (path to a book file) — for lab experiments; shipped books are compiled in and sha-pinned |
| `PXA_PXQ6_SUB` / `_SUB_HQ` / `PXA_PXQ2_SUB` / `PXA_PXQ3_SUB` | override the sub-scale LUTs (lab) |
| `PXA_PXQ6_ANCHOR_FIT` / `PXA_PXQ2_ANCHOR_FIT` / `PXA_PXQ3_ANCHOR_FIT` | anchor-fit strategy toggles in the native quantizers (lab; defaults are the shipped, gate-proven settings) |
| `PXA_PXQU_DIR` | directory for `--pxq-universal` preset `.tiers` files (default `pxa-bench/pxq-universal/`; presets are also baked into the binary) |

## 9. Diagnostics (no effect on results; may cost speed — leave OFF in production)

`PXA_EXPERT_LOG` (per-request MoE expert-routing histograms, np1 only), `PXA_PROFILE` /
`PXA_PROFILE_EVERY` / `PXA_CKPT_PROF` / `PXA_DECODE_WALL_DBG` / `PXA_SHARD_TIMING` (timing
instrumentation), `PXA_GRAPH_DUMP` (graph node dump), `PXA_MOE_DEBUG`, `PXA_SPEC_DBG` /
`PXA_DRAFT_DBG` / `PXA_MTP_DBG` (speculation tracing), `PXA_BIGCOPY_DBG` (large D2H copy
tracing), `PXA_OP_SYNC_CHECK` / `PXA_SYNC_BISECT` (per-op sync bisection for debugging async
faults), `PXA_CHATPARSE_EVERY` (chat-parse cadence).

---

*Numbers cite their config; anything without a number has no published A/B on the shipped
configs — treat its default as the tested state. Full A/B raw logs live with the bench suite
release notes.*
