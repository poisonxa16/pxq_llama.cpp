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

## 0. Three config tiers — `PXA_REFERENCE` / default / `PXA_ENHANCE` (master switches, 2026-07-21)

One knob picks the whole posture; **every per-lever env var below still overrides its own
default** — the tier only moves what a lever defaults to when its own env is unset.

| tier | how to select | what you get |
|---|---|---|
| **REFERENCE** (level 0) | `PXA_REFERENCE=1` — **wins if both are set** | every PXA lever defaults OFF/0: the whole `PXA_PXQ6_GATE` family off, `PXA_FUSE_DELTANET=0` (eager path), `PXA_VOLTA_CUBLAS_NE11=0` (MMQ-always), no G3-class levers. The pure reference kernel/dispatch paths — the bit-exact audit and A/B baseline. |
| **default** (level 1) | no env | the shipped defaults: §2's measured bit-exact winners ON + `PXA_VOLTA_CUBLAS_NE11=64` (sm_70). Behavior unchanged vs the 2026-07-21 ship. |
| **ENHANCE** (level 2) | `PXA_ENHANCE=1` | default **+** the per-arch **measured** G3-class levers whose ship gates passed: `PXA_PXQ_INT8_PREFILL=1` on **sm_61 ONLY** (the cc==610 ship gate is unchanged; **+182% prefill** measured on the 1080 Ti, §4) and `PXA_SPEC_RELAXED=1` (spec lanes only, G3). |

Every claim is per-arch measured (the per-lever rows below carry the numbers + configs). The
CUDA backend prints one startup line with the level and the per-device decisions, e.g.
`PXA level=ENHANCE | dev0 Tesla V100-PCIE-16GB(sm_70): CUBLAS64 ON [+9.4% pf] | dev1 Tesla
P100-PCIE-16GB(sm_60): defaults [bit-exact set] | dev2 GeForce GTX 1080 Ti(sm_61):
INT8_PREFILL ON [+182% pf, G3] | spec: SPEC_RELAXED ON [G3, spec lanes]`.
Implementation: `ggml/src/ggml-cuda/pxa-enhance.cuh` (`common/sampling.cpp` mirrors the level
logic for `PXA_SPEC_RELAXED`, a non-CUDA TU).

## 0b. The two postures — `PXA_MODE=balance|max` + ADAPTIVE-UB (the ship UX, 2026-07-22)

The kernel levers below are the MEANS; these two named postures are the PRODUCT. One knob:

| posture | how to select | fa / ub it fills | goal |
|---|---|---|---|
| **BALANCE** (default, the daily) | no env, or `PXA_MODE=balance` | `-fa on`, adaptive ub (2048-class on 16 GB cards) | best **decode** AND best-possible prefill **in the fa-on regime** — carried by `PXA_FA_PREFILL_SPLIT=64` (big batches build the non-FA/fa-off math even under `-fa on`) + `PXA_FA_MASK_SKIP_TILE` (bit-identical fully-masked-tile skip in the Pascal tile FA kernel). fa-on **decode is byte-untouched by construction** (decode-sized graphs never cross the split threshold; skipped tiles contribute exactly zero) — a prefill lever that costs decode is a MAX-only lever, never BALANCE. |
| **MAX** (bulk ingest) | `PXA_MODE=max` | `-fa off`, largest-fitting ub {2048→1024→768→512} | absolute max prefill; decode secondary. SPLIT/MASK_SKIP are inert at fa-off; all fa-off prefill levers (fp16-GEMM sm_60, CUBLAS64 sm_70, INT8_PREFILL sm_61 under ENHANCE) engage. |

Rules of engagement:
- **Explicit CLI always wins**: `-fa`/`-ub` (or `LLAMA_ARG_FLASH_ATTN`/`LLAMA_ARG_UBATCH`) on the
  command line are never overridden — the mode only fills UNSET flags (`llama-server` only; the
  filling + adaptive-ub live in `examples/server/server.cpp`, the kernel-lever selection in
  `pxa_mode()` in `ggml/src/ggml-cuda/pxa-enhance.cuh`).
- `PXA_REFERENCE=1` still overrides everything to the pure reference path (posture stands down).
- Both postures imply the full measured ENHANCE-class lever set for the active tier; they differ
  only in fa + which prefill carriers engage + the adaptive-ub target.
- **ADAPTIVE-UB**: at server startup the free/total VRAM of every assigned CUDA device is probed
  and the largest ub in {2048,1024,768,512} that plausibly fits next to the model's per-device
  share is chosen (≈0.5 MiB/ub-token optimistic heuristic, capped at the card-type default);
  safe fallback = card-type default (**≥15 GiB card → 2048; 11 GB 1080Ti class → 768; else 512**
  — the 11 GB fallback is hardware-verified: ub2048/1024 compute buffers OOM next to a ~10 GB
  model, ub768 fits). The server logs the chosen `mode/fa/ub` + the reason at startup
  (`PXA posture: mode=… fa=… ub=… (…)`), and the CUDA startup line reports
  `| mode=balance [fa-on serving]` / `| mode=max [fa-off ingest]`.

**Measured per-card posture table** (fair-battle rev2, n_prompt=5432 cold, temp-0 median of 3;
PXQU-16+q8head on P100/V100, PXQ2 on the 1080 Ti — 2026-07-22 windows):

| card | BALANCE (fa-on) | MAX (fa-off, largest-fitting ub) |
|---|---|---|
| **V100 16 GB** (sm_70) | prefill **1627.7** / decode **91.0–92.8** @ ub512 (+6.0% prefill from CUBLAS64; fa-on ub2048 blows up at request time until SPLIT is silicon-verified — adaptive-ub or explicit `-ub 512` is the working fa-on ceiling on a near-full single card) | prefill **2149.6** / decode 76.9 @ ub2048 (CUBLAS64 +9.6%; threshold 64 proven the true optimum: 48 ties, 32 loses, 96 forfeits the +5% [64,96) window) |
| **P100 16 GB** (sm_60) | **BALANCE (fa-on ub2048): prefill 1206 / decode 56.7** -- `PXA_FA_PREFILL_SPLIT` lifts fa-on prefill +45% (834->1206) with decode held (measured, median of 3) | **MAX (fa-off ub2048): prefill 1170 / decode 41.5** (fp16-GEMM on, banked) |
| **1080 Ti 11 GB** (sm_61) | prefill 678 / decode **65.6** @ ub768 adaptive (ub2048/1024 physically OOM; SPLIT is the staged carrier toward the ~950–1001 fa-off class) | prefill **985** / decode 64.7 @ ub768 (reproduces published 1001 within 1.6%; ENHANCE INT8_PREFILL +830% is the carrier; I8-DBUF/BN128 maturation REFUTED — see §5) |

⚠ **HONESTY GATE (2026-07-22)**: the two BALANCE carriers (`PXA_FA_PREFILL_SPLIT`,
`PXA_FA_MASK_SKIP_TILE`) are compiled clean and equivalence-argued by construction, but their
staged silicon A/B (B1/B2/B3 sha-set + decode-guard cells) has **not yet run** — they were
defaulted ON per the posture directive. Roll back instantly with `PXA_FA_PREFILL_SPLIT=0` /
`PXA_FA_MASK_SKIP_TILE=0`; run the staged cells before quoting BALANCE prefill numbers.
Determinism note (new stack fact): temp-0 output has run-to-run sha flutter even on unmodified
binaries in cuBLAS-engaged configs — determinism gates must compare **sha sets / short-gen
exactness**, not single-run sha equality.

## 1. Format enables (required to run PXQ files)

| var | default | what it does | notes |
|---|---|---|---|
| `PXA_PXQ6` | **on** | fused CUDA kernel family for the 4-bit tier (PXQ4, formerly PXQ6) and PXQ4-HQ; `=0` drops to the dequant→cuBLAS fallback | fallback is correct but ~2× slower decode; a failed table self-check auto-falls-back |
| `PXA_PXQ2` | **on** | the 2-bit (LM4) kernel family; `=0` disables | |
| `PXA_PXQ3` | **on** | the 3-bit (LM8 bit-plane) kernel family; `=0` disables | |
| `PXA_PXQ6R` | **on** | the 5-bit PXQ6 (LM32 x E16-row) kernel family; `=0` drops to dequant→cuBLAS | the quality tier (env name keeps the internal working name) |

All master gates are ON out of the box (they were mis-documented as "off" before 2026-07-21) —
zero-env users get the fused kernels on every PXQ / PXQ-UNIVERSAL file.

> **RETIRED 2026-07-21:** type ids **250** (`PXQ4-LEGACY`, the lossless MXFP4-repack slab type;
> `PXA_PXQ4` gate) and **251** (`PXQ5`, the learned-book + SE8 legacy type; `PXA_PXQ5` /
> `PXA_PXQ5_FAST` gates) were removed from the fork entirely. Loading an old id-250/251 gguf now
> fails with a clean error at gguf load ("type id N — this type was retired 2026-07-21; requantize
> from your source model with llama-quantize PXQ4 or PXQ6") instead of running. ⚠ Any on-disk
> artifact quantized as PXQ5 (e.g. `*-PXQ5.gguf`) needs a pre-2026-07-21 binary or a requant.
> Same day, the CLI/display name **PXQ6 was re-pointed to the real 5-bit LM32 tier (gguf type id
> 256, ftype 257, ~5.27 bpw)** — the ladder is now strictly PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6
> (+ PXQ_UNIVERSAL); "PXQ6HQ" survives only as a deprecated `llama-quantize` alias for PXQ4-HQ.

## 2. Recommended-ON performance levers (all bit-exact) — **default ON since 2026-07-21**

These used to require an env; they are now the compiled-in defaults (`<var>=0` reverts any one of
them to the proven reference path — each is individually bit-exact, so reverting is purely a perf
rollback). The published per-card numbers are what a zero-env user now gets.

| var | default | what it does | measured | verdict |
|---|---|---|---|---|
| `PXA_PXQ6_KSPLIT` | **on** | K1: splits the gate/up decode GEMV over K-segments with a fixed-order workspace reducer (more blocks in flight on small-R launches) | part of the published kernel set (all published decode numbers use it) | **ON** |
| `PXA_PXQ6_VECX` | **on** | K2b: float4 activation loads in the decode mmv inner loop | part of the published kernel set | **ON** |
| `PXA_PXQ6_GUFUSE` | **on** | K3a: fuses the up+gate GEMV pair + GLU epilogue into one kernel | part of the published kernel set | **ON** |
| `PXA_PXQ6_SCATFUSE` | **on** | K3b: fuses the MoE scatter/accumulate epilogue | part of the published kernel set | **ON** |
| `PXA_PXQ6_RAGTAIL` | **on** | K4: skips store-masked FMA work on ragged tail tiles | part of the published kernel set | **ON** |
| `PXA_FUSE_DELTANET` | **3** | `=3` fuses the DeltaNet (linear-attention) decode glue-kernel chain | **+3.7% decode P100 U16** (57.2→60.2→62.4 with q8 head, published config) | **ON (=3)** |
| `PXA_G2_ADDFUSE` | **on** | G2-F4: re-enables ADD+FUSED_RMS_NORM pair fusion (ne0≥256) + fuses the residual add into the MUL_MULTI_ADD epilogue (experts first, residual last — bitwise-commutative-identical) | **+1.9% decode V100** (100.1→102.0, U16-q8out quiet-window) / **+1.2% P100** (62.25→63.0, same protocol) | **ON** |

The published per-card numbers in `bench/README.md` were measured with section-2 rows 1–6 ON
(`ADDFUSE` landed after; its gain stacks on top — see the cookbook).

## 3. Available but NOT in the recommended env (measured no-gain on the published configs)

| var | default | what it does | measured | verdict |
|---|---|---|---|---|
| `PXA_PXQ6_PAIRLUT` | off | K2a: 256-entry float2 byte-pair LUT for code expansion (bit-exact) | **+0.1% P100 U16-q8out** (62.3 vs 62.3) — an earlier +4.8% reading came from a 2×P100 `-ts 1,1` 4-bit-flagship config and does not transfer | OFF (harmless; config-specific) |
| `PXA_PXQ6_PIPE` | off | K5: sm_60 2-stage register prefetch in the decode mmv (bit-exact) | no measured gain on published configs | OFF |
| `PXA_PXQ6_KSPLIT_GEN` | off | K1 variant: K-split on the generic (non-fused) mmv path; G3-class | superseded by the fused path | OFF |
| `PXA_MMVQ_MOE_NWARPS` | 1 | forces 2/4 warps in the routed MoE GEMV (mmvq) | "prove faster + shadow-clean before defaulting" — no win recorded | leave unset |

## 4. Opt-in levers with trade-offs (read before enabling)

| var | default | what it does | measured | gate class |
|---|---|---|---|---|
| `PXA_PXQ_INT8_PREFILL` | off | **V100 (sm_70) A/B 2026-07-21: −6.6% prefill vs the fp16 fused incumbent → KILLED for sm_70; gate stays sm_61.** `=1`: routes PXQ prefill GEMMs through an int8 dp4a MMQ-style tile on **sm_61 only** (codes→s8 via the snapped book, per-16/per-8 sub-scales folded into a per-tile fp32 rescale, activations q8-per-32). `=2` lifts the arch gate (TEST — sm_60 dp4a is emulated, never ship there) | **1080 Ti PXQ2 cold 5.8k-token prefill 251→709 t/s (+182%)**, 95% of the native-MMQ ceiling; decode byte-untouched (66.4/66.5) | **G3-class** (temp-0 64-tok continuation sha-identical in our gates; top-1 logits identical every spot-check; tail top-5 order can shift at p≈0.015). Flag OFF = byte-identical dispatch |
| `PXA_PXQ6_WMMA` | 0 | experimental V100 tensor-core prefill path (fp16 fragments; `=1` fp32-accum, `=2` fp16-accum twin); auto-guarded to the 4-bit tier, cc 7.0 only | **+0.97% e2e prefill** after the 256-thread launch fix — not worth it | G3-class; experimental, keep OFF |
| `PXA_VOLTA_CUBLAS_NE11` | **64** | on sm_70, routes dense quantized GEMMs with `ne11 ≥ N` to fp16 cuBLAS (HMMA tensor cores) instead of DP4A MMQ; `=0` restores MMQ-always | **default-ON 2026-07-21: public PXQ2 single-V100 prefill median +9.4% (1949→2133), won all 3 interleaved rounds; decode untouched.** Earlier internal 35B +6.5% consistent. **Re-confirmed 2026-07-22 on PXQU-16+q8head: +9.6% MAX (1962→2149.6 fa-off ub2048) and +6.0% BALANCE fa-on ub512 prefill; the kernel-level break-even ladder (ub=ne11 sweep) proves 64 is the TRUE optimum — cuBLAS wins ≥64 (+5% at 64), ties at 48, loses at 32; 96 forfeits the [64,96) window** | G3-class (prefill numerics class changes; also run-to-run nondeterministic at temp-0 — gate on sha SETS); `=0` rollback |
| `PXA_VOLTA_CUBLAS_ID_NE11` | 0 | same idea for the routed (mul_mat_id) expert GEMMs on sm_70 | measured a LOSS on the configs tried — MXFP4 experts already ride fast MMQ | leave unset |
| `PXA_P100_FP16_GEMM` | on | sm_60 dense-GEMM prefill path: fp16 dequant + GemmEx-16F (GP100 has full-rate fp16) | `=0` rolls back to fp32 SGEMM (the old, slower path). Its gain is already banked in the published P100 1213/1169 fa-off numbers. **0a hygiene 2026-07-22: now level-aware — `PXA_REFERENCE=1` really turns it OFF on sm_60** (it used to stay silently ON and contaminate reference floors); explicit env still wins | G3-class; ON is the shipped default |
| `PXA_FA_MASK_SKIP_TILE` | **on** | skips fully-`-inf`-masked 64-wide KV tiles in the Pascal tile-f16 FA kernel (port of the shipped wmma MASK_SKIP; the nb31 mask-stride lesson applied). A BALANCE carrier: engages sm_60/sm_61 under `-fa on`; inert at fa-off | bit-identical BY CONSTRUCTION (skipped tiles contribute exactly zero); **⚠ staged B1 silicon A/B (sha-set + decode-guard, target P100 fa-on ub2048 pf ≥900) has NOT yet run** — defaulted ON per the 2026-07-22 posture directive; `=0` rolls back | bit-exact (by construction; silicon gate pending) |
| `PXA_FA_PREFILL_SPLIT` | **64** (BALANCE) / 0 (MAX, REFERENCE) | per-ubatch FA regime dispatch (`src/llama-build-context.cpp` + resolver): graphs with `n_tokens ≥ N` build the non-FA batched-cuBLAS attention chain even under `-fa on` — prefill rides the fa-off math (the pre-Turing fast-prefill regime), decode/MTP-verify (< N) keep the byte-untouched FA branch. Values 1–8 clamp to 9 (decode/MTP-verify safety) | decode byte-identical by construction; **⚠ staged B2/B3 silicon A/B (target P100 fa-on ub2048 pf ≥1100, decode sha-identical) has NOT yet run** — defaulted 64-under-BALANCE per the posture directive; `=0` rolls back | prefill G3-class (regime swap), decode bit-exact |
| `PXA_MXFP4_DEQ_V2` | on | fast coalesced smem-table MXFP4→f16 dequant kernel | 150→397 GB/s dequant, bit-identical output; `=0` rolls back | bit-exact |
| `PXA_PXQ6_PRMT` | off | K2c: prmt/byte-perm **register-LUT** book decode (4-bit tiers) — 16-entry book in 8 uniform registers, `__byte_perm` nibble→fp16, zero smem. Bit-exact (memcmp all-pass, all 4 tiers) | **−11% decode on V100** (register-LUT trades bandwidth for ALU; Tesla decode is bandwidth-bound so it loses). KEEP default-OFF: it is the correctness-proven **sm_80 Marlin-tier prerequisite**, not a Pascal/Volta lever | bit-exact |
| `PXA_PXQ6_LDCS` | off | K7: `ld.global.cs` (evict-first) on the decode weight code stream | **+0.5% V100 decode = noise** (below the <1% kill line); bit-exact + harmless, left OFF. May pay on tighter-L2 cards (P100 A/B pending) | bit-exact |
| `PXA_SPEC_RELAXED` (+`_PMIN`, default 0.05) | off | relaxed speculative acceptance: accept a draft token that lands in the target's post-filter candidate set with p ≥ PMIN (instead of exact-match only). Auto-disabled for grammar/mirostat/temp≤0 | never A/B'd — window item; G3-class by design (output legitimately changes at temp>0) | G3-class |
| `PXA_PXQ6_FORCE_PREFILL` | off | TEST ONLY: bypasses the sm_60/70 prefill arch gate so correctness A/Bs can run on other archs | correctness testing only | never in production |

## 5. Documented dead ends (kept for reproducibility — measured no-gain or loss, default OFF)

| var | what it was | measured outcome |
|---|---|---|
| `PXA_G2_REDFUSE` | G2-F1: absorb the gateup ksplit-reduce + GLU into the down-mmv prologue | **−0.8% decode V100** — the 8× workspace re-read + GLU recompute costs more than the reduce it removes. KILL |
| `PXA_G2_NORMFUSE` | G2-F3: fused rms-norm emits a q8_1 sidecar so the mmvq chain skips `quantize_q8_1` | no measurable gain over ADDFUSE alone (P100 63.0→62.95); bit-exact, kept OFF |
| `PXA_G2_QUANTFOLD` | G2-F2: the DeltaNet out-gate kernel emits the q8_1 sidecar for `linear_attn_out` | same — no gain over ADDFUSE on the measured configs; bit-exact, kept OFF |
| `PXA_PASCAL_DMMV` | alternate Pascal DMMV dispatch experiment | measured loss; documented dead end |
| `PXA_CUDA_GRAPH_V2` (+`PXA_CUDA_GRAPH_LOG`, `PXA_CUDA_GRAPHS_PASCAL`) | keyed whole-token CUDA-graph replay cache (byte-identical output). Env-only opt-in — never default | **KILLED on both box arches with captures VERIFIED firing:** V100 −2..−4%; **P100 −3.9%** (65.0→62.5, public PXQ2, replays=396/400 tokens, 3 interleaved rounds). Decode is GPU-busy; replay bookkeeping is pure tax. ⚠ Measurement lesson: an earlier '+3.5% P100' reading was NOISE — the cc<Ampere arch gate silently kept captures at 0 (a graph env that captures nothing is a no-op). Never believe a graph number without `PXA_CUDA_GRAPH_LOG` showing captures>0 |
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
| `PXA_PXQ6_BOOK` / `PXA_PXQ2_BOOK` / `PXA_PXQ3_BOOK` / `PXA_PXQ6R_BOOK` | override the frozen codebook — for lab experiments; shipped books are compiled in and sha-pinned |
| `PXA_PXQ6_SUB` / `_SUB_HQ` / `PXA_PXQ2_SUB` / `PXA_PXQ3_SUB` | override the sub-scale LUTs (lab) |
| `PXA_PXQ6_ANCHOR_FIT` / `PXA_PXQ2_ANCHOR_FIT` / `PXA_PXQ3_ANCHOR_FIT` | anchor-fit strategy toggles in the native quantizers (lab; defaults are the shipped, gate-proven settings) |
| `PXA_PXQ_HEAD` | output-head type for the PXQ ftypes (`output.weight` only, NOT `token_embd`): `q8_0` (default) \| `q6_k` \| `f16`; unknown values warn and fall back to `q8_0`. Default q8_0 = **+3.0% P100 decode measured, all rounds** (q8_0 rides Pascal's fast DMMV path where K-quant heads ride the slow scalar path, and the head runs every token over the full ~151k vocab) — and q8_0 is higher precision than the old q6_k default, so speed AND quality. An explicit `--output-tensor-type` still wins |
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
