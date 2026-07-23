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
| **V100 16 GB** (sm_70) | prefill **1627.7** / decode **91.0–92.8** @ ub512 (+6.0% prefill from CUBLAS64; fa-on ub2048 blows up at request time until SPLIT is silicon-verified — adaptive-ub or explicit `-ub 512` is the working fa-on ceiling on a near-full single card). **Decode headline with MTP: 108.3** @ ub1024 fa-on (mtp n1 + lazy, steady-state; base 91.6 same session — see the MTP section; ub2048 + the MTP gguf OOMs single-card, ub1024 is the ceiling) | prefill **2358.2** / decode 77.3–78.3 @ ub2048 (**canonical close 2026-07-22**: WMMA v2 `=3` on the merged canonical build, +5.0% flag-attributable; clean confirm set 2383.5. Canonical flag-off base is now **2245.3** — the old 2149.6 row moved +4.5% from canonical churn (CUBLAS64-era), drift resolved. CUBLAS64 +9.6% banked inside; threshold 64 proven the true optimum: 48 ties, 32 loses, 96 forfeits the +5% [64,96) window) |
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
| — (PXQ1, type id 248) | always on | the sub-2-bit tier: 1-bit sign codes × the shared E16-row scales (per-row fp16 anchor + frozen SUB16 4-bit subs), 2-level book {−1,+1}, type_size 5 (1 scale byte + 4 code bytes / 32 elems), ~1.26 bpw. **Served dequant→cuBLAS GEMM in v1** — no fused kernel family, no env gate (nothing to disable). Built for `--pxq-universal` mixed maps: the 24/32 GB stretch tiers put low-importance experts at 1-bit (`pxq1` lines in the tier map, e.g. a knapsack mix like 126×pxq1/18×pxq2 for a ≤24 GB 122B-A5B) | quantize: `llama-quantize … PXQ1` (uniform) or `pxq1` rules in a `--pxq-universal` map; fixed compiled-in book, no provenance KVs |

All master gates are ON out of the box (they were mis-documented as "off" before 2026-07-21) —
zero-env users get the fused kernels on every PXQ / PXQ-UNIVERSAL file.

> **sm_86 / sm_89 (3090 / 4090 class, added 2026-07-23):** the canonical arch list is now
> `60;61;70;86;89` — binary-wide (every kernel, every quant: full 30xx/40xx support, not a
> PXQ-tier subset). The per-arch cc-gated levers (`PXA_ROUTER_FUSE` cc==7.0-only,
> `PXA_PXQ_INT8_PREFILL` cc==6.1-only, `PXA_PXQ6_WMMA` cc==7.0-only) fall through to their safe
> defaults on sm_86/89 — the build is correct on Ampere/Ada, just untuned (no arch-specific
> fast paths measured there yet).

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
| `PXA_PXQ6_WMMA` | 0 (=3 is the KEEP arm) | V100 tensor-core prefill path (fp16 fragments; `=1` fp32-accum, `=2` fp16-accum twin, `=3` **v2: double-buffer + fused GLU + BN128**); auto-guarded to the 4-bit tier, cc 7.0 only | **CANONICAL CLOSE (V100 A/B 2026-07-22, v2 merged to the canonical tree, 5432-tok cold, ub2048 fa-off, median of 3): `=3` (v2) +5.0% flag-attributable prefill (2245.3→2358.2), decode within ±1% (77.9→77.3) — KEEP.** Clean confirm set corroborates at **2383.5** (+6.2%; matches the K6-worktree 2383.8 almost exactly) — 2358.2 is the conservative primary-set figure (pooled 6-clean-round median 2368.1). **Drift RESOLVED: the earlier "worktree base +4.8%" was NOT worktree-local — the canonical flag-off base itself is now 2245.3 vs the published 2149.6 (+4.5%, pre-existing canonical churn, e.g. the CUBLAS64-era changes), matching the worktree's 2253.7.** `=1` (v1) is FLAT (+0.2%, BUILD_K6) — the whole gain is the v2 rebuild; the old "+0.97%" v1 reading stands confirmed dead. ⚠ Bench hazard learned during the confirm: a sibling-card COLD MODEL LOAD contaminates prefill (one set collapsed to 2190–2261 wide-spread during an idx4 35B load; steady-state sibling decode at 76% util is harmless) — check for fresh containers on the other card before trusting a wide-spread set | G3-class (kernel documented not bit-exact; shas flutter per round even flag-off — CUBLAS64-on config, gate on coherence not sha); v1/v2-fp16acc keep OFF |
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
| `PXA_CONV_SILU_FUSE` | B2 (2026-07-22, P100 grunt track): fold the DeltaNet conv-output SiLU into the SSM_CONV kernel epilogue, on the theory that the PXA_PROFILE-measured "conv_output_silu 14.1% of decode, 400us/call" bucket was a width pathology | **KILL — audit disproved the premise, fused kernel confirmed it.** The UNARY already runs n_tokens-wide (ne=[8192,2]); the 400us/call is a **profiler artifact** (PXA_PROFILE syncs before/after every node, so the fixed sync round-trip is billed equally to a 16 KB elementwise op and a real GEMM). Full fusion built anyway per protocol: temp-0 sha bit-identical, decode 56.5→56.1 (−0.7% ub512) / 56.4→55.6 (−1.4% ub2048) on P100 PXQU-16. Closed per the B2 kill line; the code stays only in the (removed) grunt worktree — the ggml_ssm_conv API change was not worth carrying for a dead lever |
| `PXA_ROUTER_FUSE` | B3 phase 1 (2026-07-22 synthesis): dedicated warp-per-row F32 GEMV kernel for the MoE router logits (`ffn_gate_inp`, F32×F32 ne11==1), which misses every fast dispatch path and lands on a bare `cublasSgemm` — the #1 PXA_PROFILE decode bucket (22.7% P100 / 25.5% V100) | **PER-ARCH verdict, closed 2026-07-22: KILL on sm_60, KEEP on sm_70.** P100 fair-battle A/B (PXQU-16+q8head): decode 57.3→58.2 (+1.6% ub512), 57.2→57.9 (+1.2% ub2048) — under the pre-registered <+3% kill line; the profiler bucket was queue-gap absorption (same lesson as B2). **The owed V100 A/B ran 2026-07-22 (single V100, ub512 fa-on): decode 92.1→96.8 (+5.1%), REPRODUCED on a full second run of both arms 90.3→96.6 (+7.0%); the 6 on-rounds sat 96.3–96.9, very tight. Above the 3% line → ON is worth it on sm_70** — the fuse verdict does NOT transfer across arches; gate it per-arch (sm_70 only), do not blanket-apply either verdict. G3-class (fuse reorders FP math; ON arm shows run-to-run sha flutter — ULP logit changes flap expert ties). Kernel + dispatch live at `ggml-cuda.cu` (search PXA_ROUTER_FUSE), zero cost when unset. **Tier wiring DONE 2026-07-22 (integration): ENHANCE auto-enables mode 1 = the cc==700-only ship gate (resolver `pxa_router_fuse_mode_resolve()` in `pxa-enhance.cuh`, INT8_PREFILL pattern); REFERENCE/DEFAULT stay OFF. Env always wins: `PXA_ROUTER_FUSE=0` forces OFF at any level, `=1` the sm_70 ship gate, `=2` TEST all-arch (⚠ semantics change: explicit `=1` is now arch-gated — a Pascal re-bench needs `=2`). Startup line prints the per-dev decision.** **Auto-wire SILICON-VERIFIED 2026-07-22 (canonical close): with `PXA_ENHANCE=1` and NO fuse env, startup printed `PXA level=ENHANCE | dev0 Tesla V100(sm_70): CUBLAS64 ON ROUTER_FUSE ON [+5-7% dec, sm_70]` and decode matched the headline (108.1 vs 108.3). Standalone re-confirm on the merged build: +4.6% decode (91.6→95.8, ub1024 fa-on). ⚠ Stacking fact: the fuse does NOT add on top of MTP — mtp+fuse 107.6 == mtp-alone 108.3 within noise (the fused-router win is absorbed into MTP accept-rate round-to-round variance, ±10 t/s). ENHANCE remains the recommended production switch: it matches the MTP headline while auto-enabling the fuse for the no-MTP path for free** |
| MTP `n_max>=2` on P100 (config, not env) | deeper MTP draft depths on sm_60 | **measured LOSS**: n_max=2 decode 54.9→47.4 (−14%), accept/drafted-token collapses to 0.42; n_max=3,p_min=0.5 gets back to 51.5 but still under OFF. Cause = the B4 verify tax (verify(3+)≥1.65× on P100). Use n_max=1 (see §6) |
| MTP `n_max=2` on V100 (config, not env) | deeper MTP draft depth on sm_70 | **same shape of LOSS as P100 (2026-07-22, ub1024 fa-on)**: decode 92.7 vs base 94.1 — below OFF; per-drafted-token accept collapses 0.960→0.480 (97/202 — the 2nd draft token is almost never accepted, accepted-count identical to n1's 97/101). n_max=3 skipped per protocol (n2 < base). The smaller sm_70 verify tax does not rescue depth ≥2; **n_max=1 is the sweet spot on BOTH arches** |
| B6 np2 on a SINGLE V100 (config, not env) | two server slots on one card to overlap requests (canonical build, c16384 np2 fa-on ub512) | **KILL (2026-07-22): 2 concurrent identical requests aggregate 77.5 t/s decode vs 93.3 single-stream on the same np2 server = −17% AGGREGATE** (rounds 79.0/77.5/77.3, aggregate = sum of both slots' predicted_per_second). Heavily asymmetric per-slot (~49.8 / ~27.5): one slot's decode overlaps the sibling's COLD prefill and both lose — the cold-prefill-interleaved regime is where single-card np2 dies. Contrast: the production 35B runs np2 across TWO V100s (different regime, resident-KV siblings + MASK_SKIP) — that layout is unaffected by this kill. Single V100 serving = np1; concurrency belongs at the proxy queue, not the slot count |
| `PXA_MOE_FASTTG_MAX_NY=1` + MTP (grouped-verify combo) | route Ny>1 MTP verify batches to the A1 expert-grouped path | **measured BIG LOSS on P100**: n2 decode 48.1→30.3. The grouped path loses badly at tiny Ny on sm_60; leave the default (8, fast-TG path) |
| B13 GPU_TOPK_SAMPLER | GPU radix top-k=100 select + 100-pair D2H to kill the CPU sampler wall | **KILLED by measurement before build** (specdecode grunt 2026-07-22): sampler = 0.43 ms/tok at top_k=100 on P100 (1.8% of the 23.4 ms decode wall) — 18× under the 1 ms kill line. The gpt-oss top_k=0 lesson does not transfer to top_k=100 |
| B1 ngram spec activation (wikitext/synthetic-agentic matrix, P100) | in-tree `--spec-type ngram-mod / ngram-map-k4v` sweeps | **no clean keeper**: every ngram-mod config either regressed cold-prose >1%, regressed agentic hugely (−36..−45% at low hit rates — failed drafts are pure verify-tax on P100), or broke sha. Near-miss: `ngram-map-k4v:n_max=64,ngram_size_n=8,ngram_size_m=8,ngram_min_hits=2` = bit-exact + +2.5% on the synthetic agentic transcript, −1.8% cold — retest against a REAL tool-call transcript before any verdict |
| `PXA_SPEC_SMALLN` | B4 v1 (2026-07-22 synthesis, from-scratch kernel `pxa-smalln.cu`): warp-per-row multi-column dequant-FMA GEMV for the dense MXFP4/q8_0 backbone at ne11 2..8 on cc<70 — one weight pass, R fp32 accumulators, meant to beat emulated-dp4a MMVQ at spec-verify shapes | **KILL as built, measured honestly**: P100 single-card MTP n1 decode 63.3→49.9 (−21%), n2 48.1→39.1. Root cause: v1 reads the code stream in scalar 1-byte loads (16 B/warp-iter) vs mmvq's vectorized 128-bit streams — the mmvq verify tax at k=2 is only 1.28×, leaving less headroom than the B4 audit implied at k=4. Zero-cost when unset (sanity cell: no-MTP decode 57.0, baseline sha, path never fires at ne11=1). Kept in-tree default-OFF; the un-attempted next step is uint4 code loads (one thread = one full 16 B block) + half2 y staging — only worth a window if a future card/model shows verify(2) ≫ 1.3× |
| B4 SPEC_SMALLN cuBLAS-redirect variant | route P100 dense ne11 2..8 to the banked fp16-cuBLAS path | **KILL** (specdecode grunt 2026-07-22): verify(4)/verify(1) got WORSE, 1.646→2.28× — per-call dequant+setup dominates at tiny N. Superseded by the custom `PXA_SPEC_SMALLN` multi-column kernel (§6) |

| `PXA_CUDA_GRAPH_V2` (+`PXA_CUDA_GRAPH_LOG`, `PXA_CUDA_GRAPHS_PASCAL`) | keyed whole-token CUDA-graph replay cache (byte-identical output). Env-only opt-in — never default | **KILLED on both box arches with captures VERIFIED firing:** V100 −2..−4%; **P100 −3.9%** (65.0→62.5, public PXQ2, replays=396/400 tokens, 3 interleaved rounds). Decode is GPU-busy; replay bookkeeping is pure tax. ⚠ Measurement lesson: an earlier '+3.5% P100' reading was NOISE — the cc<Ampere arch gate silently kept captures at 0 (a graph env that captures nothing is a no-op). Never believe a graph number without `PXA_CUDA_GRAPH_LOG` showing captures>0 |
| `PXA_CUDA_GRAPH_BATCH` / `_MOE` / `_LRU` / `_REARM` / `_MAX_NY` / `PXA_CUDA_GRAPHS_PASCAL` | earlier opt-in CUDA-graph capture experiments for small multi-token batches | no shipped gain; diagnostic lineage of GRAPH_V2 |
| `PXA_PXQ_I8_RAGTAIL` (grunt track '1080ti', 2026-07-22) | B17-adjacent: port of the fp16 kernel's K4 RAGTAIL (bit-exact ragged-tile FMA skip, `PXA_PXQ6_RAGTAIL`) into the sm_61 int8 dp4a prefill tile (`pxq6i8.cuh`/`k_pxqi8_gemm_grouped`). Diagnostic instrumentation (`PXA_PXQI8_DEBUG=1`) confirmed the theory's premise: at ub768/PXQ2, MoE routing (n_as=256 experts, ~6144 routed token-instances/ubatch) leaves **~91% of tiles ragged, averaging ~64% wasted row-slots** per tile — exactly the regime RAGTAIL should help. **Measured anyway: −2.9% prefill** (963.0 t/s vs 991.7 t/s baseline, fair-battle rev2 protocol, 1080Ti/PXQ2/ub768/fa-off, median of 3, `PXA_PXQ_INT8_PREFILL=1` both arms). **KILL.** Root cause: this tile is not FMA-throughput-bound (matches the prior 2026-07-22 DBUF/BN128 audit's "223-228 regs, 0 spills, 4 blocks/SM — ILP-saturated, not latency-bound" finding) — the per-`kb`-iteration smem stage/`__syncthreads()` pair dominates the wall, so skipping the FMA consumption loop saves nothing and the extra per-thread branch (`if (fma_on)`) is pure tax. Also confirms **B17 stream-K was correctly declined without a build**: the same debug pass showed grid = panels(8) × tiles(~259–276) ≈ 2100+ blocks per up/gate launch vs the 28-SM×4-blocks/SM = 112-block concurrency ceiling — many wave-serialized launches already, not SM-starved, so adding blocks via a K-split has no occupancy upside to capture. Both findings reframe **B18 MOE_ALIGN** for this fork: the fixed per-tile launch/staging/sync overhead (not ragged-row compute) is the likely wall, so a sort-and-pad port in the vLLM style is unlikely to pay here either without first attacking tile *count*/*fixed-cost*, not row occupancy — **not built this window** (recommend profiling the per-kb sync/stage cost with `ncu` before any further MoE-tile investment here; the existing tile design already avoids launching empty-token tiles, which is MOE_ALIGN's other headline benefit). Code kept in `pxq6i8.cuh` (gated OFF by default, zero cost when unset) as a clean, reusable diagnostic (`PXA_PXQI8_DEBUG`) + a validated-negative kernel variant so nobody re-builds this blind. |

### MTP speculative decode — MEASURED KEEPER on P100 AND V100 (2026-07-22; V100 battery closed same day)

**Config: `--spec-type mtp:n_max=1,p_min=0.0` + `PXA_MTP_LAZY_WARMUP=1`.** Model = an MTP-grafted
fusion2 (nextn tail layer, qwen35moe `nextn_predict_layers=1`). Measured, fair-battle rev2 (5432-tok
cold wikitext, n_predict=200, temp0, median of 3):

| cell | prefill | decode | vs OFF |
|---|---|---|---|
| P100 single-card (U16-q8out+MTP graft, ub512 fa-on) OFF | 919.6 | 56.8 | — |
| P100 single-card, mtp n1 + lazy | 923.8 | **63.4** | **decode +11.6%, prefill flat** |
| 2×P100 (PXQ4-MTP 19GB, ub512 fa-on) OFF | 938.2 | 54.9 | — |
| 2×P100, mtp n1 + lazy | 924.7 | **59.1** | **decode +7.7%** |
| V100 single-card (U16-q8out+MTP graft, ub1024 fa-on) OFF | 2131.9 | 94.1 | — |
| V100 single-card, mtp n1 + lazy | 2047.8 | **107.5** | **decode +14.2%, prefill −3.9% (fa-on)** |
| V100 prefill-regime control (fa-off ub1024) OFF | 2009.1 | 76.1 | — |
| V100 prefill-regime control, mtp n1 + lazy | 1966.8 | 97.4 | prefill −2.1% = flat within round noise |

Draft acceptance 0.78–0.79 per drafted token (wikitext continuation). Coherence verified (clean
technical prose at temp-0). G3-class: run-to-run sha flutters between a small sha set on the spec
arms (batch-shape fp flutter); MTP-OFF on the grafted model is **bit-identical to the ungrafted
baseline** (same sha 384ec84d3aa7c001) — the graft is output-transparent when speculation is off.
- **Without `PXA_MTP_LAZY_WARMUP=1` MTP costs −33% prefill** (938→624) — the lazy env is mandatory.
- **OOM: single-card 16GB + MTP + ub2048 does NOT fit** (cuMemCreate OOM on first prefill).
  MTP single-card runs ub<=1024 (measured ub1024: prefill 1110 / decode 63.0, accept 0.87 -- the
  single-card MTP balance posture); the 2-card split takes ub2048 fine.
- **n_max=1 is the P100 sweet spot** — deeper drafts lose (see §5 kill rows): the B4 verify tax
  (verify(2)/verify(1)=1.28×, verify(4)=1.65×, measured 2026-07-22) eats the extra accepts.
- Model artifacts (built this window via gguf surgery):
  `fusion2-35b-U16-q8out-MTP.gguf` (14.6GB, single-card; base
  U16-q8out + blk.40 tail/nextn from the ornith MXFP4-MTP donor) and
  `fusion2-35b-PXQ4-MTP-fixed.gguf` (19.2GB, 2-card; the retired-id-250 blk.40 experts of
  `fusion2-35b-PXQ6-MTP-clean` swapped back to the donor's plain MXFP4 — byte-size-identical swap).
- **V100 CLOSED 2026-07-22 (same two arms, single V100, ub1024 fa-on): mtp n1 + lazy = decode
  94.1→107.5 (+14.2%), accept 0.960 (97/101) on the reading-comprehension workload — the expected
  larger sm_70 win, confirmed.** Caveats, recorded honestly: (1) ~~the n1 decode rounds trended
  96.5→107.5→117.6 (warmup/clock ramp) — re-median on a warmed card before engraving~~ →
  **RESOLVED by the steady-state battery below: the ramp settles into a stable ~105–113 band;
  the old 107.5 median was honest.** (2) fa-on serving pays −3.9% prefill for the +14.2% decode
  (only the fa-off control regime is flat at −2.1%). (3) The 0.960 accept is prompt-dependent
  (P100 wikitext measured 0.78–0.79; the steady-state battery read 0.79–0.80 cumulative) — do not
  quote it as the general rate. (4) The MTP-vs-base temp-0 sha-identity gate could NOT be applied
  on this V100 build — the BASELINE itself flutters shas per round (CUBLAS64-on sm_70 nondet), so
  the correctness gate was temp-0 coherence (verified by reading the output), weaker than the P100
  window's bit-identity proof. (5) ~~ub2048 untested on V100~~ → **tested: single V100 + the
  13.6GB MTP gguf + ub2048 = OOM at context creation** (verbatim: `allocating 2004.00 MiB on
  device 0: cudaMalloc failed: out of memory … failed to allocate compute buffers`) — same shape
  as the P100 precedent; ub1024 is the single-card ceiling. n_max=2 is a KILL sub-arm on V100 too
  (92.7, accept 0.480 — see §5).
- **⭐ STEADY-STATE BATTERY = the canonical-close V100 decode numbers (2026-07-22, merged canonical
  build, single V100, ub1024 fa-on, np1).** Protocol upgrade: 5 rounds in ONE server session
  (`fb-cell5.sh`), median of the LAST 3 — back-to-back `fb-cell.sh` invocations tear down/reload
  the model between rounds and re-cool clocks, breaking steady-state. Cells:

  | cell | decode | prefill | notes |
  |---|---|---|---|
  | base (no spec, no fuse) | 91.6 | 2031.7 | rounds 91.6–92.0, flat — no ramp on the base |
  | mtp n1 + lazy | **108.3** | 2107.4 | **+18.2%**; accept 0.802 cumulative; steady 105.6–109.4 after a 2-round ramp |
  | router-fuse only | 95.8 | 2125.3 | +4.6% standalone (startup: `ROUTER_FUSE ON [+5-7% dec, sm_70]`) |
  | mtp + fuse (env) | 107.6 | 2013.6 | == mtp-alone within noise — the fuse does NOT stack on MTP (accept variance ±10 t/s dominates; last-3 spread 96.9–117.6) |
  | mtp + `PXA_ENHANCE=1` (no fuse env) | 108.1 | 2031.4 | **auto-wire verified** — startup printed `level=ENHANCE … ROUTER_FUSE ON` + `SPEC_RELAXED ON`, matches the headline |

  **The publishable V100 decode headline is 108.3 t/s (mtp n1 + lazy), +18.2% over the 91.6
  same-session base — supersedes the 107.5 3-round figure.** Recommended production switch =
  `PXA_ENHANCE=1` on top of the MTP config: statistically identical to the headline (108.1) and
  it auto-enables the sm_70 fuse for free. All cells coherence-gated (clean English temp-0
  continuations; shas flutter on every cell incl. base, per the CUBLAS64 sm_70 protocol note).

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

## 10. Architecture support: GLM-4.7-Flash (`glm4_moe_lite`)

`Glm4MoeLiteForCausalLM` (HF `model_type: glm4_moe_lite`, e.g. `zai-org/GLM-4.7-Flash`) is a
**different architecture from GLM-4.5/GLM-4.6** (`Glm4MoeForCausalLM`, `model_type: glm4_moe`,
loaded here by `build_glm4_moe()`). Flash is DeepSeek-V2/V3-lineage — MLA attention, sigmoid
(`noaux_tc`) MoE gating with a score-correction bias, and a NextN/MTP tail — not the GQA
attention `build_glm4_moe()` implements. Mapping it onto that graph produces shape mismatches
or silent rope garbage; it must NOT be treated as a `glm4_moe` variant.

- **Converts as `deepseek2` (MLA), not a new arch.** `convert_hf_to_gguf.py`:
  `Glm4MoeLiteModel(DeepseekV2Model)` (`model_arch = gguf.MODEL_ARCH.DEEPSEEK2`), registered
  `@Model.register("Glm4MoeLiteForCausalLM")`. Two gaps closed on top of the existing
  `DeepseekV2Model`: (1) the HF config omits `scoring_func` (uses `topk_method: noaux_tc`
  instead) where `DeepseekV2Model.set_gguf_parameters` hard-indexes it — the subclass
  `setdefault`s it to `"sigmoid"` (correct: the `e_score_correction_bias` tensor is present,
  and the loader's own GLM-4.7-Flash 47-layer heuristic already defaults to sigmoid gating
  when the KV is missing); (2) the tokenizer pre-hash `cdf5f353...` (GLM-4.7-Flash's own
  vocab, distinct from GLM-4.5's `9ca2dd61...`) is mapped to `res = "glm4"` — the fork's
  `tokenizer.ggml.pre == "glm4"` vocab path already existed and needed no changes. Everything
  else (expert 3D-stacking into `ffn_{gate,up,down}_exps`, `kv_b_proj` → `attn_kv_b` +
  transposed-split `attn_k_b`/`attn_v_b`, `e_score_correction_bias` → `exp_probs_b.bias`,
  `q_lora_rank`/`kv_lora_rank`/`key_length`/`value_length`/`rope.dimension_count` KV writes) was
  already correct in `DeepseekV2Model` — this is a converter-registration gap, not a loader gap.
- **Loads on the existing `LLM_ARCH_DEEPSEEK2` graph — no loader changes needed.** The loader
  already special-cases the 47-layer shape (`n_layer==47` → sigmoid gating default, `is_lite`
  heuristic correctly excludes it so `q_lora_rank` is read) and already builds `wk_b`/`wv_b` +
  shared-expert tensors generically. Verified end-to-end: a mainline-converted (unsloth)
  GLM-4.7-Flash GGUF loads clean and produces coherent temp-0 output under this fork; the
  fork's own converter output was verified separately against a synthetic fixture (below).
- **Run flags: `-fa on -mla >= 1` is REQUIRED for `-sm` multi-GPU on this (and any MLA) arch** —
  the multi-GPU graph path aborts otherwise. Bake `-fa on -mla 3` (or `-mla 1`) into any launch
  config; a temp-0 `-mla 1` vs `-mla 3` A/B on the same GGUF must be token-identical (divergence
  there is a graph bug, not an expected quant/precision difference).
- **NextN/MTP layer is skipped at conversion, by design.** `model.layers.<num_hidden_layers>.*`
  (the NextN/MTP block: `eh_proj`/`embed_tokens`/`enorm`/`hnorm`/`shared_head.{head,norm}` plus
  a full attn+MoE block) is dropped by the existing `DeepseekV2Model.modify_tensors` layer-index
  guard — this fork's native MTP tail (`build_glm4_moe_mtp()`) is hard-gated to `LLM_ARCH_GLM_DSA`
  only, so GLM-4.7-Flash gets no speculative-decode head from this pass; wiring the fork's tail
  to the NextN block is future work, not part of this conversion path.
- **PXQ4/PXQ6 quantize the routed experts natively, no code changes.** `pxq4_tensor_eligible`
  (name ends `_exps.weight`, `ne[1] % 64 == 0`, `ne[0] % 32 == 0`) is purely name/shape driven —
  GLM-4.7-Flash's `ffn_{gate,up}_exps` (`ne=[2048,1536,64]`) and `ffn_down_exps`
  (`ne=[1536,2048,64]`) qualify with zero arch-specific handling.
- **MLA small-tensor quant lever — `attn_v_b` is the one gap.** `llama-quantize`'s legacy
  loose substring match `name.find("attn_k") != npos` (written for classic single `attn_k`
  GQA tensors) incidentally also matches `attn_k_b`, `attn_kv_a_mqa`, and `attn_kv_b` (all
  contain `"attn_k"` as a substring) — combined with `n_expert >= 4` this already forces those
  three MLA tensors to `q8_0` for free, no code change needed. `attn_v_b` does **not** match
  (no `"attn_k"` substring, and the exact-suffix `"attn_v.weight"` check doesn't fire either),
  so it rides plain MXFP4 (bs32; legal, `ne[0]=192` on the real model is not 256-superblock
  divisible but MXFP4/q8_0 don't care). If a quality gate ever flags the MLA path, force
  `attn_v_b` to `q8_0` explicitly (`--attn-v-type q8_0` or a `--custom-quants` regex override);
  `attn_k_b`/`attn_kv_a_mqa`/`attn_kv_b` need no such override.
- **CI conversion smoke test:** `tests/test-glm47flash-convert.sh` +
  `tests/fixtures/glm4-moe-lite-tiny/` — a synthetic 2-layer (`num_hidden_layers=2`,
  `first_k_dense_replace=1`) fixture carrying every real tensor-name pattern including the
  NextN tail, the REAL tokenizer (so the `cdf5f353...` pre-hash actually fires), and a
  config.json that deliberately OMITS `scoring_func` (the exact key gap this conversion path
  must survive). Asserts an exact tensor-name-set match (36 tensors; nothing missing, nothing
  extra, no NextN leak), `general.architecture==deepseek2`, and the sigmoid/MLA/tokenizer KVs.
  CPU-only, no GPU, no model download — safe on every commit.
- **Verified on real weights (2026-07-22).** `llama-quantize --allow-requantize` q8_0 → PXQ4 on
  the real 30.159 B model: 138/138 `_exps` take PXQ4 native (embedded tensor type 252, E16-row
  scales, tier core/bs16), head → q8_0, `attn_k_b`/`attn_kv_a_mqa`/`attn_kv_b` keep q8_0,
  `attn_v_b` → MXFP4, routers + `exp_probs_b` stay F32 — exactly the fixture prediction. Output
  15.307 GiB / 4.360 bpw (tensor mix: 281 f32, 142 q8_0, 330 mxfp4, 138 pxq4). Loads split across
  2× P100-16GB (`-ngl 99 -sm layer -fa on -mla 3 -c 8192`, 8279+7234 MiB weights, 423 MiB KV) with
  temp-0 coherent output on capital-continuation / factual-QA / code-reasoning prompts. mla A/B on
  a 1.5k-token cold prompt (P100 pair): prefill 686 t/s (`-mla 3`) vs 139 t/s (`-mla 1`) — ~4.9× —
  decode parity (~10.5 t/s at 1.5k fill; 24–28 t/s near-empty). `-mla 3` is the setting to ship.

---

*Numbers cite their config; anything without a number has no published A/B on the shipped
configs — treat its default as the tested state. Full A/B raw logs live with the bench suite
release notes.*
