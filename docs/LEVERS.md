> ## STOP — users need exactly two envs
> **PXA_ENHANCE=1** (the tune) and optionally **PXA_MODE=balance|max** (posture). Every other env below is an
> internal experiment record — including measured LOSSES kept so nobody rebuilds them blind. Setting lab knobs
> manually bypasses per-arch gates and typically degrades performance (e.g. RAGTAIL is a measured loss on sm_61).
> This file is the lab notebook, not a settings menu.

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

## ⚠ 0a. SUPERSEDED NUMBERS — read before trusting any row below (2026-07-28)

Three engine fixes in this release changed **what the incumbent code path is**, so rows measured
against the previous incumbent no longer describe the shipping binary. Corrections, in order of how
badly they would mislead you:

| row | what it says below | what it measures NOW | why |
|---|---|---|---|
| `PXA_PXQ_GEMM_2D=2` (sm_70) | +2.30% V100 prefill | **−18.6% on dense sm_70**, **+0.1% (neutral) on MoE sm_70** | It was measured against the *pre-coalescing* `k_pxq6_dequant_matrix`. That kernel wrote one thread per row — 32 sectors per 64 useful bytes — so beating it was easy. Now that the dequant is coalesced and cuBLAS keeps its HMMA GEMM, the half2 tile loses on Volta. **Ship arch-split: ON for sm_60 (+35% P100 dense prefill), hard-veto on cc>=700.** |
| `PXA_PXQ4_2D_SPLIT_TARGET` | tuned default, K8-2D split | **+7.6% decode on a 122B A5B MoE, no-op on a 35B MoE**. ~~changes greedy output at temp 0~~ **cured 2026-07-28 by PXQ_CANON_v1: S no longer affects output bits** | Workload-dependent. Arch-tuned defaults shipped 2026-07-28: 16×nsm on cc 7.0, 8×nsm on cc 6.0, 2×nsm elsewhere (from the sm70-stack sweeps). |
| dense prefill baselines (61.7 V100 / 77.4 P100) | quoted in several rows | **dead numbers** | The K≤11264 smem cliff made every wide-K tensor fall to dequant+cuBLAS per token; that fallback path costs 18×. Any plan written against those baselines is planning against a bug. |

**The general lesson, which applies to every row here:** a lever's measured value is relative to the
incumbent it replaces. When a fix improves the incumbent, every lever that was beating the old
incumbent must be re-measured. **Three of ours flipped sign or went to zero.**

### The cell matters more than the lever

The same binary change measured **+56.2% decode on a 122B A5B MoE** and **+12.3% on a 35B MoE**,
because the fraction recovered depends on how much stall was present to begin with. And the codec
comparison inverts by workload on identical silicon:

| cell (2×V100) | PXQ4 vs MXFP4 |
|---|---|
| **MoE** (Fusion4-35B, PXQ4 17.9711 GiB vs MXFP4 17.7978 GiB) | **+18.9% prefill, +7.7% decode** — PXQ wins *while being 0.97% larger* |
| **dense** (Qwable-27B, genuine byte parity 4.2526 vs 4.2500 bpw) | −4.57% decode — PXQ loses |

The dense-decode ceiling is structural, not a tuning failure: MXFP4 dense decode runs at **76% of
HBM peak with DRAM traffic equal to weight bytes**, i.e. bandwidth-bound on weights. At equal bytes
the ceiling is a tie by construction, so a dense-decode ratio win requires **fewer bytes at
equal-or-better error**, not a faster kernel. MoE is not bound the same way, which is why the kernel
work wins there. **Quote the cell, not an average.**

---

## 0. Three config tiers — `PXA_REFERENCE` / default / `PXA_ENHANCE` (master switches, 2026-07-21)

One knob picks the whole posture; **every per-lever env var below still overrides its own
default** — the tier only moves what a lever defaults to when its own env is unset.

| tier | how to select | what you get |
|---|---|---|
| **REFERENCE** (level 0) | `PXA_REFERENCE=1` — **wins if both are set** | every PXA lever defaults OFF/0: the whole `PXA_PXQ6_GATE` family off, `PXA_FUSE_DELTANET=0` (eager path), `PXA_VOLTA_CUBLAS_NE11=0` (MMQ-always), no G3-class levers. The pure reference kernel/dispatch paths — the bit-exact audit and A/B baseline. |
| **default** (level 1) | no env | the shipped defaults: §2's measured bit-exact winners ON + `PXA_VOLTA_CUBLAS_NE11=64` (sm_70). Behavior unchanged vs the 2026-07-21 ship. |
| **ENHANCE** (level 2) | `PXA_ENHANCE=1` | default **+** the per-arch **measured** G3-class levers whose ship gates passed: `PXA_PXQ_INT8_PREFILL=1` on **sm_61 ONLY** (the cc==610 ship gate is unchanged; **+182% prefill** measured on the 1080 Ti, §4) and `PXA_SPEC_RELAXED=1` (spec lanes only, G3). **2026-07-29: plus the model-adaptive auto-set — see §0c** (`PXA_ROUTER_FUSE` gated pure-sm_70 × MoE, `PXA_PXQ_MMVQ` auto for PXQ4-bearing models on DP4A devices, `PXA_MTP_LAZY_WARMUP` default ON, server auto-samplers). |

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
| **BALANCE** (default, the daily) | no env, or `PXA_MODE=balance` | `-fa on`, adaptive ub (2048-class on 16 GB cards) | best **decode** AND best-possible prefill **in the fa-on regime** — carried by `PXA_FA_MASK_SKIP_TILE` (bit-identical fully-masked-tile skip in the Pascal tile FA kernel). ⚠ corrected 2026-07-30: `PXA_FA_PREFILL_SPLIT` is **no longer a BALANCE carrier** — it has been experimental OPT-IN (default 0) since 2026-07-24 because its non-FA prefill chain inflates the compute buffer ~2.35× and OOMs 16 GB cards at ub2048; opt in with `PXA_FA_PREFILL_SPLIT=64` when VRAM headroom allows (see its §4 row). fa-on **decode is byte-untouched by construction** — a prefill lever that costs decode is a MAX-only lever, never BALANCE. |
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
| **P100 16 GB** (sm_60) | **BALANCE (fa-on ub2048): prefill 1206 / decode 56.7** -- measured WITH `PXA_FA_PREFILL_SPLIT=64` explicitly set (+45% fa-on prefill, 834->1206, decode held; median of 3). ⚠ SPLIT is opt-in since 2026-07-24, so out-of-the-box BALANCE prefill is the 834-class number unless you set it | **MAX (fa-off ub2048): prefill 1170 / decode 41.5** (fp16-GEMM on, banked) |
| **1080 Ti 11 GB** (sm_61) | prefill 678 / decode **65.6** @ ub768 adaptive (ub2048/1024 physically OOM; SPLIT is the staged carrier toward the ~950–1001 fa-off class) | prefill **985** / decode 64.7 @ ub768 (reproduces published 1001 within 1.6%; ENHANCE INT8_PREFILL +830% is the carrier; I8-DBUF/BN128 maturation REFUTED — see §5) |

⚠ **HONESTY GATE (2026-07-22; amended 2026-07-30)**: `PXA_FA_MASK_SKIP_TILE` is compiled clean
and equivalence-argued by construction, but its staged silicon A/B (B1/B2/B3 sha-set +
decode-guard cells) has **not yet run** — it was defaulted ON per the posture directive; roll
back instantly with `PXA_FA_MASK_SKIP_TILE=0`. `PXA_FA_PREFILL_SPLIT` was demoted to opt-in
(default 0) on 2026-07-24 — see its §4 row — so it is no longer a shipped default awaiting
silicon; its A/B remains un-run and is owed when someone opts in.
Determinism note (new stack fact): temp-0 output has run-to-run sha flutter even on unmodified
binaries in cuBLAS-engaged configs — determinism gates must compare **sha sets / short-gen
exactness**, not single-run sha equality.

## 0c. Model-adaptive auto-set — (device × model) decisions + the decision ledger (2026-07-29)

`PXA_ENHANCE=1` is now adaptive to the **loaded model**, not just the device set. At model load the
loader registers a **model profile** with the backend (arch name, model class, expert count, MTP
head, vocab size, and a census of MMVQ-eligible PXQ4/PXQ4HQ tensors); every ENHANCE auto-set that
depends on the model re-resolves against it. Nothing changes at REFERENCE or DEFAULT level — with
`PXA_ENHANCE` unset the engine behaves exactly as before this layer existed, and **every per-lever
env var still overrides its own auto-decision at any level**.

Model classes: `dense` (no experts), `moe`, `moe-hybrid-recurrent` (Gated-DeltaNet class:
`qwen35moe`, `qwen3next`), `moe-hybrid-swa` (full/SWA interleave: `gpt-oss`, `laguna`).

**The decision ledger.** Silent failure is this engine's recurring defect class (a documented lever
absent from the binary, a silent `--custom-q` demote, 12 silently-ignored type flags). So every auto
decision is LOUD: once both the device topology and the model profile are known, the backend prints
one `PXA model=…` line plus one `PXA_AUTO: <LEVER>=<value> (<reason>; override <ENV>)` line per
lever — including levers that CANNOT engage, which say why (`OFF: dense model — no router GEMV
exists`, `INERT: no sm_60 device`). A lever that fails to engage never no-ops quietly.

What ENHANCE now auto-sets from the (device × model) pair, on top of the §0 table:

| decision | keyed on | auto value | basis |
|---|---|---|---|
| `PXA_ROUTER_FUSE` | pure-sm_70 topology **× MoE model** | 1 on pure-sm_70 × MoE; 0 on dense (no router GEMV exists), 0 on mixed (T4 measured −3.1%), 0 without sm_70 | +5.1..7.0% dec, single V100 (§4 row) |
| `PXA_PXQ_MMVQ` | model PXQ4/PXQ4HQ census **× DP4A capability** | 1 when the model carries PXQ4/PXQ4HQ tensors and an sm_70+ device is present; 2 on an all-sm_61 fleet; 0 on a pure-sm_60 fleet (no DP4A — the emulation loses) or when the census is 0 | +13.7% dense / +6.7% MoE-attn-PXQ4 decode (§4 row); fidelity-neutral paired |
| `PXA_MTP_LAZY_WARMUP` | MTP lever, self-gating on MTP-active runs | ON at ENHANCE (explicit `=0` rolls back; REFERENCE off) | temp-0 **bit-identical**; without it MTP costs −33% prefill (§6 MTP section) |
| server sampler defaults | gguf `general.architecture` (server-side, `PXA_AUTO_SAMPLERS`) | see §6 row | top_k=0 measured to HALVE decode on a ~201k vocab |
| server `-ts` fill (`PXA_AUTO_TS`) | exactly-2-device mixed sm_70+sm_60, `-ts` unset | `1.4,0.6` V100-heavy | +9.78% decode (T4 cell: PXQ4-35B split V100+P100, `-sm layer`, 588-tok prompt, 2026-07-23); other topologies deliberately untouched — the basis does not generalize |
| `PXA_FUSE_DELTANET` | ledger relevance only | unchanged (default 3) — ledger states `INERT on this arch` for non-DeltaNet models | +3.5% P100 decode where active |
| `PXA_PXQ_GEMM_2D` | sm_60 present **× dense × PXQ-bearing model** | auto mode 1 (sm_60-only kernel) | +35% P100 dense prefill (coalesced-binary cell, 2026-07-28); MoE deliberately stays OFF — the post-coalescing sm_60 MoE cell is unmeasured and the sm_70 flip (−18.6%) proves the coalescing fix can invert a win |

Implementation: `ggml_pxa_set_model_profile` (`ggml.h`/`ggml.c`, registered from `src/llama.cpp`
before any graph is built), consumed by `pxa-enhance.cuh` resolvers cached against the profile
generation (the profile can land after CUDA init in the server flow — one-shot statics would have
frozen the pre-model decision). The ledger prints from whichever of {CUDA init, model registration}
happens second.

## 1. Format enables (required to run PXQ files)

| var | default | what it does | notes |
|---|---|---|---|
| `PXA_PXQ6` | **on** | fused CUDA kernel family for the 4-bit tier (PXQ4, formerly PXQ6) and PXQ4-HQ; `=0` drops to the dequant→cuBLAS fallback | fallback is correct but ~2× slower decode; a failed table self-check auto-falls-back |
| `PXA_PXQ2` | **on** | the 2-bit (LM4) kernel family; `=0` disables | |
| `PXA_PXQ3` | **on** | the 3-bit (LM8 bit-plane) kernel family; `=0` disables | |
| `PXA_PXQ6R` | **on** | the 5-bit PXQ6 (LM32 x E16-row) kernel family; `=0` drops to dequant→cuBLAS | the quality tier (env name keeps the internal working name) |
| — (PXQ1, type id 248) | always on | the sub-2-bit tier: 1-bit sign codes × the shared E16-row scales (per-row fp16 anchor + frozen SUB16 4-bit subs), 2-level book {−1,+1}, type_size 5 (1 scale byte + 4 code bytes / 32 elems), ~1.26 bpw. **Served dequant→cuBLAS GEMM in v1** — no fused kernel family, no env gate (nothing to disable). Built for `--pxq-universal` mixed maps: the 24/32 GB stretch tiers put low-importance experts at 1-bit (`pxq1` lines in the tier map, e.g. a knapsack mix like 126×pxq1/18×pxq2 for a ≤24 GB 122B-A5B) | quantize: `llama-quantize … PXQ1` (uniform) or `pxq1` rules in a `--pxq-universal` map; fixed compiled-in book, no provenance KVs |
| `PXA_PXQ1_REP_GUARD` | auto (arms at ENHANCE) | **PXQ1 repetition guard** (2026-07-23): the 1-bit tier measurably loops on open-ended prompts (coding answers stay clean — measured on a 122B PXQU24 mixed file). When the loaded gguf contains ANY PXQ1 tensor AND the config level is **ENHANCE**, sampler init fills repetition defenses as DEFAULTS: `repeat_penalty` 1.0→1.15, `repeat_last_n` 64→256, DRY `dry_multiplier` 0→0.8 (base/allowed-length untouched). **User-set values always win — only knobs still at their defaults are filled.** `=0` forces off, `=1` forces on at any level. Visible as `PXQ1 content detected: repetition guard ON […]` at sampler init (the loader also logs detection at model load) | REFERENCE/DEFAULT levels leave sampling untouched; models with no PXQ1 tensors never trigger it. ⚠ **2026-07-30 — code/doc drift found, measured, and fixed:** on 2026-07-24 the code silently broadened the auto-arm to ANY PXQ tier (this row still said PXQ1-only). Measured harm on a high-quality tier (Laguna-S-2.1 PXQ4-core, 4×P100+1080Ti `-ts 8,8,8,8,3`, ctx 8192, `/v1/chat` temp 0 top_k 1, fresh server per arm, ×3 reps): DEFAULT answers 4183×391 = **1,635,553 (correct ×3)**; ENHANCE answered **1,635,593 (wrong ×3)** — the filled repeat/DRY penalties suppress the legitimate repeated digit. Both-direction conviction: ENHANCE+`PXA_REP_GUARD=0` → correct ×3; DEFAULT+`PXA_REP_GUARD=1` → wrong ×3. This flip was chased for a night as an "sm_61/1080Ti numerics bug" (INT8_PREFILL, MMVQ, graphs-off, and the 7-card topology were each A/B-exonerated — all null arms). The auto-arm is now PXQ1-content-only again (matching this row); the broadening's original motive (the Fusion4 prose attractor, H1) was separately root-caused to `PXA_PXQ4_2D_SPLIT` non-bit-exactness and fixed by `PXQ_CANON_v1`. `=1` still forces any-PXQ arming; fixed-binary verification: ENHANCE→correct ×3 + guard not applied, forced→wrong ×3 + `[FORCED]` log. ⚠ COROLLARY for past numbers: every gauntlet/eval served with `PXA_ENHANCE=1` on a PXQ file between 2026-07-24 and 2026-07-30 ran with repeat_penalty 1.15 / last_n 256 / DRY 0.8 silently filled for requests that did not set them |

All master gates are ON out of the box (they were mis-documented as "off" before 2026-07-21) —
zero-env users get the fused kernels on every PXQ / PXQ-UNIVERSAL file.

> **sm_86 / sm_89 (3090 / 4090 class, added 2026-07-23):** the canonical arch list is now
> `60;61;70;86;89` — binary-wide (every kernel, every quant: full 30xx/40xx support, not a
> PXQ-tier subset). The per-arch cc-gated levers (`PXA_ROUTER_FUSE` cc==7.0-only,
> `PXA_PXQ_INT8_PREFILL` cc==6.1-only, `PXA_PXQ6_WMMA` cc==7.0-only) fall through to their safe
> defaults on sm_86/89 — the build is correct on Ampere/Ada, just untuned (no arch-specific
> fast paths measured there yet).

> **deepseek2 / MLA posture auto-wire, cc-aware (2026-07-23, refined):** on a
> `deepseek2`-arch gguf (GLM-4.7-Flash / DeepSeek class, MLA attention) `llama-server`'s
> posture layer defaults `-mla 3` on every arch, and defaults `-fa` **per visible CUDA
> fleet compute capability**: **ON** everywhere EXCEPT when every visible device is cc 610
> (sm_61 / P40, GTX 10-series), where it defaults **OFF**. Reason: fa-off on MLA degrades
> catastrophically with context on most archs (community-measured P40: 37 → 3.3 t/s by
> 36k ctx), BUT a second community P40 measurement found the MLA fa-on kernel itself is
> 75-326% SLOWER decode on sm_61 (fp16-starved, 1:64 rate) than fa-off there — so sm_61
> flips the default. `-fa on` still defaults ON under `PXA_MODE=max` for deepseek2 on
> sm_60/70+ (logged: `PXA posture: mode=max but arch=deepseek2 — fa kept ON (MLA requires
> it)`); on an all-sm_61 fleet that max-mode exception does NOT fire. Explicit `-fa`/`-mla`
> always win on every arch; the fa-off warning is suppressed on an all-sm_61 fleet only.
> See docs/KNOWN-ISSUES.md.

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
| `PXA_PXQ6_KSPLIT_GEN` | **arch default: S=4 on sm_60/sm_70, off elsewhere** (2026-07-26; explicit env always wins, incl. `=0`) | K1b variant: generic S-chunk K-split (S in 2/4/8) on the **gate/up fused** decode path (`k_pxq6_gateup_mmv_ksplit_gen`). **Bit-exact since 2026-07-28 (PXQ_CANON_v1):** every decode-mmv variant (unsplit, K1 re-grid, S-split, FUSERED finisher, REDFUSE) now computes ONE fixed, shape-only summation order (NFIX fixed K-chunks, lane-major fold), so toggling this lever — or changing S — does not change output bits. (Historical: before that date the S-chunk fold differed from the reference kernel and this row was G3-class.) Why it wins where K1 does not: the K1 64-thread form spreads the SAME warp count over 4× the blocks, while this 256-thread S-chunk form multiplies warps-in-flight by S and divides the per-block x-stage by S | measured 2026-07-26, F35B pure-PXQ4 fill ~6k on the shipped-lever binary: 2×V100 decode 86.0 → 87.9/88.3 (**+2.6%**, S=8 worse at 86.5), 2×P100 59.0 → 60.8 (**+2.9%**). Numerics gated with the K8-2D split in one arm: 64-chunk paired NLL (35B pure-PXQ4, `ppl-eval-half.txt`), 0 NaN, aggregate −0.1% (inside noise, favors ON), max per-position 0.28% — reassociation jitter, zero-centered. Tier transfer (ship levers vs all-off, fill ~6k, screening grade): PXQ2 2×P100 n=2/arm 58.16–58.40 → 59.86–60.51 (+3.3% median); PXQU16 2×P100 n=2/arm 56.62–56.77 → 57.22–57.62 (+1.1%); PXQ2 single V100 (idx-pinned, arms interleaved A/B/A/B, sibling card under an ncu run — screening hygiene caveat) n=4/arm 86.98–88.80 → 93.17–93.44 = **+6.1% median, complete rank separation**. `PXA_PXQ6_LDCS=1` re-tested on the new geometry (2×P100, n=2): 60.13–60.71 vs 60.92 ship — flat/negative, stays OFF | S=4 ON (sm_60/70) |
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
| `PXA_FA_PREFILL_SPLIT` | **0 — EXPERIMENTAL OPT-IN ONLY** (⚠ row corrected 2026-07-30: it previously claimed "64 under BALANCE" while both resolvers have returned 0 since 2026-07-24/merged 07-28 — doc/code drift caught by the A6 audit) | per-ubatch FA regime dispatch (`src/llama-build-context.cpp` + resolver): graphs with `n_tokens ≥ N` build the non-FA batched-cuBLAS attention chain even under `-fa on` — prefill rides the fa-off math (the pre-Turing fast-prefill regime), decode/MTP-verify (< N) keep the byte-untouched FA branch. Values 1–8 clamp to 9 (decode/MTP-verify safety). **Why no longer a default:** the non-FA prefill chain inflates the compute buffer ~2.35× (1956 → 4607 MiB measured) and OOMs 16 GB cards at ub2048. The +45% P100 fa-on prefill (834→1206) is real but must be bought explicitly (`PXA_FA_PREFILL_SPLIT=64`) with the VRAM headroom to pay for it | decode byte-identical by construction; the staged B2/B3 silicon A/B has still not run | prefill G3-class (regime swap), decode bit-exact |
| `PXA_MXFP4_DEQ_V2` | on | fast coalesced smem-table MXFP4→f16 dequant kernel | 150→397 GB/s dequant, bit-identical output; `=0` rolls back | bit-exact |
| `PXA_PXQ6_PRMT` | off | K2c: prmt/byte-perm **register-LUT** book decode (4-bit tiers) — 16-entry book in 8 uniform registers, `__byte_perm` nibble→fp16, zero smem. Bit-exact (memcmp all-pass, all 4 tiers) | **−11% decode on V100** (re-screened −17% n=2 in the 2026-07-26 KSG4 ship config — same verdict, dead). ⚠ **CAUSE CORRECTED 2026-07-27: the original reading "Tesla decode is bandwidth-bound so it loses" is DISPROVEN by the ncu roofline** (`ncu-pxq4-mmv.log`, V100 decode instances of the mmv family): `dram__throughput` **5.96–35.62% of peak** — nowhere near a memory ceiling — with achieved occupancy **19.8–38.6%, Block Limit = Registers (6 blocks/SM)**. The kernel is **register-limited and latency-bound**; PRMT loses because its extra uniform-register book + integer ops worsen exactly the register/issue pressure that is the real constraint. This misreading aimed four sm_70 attacks at a bottleneck that was not there (`CUDA_GRAPH_V2` −2..−4%, `INT8_PREFILL` −6.6% prefill, PRMT −11%, `FUSERED` −1.6%). KEEP default-OFF: it is the correctness-proven **sm_80 Marlin-tier prerequisite**, not a Pascal/Volta lever | bit-exact |
| `PXA_PXQ6_LDCS` | off | K7: `ld.global.cs` (evict-first) on the decode weight code stream | **+0.5% V100 decode = noise** (below the <1% kill line); bit-exact + harmless, left OFF. May pay on tighter-L2 cards (P100 A/B pending) | bit-exact |
| `PXA_SPEC_RELAXED` (+`_PMIN`, default 0.05) | off | relaxed speculative acceptance: accept a draft token that lands in the target's post-filter candidate set with p ≥ PMIN (instead of exact-match only). Auto-disabled for grammar/mirostat/temp≤0 | never A/B'd — window item; G3-class by design (output legitimately changes at temp>0) | G3-class |
| `PXA_PXQ4_2D` (+ `PXA_PXQ4_2D_MAX_NY`, default 8) | **on** (was `off` until BACKBONE_REV 2, 2026-07-26 — every rev-2 file carries PXQ 2D weights, so the old default would have put attention/shexp/dense-FFN on a per-token dequant→cuBLAS expansion; `=0` restores the old default for A/B) | Routes a plain 2D (non-MoE) MUL_MAT on a PXQ slab weight — attention projections, shared-expert FFN, `ssm_out`, `token_embd`, `nextn.eh_proj` — into the PXQ decode mmv (the E==1 case of the expert-stacked kernel, fed a one-entry zero-`ids` buffer) instead of dequant→cuBLAS. Only relevant to files quantized with `PXA_PXQ_NATIVE` beyond the routed experts (see §8). `MAX_NY` caps the token count claimed; above it the per-call dequant amortizes over the batch and the driver declines (do NOT raise it into prefill range — at ny=2048 the mmv would re-decode the weight once per token) | shared-experts-native file: decode **63.6 → 88.2 t/s** at the shexp conversion (recovers the −39.7% per-token dequant fallback). attn-native file, 2×V100 `-ts 1.05,0.95` fill 5739 n=8 interleaved: decode 85.7 t/s, prefill 1759.5 | bit-exact (same kernel instantiations, same fp32 chains as the MoE mmv); **ON for any PXQ-native-attn/shexp file** |
| `PXA_PXQ4_2D_KSPLIT` (+ `PXA_PXQ4_2D_KSPLIT_MINBLK`, default = device SM count) | off | K1-2D: re-grids the 2D decode mmv's four in-block k-segment chains (`kb ≡ kseg mod 4`) into four 64-thread blocks writing fp32 partials to the persistent KSPLIT workspace, plus a fixed-order reducer. Same total warps, 4× the blocks — the medicine for a launch whose `panels*ny` cannot cover the SMs. Fires only when `panels*ny < MINBLK`; `MINBLK=0` never splits, a huge value always splits (pure A/B knob, no rebuild). Inert unless `PXA_PXQ4_2D=1` | **FLAT on 2×V100, measured LOSS on 2×P100 — hence default OFF.** attn-native Fusion4-35B, fill 5739 / n_predict 256 / temp 0, one server instance per arm, medians of 3. **2×V100** (`-ts 1.05,0.95`, n_SM=80): off **85.34 / 85.06**, threshold-on (fires on the 40 `panels=64` + `panels=32` nodes) **84.88 / 84.90**, always-split (`MINBLK=100000`, all 82 nodes incl. `panels=128`) **85.14 / 85.24** — every arm inside the ±0.5% instance-to-instance drift measured between two *identical* code paths (pre-patch 85.09 vs lever-off 85.34). Prefill untouched (1744–1767 all arms; prefill declines at `ny > MAX_NY` long before reaching this code). **2×P100** (`-ts 1,1`, n_SM=56, medians of 2): off **57.86**, always-split **53.65 = −7.3%**. Verdict: the 2D decode mmv is **not** occupancy-limited at panels 32–128. Firing was proven, not assumed (the one-shot `PXA_PXQ4_2D_KSPLIT dev0: FIRING (panels=… )` stderr line); the 64-thread form's 4× redundant full-K smem staging (32 serial global loads per thread in the prologue vs 8 in the 256-thread form) plus one extra reducer launch per firing node cancels the extra-SM win. Un-tested shapes where the occupancy model still predicts a gain: shexp-native `panels=8`, and low-SM cards. **Independently re-measured (n=8/arm, 4 interleaved blocks, fresh server per arm per block, 2×V100): off 85.43 (IQR 0.59), threshold-on 84.58 (IQR 0.25) = −1.00% with complete rank separation (Mann-Whitney U 64/64, and separately within each block), always-split 85.09 = −0.41% (overlapping). At fill 50 — where the occupancy model predicts the *largest* win — off 93.07, threshold-on 92.30, always-split 93.37: no starvation win appears at any fill.** So at its shipped threshold this is a small but statistically clean regression, not merely flat | bit-exact: identical per-thread fp32 chains and identical ascending 4-term reduction as `k_pxq6_mmv` — temp-0 sha256 unchanged vs the pre-patch binary on 26 completions across 11 arms and both arches (sm_70 + sm_60), incl. the forced always-split arm. `=1` enables; unset is the exact pre-split single-launch behaviour. ⚠ **That sha gate is non-discriminating with the harness prompt** (5500 words of a repeating pangram): the identical hash `18a4c492…` is also produced by the *MXFP4-baseline model file*, i.e. by a different attention quantization entirely. It proves coherence and unchanged flag-off dispatch, not kernel equality. The bit-exactness claim rests on the construction argument (identical per-thread fp32 chains, identical fixed ascending 4-term reduction); no perplexity or logit-level gate was run on this lever |
| `PXA_PXQ4_2D_SPLIT` (+ `PXA_PXQ4_2D_SPLIT_TARGET`, default = 2× device SM count) | **on** (2026-07-26; `=0` restores the unsplit dispatch, which byte-reproduces pre-split output) | K8-2D: S-way K-chunk split for the 2D decode mmv (`k_pxq6_mmv_ksplit_gen` + `k_pxq_mmv_reduce_s`), the successor to `PXA_PXQ4_2D_KSPLIT` that attacks what that lever could not. nvprof gpu-trace on a rev-2-backbone file (Laguna-XS, 2×P100) localized the 2D decode tax to two launch shapes: R=2048/K=8192 (`attn_output`, dense `ffn_down`) = 32 blocks whose **33.8 KB full-K x-stage caps occupancy at 1 block/SM** → 127.2 µs/call = **87 GB/s**, and R=512/K=2048 (`ffn_{up,gate}_shexp`) = **8 blocks total** → 34.2 µs = **20 GB/s** — while the SAME dot path with a healthy grid (R=8192/K=2048 `attn_q`: 128 blocks, 9 KB) ran 53.0 µs = **210 GB/s and beat MXFP4's mmvq (161 GB/s) per byte on the same-shape tensor**. The split multiplies blocks by S (smallest power of two reaching `SPLIT_TARGET` blocks incl. the ny axis, capped at 16 and at ≥4 slabs per chunk) and divides the x-stage by S, fixing both starvation modes; S=1 shapes keep the bit-exact unsplit kernel unchanged. **PXQ_CANON_v1 (2026-07-28): the split is now BIT-EXACT vs unsplit.** Every variant writes raw per-(fixed-chunk, lane) partials (NFIX×KSEG slots per row, NFIX shape-only) and one canonical fold produces the value — identical bits at any S, on any device, with the lever on or off. This closed a real production defect: on the 122B with PXQ4 attention, the pre-canon split path turned a fabrication probe into a 962×-8-gram runaway that hit the 2048-token cap (SPLIT=0 answered cleanly in 343 tokens); the aggregate-NLL G3 gate that certified the lever was structurally blind to that failure mode (aggregate −0.1% while one near-tie flip enters a repetition attractor). ⚠ One-time re-baseline: the canonical order differs from BOTH pre-canon paths, so temp-0 output vs pre-2026-07-28 binaries changes once (PPL-verified neutral); from now on it is stable across split decisions. Unlike `PXA_PXQ4_2D_KSPLIT` (same warps spread thinner + 4× x-stage → measured flat/loss, stays OFF) this adds warps AND shrinks smem | PROTOCOL (fill 6401, n_predict 256, temp 0, **n=8/arm, 4 interleaved blocks, fresh server per arm per block**, 2×P100 `-ts 1,1`, all contrasts complete-rank-separated): rev-2 Laguna-XS decode unfixed **43.47** (IQR 0.15) → fixed **58.70** (0.34) = **+35.1%**, vs legacy-backbone 60.49 (0.24) — i.e. the rev-2 fidelity backbone's decode cost drops from **−28.1% to −3.0%**. Pure-PXQ4 F35B same protocol: 2×P100 decode MXFP4 53.10 (0.44) vs PXQ4-ship **60.92 (0.11) = +14.7%**, prefill 497 vs **783 = +57.5%**. 2×V100 same protocol (n=8/arm, ncu-stamped clean cards, complete rank separation): decode MXFP4 **89.16** (IQR 1.69) vs PXQ4-ship **85.17** (0.26) = **−4.5%** (from −23.7% pre-fix on the same config, screening n=3: 69.4), prefill 1654 vs 1618 = −2.2% (overlapping IQRs). **PAIRED RERUN on the shipping binary (2026-07-27, llama-server md5 `6940b347`, libggml `2996fc25` — the FUSERED-default-OFF build; both arms same run, same protocol): V100 decode MXFP4 89.12 (IQR 0.91) vs PXQ4 87.33 (0.46) = −2.0%, rank separation NOT complete (pxq max 88.02 > mxfp4 min 87.72); P100 52.56 (0.17) vs 60.52 (0.27) = +15.1%, rank-separated. The PXQ arm gained +2.5% between binaries with the MXFP4 control flat — the only source delta is the dormant FUSERED tail in the split kernels (codegen/register-allocation shift, not a designed change), so −2.0% is the current binary's honest figure and −4.5% is the prior binary's.** Short-fill smoke: Laguna 44.0 → 70.5. Numerics: 64-chunk paired NLL vs off-arm, 0 NaN, aggregate −0.1%, max per-position 0.28% (zero-centered reassociation jitter); temp-0 divergence begins at a near-tie token, off-arm byte-reproduces the pre-patch binary. **All of that is history as of 2026-07-28: split==unsplit bitwise (PXQ_CANON_v1), verified on Fusion4-35B-PXQ4N-attn across 5 dispatch variants (unsplit / default split / forced-max split / K1 re-grid / gateup-gen S=2) × 3 prompts × greedy-384 + fixed-seed temp-1.0 sampling — all hashes identical — and on the 122B kaskal fabrication-probe regression (see the row text)** | bit-exact (PXQ_CANON_v1), ON |
| `PXA_PXQ_SPLIT_FUSERED` | **off — measured LOSS on both arches, kept as a documented dead end** (`=1` enables) | fence-and-flag fused finish for the S-split reducers: the last split block of each node performs the final chunk reduction in-kernel (threadFenceReduction pattern, same ascending order as the standalone reducers — output verified byte-identical), eliminating the `k_pxq_mmv_reduce_s` launch (230/token) and the gateup reduce launch (40/token). Motivated by an nvprof reading of GPU-busy parity between the codec arms with the deficit apparently in per-token launch count | protocol (F35B matched bytes, fill 5992, n=8/arm, 4 interleaved blocks, fresh server per arm per block, rank-separated): fused-ON decode **83.81 vs 85.17 = −1.6% (2×V100)**, **59.45 vs 60.92 = −2.4% (2×P100)**, MXFP4 control arms flat (+0.6%/−0.9%). Post-mortem: the finisher block reduces S·R floats ALONE while the rest of the machine drains — the wide reducer does the same work across R/256 parallel blocks in ~2.4 µs — and the launches it saves were already hidden by async submission. ⚠ Pattern note: this is the SECOND independent attack on decode launch overhead to come back negative on these cards (`PXA_CUDA_GRAPH_V2` was −2..−4% with captures verified firing). Decode launch count is NOT where the V100 deficit lives; do not attack it a third time without new evidence | byte-identical when enabled (verified); OFF |
| — (launch-bounds min-blocks experiment, NOT a lever — closed 2026-07-27) | reverted, not in tree | `__launch_bounds__(256, N)` min-blocks forcing on the three decode mmv kernels (`k_pxq6_mmv`, `k_pxq6_mmv_ksplit_gen`, `k_pxq6_gateup_mmv_ksplit_gen`), aimed at the ncu register finding (achieved occupancy 19.8–38.6%, Block Limit = Registers at 6 blocks/SM) | **decisive KILL at both tested points** (Laguna-XS rev-2, fill 6401, fresh server per arm per block): `(256,7)` n=8/arm interleaved — 2×V100 base 88.29 (IQR 0.64) vs 63.87 (1.04) = **−27.7%, rank-separated**; 2×P100 59.20 vs 58.52 = −1.2% (one P100 log tag reads `lb256x8` from a lazily-re-read script — the binary was the (256,7) snapshot, libggml `1d0efbd7`). `(256,8)` cut short at n=2 on the same cliff (63.6/63.9 vs 88.5/88.8, −28%). Verdict: the kernels' ~40+ live registers are load-bearing — capping to 36 or 32 regs/thread spills to local memory and destroys sm_70 decode. **Occupancy here cannot be bought with register caps**; the accidental +2.5% from the dormant FUSERED tail is therefore scheduling/ILP drift, not occupancy. **V100 codec lane CLOSED at −2.0%** (paired rerun, no rank separation — a median gap inside touching distributions; P100 +15.1% separates cleanly). Reopening evidence for a future session: `ncu-pxq4-mmv.log` (latency-bound, DRAM ≤36% of peak, register-limited) + this row | closed, reverted |
| — (Q-G2 wrel gate status, 2026-07-27) | — | `pxa-bench/pxqu_wrel.py` ships (commit 62473735) but is currently NOT runnable: its numpy oracle `pxqu_lab.py` (the internal calibration lab it imports) is not on the box — searched the lab dirs, the archived squeeze tree and the ik trees; the surviving `quant_lab.py` is the PXQ1 design lab with a different API. The ornith imatrix fixtures survive locally. Q-G1 (`pxqu_golden.py`, the stricter byte-parity gate) builds and runs PASS on all five input classes against the current-tree quantizers | oracle lost; Q-G1 is the running gate |
| `PXA_MTP_PREFETCH` | off (merged 2026-07-27 from the mtp-overlap lane) | async MTP companion commit: a dedicated `ctx_mtp` worker thread + `_async` variants of the accepted-hidden-rows commit; the wrapper submits the EXACT serial commit and returns immediately, overlapping the companion decode with `process_token` + the streaming write. Per-seq futures with drain calls at every other `ctx_mtp` touch point keep ordering; `=0` (default) is byte-for-byte the serial path | **measured NULL where measured (2026-07-27): F35B PXQ4-MTP, `--spec-type mtp:n_max=1`, fill ~6k, n=8/arm, 4 interleaved blocks, fresh server per arm per block, 2×V100 `-sm layer`: pf0 median 113.39 (111.33–114.72) vs pf1 113.31 (109.47–114.68) = −0.07%, fully overlapping, accept rate identical 0.96154 both arms.** ⚠ Caveat, prominently: 2×V100 `-sm layer` is the regime where this lever class is KNOWN to disappear (`PXA_SPEC_1ROW`: +6.6% single-card → +0.7% noise once split). Null where measured, **untested where it could pay** — the single-card dense-MTP verdict (27B vehicle) is owed | OFF |
| `PXA_F16_GEMV` | **on** (2026-07-26; `=0` restores the cuBLAS chain) | small-R F16 decode GEMV (`k_pxa_gemv_f16`): an F16×F32 `ne11==1` node misses every fast dispatch path (dmmv/mmvq need a quantized src0; batched-cuBLAS needs a real batch dim) and lands on the generic cuBLAS chain — convert x f32→f16 + `gemmSN_TN_kernel_half` + convert dst + a cpy = measured **31.5 µs + 3 helper kernels per call** (nvprof, 2×P100) for what is a ~0.3 MB tensor read. The rev-2 backbone's per-head `attn_gate` → `f16` promotion creates exactly this node per layer (40/token on Laguna-XS ≈ **1.5 ms/token**, the bulk of the `lite` recipe's measured decode cost). The kernel does the row dot directly: 4 warps/row, f16 weights × f32 activations, fp32 per-thread partials, fixed warp/smem reduction tree. Restricted to 2 ≤ R ≤ 512, contiguous, `ne11==1`, so it can never intercept a real dense projection; `ne01==1` stays with `mul_mat_1row` | isolated contribution inside the rev-2 Laguna fix above (70.5 with, ~68.5 without, short-fill n=1 — the 2D split dominates; the chain it removes is 1.5 ms/token of GPU time plus its CPU launch pressure). NOTE the replaced cuBLAS chain rounds x AND the result through f16; this path keeps both f32, so it is numerically DIFFERENT (strictly tighter) — behavior-gauntlet class, not sha class | G3-class, ON |
| `PXA_PXQ_GEMM_2D` <br>**⚠ mode 2 SUPERSEDED on sm_70 — see §0a; the sm_70 numbers in this row were measured against the pre-coalescing dequant and are −18.6% against the current one. Mode 2 is CLAMPED to sm_60 in code.** | 0 (off) | E==1 **prefill** GEMM for plain 2D PXQ weights (attention projections, shared-expert FFN, `ssm_out`, `token_embd`, `nextn.eh_proj`) in the `ny > PXA_PXQ4_2D_MAX_NY` window - the exact complement of `PXA_PXQ4_2D`'s decode claim, so on a gated arch a PXQ 2D weight never reaches dequant->cuBLAS at any batch size. Reuses `k_pxq6_gemm_grouped` unchanged with a one-expert tile map (`tile.e == 0` => `pxq6_panel()` degenerates to `W + p*stride`, the same degeneration the mmv driver gets from its one-entry zero-`ids` buffer); the map is built by a device kernel rather than the MoE driver's pageable H2D, so it stays legal under stream capture. `=1` sm_60 only, `=2` sm_60 + sm_70 (any fast-fp16 pre-Turing arch). Declines `GGML_PREC_F32` nodes - the cuBLAS fallback honours that with an fp32 SGEMM and this tile accumulates in half2 (MUL_MAT_ID never sets it, which is why the MoE driver has no such check). Only relevant to files quantized with `PXA_PXQ_NATIVE` beyond the routed experts (see §8) | attn-native Fusion4-35B, fill 5739 / n_predict 256 / temp 0, **n=8 per arm, 4 interleaved blocks, fresh server per arm per block**, `-c 8192 -b 2048 -ub 2048`: **2xV100** `-ts 1.05,0.95` mode 2 - prefill **1718.3 -> 1757.9 t/s = +2.30%**, decode 85.08 -> 85.41 (flat). **4xP100** `-ts 1,1,1,1` mode 1 - prefill **799.6 -> 806.7 t/s = +0.89%**, decode 57.24 -> 57.22 (flat). Complete rank separation within *both* the cold-rep and warm-rep strata on *both* arches (firing proven per device by the one-shot `PXA_PXQ_GEMM_2D dev0: FIRING (...)` stderr line), so the direction is real - but both are **below the +3% pre-registered keep line, hence default OFF**. Decode is untouched by construction (`ny <= MAX_NY` is declined here and owned by `PXA_PXQ4_2D`). WARNING: the design spec predicted a *multi-x regression* on sm_70 and would have shipped this sm_60-only; that was **wrong**, and why is worth keeping: it weighed this half2 tile (~15-20 TF) against cuBLAS HMMA (~90 TF) *alone*, but the incumbent is `k_pxq6_dequant_matrix` (32 scalar 2-byte stores per thread at a row stride of `K`) **plus** cuBLAS. Corollary for the prefill deficit: claiming *both* halves of the incumbent buys only +2.3%, so coalescing that dequant while KEEPING cuBLAS's HMMA GEMM should dominate this on Volta. **Independently re-measured (n=8/arm, 4 interleaved blocks, fresh server per arm per block): 2×V100 prefill 1747.2 → 1753.5 pooled, per-block paired median +1.53% (range −2.65%…+2.30%, 6/8 blocks positive) — i.e. the +2.30% above is the top of the range, not its centre; with `PXA_PXQ4_2D_KSPLIT=1` also on, 1727.1 → 1751.7 = +1.48%, 8/8 blocks positive. 4×P100 `-ts 1,1,1,1` 800.1 → 807.8 = +0.93% paired, 8/8 positive (confirms the +0.89%). Decode flat on both (−0.19% / −0.01%).** | **G3-class - gate on perplexity, not sha.** Strict-k half2 chain vs dequant + `cublasGemmEx(COMPUTE_16F)`: different accumulation order and intermediate handling. Gated: 200-chunk paired perplexity (`-c 512 -b 512 -ub 512 --seed 1`, `ppl-eval-half.txt`, 2xP100, same binary/model/cards) - **off 5.7880 +/-0.06762 vs on 5.7842 +/-0.06753**, delta -0.0038 = 18x inside the error bar. **The sm_70 path was gated separately** (mode 2 is the arm that enables it, and it was not covered by the sm_60 run above): 2×V100 `-ts 1.05,0.95`, 200 chunks, two independent replicates per arm — off **5.7713 ±0.06737** (both runs), on **5.7692 ±0.06732** (both runs), delta −0.0021 = 32× inside the error bar, firing proven per device. Flag OFF is byte-identical dispatch. ⚠ The temp-0 sha256 is NOT a meaningful gate here — the same hash is produced by the MXFP4-baseline *model file* — so the quality claim rests on the two perplexity runs alone |
| `PXA_PXQ6_FORCE_PREFILL` | off | TEST ONLY: bypasses the sm_60/70 prefill arch gate so correctness A/Bs can run on other archs | correctness testing only | never in production |

### New in 2026-07-28 — both default OFF

| lever | default | what it does | measured | gate |
|---|---|---|---|---|
| `PXA_PXQ_MMVQ` | `0` (off); **ENHANCE auto-sets 1 (or 2 on an all-sm_61 fleet) when the loaded model carries PXQ4/PXQ4HQ tensors and a DP4A-capable device is present — §0c (2026-07-29); explicit env always wins** | Routes the **decode** GEMV for the PXQ4 / PXQ4HQ tiers to the stock q8_1 **MMVQ** kernel (`mul_mat_vec_q`) instead of the bespoke PXQ 2D driver, and lets the fused up+gate pair reach `ggml_cuda_op_fused_mul_mat_vec_q_id` — one kernel walking the activation once for both, which is the fusion MXFP4 already gets and the per-operand divert throws away. Only the MMVQ batch window (`ny <= MMVQ_MAX_BATCH_SIZE`) is handed over; prefill keeps the 2D drivers untouched. `=1` cc>=CC_VOLTA, `=2` cc>=610 (real DP4A only — sm_60 would run the DP4A *emulation* and lose). Declines if `PXA_PXQ6_BOOK` / `PXA_PXQ6_SUB` / `PXA_PXQ6_SUB_HQ` are set: this TU holds frozen copies of the book/SUB tables and a runtime override would silently diverge from the fused kernels. | **MEASURED 2026-07-28.** Qwable-27B dense PXQ4core, 2×V100, `-c 4096`, n=7, same gguf in every arm: bespoke `mmv` **29.787** → `MMVQ=1` **33.861** = **+13.7%** (MXFP4 on the same config: 36.401, so the gap closes from −18.2% to −7.0%). Prefill also rises 541.3 → 554.7, free, since the flag only claims the decode batch window. `MMVQ=2` is identical to `=1` (33.854), so the DP4A-only variant adds nothing over the Volta gate. **PXQ4core gains far more than PXQ4HQ** (33.861 vs 30.969). On **MoE** (Fusion4-35B) the flag is a no-op unless attention is moved off PXQ6 first — PXQ6 is not MMVQ-registered, so `attn=PXQ6 + MMVQ` is +0.4% while `attn=PXQ4 + MMVQ` is **+6.7%** (93.20 → 99.47). **QUALITY: neutral.** Paired perplexity **at `-b 8 -ub 8`** (see the warning below), 30 chunks, held-out half: dense 5.2113±0.15741 → 5.2149±0.15776 (Δ +0.0036, 44× inside the bar); MoE 5.9029±0.17947 → 5.8998±0.17937 (Δ −0.0031, 58× inside). Opposite signs, each ~2% of its own bar — noise, not a systematic cost. ⚠⚠ **DO NOT gate this lever with default-batch perplexity.** `llama-perplexity` is pure prefill at `-b 512`; the MMVQ dispatch gate is `ne11 <= MMVQ_MAX_BATCH_SIZE` (=8), so at default batch the kernel **never fires** and both arms return bit-identical PPL (5.1599±0.05895 on ours) — a false PASS from a measurement in which the feature was off. The `b=8` arms DIFFER, which is what proves the kernel engaged. This applies to any decode-window lever, not just this one. Background: ncu on the bespoke PXQ mmv shows it **register-limited at 19.8–38.6% occupancy** — precisely the failure mode MMVQ's design avoids — and PXQ was absent from all three MMVQ registration points while MXFP4 rides it on sm_70. Cost is ~1 `vec_dot` plus 3 switch arms. Enable and measure on your own cell before trusting it. | G3-class — different accumulation path; gate on perplexity, not sha |
| `PXA_PXQ_KV` | unset (= `q8_0`, table default). ⚠ This row described a lever that was NOT in any shipped source until 2026-07-28 (docs/code drift — the audit found 0 hits in six branches and an unconditional pin in `llama-quantize.cpp`); it is now IMPLEMENTED on the codec-fixes branch exactly as documented | Overrides the BACKBONE_REV 2 pin that holds `attn_k` / `attn_v` / `attn_v_b` at `q8_0`. Accepts `q8_0｜pxq4｜pxq4hq｜pxq6｜mxfp4`. **Quantizer-side, not runtime.** ⚠ Three gates had to agree before this could work, and each failed *silently*: (1) `--custom-q` registers a rule but `pxa_pxq_backbone_type()` never receives `params`, so the table wins regardless; (2) the rev-2 table pins K/V; (3) `pxq4_tensor_eligible()` excludes attn_k/v **by name** and demoted the tier straight back. All three now clear when the env names a `pxq*` tier. | On Qwable-27B, `attn_k` per layer **10.00 MiB → 2.66 MiB** (vs 5.00 MiB at q8_0); 17 layers each of k and v. Speed effect **NOT yet measured** — the first arm built against a pre-patch binary and came out **byte-identical to its control**, i.e. it tested nothing. | Changes the artifact, not the kernel. **Diff the tier table of the arm against its control before benching** — that is what caught the null arm |

**Method note that generalises.** An A/B arm whose tier table matches its control in every class did
not test your flag; it tested nothing, and it will read as "the lever is worth zero". Dump and diff
tier tables before spending a bench window. Likewise, put a **known-answer coherence gate before any
timing** — an empty completion scores perfectly on tokens/sec — and make that gate tolerant of a
`<think>` block, or reasoning models will report false failures.

---

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

### PXA_SPEC_1ROW + PXA_CUBLAS_EAGER_INIT — MEASURED, both default ON (2026-07-23 fair-battle)

**Origin:** an r/unsloth MTP-prefetch report, checked out line-for-line against our dispatch — the
ne01==1 F32 shared-expert-gate GEMV at MTP spec-verify batch sizes (Ny=2..8) missed every fast
path (dmmv/mmvq/mmq need a quantized src0, batched-cublas needs `src1->ne[2]*ne[3] > 1`, the old
`mul_mat_1row` needed `ggml_nrows(src1)==1`, ROUTER_FUSE needs a `ne01` in [2,4096]) and fell
through to a bare `cublasSgemm` every spec-verify decode step — plus a separately-reported
intermittent `CUBLAS_STATUS_INVALID_VALUE` from cuBLAS's lazy workspace allocation on a
near-full card (external sm_86 report; not reproduced on this box's Tesla-only fleet).

| var | default | what it does | measured | gate class |
|---|---|---|---|---|
| `PXA_SPEC_1ROW` | **on** | extends the single-output-row GEMV (`mul_mat_1row`) from `ne11==1` only to spec-verify batch sizes `Ny<=8` (one CUDA block per token column instead of one block total); `=0` restores the old ne11==1-only dispatch, the missed shapes fall back to cuBLAS as before | **V100 single-card, ub1024 fa-on, MTP n1: +6.6% decode (110.64 vs 103.82 t/s off)** — 6-round fair-battle, median of rounds 3-6. **Flat/harmless on P100 ub512 (+0.06%, noise) and on a 2xV100 `-sm layer` split (+0.7%, noise; also flat under `-ts` PXQ4-MTP, 102.34 vs 101.66)** — the single-GPU win doesn't transfer once the model is split across cards, but nothing regresses. `Ny==1` launches the identical arithmetic DAG as before (bit-identical). `Ny` 2..8 replaces cuBLAS's tiled reduction with a per-thread fp32 dot product | **G3-class, gated like `PXA_PXQ_INT8_PREFILL` (logit-identity + accept-parity, not a sha gate).** Gate 1 (MTP off, temp-0 sha, 1 V100): 5/5 bit-identical, patched vs baseline. Gate 2 (MTP on, 3 arms, 1 V100): baseline == patched+`SPEC_1ROW=0` exact on all 5 prompts (identical sha + identical draft/accept counts, 0.8036 both); patched-default diverges from both on 2/5 prompts — confirmed via logprobs to be a genuine float-ULP near-tie flip (top-2 candidates within ~0.01-0.02 nats at the first diverging token) between the fp32 dot product and cuBLAS's tiled reduction, not a bug. Both diverged responses stay coherent and land on the same correct final answer, phrasing only. MTP accept-rate held (0.8036 off/baseline vs 0.8482 default-on; the higher total draft count is a longer path, not degraded acceptance) |
| `PXA_CUBLAS_EAGER_INIT` | **on** | creates each device's cuBLAS handle + a preallocated `cublasSetWorkspace` buffer (cuBLAS's own default size, 4 MiB pre-Hopper / 32 MiB Hopper+) at backend init, before weights fill VRAM, instead of lazily on first use; `=0` restores stock lazy creation. Alloc failure at init falls back to stock lazy behavior silently | **Perf-neutral by design, measured: P100 ub512 +0.05% (noise); V100 ub1024 −1.5% (110.64→112.36 off), within the fair-battle's own ~2.7% round-to-round noise band.** Measured VRAM cost ~12 MiB/device (a bit above the theoretical 4 MiB workspace, still trivial). All 13 server starts across both cards in the fair-battle came up healthy in 11-12s, no OOM introduced at any point (V100 ub1024 had 995 MiB headroom, P100 ub512 had 1577 MiB) — this box's Tesla cards never sat near-full enough to reproduce the lazy-alloc failure the fix targets, so the stability claim is unverified-but-free here, not disproven | bit-exact (handle/workspace plumbing only, no math path touched) |

Full protocol + raw per-round numbers: `bench/spec1row-fairbattle` results are summarized above;
the same-worktree drift-free A/B build (2 target files only) ran on `build-spec1row` vs
`build-baseline`, model PXA-Fusion4-35B-PXQU16-MTP.

## 6. Speculative-decode / MTP / server levers (engine features, model-dependent)

| var | default | what it does |
|---|---|---|
| `PXA_MTP_LAZY_WARMUP` | off; **ENHANCE default ON (2026-07-29, §0c; `=0` rolls back; REFERENCE off)** | `=1`: skips the per-prompt-batch MTP companion warmup and stops flagging every prompt token as an MTP output (large prefill win on MTP models; temp-0 bit-identical; eager mode byte-unchanged when unset) |
| `PXA_AUTO_SAMPLERS` | unset → engages only under `PXA_ENHANCE=1` (never under `PXA_REFERENCE=1`); `=0` forces off, `=1` forces on at any level | **Server-side model-family sampler DEFAULTS** (2026-07-29): fills only sampler fields the CLI left unset (per-field `--temp/--top-k/--top-p/--min-p` explicitness check; per-request params still win). Family table: `gpt-oss` → temp 1.0 / top_k **100** / top_p 1.0 (top_k=0 — the "official" gpt-oss setting — was measured to **HALVE decode**: 31 vs 65 t/s, full ~201k-vocab CPU sort per token, gpt-oss-120b 6-card cell 2026-07-04); `qwen35moe`/`qwen3next`/`qwen35` → the official Qwen no-think set temp 0.7 / top_k 20 / top_p 0.8 / min_p 0; `laguna` → the same Qwen-family set (HEURISTIC — no per-model sampler sweep yet, the log says so). Unknown arch → stock defaults kept, logged. Plus a generic guard: a resolved `top_k<=0` is clamped to 100 (loud). Every applied default logs value+reason+override. |
| `PXA_MTP_ADAPTIVE` (+`_K`) | off | adaptive draft-length cap for MTP speculation (per-slot acceptance feedback) |
| `PXA_NGRAM_RESET_STREAK` | 3 | streak threshold for the ngram-speculator map reset; `0` = never reset on acceptance (helps varied-writer models keep the map warm) |
| `PXA_NP_SPEC_GATE` | off | opt-in gate for speculation under np>1 (shelved feature; leave unset) |
| `PXA_SPEC_RELAXED` (+`_PMIN`) | off | relaxed draft-acceptance experiment (G3-class; not recommended) |
| `PXA_SHARED_MTP_BATCH_COMMIT` | on | batches MTP commit work across slots; `=0` restores fully-serial behavior (rollback knob) |
| `PXA_MOE_FASTTG_MAX_NY` | 8 | max verify-batch Ny that stays on the per-token fast-TG path; `=1` routes Ny>1 MTP verify batches to the expert-grouped batched path (weights read once per traversal) |
| `PXA_MOE_GROUPED` / `_VERIFY` / `PXA_MOE_BATCHED_VERIFY` | off | A1 expert-grouped batched-MoE verify kernels + shadow-verify harness (G3-class; incompatible with graph capture). ⚠ fixed 2026-07-30: `PXA_MOE_GROUPED_VERIFY` was presence-tested (`=0` still enabled shadow-verify); it is now value-tested like every other lever |
| `PXA_PROMPT_INTERLEAVE` | on | co-decodes resident slots while another slot prefills; `=0` reverts to serialize-behind-decode (ops kill-switch) |
| `PXA_HEALTH_STALL_MS` | 60000 | `/health` reports stalled if a queued probe can't be served within the deadline (`0` = off) — keeps health honest instead of parking an HTTP worker |
| `PXA_WEDGE_EXIT_MS` | **180000** (⚠ doc previously said 0 — the code has always defaulted to 180000/ON) | in-server stall watchdog: one `llama_decode` stuck longer than this across 3 consecutive 5 s checks triggers the runtime-appropriate action (`0` = off). **Container-aware since 2026-07-30:** in a container → `_exit(42)` (the orchestrator restarts); bare metal → an in-process recovery attempt first (`llama_decode_stop()` soft-wedge abort at strike 3), then `_exit(41)` at strike 6 with a loud "no supervisor detected — restart manually" banner. Exit codes are distinct (41 bare / 42 container) so logs can tell the contracts apart. The monitor never forks or re-execs in either mode. |
| `PXA_IN_CONTAINER` | unset (auto-detect) | forces the container verdict used by `PXA_WEDGE_EXIT_v1`: `=1` container contract, `=0` bare-metal contract. Precedence: this override > auto-detection (`/.dockerenv`, `/run/.containerenv`, `container=` in `/proc/1/environ`, docker/containerd/kubepods/libpod/lxc in `/proc/1/cgroup`, container paths in `/proc/self/mountinfo`). The verdict + evidence is printed once at startup (`PXA_CONTAINER_AWARE_v1: runtime=...`) |
| `PXA_PORT_GUARD` | on | pre-bind probe: if a LIVE listener already answers on the target host:port the server refuses to start with an actionable error instead of silently coexisting/fighting (the 2026-07-30 bare-metal duplicate-server incident); `=0` bypasses |
| `GGML_NO_BACKTRACE` | unset | `=1`: on abort, skip the fork-a-debugger backtrace entirely (symbols-only). The fork path is now safe (`PXA_BT_NOFORK_v1`: child `_exit`s instead of `exit`ing — the old `exit()` could deadlock a fork child of the multithreaded CUDA process into an immortal duplicate server; parent waits max 15 s then SIGKILLs), but production servers may still prefer to skip the debugger attach, which stops the process while gdb runs |

## 7. Multi-GPU / partition levers

| var | default | what it does |
|---|---|---|
| `PXA_EXPERT_SHARD` | unset | comma list of device indices: shards expert tensors of the listed home devices across the group (consumed by the CUDA MoE up/gate shard branch). Unset = bit-identical stock placement |
| `PXA_REPLICATE_RECURRENT` | off | replicates recurrent (DeltaNet) state full-head per device instead of head-splitting — trades memory for no cross-device reduce |
| `PXA_REDUCE_CAPTURE` | off | allows the cross-device reduce inside graph capture once the per-device events exist |

## 8. Quantizer inputs (build-time, `llama-quantize`)

| var | what it does |
|---|---|
| `PXA_PXQ_COMPOSITION_OVERRIDE` (env) / `--pxq-composition-override` (CLI) | **SAFETY GATE, not a performance knob — read before ever setting it.** After every PXQ-target quantize the writer sums output bytes by type and asserts (a) PXQ-family bytes ≥ 50% of the file and (b) a uniform PXQ target actually produced bytes of its named tier. On failure it **deletes the mislabelled output** and aborts with `PXQ composition assertion: target <FTYPE> produced <X>% PXQ-family bytes…`. This assertion exists because exactly such a file shipped once: an artifact labelled "PXQ4" that was 27% PXQ / 68% MXFP4 with attention and the shared expert carrying none, which entered a public README as *"tier/codec the only variable"* and stood for weeks (the retracted 104.06). **If you trip this assertion and reach for the override, you are about to ship the same mislabel — rename the output honestly instead.** (Documented 2026-07-30; the guard itself shipped 2026-07-28, `src/llama-quantize.cpp`.) |
| `PXA_PXQ6_BOOK` / `PXA_PXQ2_BOOK` / `PXA_PXQ3_BOOK` / `PXA_PXQ6R_BOOK` | override the frozen codebook — for lab experiments; shipped books are compiled in and sha-pinned |
| `PXA_PXQ6_SUB` / `_SUB_HQ` / `PXA_PXQ2_SUB` / `PXA_PXQ3_SUB` | override the sub-scale LUTs (lab) |
| `PXA_PXQ6_ANCHOR_FIT` / `PXA_PXQ2_ANCHOR_FIT` / `PXA_PXQ3_ANCHOR_FIT` | anchor-fit strategy toggles in the native quantizers (lab; defaults are the shipped, gate-proven settings) |
| `PXA_PXQ_HEAD` | output-head type for the PXQ ftypes (`output.weight` only, NOT `token_embd`): `q8_0` (default) \| `q6_k` \| `f16`; unknown values warn and fall back to `q8_0`. Default q8_0 = **+3.0% P100 decode measured, all rounds** (q8_0 rides Pascal's fast DMMV path where K-quant heads ride the slow scalar path, and the head runs every token over the full ~151k vocab) — and q8_0 is higher precision than the old q6_k default, so speed AND quality. An explicit `--output-tensor-type` still wins |
| `PXA_PXQ_NATIVE` | widens PXQ slab-tier eligibility beyond the routed expert stacks (`*_exps.weight`, always eligible) for the “100% native PXQ, no inherited MXFP4” work. Comma-separated class list: `shexp` (`ffn_{up,gate,down}_shexp.weight`), `attn` (`attn_q` / `attn_qkv` / `attn_output` / `attn_gate.weight`), `embd` (`token_embd.weight`), `ssm` (`ssm_alpha` / `ssm_beta` / `ssm_out.weight` — **risky, recurrent state**), or `all`. Unset (default) = historical behaviour, routed experts only. The geometry gate still applies to every class (`rows % 64 == 0 && K % 32 == 0`); a tensor that fails it is demoted to MXFP4 by the caller, so a bad list can never produce an unloadable file. **Quality:** `attn` measured *better* than the tier it replaces — native-attn PXQ4 beats PXQ6 on a 4000-chunk paired perplexity (t = +15.64) while being 4 GB smaller. **Speed:** the resulting file needs `PXA_PXQ4_2D=1` at runtime or its attention nodes fall back to dequant→cuBLAS; even with it, 2×V100 decode is −10% and prefill −12% vs the MXFP4-attention file, while 4×P100 is at decode parity (+0.5%) and −2.5% prefill (see §4 and the kernel-parity report) ⚠ **Superseded as a default by `PXA_PXQ_BACKBONE` (BACKBONE_REV 2, 2026-07-26):** the rev-2 table now promotes `attn` / `shexp` / dense-FFN / `token_embd` in every PXQ tier without this lever, and to a *higher* tier than this lever gives (PXQ6 rather than the file's own expert tier). `PXA_PXQ_NATIVE` remains a research override for widening eligibility to classes the table does not claim (notably `ssm`). ⚠ **To reproduce a pre-2026-07-26 `PXQ4N-attn`-style artifact byte-for-byte you must ALSO set `PXA_PXQ_BACKBONE=legacy`** — otherwise the backbone table wins and you are measuring a different file than the one the historical number came from. A second latent trap, fixed the same day: `embd` never actually fired (its name test was a strict-suffix match that cannot match the top-level `token_embd.weight`), so any "embd is a null" result predating the fix is null **by construction**, not evidence |
| `PXA_PXQU_DIR` | directory for `--pxq-universal` preset `.tiers` files (default `pxa-bench/pxq-universal/`; presets are also baked into the binary) |
| `--imatrix` (behaviour note, not a lever) | every PXQ codec **consumes** the importance matrix: the anchor pick and the code/sub-scale selection both minimise a diagonal-weighted SSE (`err += w[i]·e²`), per-expert columns when `imx_size == K·E` and a shared column when `== K`. Since 2026-07-26 a **dead-column guard** sits in front of it: an all-zero (or non-finite / negative) column would make *every* candidate score exactly `0.0`, silently collapsing the argmin into its tie-break, so such a column is fit **unweighted** instead and the total is reported at completion. This is the normal case for routed experts that never fired during calibration — expected on a 256-expert MoE — but a large count means the corpus is too thin for the expert count. Corpus guidance: ≥2000 chunks, and compute the imatrix with `-sm graph` (pure prefill, measured +163%) |
| `PXA_PXQ_BACKBONE` | **the backbone allocation table (BACKBONE_REV 2, default ON since 2026-07-26).** Before rev 2 every PXQ tier quantized its routed experts natively and then flattened *everything else* — attention, `token_embd`, shared experts, dense FFN — to flat MXFP4. Measured on Laguna-S-2.1 vs the stock `Q4_K_M` recipe from the same bf16 source: `attn_output` **3.2×** the error (0.1157 vs 0.0362), `attn_gate` **1.6×** at the worst absolute value (0.1609), `attn_q` / `token_embd` / `ffn_*_shexp` 1.6× each — while the PXQ6 file was **8% BIGGER** than the Q4_K_M it lost to. Rev 2 replaces the flatten with a per-class table: `attn_q`/`attn_qkv`/`attn_output`/per-channel `attn_gate`/`ffn_*_shexp`/dense `ffn_*` → the native PXQ tier one notch above the experts (PXQ6 for the PXQ4/PXQ6 tiers, PXQ4HQ for PXQ3, PXQ4 for PXQ2); `attn_k`/`attn_v` → `q8_0` (unchanged, already at Q4_K_M parity); **per-HEAD `attn_gate` (`ne[1] ≤ 256`) → `f16`** — the Laguna killer, 72 softplus scalars per layer at 0.03% of the file whose 16% error becomes a 31–38% functional error on every head; `token_embd` → `q6_k` (a row *gather*, never a GEMM, so the Pascal "k-quants are slow" rule does not apply); `output` → `q8_0` via `PXA_PXQ_HEAD` (unchanged); geometry failures land on `q8_0` instead of a silent MXFP4 demotion. `ssm_*`, `nextn.*` (MTP), the router and the norms are untouched. **Cost: +0.45 GiB on a 62.0 GiB Laguna PXQ4 = +0.73%.** Values: unset/`1`/`v2` = rev 2 [default]; `legacy`/`0` = the exact pre-rev-2 recipe (byte-reproduces old artifacts — use it for any A/B against a pre-2026-07-26 file); `lite` = **only the promotions that are free at decode** — per-head `attn_gate`→`f16`, `token_embd`→`q6_k`, `attn_k`/`attn_v`→`q8_0`, geometry-fail→`q8_0` — leaving the dense GEMM backbone on MXFP4. This keeps the class that actually corrupted Laguna (the per-head gate) at essentially zero decode cost, and is the option to reach for if the full table's decode cost is unacceptable on your cards; `hq` = the pre-registered fallback, PXQ4HQ instead of PXQ6 on the 4-/5-bit tiers (≈82% of the modelled gain at +0.26 bpw instead of +1.02, and PXQ4HQ — unlike PXQ6 — has a CPU panel-dequant, so the file stays partial-offload capable); `universal` = additionally apply the table to `PXQ_UNIVERSAL` and `PXQ1` (off by default: a PXQU tier map is user-authored per-tensor and neither tier has measured backbone evidence). Recipe-level only — **no on-disk slab layout changes, so every previously shipped PXQ file still loads.** Every file records which table built it in the `pxa.pxq.backbone_rev` / `pxa.pxq.backbone_map` gguf KVs. ⚠ A rev-2 file needs `PXA_PXQ4_2D` at runtime (now default ON) or its promoted backbone falls back to a per-token dequant→cuBLAS — **measured 45.2 → 9.3 t/s, −79%**, on a rev-2 Laguna-XS on 2×P100. ⚠⚠ **Decode cost of the full table, measured and unresolved:** same file/cards/harness (n_predict 128, 4 reps, one instance — indicative, NOT the fill-5739 n≥8 interleaved standard), decode is legacy-backbone **71.6** t/s, `hq` **51.6** (−28%), full PXQ6 table **45.2** (−37%), and `PXA_PXQ4_2D_KSPLIT=1` makes it worse still at **32.1** (extending that lever's known P100 loss to the previously-untested `panels=8` shexp shapes — leave it OFF). Cause: in a 256-expert MoE only ~8/256 of the expert bytes are touched per token, so the always-resident backbone is roughly HALF the per-token decode traffic, and the PXQ 2D decode mmv is slower per byte than MXFP4's mmvq/DMMV on Pascal. Confirmed like-for-like on the real artifact — Laguna-S-2.1 PXQ4 built from the same bf16 source, 6 Tesla cards, `-c 4096`, n=1: shipped legacy-backbone **34.8** t/s vs rev-2 **26.6** t/s = **−23.6%**. Until that kernel is competitive, `lite` is the speed-preserving way to take the gate/embedding half of the fix |
| `PXA_PXQ_ERRBUDGET` (+ `_REF`, `_MAXELEM`) | **default off** (it costs a full dequant pass). Quantize-time acceptance gate: dequantizes each written tensor straight back, accumulates relative RMS error vs the f32 source **per tensor class**, prints a table at completion and writes `<output>.errbudget.tsv`. `PXA_PXQ_ERRBUDGET_REF=<tsv>` compares against a stored budget (e.g. the `Q4_K_M` control's) and WARNs at **>1.5×** on any class; `PXA_PXQ_ERRBUDGET_MAXELEM` caps per-tensor sampling to whole rows (default 64M elements, `0` = every element). This is the instrument that would have caught the corrupt Laguna artifacts on disk — the signature (`attn_gate` 0.1609, `attn_output` at 3.2× the control) was in the numbers before a single token was generated. **Limitation, stated not hidden:** the PXQ slab types are CUDA-only and have no CPU `to_float`, so expert classes report `n/a` — the report covers the BACKBONE, which is the entire surface the bug lived on and the entire surface rev 2 changes. Row-interleaved (`*_R4`) types are also reported `n/a` rather than decoded row-wise incorrectly |

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

## Updates — 2026-07-28 — PXQ_CANON_v1, merged sm70-stack levers, backbone rev additions

| lever | default | what it does | measured | gate |
|---|---|---|---|---|
| `PXA_PXQ_DENSE_GATEUP` | **on** (merged from `sm70-stack` 2026-07-28) | Dense `GGML_OP_FUSED_UP_GATE` nodes with PXQ-typed up/gate now reach the fused `k_pxq6_gateup_mmv` / `_ksplit_gen` family through the one-entry `{0}` ids idiom instead of splitting into two `k_pxq6_mmv` calls + an elementwise GLU. SILU only; declines (with a one-shot FIRING/DECLINED log) for anything it cannot serve. ⚠ NOT bit-exact vs the unfused path (GLU association: fused vs split-node order) — a PATH choice, deterministic under a fixed config; `PXA_PXQ_DENSE_GATEUP=0` restores the old path. | Qwable-27B dense, 2×V100: part of the −21.6% → −4.25% recovery; 2×P100 stack total **+45.8% vs MXFP4** (14.268 → 20.796 t/s). Composes with the S-split; under PXQ_CANON_v1 the S choice no longer affects its output bits. | behaviour-gauntlet (fused-vs-unfused association), ON |
| `PXA_PXQ_DENSE_GATEUP_SPLIT` / `PXA_PXQ_DENSE_GATEUP_TARGET` | split follows arch default (cc 7.0: never split; else shared target) | The fused gateup carries two weights per block, so its grid does 2× the work of the mmv grid; sm_70 is past the knee at typical dense shapes and never splits (+0.85%), sm_60 keeps the shared target (−8.6% if disabled). | measured on Qwable-27B PXQ4core (see the sm70-stack merge commits) | bit-exact under PXQ_CANON_v1 |
| `PXA_PXQ4_2D_SPLIT_TARGET` (defaults) | **16×nsm on cc 7.0, 8×nsm on cc 6.0, 2×nsm elsewhere** (2026-07-28, was 2×nsm everywhere) | Measured sweeps on both Teslas showed the old 2×nsm default sat far below the knee (V100: 160→27.89 t/s vs 1280→30.70; P100: 112→14.81 vs 448→19.17). Under PXQ_CANON_v1 the retune is output-neutral (S does not change bits) — pre-canon it silently changed greedy output, which is why it was never shipped as a default before. | see the `bd0977e` sweep table | bit-exact under PXQ_CANON_v1 |
| `PXA_PXQ_BACKBONE=core` (new token) | off (rev-2 table default) | GEMM backbone (attn_q/qkv/output/per-channel gate, shexp, dense FFN) at the byte-parity **PXQ4 core** tier (4.2526 bpw vs MXFP4 4.2500) instead of PXQ6 — every backbone class becomes `PXA_PXQ_MMVQ`-eligible, which is the sm_70 recipe (attn PXQ6+MMVQ = +0.4%, attn PXQ4+MMVQ = +6.7% MoE decode). Previously only expressible via `--custom-q`. | MoE fidelity/speed arms: see the 2026-07-28 arm matrix in the codec-fix report | quantizer-side |
| `PXA_PXQ_KV` | unset (=q8_0) | **Implemented 2026-07-28** (was a docs-only phantom — the audit found 0 source hits and an unconditional pin). Overrides the rev-2 K/V pin; accepts `q8_0`/`pxq4`/`pxq4hq`/`pxq6`/`mxfp4`. `mxfp4` restores true byte parity vs a flat-MXFP4 legacy control (whose K/V are MXFP4, not q8_0 — the pin rationale was wrong for that comparison). | artifact effect verified earlier (attn_k 10.00 → 2.66 MiB/layer on Qwable); speed/quality of the pxq tiers on K/V still unmeasured | quantizer-side |

**Backbone rev additions (2026-07-28, same branch):** `ssm_alpha`/`ssm_beta` (geometrically impossible for the 64-row PXQ panel) and non-expert `nextn.*` MTP companions now land on **q8_0** instead of flat MXFP4 (zero-MXFP4 rule; +0.01% file size). `ssm_out` stays on the legacy landing pending measurement — build the native arm with `--custom-q "(ssm_out\.weight)=pxq4"`. **Explicit `--custom-q` PXQ targets are now honoured in every backbone mode whenever slab geometry allows** — before this, `PXA_PXQ_BACKBONE=legacy` + a custom PXQ rule silently demoted back to MXFP4 and the arm came out byte-identical to its control (two agents measured "nothing" this way).

**⚠ Gate-methodology warning (systemic, 2026-07-28):** any DECODE-window lever (dispatch gated on `ne11 <= 8`-class conditions) that was certified with default-batch `llama-perplexity` was certified by a run in which the lever NEVER EXECUTED — `-b 512` perplexity is pure prefill, both arms are bit-identical, and the gate returns a false PASS. Gate decode levers at `-b 8 -ub 8` (the arms DIFFERING is what proves the kernel engaged), or with temp-0 + fixed-seed-sampling generation hashes as PXQ_CANON_v1 was. Audit note: `PXA_PXQ4_2D_SPLIT`/`PXA_PXQ6_KSPLIT_GEN` were certified partly through 64-chunk NLL aggregates; their real generation-level failure mode (the 122B kaskal runaway) was invisible to that gate and is now covered by the canon bit-exactness regression instead.

## Updates — 2026-07-29 — ship-recipe measurements (all on the merged engine, main c99f1d9 lineage)

- **`PXA_PXQ_MMVQ` — measured for the ship decision (35B MoE, 2×V100 ts1.05,0.95, fill 6018, n=9/arm
  interleaved):** core-backbone file 75.97 → **93.43-93.89 t/s (+23%)**; ship file (core+ssm_out→PXQ4)
  75.07 → **95.17 (+27%, legacy-recipe parity)**. Fidelity, HONEST decode-window protocol (`-b 8 -ub 8`,
  300 paired chunks, V100 — the kernel provably engaged: series differ): **+0.053% ppl (t=17)** — real,
  precisely quantified, and 1/50th of the recipe's −2.72% quality win. **Ships ON in the 35B serve env.**
  ⚠ CAVEATS THAT NOW MATTER (default-on ship): (1) the MMVQ TU carries **frozen copies** of the book/SUB
  tables and does NOT honour `PXA_PXQ6_BOOK`/`_SUB`/`_SUB_HQ` runtime uploads — the dispatch declines when
  those envs are set, so an override experiment silently reverts to the bespoke path; (2) `=1` gates on
  cc≥7.0 — on sm_60 it is an intentional no-op (P100 keeps the bespoke path, which beats MXFP4 there);
  a "null" measured on P100s at `=1` is null by ARCH GATE, not evidence (this bit us once this session).
- **`PXA_PXQ_KV=pxq4` — first-ever measurement (the lever was a docs-only phantom until 2026-07-28):**
  artifact verified (all 20 attn_k/v tensors land PXQ4, file −11 MB vs q8_0 pin); paired 1000-chunk ppl
  + V100 speed cells queued (`nll-A6-kv4.txt` / speed matrix follow-up) — row to be completed with those
  numbers before any public release.
- **`PXA_PXQ4_2D_SPLIT` / canon price:** the coordinator's 2×2 on the published 35B artifact priced
  PXQ_CANON_v1 at **−1.9% decode** on the bespoke MoE path (the ~27% regression hypothesis was tested
  and killed; the 104.06 figure it was anchored to was retracted as an artifact-composition error).
  Bit-exactness costs 1.9% and is worth it: it retires the entire config-dependent greedy-flip class
  (S, target, device SM count, lever toggles).

## Updates — 2026-07-30 — container-aware wedge exit + fork root cause + strict quantize type args

- **`PXA_BT_NOFORK_v1` (ggml.c):** root cause of the bare-metal duplicate-server incident
  (DGX-1, 2026-07-30). `ggml_abort` → `ggml_print_backtrace()` forks to attach gdb; with no
  debugger on PATH the child called `exit()` — atexit/dtor handlers in a fork child of a
  multithreaded CUDA process deadlock on mutexes held at fork time, leaving an immortal
  duplicate of the server (parent's argv, PPID = parent, holding a dup of the listening
  socket) while the parent blocked forever in `waitpid`. Fixes: child `_exit`s, parent waits
  ≤15 s then SIGKILLs, `GGML_NO_BACKTRACE` honored, `fork()` failure falls back to symbols.
- **`PXA_CONTAINER_AWARE_v1` + `PXA_IN_CONTAINER` + exit codes 41/42:** see the section-6 rows.
- **`PXA_PORT_GUARD_v1`:** see the section-6 row.
- **`PXA_UTF8_FINAL_v1` (server):** final responses no longer 500 when a generation stops
  mid-multibyte-codepoint (`n_predict` cap / `ignore_eos`) — the incomplete trailing UTF-8
  bytes are dropped from the final payload exactly as the streaming path already did
  per-chunk. Closes the "incomplete UTF-8 string; last byte: 0x.." open bug (2026-07-29).
- **`PXA_TYPEARG_STRICT_v1` (`llama-quantize`):** all 12 `--*-type` flags now HARD-FAIL on an
  unparseable type name instead of silently ignoring the flag (exit 0, wrong artifact — the
  2026-07-29 open bug), and type names match case-insensitively (`q6_k` == `q6_K`).

### 2026-07-30 — previously UNDOCUMENTED levers (the A14 inventory closed out)

Every row here existed in source with no doc row (flagged by the 2026-07-29 A14 audit).
Documented so the sm_70 dead ends are reproducible instead of folklore:

| var | default | where | what it does |
|---|---|---|---|
| `PXA_PXQ_MMVQ_ROWS` | **4** (accepts 1\|2\|4\|8\|16) | `pxq-mmvq.cuh` | rows-per-block for the PXQ MMVQ decode kernel. ⚠ ROWS≥8 is a MEASURED DEAD END on sm_70 (register-bound: ROWS=1 62 reg/0 B spill → ROWS=16 183 reg/256 B spill; the "37.89 t/s ROWS=8 beats MXFP4" figure was FABRICATED — no on-disk log). This knob is how those dead ends were expressed |
| `PXA_PXQ_MMVQ_VDR` | **2** (accepts 2\|4) | `pxq-mmvq.cuh` | values-per-dot-iteration for the same kernel. VDR=4 measured −4 to −6% — dead end, kept for reproducibility |
| `PXA_MOE_GROUPED_VERIFY` | off | `grouped_moe_verify.cuh` | shadow-verify for the A1 grouped MoE path: grouped writes a private scratch, per-token path stays authoritative, mismatches print. ⚠ **Fixed 2026-07-30: this was PRESENCE-tested (`=0` still enabled it — the only PXA lever where `=0` meant ON); it is now value-tested like every other lever** |
| `PXA_MTP_ADAPTIVE_K` | 0 | `common/speculative.cpp` | acceptance-EMA adaptive draft depth (companion to `PXA_MTP_ADAPTIVE`) |
| `PXA_SPEC_RELAXED_PMIN` | 0.05 | `common/sampling.cpp` | p_min floor for the `PXA_SPEC_RELAXED` experiment (G3-class; not recommended) |
| `PXA_REP_GUARD` | level default | `common/sampling.cpp` | repetition-attractor guard; supersedes `PXA_PXQ1_REP_GUARD` (back-compat alias). ⚠ 2026-07-30: ENHANCE auto-arm narrowed back to PXQ1-bearing files only — the any-PXQ broadening measurably flipped a temp-0 exact answer on PXQ4 (see the §1 row for the full measurement); `=1` forces any-PXQ, `=0` disables |
| `PXA_ENHANCE_DBG` | off | `pxa-enhance.cuh` | prints the detected topology + every chosen ENHANCE config (diagnostic; use it to audit what the adaptive layer decided) |
| `PXA_CUDA_GRAPH_MOE` / `_LRU` / `_REARM` / `_BATCH_MAX_NY` | — | `ggml-cuda.cu` | CUDA-graph capture family for the MoE decode path (graphs measured null ×3 on the sm_70 dense cell — see §G of the action register — but the knobs are live for other cells) |
| `PXA_PXQ6R_ANCHOR_FIT` / `PXA_PXQ6R_SUB` | — | `pxq6.cuh` | PXQ6R tier lab knobs (anchor-fit strategy / sub-scale LUT override) |

### 2026-07-30 late — sampler blast radius + the checkpoint flags documented

| var / flag | default | what it does |
|---|---|---|
| `PXA_SAMPLE_SOFTFAIL_v1` (behavior, no env to enable) / `PXA_SAMPLE_ABORT` | soft-fail ON; `PXA_SAMPLE_ABORT=1` restores the old fatal | An unsampleable probability vector (NaN-cascade from garbage logits) used to `GGML_ABORT` the WHOLE server at `llama-sampling.cpp` — one poisoned slot killing every co-resident generation, the same blast-radius class as the old ret=-3 unwind. Observed live 2026-07-30 on the DGX teacher (hy_v3, checkpoint-restore → full-reprocess → "Failed to sample token" → abort → the abort-path fork hang). Now: forensic dump on first occurrence (`probabilities.txt` kept), loud `PXA_SAMPLE_SOFTFAIL_v1` log, deterministic fallback token (max finite logit, else first candidate) — the request may degenerate, the server survives |
| `--ctx-checkpoints N` | **32** per slot | max recurrent/KV context checkpoints per slot (0 disables checkpointing). ⚠ Load-bearing operational knowledge: **`--ctx-checkpoints 0` was the H2 mitigation** for the recurrent-checkpoint contamination on `qwen35moe` hybrids **on pre-2026-07-28 binaries**; the canonical tree carries `PXA_CKPT_HYBRID_ROLLBACK_v1` which fixes the contamination properly (rollback gate on `seq_pos_max`, checkpoint match on `cur.pos_max`). Any binary WITHOUT that fix (e.g. the DGX `build-pxa-new`, `build-mmfast`) must either serve `cache_prompt:false` client-side or be upgraded — the 2026-07-30 DGX fatal chain started exactly here. Note some older builds do not have this flag at all: check `--help` before relying on it |
| `--ctx-checkpoints-interval N` | 512 | min tokens between checkpoints |
| `--ctx-checkpoints-tolerance N` | 5 | tokens-before-full-prompt at which a checkpoint is created |
| `--ctx-checkpoints-eviction NAME` | variance | eviction strategy: `fifo`, `variance`, `auto`(=variance); variance keeps uniform positional coverage |

### 2026-07-30 latest — the "sm_61 arith flip" was the PXQ repetition guard (dead-end + fix, fully measured)

Verification class: **root-caused with both-direction A/B; fix verified on the shipped binary.**
Cell for every arm: Laguna-S-2.1 **PXQ4-core**, 4×P100 + 1080 Ti, `-ts 8,8,8,8,3`, `-c 8192
-np 1 -b 2048 -ub 2048 -fa on` f16 KV `--jinja`, `/v1/chat/completions` temp 0 / top_k 1 /
`cache_prompt:false`, fresh server per arm, 3 reps per arm (all arms ×3 identical).

| arm | env | 4183×391 | verdict |
|---|---|---|---|
| B | none (DEFAULT) | **1,635,553 ✓** | control |
| C | `PXA_ENHANCE=1` +MMVQ+graphs-off | 1,635,593 ✗ | repro of the "flip" |
| D | C + `PXA_PXQ_INT8_PREFILL=0` | 1,635,593 ✗ | **INT8_PREFILL exonerated** |
| E1 | `PXA_ENHANCE=1` alone | 1,635,593 ✗ | ENHANCE is the carrier |
| E2 | `GGML_CUDA_DISABLE_GRAPHS=1` alone | 1,635,553 ✓ | null |
| E3 | `PXA_PXQ_MMVQ=1` alone | 1,635,553 ✓ | null |
| F1 | ENHANCE + `PXA_SPEC_RELAXED=0` | 1,635,593 ✗ | exonerated |
| F2 | ENHANCE + `PXA_AUTO_SAMPLERS=0` | 1,635,593 ✗ | exonerated |
| G1 | ENHANCE + `PXA_REP_GUARD=0` | **1,635,553 ✓** | **conviction (direction 1)** |
| G2 | DEFAULT + `PXA_REP_GUARD=1` | 1,635,593 ✗ | **conviction (direction 2)** |
| H1 | fixed binary, ENHANCE | **1,635,553 ✓** | fix verified |
| H2 | fixed binary, `PXA_REP_GUARD=1` | 1,635,593 ✗ + `[FORCED]` log | force path intact |

Also measured: the ORIGINAL failing 7-card cell (2×V100+4×P100+1080Ti, `-ts 1,1,8,8,8,8,3`,
same file/env as the incident incl. graphs-off) answered **correct ×3** when served standalone
— the incident's 4/5 ran co-resident with a second brain on the V100s; with the guard armed
the flip is near-tie/cell-fragile, which is exactly why it masqueraded as a numerics bug.
**Dead ends recorded so nobody re-chases:** sm_61/1080Ti DP4A numerics, INT8_PREFILL,
MMVQ, CUDA-graphs-off, and 7-card topology are ALL exonerated for this symptom.
