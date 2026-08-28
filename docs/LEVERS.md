# PXA levers — the supported `PXA_*` environment variables

Every lever below is read by the shipped engine, and every number quoted with it was
measured on the reference hardware described in [`bench/README.md`](../bench/README.md)
(Tesla P100-PCIE-16GB / Tesla V100-PCIE-16GB / GTX 1080 Ti). Where no number exists the
row says **unmeasured** rather than guessing.

**This document is the supported surface.** The engine source contains several hundred
other `PXA_*` names — experiment records, diagnostics, revert switches and per-fix guards
kept for the paper trail. They are not a configuration API, they are not documented here,
and setting one by hand overrides the per-architecture gating that the levels in §1 exist
to do for you. If a variable is not in this file, leave it unset.

---

## §1 The three config levels

Set at most one. They move the *default* every other lever falls back to; an explicit
per-lever variable always wins over the level.

| variable | level | what it selects |
|---|---|---|
| `PXA_REFERENCE=1` | 0 REFERENCE | every PXA lever OFF/0 — the pure reference kernel and dispatch paths. The bit-exact audit and A/B baseline. Also stands the server posture layer (§4) down entirely. |
| *(neither set)* | 1 DEFAULT | the shipped defaults: the §2 bit-exact winner set, plus `PXA_VOLTA_CUBLAS_NE11=64`. This is what the published per-card numbers assume. |
| `PXA_ENHANCE=1` | 2 ENHANCE | DEFAULT plus the per-architecture measured levers whose ship gates passed. |

`PXA_REFERENCE` wins if both are set.

What ENHANCE adds, and the ship gate each one passed:

| lever | scope | measurement |
|---|---|---|
| `PXA_PXQ_INT8_PREFILL=1` | **sm_61 only** | +182% prefill on a 1080 Ti (251 → 709 t/s, PXQ2, cold 5.8k prompt, `-ub 768`); decode byte-untouched |
| `PXA_ROUTER_FUSE=1` | **sm_70 only** | +5.1–7.0% decode on V100; a −1.6% *loss* on sm_60, so Pascal stays off |
| `PXA_SPEC_RELAXED=1` | speculative lanes only | — |
| `PXA_PXQ_MMVQ` | PXQ4/PXQ4HQ model × DP4A-capable device | see §3 |
| `PXA_MTP_LAZY_WARMUP` | when MTP is active | see §5 |

ENHANCE is **(device × model) adaptive**: selection reads the loaded model's tensor census
as well as the device fleet, and every auto-set decision is printed at startup with its
reason, so the configuration actually in force is auditable instead of inferred.

Value semantics: gates are **value-tested, not presence-tested**. `PXA_FOO=0` disables the
lever; it does not enable it by virtue of being set.

---

## §2 The bit-exact default winner set (ON out of the box)

These are the fused PXQ kernels. Each is **bit-exact** against the unfused reference path —
`memcmp`-identical device output, proven per kernel in
[`bench/determinism-gates.md`](../bench/determinism-gates.md). Setting any one to `0`
reverts that kernel to the reference path.

| lever | default | what it fuses |
|---|---|---|
| `PXA_PXQ6_KSPLIT` | **1** | K1 decode gate/up K-split, bit-exact kseg form |
| `PXA_PXQ6_VECX` | **1** | K2 `float4` activation loads |
| `PXA_PXQ6_GUFUSE` | **1** | K3 fused up+gate GEMM with GLU epilogue |
| `PXA_PXQ6_SCATFUSE` | **1** | K3 down-GEMM scatter fusion |
| `PXA_PXQ6_RAGTAIL` | **1** | K4 ragged-tile FMA skip |
| `PXA_G2_ADDFUSE` | **1** | G2-F4 residual-add fusion (ADD + FUSED_RMS_NORM pair, MUL_MULTI_ADD epilogue). +1.9% decode V100, +1.2% P100 |
| `PXA_FUSE_DELTANET` | **3** | DeltaNet decode glue fusion, bitmask: bit0 = qk-norm/state-writeback cluster, bit1 = out-gate rms+silu. +3.7% P100 decode on the published PXQU-16 config. `=0` restores the eager path |
| `PXA_PXQ1` | **1** | the PXQ1 fused decode dispatch. `=0` returns to dequant + cuBLAS per token — measured 36.0 → 11.8 t/s on a 122B-A10B PXQU24 artifact. A one-shot sign-book self-check disables the fused path on its own if it ever fails |
| `PXA_VOLTA_CUBLAS_NE11` | **64** | the `ne11` threshold at which Volta hands off to cuBLAS |

Format families: `PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1` enable the PXQ format families. Set all
three for a `PXQ_UNIVERSAL` / mixed-tier model. (The env names keep the pre-rename `PXQ6`
identifier for the 4-bit tier — see [`RENAME-MAP.md`](RENAME-MAP.md).)

---

## §3 Config-specific and opt-in levers (default OFF)

| lever | default | scope and measurement |
|---|---|---|
| `PXA_PXQ_MMVQ` | auto | Routes PXQ4/PXQ4HQ decode to the stock q8_1 MMVQ kernel. **+13.7% dense decode** (29.787 → 33.861, 2×V100), +6.7% on MoE with PXQ4 attention. Quality-neutral (paired PPL Δ +0.0036 dense / −0.0031 MoE — opposite signs, i.e. noise). **G3-class: token output changes**, so set `=0` if you need bit-reproducibility. Auto-arms at DEFAULT level when the model carries PXQ4/PXQ4HQ tensors and a DP4A-capable device is present: mode 1 if any sm_70+ card is in the fleet, mode 2 on an all-sm_61 fleet; a pure sm_60 fleet stays **off** because its DP4A is emulated. ⚠ Do not gate this lever with default-batch perplexity: `llama-perplexity` at `-b 512` is pure prefill and the MMVQ dispatch gate is `ne11 <= 8`, so the kernel never fires and both arms return identical perplexity — a false pass. Applies to any decode-window lever. |
| `PXA_PXQ_INT8_PREFILL` | 0 | int8 DP4A prefill tile for **sm_61** (GTX 10-series), where the fp16-family path has no fast dot product. +182% prefill on a 1080 Ti; decode byte-untouched; flag-off dispatch byte-identical. Not bit-exact vs the fp16 path (int8 activation quantization), hence opt-in. `=2` lifts the arch gate for testing — **do not ship on sm_60**, its DP4A is emulated. Armed automatically by `PXA_ENHANCE` on sm_61 only. |
| `PXA_PXQ_GEMM_2D` | auto | `=2` is **clamped to sm_60** (+35% dense prefill there). Its earlier +2.30% sm_70 figure was measured against the pre-coalescing dequant; against the current one sm_70 is **−18.6%** on dense. Auto-arms only for sm_60 × dense × PXQ-bearing tensors, not on device class alone. |
| `PXA_FA_MASK_SKIP_TILE` | auto | Skip fully-masked KV tiles in the tile-**f16** flash-attention kernel (contribution is exactly zero, so the skip is bit-identical). **Engages on sm_60 only**, and then only at `GGML_PREC_DEFAULT` with Q rows > 8 and head-dim ≠ 256. It does **not** engage on sm_61 — the BALANCE-mode win explicitly excludes all of sm_61. |
| `PXA_FA_MASK_SKIP_TILE_F32` | 0 | the same skip for the tile-**f32** kernel; this is the sm_61 / F32-precision equivalent of the row above. Bit-identical. |
| `PXA_FA_PREFILL_SPLIT` | 0 | **no auto-default at any level or posture.** The non-FA prefill chain inflates the compute buffer ~2.35× and OOMs 16 GB cards at `ub2048`, so the earlier BALANCE/ENHANCE auto-default was withdrawn. |
| `PXA_ROUTER_FUSE` | auto | router-GEMV dispatch fusion. **sm_70 only**: +5.1–7.0% decode on V100, −1.6% on sm_60. Armed by `PXA_ENHANCE` on sm_70. |
| `PXA_SPEC_1ROW` | **1** | extends the single-output-row GEMV to MTP spec-verify batch sizes (`Ny<=8`), which previously fell through to a bare `cublasSgemm` every spec-verify decode step. +6.6% decode on a single V100 (110.64 vs 103.82 t/s, ub1024, fa-on, MTP n1); flat on P100 and on a 2×V100 split. `=0` restores the old dispatch. |
| `PXA_CUBLAS_EAGER_INIT` | **1** | creates each device's cuBLAS handle and workspace at backend init instead of lazily mid-inference. Perf-neutral, ~12 MiB/device; prevents a lazy-allocation failure on a near-full card. |
| `PXA_PXQ6_WMMA` | 0 | experimental V100 tensor-core prefill path, auto-guarded to the 4-bit tiers. Honest measured gain after the launch-geometry fix: **+0.97% prefill**. Kept for experimentation; not part of the recommended set. |
| `PXA_PXQ4_2D_SPLIT` | auto | the S-split 2D decode driver. Split, unsplit and any `S` are **bitwise identical by construction** (canonical fold in the reducer). The canonicalization changed rounding once versus pre-canon binaries — a one-time, documented re-baselining, **PPL-verified unchanged**. |
| `PXA_G2_NORMFUSE`, `PXA_G2_QUANTFOLD` | 0 | q8_1 sidecar producers (fused rms-norm / DeltaNet out-gate). Bit-exact (temp-0 sha identical on/off) but **no measured gain**, so default off. |
| `PXA_G2_REDFUSE` | 0 | absorbs the gate/up ksplit-reduce + GLU into the down-mmv staging prologue. Bit-exact, **measured a loss**. Kept for the record. |
| `PXA_PXQ6_PAIRLUT`, `_PIPE`, `_PRMT`, `_LDCS`, `_SHFL`, `_ROWX2`, `PXA_PXQ3_PAIRLUT` | 0 | further bit-exact decode-mmv variants that did not beat the shipped set on the reference cards. Off by default; available for A/B on other silicon. |
| `PXA_PASCAL_DMMV` | 0 | a documented dead end — measured a loss. Kept so it is not rediscovered as a fresh idea. |
| `PXA_CUDA_GRAPH_V2`, `PXA_CUDA_GRAPH_LOG`, `PXA_CUDA_GRAPH_MOE`, `_LRU`, `_REARM`, `_BATCH_MAX_NY`, `PXA_PXQ_DISPATCH_DBG`, `PXA_EXPERT_LOG` | 0 | **diagnostic / unmeasured.** CUDA-graph replay semantics and instrumentation. Measured neutral-to-negative on the reference cards — instrumentation honesty, not a speed claim. `PXA_EXPERT_LOG` prints per-request MoE expert-routing histograms and is `np1` only. |

---

## §4 Server posture, micro-batch and split

The posture layer **only fills flags the CLI left unset**. An explicit `-fa` / `-ub`
(or its `LLAMA_ARG_*` env form) always wins, and `PXA_REFERENCE=1` stands the whole layer
down. It prints one line at startup: `PXA posture: mode=… fa=… ub=… (…)`.

| variable | default | effect |
|---|---|---|
| `PXA_MODE=balance` | default | fa **on**, `ub` 2048-class. Best decode and the best prefill available *inside* the fa-on regime. The daily serving posture. |
| `PXA_MODE=max` | — | fa **off**, largest-fitting `ub`. Absolute maximum prefill; decode is secondary. For bulk ingest. |

`PXA_MODE` moves no kernel-lever default — its only consumers are the posture flags above
and the startup report.

**Architecture exception (`deepseek2` / MLA).** `-mla 3` always defaults on a `deepseek2`
GGUF. `-fa` defaults **on** on every fleet containing an sm_60 or sm_70+ device — including
under `PXA_MODE=max`, which logs the exception — because MLA without flash attention
re-materializes full attention matrices and decays catastrophically with context. On an
**all-sm_61** fleet `-fa` defaults **off** instead: sm_61 is fp16-starved (1:64) and the MLA
FA kernel is measured 75–326% *slower* there. Explicit flags win in both directions. See
[`KNOWN-ISSUES.md`](KNOWN-ISSUES.md).

### ADAPTIVE-UB — the card-type table

With `-ub` unset the engine probes free/total VRAM on each assigned device and picks the
largest `ub` in {2048, 1024, 768, 512} that plausibly fits beside that device's share of the
model. The safe fallback, and the primary selector, is the **card-type default**:

| card | `-ub` |
|---|---|
| ≥ 15 GiB (P100 / V100 16 GB class) | **2048** |
| ≥ 10 GiB (11 GB 1080 Ti class) | **768** |
| anything smaller | **512** |

The 11 GB row is not conservatism: a `ub2048` compute buffer is ~1.9 GiB and cannot allocate
next to a resident ~10 GiB tier on an 11 GB card — verified for both PXQ2 and the IQ2
incumbent. Decode is `ub`-insensitive.

⚠ **`-ub` and `-ts` are coupled.** llama.cpp folds a per-device compute allowance into the
`-ts` walk, so raising `-ub` **repacks the layers** and a split tuned at one `-ub` can
overflow a card at another. If you force one, re-derive the other.

⚠ **One global `-ub` across a heterogeneous pool is wrong by construction** — the table above
wants different values on different cards and the CLI carries a single value.
`tools/pxa-launch.py` therefore passes **no** `-ub` and lets adaptive-ub probe each device.

### Tensor split

| variable | default | effect |
|---|---|---|
| `PXA_AUTO_TS` | on | fills `-ts` when it is unset. On an **exactly-2-device mixed sm_70 + sm_60 pair** it fills `-ts 1.4,0.6`, measured **+9.78% decode**. That figure comes from **one cell** (a PXQ4 35B split across a V100 and a P100) and does not generalize — it is not applied to other topologies. |

### KV-cache types

Symmetric `-ctk`/`-ctv` pairs are the tested configuration. At head-dim 128 the **only**
compiled *asymmetric* FA-vec pairs are `q8_0/q6_0`, `q8_0/iq4_nl` and `q6_0/q5_0`. Any other
asymmetric pair does not fall back — it **hard-aborts at request time** with
`Unsupported KV type combination for head_size 128`. No measurement exists for any
asymmetric pair: it is an unbenched VRAM trade, not a free one.

---

## §5 Speculative decoding and MTP

| variable | default | effect |
|---|---|---|
| `PXA_MTP_LAZY_WARMUP` | armed by `PXA_ENHANCE` | **mandatory whenever MTP is active.** Without it MTP costs **−33% prefill**. |
| `PXA_MOE_FASTTG_MAX_NY` | **8** | leave it at the shipped value. `=1` with MTP verify measured **48.1 → 30.3 t/s** on P100. |
| `PXA_MTP_DRAFT_RESERVE_CLAMP` | 0 | upstream MTP draft-generation KV-reserve clamp, ported default-off. |
| `PXA_NP_SPEC_GATE` | 0 | speculative-decode gating under `np>1`; swept by `bench/multislot-throughput.sh`. |
| `PXA_SPEC_RELAXED` | armed by `PXA_ENHANCE` | relaxed acceptance on speculative lanes only. |

**`--spec mtp:n_max`.** `n_max >= 2` is a measured loss on both architectures: P100
54.9 → 47.4 t/s (−14%, acceptance 0.42); V100 92.7 vs 94.1 (acceptance 0.960 → 0.480).
**Use `n_max=1`.** A bare `--spec mtp` expands to `mtp:n_max=1`.

**MTP on a sparse MoE is a measured loss even at `n_max=1`:** −8.6% (n_max=1) and −29.8%
(n_max=2), despite 0.800 acceptance.

**MTP and n-gram drafting have opposite verdicts on this model class** — n-gram +23.0% on
code and +4.6% on prose, against MTP's −8.6% / −29.8%. They are not substitutes for each
other; ask for the one you want by name.

---

## §6 Quantizer levers (`llama-quantize`)

| variable | default | effect |
|---|---|---|
| `PXA_PXQ_BACKBONE` | `v2` | comma-separated tokens selecting the backbone allocation table for attention / router / embeddings. `v2` (or `1`) = the rev-2 table on PXQ2/PXQ3/PXQ4/PXQ4HQ/PXQ6. `legacy` (or `0`) byte-reproduces pre-rev-2 recipes. `hq` substitutes the PXQ4HQ backbone for PXQ6 on the 4-/5-bit tiers — the pre-registered fallback, ~82% of the modelled gain at +0.26 bpw instead of +1.02, and PXQ4HQ has a CPU panel-dequant so the file stays partial-offload capable. `core` puts the GEMM backbone at the byte-parity PXQ4 core tier (MMVQ-eligible). `lite` keeps only the promotions that cost nothing at decode. `universal` additionally applies the table to `PXQ_UNIVERSAL` and PXQ1 — off by default, because a PXQU tier map is user-authored per tensor and PXQ1 is a closed tier; neither has measured backbone evidence. Anything failing the slab geometry falls back to Q8_0, never to a silent MXFP4 demotion. The chosen map is written to the GGUF as `pxa.pxq.backbone_map`. |
| `PXA_PXQ_KV` | `q8_0` | type for `attn_k` / `attn_v` / `attn_v_b`. Accepts `q8_0`, `pxq4`, `pxq4hq`, `pxq6`, `mxfp4`. The `q8_0` default is parity with the shipped files and with Q4_K_M; `mxfp4` restores byte parity against a flat-MXFP4 legacy file for A/B work; the PXQ tiers are the native option. |
| `PXA_PXQ_COMPOSITION_OVERRIDE` | 0 | lifts the assertion that the PXQ family must hold ≥ 50% of the file. Required for mixes whose bulk is gather tables rather than panel-codec tensors — without it such a run deletes its own output after hours of work. |

**Backbone note for MoE.** `BACKBONE_REV 2` promotes attention to PXQ6, which costs **12.2%
MoE decode** and buys no detectable fidelity on the model it was measured on (PXQ6 attention
5.6810 ± 0.065 vs PXQ4 attention 5.6766 ± 0.065). Shipping attention at **PXQ4** recovers 6.7
of those points and makes the class MMVQ-eligible. Do **not** revert attention to MXFP4 for
the remaining points — that re-opens the 3.2× error regression rev-2 exists to prevent.

**Recommended:** add `--output-tensor-type q8_0`. The single lm_head GEMV is ~14% of the
Pascal decode wall (int8 is emulated there); a q8_0 head costs +123 MB over the default and
measured **+5.2% decode on P100** (57.2 → 60.2 t/s on PXQU-16) at quality ≥ the default head.

---

## §7 Server robustness

All of these default to the safe behaviour; the variable exists to restore the old one.

| variable | default | effect |
|---|---|---|
| `PXA_SAMPLE_ABORT` | 0 | `=1` restores the old fatal `GGML_ABORT` on an unsampleable distribution (in practice a NaN cascade from invalid logits). By default the server keeps the forensic dump, falls back to the finite argmax, and degrades only that request instead of killing every co-resident generation. |
| `PXA_SOFTFAIL_MAX_CONSEC` | — | consecutive soft-fail budget per slot before the slot is stopped, so one degenerate request cannot spin. |
| `PXA_PORT_GUARD` | 1 | refuses to start when a live listener already answers on the target port, and names the cause. `=0` bypasses. |
| `PXA_IN_CONTAINER` | auto | overrides container detection (`0` or `1`). Exit-and-let-the-orchestrator-restart is only a valid contract when an orchestrator exists; bare metal gets an in-process recovery attempt and a distinct exit code instead. The verdict is logged once at startup. |
| `PXA_CKPT_HYBRID_ROLLBACK` | 1 | hybrid-recurrent checkpoint rollback. |
| `PXA_SWA_HINT` | 1 | `=0` restores the pre-fix suppression of the KV-min/max hint on sliding-cache layers, without a rebuild. |
| `PXA_REP_GUARD` | scoped to PXQ1 | damps the 1-bit degeneration loop. It is **PXQ1-scoped**: arming it on any PXQ artifact was the root cause of reported arithmetic flips on sm_61, where it penalised correct repeated digits in ordinary output. |
| `PXA_JINJA_LEGACY_LOOP_SCOPE` | 0 | restores the pre-fix jinja for-loop scoping. |
| `PXA_PARALLEL_LOAD` | 0 | opt-in parallel weight loading, `--no-mmap` only. Unset/`0` keeps the serial path, `1` selects 8 workers, `2..64` an explicit count. With mmap the upstream rewrite serializes every tensor behind one mutex, so that path is kept serial and the loader warns once. |
| `PXA_MTMD_STBIR` | 0 | stb_image_resize2 SIMD resizers plus the reference bicubic Qwen-VL / Gemma4V preprocessing. One switch, because the reference "bicubic" is a filtered Catmull-Rom that only the stbir path provides. |

---

## §8 The recommended set, in full

For a PXQ model on Pascal/Volta, this is the whole configuration:

```bash
PXA_ENHANCE=1 \
PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1 \
./build/bin/llama-server -m model.gguf -ngl 99 -c 8192 --jinja
```

`PXA_ENHANCE=1` selects the measured-good levers per card — mixed-card boxes get per-GPU
decisions — and prints the decision ledger. The `PXA_PXQ6/2/3` family switches enable the
PXQ format families; set all three for a mixed / `PXQ_UNIVERSAL` model. Add
`PXA_MODE=max` only for bulk ingest.

Everything in §2 is already on. Everything in §3 that applies to your silicon is already
armed by `PXA_ENHANCE`. If you are chasing a specific number, change **one** lever, and
gate it with a harness that actually reaches the kernel you changed — see the warning on
`PXA_PXQ_MMVQ` in §3, and [`bench/determinism-gates.md`](../bench/determinism-gates.md).
