# pxq_llama — v2026.08.28-rc3

Release candidate from [PXA Network](https://pxanetwork.com). Supersedes
**v2026.08.25-rc2**; **47 commits**.

This release adds support for a hybrid-attention MoE architecture that upstream
llama.cpp cannot load at all, a mixed per-tensor quantization tier
(`PXQ_UNIVERSAL`), two correctness fixes — one of which affected **every**
architecture — and a launcher that now configures a heterogeneous card pool on its
own. The llama.cpp engine and the PXQ4 vLLM backend ship from one tree.

**Nothing in this release changes numerics on an existing PXQ4 deployment.** Every
new code path is either a new architecture, a new quantization tier, or a guard that
declines where it previously faulted.

---

## How to read the numbers in this document

Every performance figure below names its **instrument** and its **hardware class**.
None of them are estimates, and where nothing was measured the text says so instead
of guessing.

The reference hardware is a single workstation of used datacentre cards — the same
one described in [`bench/README.md`](../bench/README.md), which is where the
reproduction scripts live:

- **Tesla P100-PCIE-16GB** (sm_60), **Tesla V100-PCIE-16GB** (sm_70),
  **GeForce GTX 1080 Ti 11GB** (sm_61)
- NVIDIA driver 580.142, CUDA 12.8.1, built and run inside
  `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- PCIe x4 per card (bifurcation risers). Decode is single-card-resident and
  unaffected; only multi-card `-sm layer` hand-off touches the bus.

Match the class, not the machine. If your numbers differ on comparable silicon,
that is worth an issue.

**Build gate for this tag.** The tree was booted on the new architecture before the
tag was cut: 49/49 layers offloaded, KV allocated at exactly 3840.00 MiB, the
PXQ2/PXQ3/PXQ6 fused kernels engaged, a nine-item factual probe answered correctly
with a clean EOS, and decode at 26.86 tok/s. Instrument: `llama-server` +
`llama-cli`, PXQU artifact, five cards (4× Tesla P100 + 1× GTX 1080 Ti), 160k
context.

---

## Read this first

**This release adds a model architecture that upstream cannot load.**
Qwen3.8-Flash-Next declares `general.architecture = qwen4exp`. Current mainline
llama.cpp (`ghcr.io/ggml-org/llama.cpp:server-cuda`, ggml 0.19.0) answers:

```
llama_model_load: error loading model: unknown model architecture: 'qwen4exp'
```

Mainline registers `qwen3next` and has none of the `hc_*` hyper-connection or
`ple_*` tensors this architecture is built from. That is a statement about
coverage, not a benchmark: **there is no engine-vs-engine number for this model,
because there is no other engine to compare against.**

**Two latent correctness bugs are fixed, and one of them affected every model.**
`inp_ple_rows` / `inp_ple_conv_hist` were read on every architecture but assigned
only by the `qwen4exp` graph builder, with no initialiser — so on any *other* model
the guard dereferenced a wild pointer. Measured 1 abort in 3 runs before, 0 in 15
after. The older binary was not correct, only lucky.

**The launcher will now pick a working config for a heterogeneous card pool.**
`--model` plus `--cards` is enough: it derives the tensor split, the micro-batch and
the CPU offload itself, prints the evidence behind each choice, and refuses rather
than guessing.

---

## Both engines, one tree

| piece | what it is |
|---|---|
| `tools/vllm-pxq4/` | the PXQ4 quantization backend / vLLM plugin, with its own build and conversion tooling |
| `tools/pxa-launch.py` | the unified launcher over both engines ([`docs/LAUNCHER.md`](LAUNCHER.md)) |
| `scripts/pxa-serve-sm70.sh` | PXQ4 on vLLM, Tesla V100 class ([`docs/PXA-SM70-SERVING.md`](PXA-SM70-SERVING.md)) |
| `scripts/pxa-serve-sm60.sh` | PXQ4 on vLLM, Tesla P100 class ([`docs/PXA-SM60-SERVING.md`](PXA-SM60-SERVING.md)) |
| `scripts/pxa-serve-flashnext.sh` | Flash-Next PXQU on llama.cpp, multi-card, 160k context |
| `pxa/pxq4/` | six prebuilt kernel libraries (four sm_60, two sm_70), a README describing what each one is, and a MANIFEST carrying sizes and md5s so a deployment can be verified against the artifacts that produced the recorded numbers |

Every serving script takes `MODEL=` and `--help` and runs the shipped binary
directly; a container is opt-in via `IMAGE=`. Both vLLM paths were booted from this
tree and verified end to end — including **tool calling**, which needs
`--enable-auto-tool-choice` and `--tool-call-parser qwen3_coder`. Without those two
flags every `tools=` request returns 400. See *Operating notes*.

---

## New architecture: qwen4exp (Qwen3.8-Flash-Next)

48 layers, 512 experts with 10 active, hybrid attention, and two structures that do
not appear in any other architecture this engine supports.

| piece | what it is | why it needed new code |
|---|---|---|
| Per-layer embeddings (PLE) | `per_layer_token_embd`, 51.2e9 elements, ~51 GiB | a pure `GET_ROWS` gather — one lookup per token per head, no GEMM. It belongs in host RAM; the n-gram hash that indexes it needs int64 multiply and xor, which ggml does not have, so it is computed host-side |
| Hyper-connections | `hc_*`, a wide 10240 residual with 4 streams | the mixer **replaces** all norms; the head mixer **is** the output norm |
| GDN linear attention | 36 of 48 layers | only 12 layers keep a KV cache (`full_attention_interval = 4`) |

### Converter

`convert_qwen4exp.py` takes HF safetensors to GGUF, streaming, and carries a
213-case self-test (`--self-test <reference.gguf>`) so a conversion can be gated
before it is quantized.

Four layout rules were each established by byte-comparison against a reference
GGUF. Each one silently corrupts the model if missed — the file loads and generates
plausible-looking garbage:

- `PERM48 = arange(48).reshape(16,3).T.ravel()` applies to `ssm_a`, `ssm_dt.bias`,
  `ssm_alpha`, `ssm_beta`, every `attn_gate`, the v-portion of `attn_qkv` and of
  `ssm_conv1d`, and the **input** dim of `ssm_out`
- `ssm_a = -exp(A_log)`, under that permutation
- every RMSNorm weight is stored as `(w − 1)` — `linear_attn.norm` is the sole
  exception
- `ssm_conv1d` permutes **only** its v-portion

**Scope, stated plainly:** the vision tower (333 tensors) and the MTP head (31
tensors) are **not** converted. Text generation only.

---

## PXQ_UNIVERSAL: a mixed per-tensor tier map

A PXQU artifact carries different PXQ tiers on different tensors, driven by a
`.tiers` map ([`docs/PXQU-CONVERT.md`](PXQU-CONVERT.md)). The Flash-Next build is
96.77 GiB across 1224 tensors:

| type | tensors | bytes | share |
|---|---:|---:|---:|
| q8_0 | 123 | 51974 MiB | 52.5% |
| pxq3 | 106 | 34606 MiB | 34.9% |
| pxq2 | 38 | 8574 MiB | 8.7% |
| pxq4 | 276 | 1460 MiB | 1.5% |
| f16 | 290 | 1220 MiB | 1.2% |
| q6_K | 2 | 995 MiB | 1.0% |
| f32 | 389 | 250 MiB | 0.3% |

The q8_0 share is almost entirely the CPU-resident PLE table. **Excluding it, the
GPU-resident bytes are 94.7% PXQ-family** — which is why this build needs
`--pxq-composition-override`: the composition floor measures whole-file share, and
that is the wrong measure for an architecture whose largest tensor never reaches a
GPU.

Two guards were added to the quantizer, both because the failure was hit for real:

- **panel codecs are refused on row-gather tensors.** A panel-packed codec on a
  `GET_ROWS` target gathers nonsense.
- **any tensor with `ne0 % 32 != 0` is ruled to f32**, with a preflight that
  refuses to start rather than dying 59 tensors into a multi-hour run.

**Quality caveat, unresolved.** The build emits 154 `pxq3 reconstruction ceiling
0.8957 × row absmax` warnings: row-peak weights are mildly clipped, so expect
elevated top-1% error. This artifact was quantized **without an imatrix**. Both
correctness batteries passed, but if output quality is ever questioned on a PXQU
build, this is the first thing to suspect.

---

## Correctness fixes

| fix | symptom | evidence |
|---|---|---|
| `inp_ple_rows` / `inp_ple_conv_hist` initialised | wild-pointer dereference on **any non-`qwen4exp` model**; a 0.6B dense model aborted on a PLE assert after an unrelated relink | 1 abort / 3 runs before, 0 / 15 after, correct output in all 15 |
| `PXA_ROUTER_FUSE` mode-3 base-pointer guard | the guard checked `ne00 % 4` and row stride `nb[1] % 16`, but never that `src0->data` / `src1->data` were themselves 16-byte aligned, while the kernel casts both to `float4*` — a misaligned-address **abort**, not a wrong number | default-OFF path; it now declines the fused path instead of faulting |

The first one carries the more useful lesson: the failure moved with an unrelated
relink, because the value tracked heap layout. **When a failure moves with a
relink, suspect uninitialised memory before you suspect the change.**

---

## Performance

### llama.cpp engine, qwen4exp

**Instrument:** `llama-cli`, `--temp 0 --ignore-eos`, 2 repetitions, 2931-token
prompt for the prefill figure. (Earlier prefill figures on this architecture came
off a 15-token prompt and measured fixed overhead, not throughput. They are
withdrawn.)

**PXQ4, six cards** — 4× Tesla P100-PCIE-16GB + 2× Tesla V100-PCIE-16GB, PLE
resident in host RAM:

| `-ub` | prefill | decode |
|---|---|---|
| 512 (adaptive default) | 337–366 tok/s | 30.6–30.9 tok/s |
| **1024** | **430–439 tok/s** | 30.6–31.1 tok/s |
| 2048 | OOM — needs its own `-ts` | — |

`-ub 1024` is **+20% prefill for nothing**: the decode ranges overlap, so it is not
a trade. The launcher now emits it for this architecture only; it is not
generalised to other architectures, which have different activation shapes.

Decode reached **32.4 tok/s** on a short prompt on the same six cards, via two
levers committed in this release:

- a byte-proportional six-card `-ts` rebalance that moves 4 layers off the Pascal
  cards onto the Volta cards;
- a dedicated small-K / large-R f16 GEMV for the hyper-connection up-projection.

**PXQU, four Tesla P100-PCIE-16GB:** 27.79 tok/s decode; prefill 411–414 tok/s at
`-ub 1024`, 424–430 at `-ub 2048`.

**PXQU, five cards** (4× Tesla P100 + 1× GTX 1080 Ti), 160k context: 26.86 tok/s
decode, nine-item factual probe correct, clean EOS.

**PXQU is slower than PXQ4 and should not be read as a speed win.** It is a
footprint win: 46702 MiB of GPU-resident weights against the PXQ4 build's 65032 MiB
for the same model — four or five cards instead of six, with room for a 160k
context.

### PXQ4 on vLLM

Unchanged this cycle; restated here with the instrument, because these are the
numbers the shipped launch scripts default to.

**Instrument:** vLLM server, PXQ4 artifact, TP=2, concurrency ladder as noted; each
arm correctness-gated **before** its speed number was recorded (a fast wrong answer
is not a result). Full sweeps in the two serving documents.

| hardware class | script | single-stream decode | aggregate @8 |
|---|---|---|---|
| 2× Tesla V100-PCIE-16GB (sm_70, PCIe host bridge, no P2P) | `pxa-serve-sm70.sh`, default `agg` profile | 50.98 tok/s | 135.78 tok/s |
| the same pair, `PXA_PROFILE=single` | `pxa-serve-sm70.sh` | 51.46 tok/s | 132.13 tok/s |
| 2× Tesla P100-PCIE-16GB (sm_60) | `pxa-serve-sm60.sh` | 24.0–26.4 tok/s | 70.0–72.1 tok/s |

Both shipping launchers pin the `v10` kernel libraries. The sm_60 container recipe
recorded in the launcher's image table still named the stale `v8`; it is corrected
to `libpxq4_sm60_v10.so` in this release, so the recipe and the launchers agree.
**Not v11.** `libpxq4_sm60_v11.so` with `PXQ4_GEMM2D=1` reaches 300.1 tok/s prefill
on the Tesla P100 pair — **+37%** — and **fails** raw-prompt correctness at 87.5%
first-token agreement. It ships in `pxa/pxq4/kernels/` behind that flag, default
off, and must be re-gated for quality before anyone turns it on. A prefill win that
changes what the model says is not a win.

---

## Launcher

`pxa-launch.py` handles the new architecture. Four defects are fixed, three of
which produced a confident wrong answer rather than saying UNMEASURED:

1. **KV per token was 4× too high on hybrid attention.** The flat
   `n_layer × n_kv × (d_k + d_v)` form assumes every layer caches KV. `qwen4exp`
   caches on 12 of 48. It priced a 160k context at 15.0 GiB instead of 3.75. The
   launcher now reads `attention.compress_ratios`, falling back to
   `full_attention_interval`. The corrected formula predicts 24576 B/token, and the
   engine allocated **exactly** 3516.00 / 3840.00 / 6144.00 MiB at
   c = 150016 / 163840 / 262144 — three exact hits.
2. **The VRAM fit rule refused a model that fits.** It compared *file* bytes against
   VRAM, but 50.66 of the 96.77 GiB is host-resident PLE; the GPU-resident remainder
   is ~46.1 GiB and runs on five cards. The rule now compares GPU-resident bytes,
   and says so explicitly when it cannot determine a tensor's block geometry.
3. **`-ot` was never emitted**, so the PLE table was offloaded with everything else
   and the load died on a single 16089.57 MiB `cudaMalloc`. The launcher now emits
   `-ot per_layer_token_embd\.weight=CPU`, anchored on the full tensor name — a
   loose `ple` regex also matches the tiny `ple_key` / `ple_conv1d` / `ple_norm_*`
   tensors, which must stay on the GPU.
4. **No `-ts` was derived**, so a heterogeneous pool got an even split — which puts
   ~9.2 GiB of weights on a card with 7.8 GiB free.

`auto_tensor_split()` now derives the split from **real free VRAM** per card,
charges the head card its larger compute buffer (llama.cpp places the output head on
the last device in PCI order), reserves decode headroom, and **declines loudly** if
any card has no room left:

```
AUTO -ts DECLINED: card(s) [0, 1] have no capacity left after 1200 MiB headroom
+ compute buffers at ctx=16384. Free a card, drop -c, or pass --ts by hand.
```

It was validated against a hand-tuned split already known good: the launcher
independently produced `111,267,95,267,260` against the hand-tuned
`110,268,94,268,261` — within 1 part in 1000 per card — and the deployment was then
restarted on the launcher's own numbers and generated correctly.

---

## Operating notes

These four cost real time to learn. They are properties of the engine and the
hardware, not bugs in this release.

**A config that loads is not a config that runs.** At c = 262144 a five-card
Flash-Next deployment loaded cleanly, printed every buffer, then died on the *first
token* with `CUDA error: out of memory` inside `llama_decode` — the split had left
one card 51 MiB. Decode allocates transient buffers that the init-time report does
not include. Leave ~1200 MiB free per card **after** load. The same configuration at
c = 163840 keeps 807–1803 MiB free per card and runs.

**Run this model from local NVMe, never from a parity RAID array or a network
share.** `per_layer_token_embd` is CPU-resident and mmap'd, so every decode token
faults PLE pages off whatever disk holds the file. Off a parity array that is
~17 MB/s, and the process wedges in uninterruptible `D` state inside the md driver,
where `SIGKILL` cannot be delivered until the read drains.

**`-ub` and `-ts` are coupled.** `-ts` partitions bytes, not layers, and llama.cpp
folds a per-device compute allowance into the same walk — so changing `-ub` repacks
the layers, and a split tuned at one `-ub` can overflow a card at another. That is
exactly how `-ub 2048` OOM'd a six-card deployment whose split was tuned at
`-ub 512`. If you force `--ub`, re-derive `--ts`.

**Tool calling needs two flags.** Without `--enable-auto-tool-choice` and
`--tool-call-parser`, every `tools=` request returns 400. The parser is
`qwen3_coder`, **not** `hermes`: hermes expects JSON inside `<tool_call>`, while
this chat template emits XML `<function=NAME><parameter=K>V`. Choosing hermes would
have returned *empty* tool calls rather than an error, which is worse.

---

## Known limitations

Honest list. None of these are hidden behind a flag.

- **qwen4exp is text-only here.** The vision tower (333 tensors) and the MTP head
  (31 tensors) are not converted.
- **The PXQU Flash-Next build was quantized without an imatrix**, and emits 154
  `pxq3` reconstruction-ceiling warnings. Expect elevated top-1% weight error;
  re-quantizing with an imatrix is open work.
- **`-ub 2048` on the six-card PXQ4 configuration needs its own `-ts`.** It is not
  derived, and the default split OOMs at that micro-batch.
- **The hybrid-attention KV correction is MEASURED for `qwen4exp` only.**
  `qwen35moe` takes the same correction but remains tagged `[INFERRED]` — nobody has
  booted it.
- **QSA top-k sparsity is not wired.** Full-attention layers attend densely. Not a
  bug at short context; real at length.
- **PXQU is a footprint tier, not a speed tier.** It is measurably slower than PXQ4
  on the same model.
- **No engine-vs-engine comparison exists for `qwen4exp`,** because no other engine
  loads it. Every number in this document for that architecture is
  configuration-vs-configuration on our own engine.
- **The v11 sm_60 kernel is faster on prefill and fails first-token quality
  gating.** It ships default-off and must be re-gated before anyone enables it.
- Standing engine-level constraints — PXQ requires full GPU residency, and imatrix
  capture on partial-offload configurations is broken upstream — are documented in
  [`docs/KNOWN-ISSUES.md`](KNOWN-ISSUES.md).

---

## Attribution

pxq_llama is a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp),
itself a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) / ggml.
Upstream copyright, license terms and the provenance of every ported change are in
[`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE) and
[`LICENSING.md`](../LICENSING.md). PXQ, the PXQ CUDA kernels and the PXQ4 vLLM
backend are PXA Network's own work.
