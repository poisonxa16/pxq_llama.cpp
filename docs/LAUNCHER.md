# `pxa-launch` — the unified launcher

`tools/pxa-launch.py` is one entry point in front of two runtimes. It reads your
model file, looks at your cards, and picks between **pxq_llama** (the llama.cpp
engine in this tree) and **vllm-pxq4** (the branded vLLM backend in
`tools/vllm-pxq4`) — then prints the decision, the evidence behind it, and the
exact command before running anything.

Two runtimes serve one quant family. pxq_llama runs on every card the PXQ family
supports. vllm-pxq4 runs wherever its image has PXQ4 kernels for the card, and
brings real data parallelism that llama.cpp's `-sm layer` does not have. Choosing
by hand means remembering which card is which, whether the model is dense or MoE,
which PXQ tier is actually inside the file, and which engine wins at the
concurrency you actually serve at. That is what this script is for.

## Design rule: never magic

The launcher is built around four commitments, and they explain most of its
behaviour:

1. **It prints the decision, the evidence, and the command** before executing.
2. **It refuses rather than silently dropping** a parameter that does not
   translate between engines.
3. **It says `UNMEASURED` out loud** instead of guessing quietly.
4. **It makes no claim it cannot back**, including no health claim about the instance
   it starts — it `exec`s the server, so it cannot observe anything afterwards.

A launcher that quietly picks differently turns every performance question into a
debugging exercise about the launcher.

---

## Quickstart

```bash
# See the decision and the command; run nothing.
python3 tools/pxa-launch.py --model /models/MyModel-PXQ4.gguf --explain

# Serve it, single stream, on card 0.
python3 tools/pxa-launch.py --model /models/MyModel-PXQ4.gguf --cards 0

# Serve 8 concurrent slots across two cards.
python3 tools/pxa-launch.py --model /models/MyModel-PXQ4.gguf --cards 0,1 --np 8

# Exercise the decision table against this machine's real cards.
python3 tools/pxa-launch.py --selftest --model /dev/null
```

> `--model` is required by the argument parser even for `--selftest`, which
> ignores its value. Pass any path.

---

## How a claim is tagged

Exactly three tags, no fourth category. They appear in the source comments *and*
in the printed output, so you can tell a measured branch from a guess without
leaving the file.

| Tag | Meaning |
|---|---|
| `MEASURED` | A number from a correctness-gated boot on the PXA reference bench, carrying the id of the bench row that produced it. |
| `[INFERRED]` | A branch taken from an **adjacent** measurement. Never a new number. |
| `UNMEASURED` | Nothing was measured. The launcher says the word, and then either stops or asks for `--accept-unmeasured`. |

**Bench row ids** are stable labels for individual gated boots, quoted inline
beside every number they back:

- `D1` / `D3` — dense 27B PXQ4, 2× P100 sm_60 (D1 llama.cpp, D3 vLLM)
- `M1` / `M4` / `M7`–`M9` — MoE 35B PXQ4, 2× P100 sm_60
- `np1`, `np4`..`np8` — the MoE crossover sweep, 11 gated boots on one 2× P100 pair

Every boot behind a row was gated on short-prompt correctness **before** its
number was kept. An ungated boot has no row and appears nowhere.

---

## The decision pipeline

The launcher runs seven stages in a fixed order. **What cannot run is settled
before what is fastest** — refusals come first, always.

### 1. Resolve the artifact

`model_kind()` classifies the path before anything else reads it. The taxonomy is
deliberately fine-grained so the error names the real problem:

| kind | meaning |
|---|---|
| `gguf` | a readable GGUF file |
| `gguf_broken` | a GGUF whose header will not parse |
| `vllm_dir` | a PXQ4-converted directory (safetensors + `quantization_config.quant_method`) |
| `hf_dir` | an unquantized or foreign-quantized HF checkpoint |
| `lora_dir` | an adapter directory with no base model |
| `weightless_dir` | `config.json` and no weights |
| `not_a_model_file` | an existing file that is not servable — one safetensors shard, a `.tiers` map, an `.imatrix` |
| `missing` / `not_a_model` | nothing there, or nothing recognisable |

A single shard (`model-00003-of-00014.safetensors`) is detected by name and the
message tells you to pass the directory instead.

### 2. Inspect the GGUF — header only

`gguf_header()` reads the KV block and the tensor directory. **No tensor data is
read and no GPU is touched.** From that walk the launcher derives:

- `general.architecture`, `<arch>.expert_count` (→ dense vs MoE),
  `<arch>.context_length` (the trained context)
- the **tensor-type histogram** — printed as the `compose:` line
- MTP/nextn head presence, by tensor walk
- DeltaNet/linear-attention presence
- vision tensors
- the per-layer embedding table, if present
- KV bytes/token, by arithmetic over header fields

### 3. Detect the PXQ tier — from tensors, not from a KV

**This is the single most important design decision in the file.**

There is no generic `pxa.pxq.tier` key to read. `llama-quantize.cpp` rewrites
*every* PXQ tier to `LLAMA_FTYPE_MOSTLY_MXFP4` (=38) before writing it, so
`general.file_type` cannot identify a tier — verified over a 138-file library:
138/138 report 38 or a K-quant id, **0 yield a tier**. And PXQ1, the one tier this
launcher exists to refuse, writes no provenance KV at all.

So the ground truth is the **per-tensor ggml type histogram**:

| ggml type id | tier |
|---|---|
| 248 | PXQ1 |
| 252 | PXQ4 |
| 253 | PXQ4-HQ |
| 254 | PXQ2 |
| 255 | PXQ3 |
| 256 | PXQ6 |

A file carrying more than one PXQ type is `PXQ_UNIVERSAL` — a mixed per-tensor
tier map. Non-PXQ types that legitimately appear alongside (f32, f16, bf16, q8_0,
q6_K, MXFP4) are backbone carriers and are counted separately.

This mirrors the engine, which detects PXQ1 by tensor type in
`src/llama-model-loader.cpp:527`. The provenance KV **is** read as well, and any
conflict between the two signals is printed and never resolved silently — the
tensor walk wins, because that is what the loader dispatches on.

Verified against real artifacts: 5/5 PXQ GGUFs yield a tier this way; 0/5 through
`general.file_type`.

### 4. Choose the engine

**Structural gates first** — what each engine can physically read:

| Condition | Engine | Why |
|---|---|---|
| tier is PXQ1 | **refuse** | R-01 — loads, passes composition gates, generates incoherent text |
| tier is PXQ2/PXQ3/PXQ4-HQ/PXQ6/UNIVERSAL | llama | vLLM implements PXQ4 **only** |
| no PXQ tensors at all (stock K-quant, MXFP4, f16) | llama | vLLM's PXQ4 backend has nothing to load |
| model is a raw `.gguf` | llama | vLLM needs a converted artifact |
| no vLLM-eligible card | llama | the failed probe is named |
| only *some* selected cards eligible | llama | vLLM cannot span a mixed-arch selection on a single-arch image, and dropping cards would silently change the parallel degree |
| selection spans >1 compute capability | llama | the vLLM command carries one `--attention-backend` |
| single GPU | llama | no parallelism to gain, and lower single-stream overhead |
| converted vLLM dir but llama.cpp wins | **refuse** | R-29 — llama.cpp cannot read safetensors |

**Only if both engines can genuinely run it** does the launcher pick on measured
performance.

#### Dense models → vLLM

vLLM wins dense at every workload measured (27B PXQ4, 2× P100 sm_60, rows D1/D3):

| | vLLM | llama.cpp | ratio |
|---|---|---|---|
| single-stream decode | 24.01 | 13.7 | 1.75× |
| aggregate decode @8 | ~70 | 12.4 | 5.6× |
| prefill | ~225 | 156.5 | 1.44× |
| aggregate decode @4 | *UNMEASURED* | 12.0 | — |

The launcher prints a standing caveat with this: the llama.cpp side (D1) is **one
boot**, below this bench's own two-boot bar, and the graphs-on dense arm (D2) was
never launched. The *direction* is not in doubt; the exact ratios are single-boot.

#### MoE models → a split instance, decided by concurrency

This is the important branch. MoE 35B PXQ4, 2× P100 sm_60:

| `--np` | llama.cpp | vLLM | winner | margin |
|---|---|---|---|---|
| 1 | 95.6 | 30.4 | llama.cpp | 3.14× |
| 4 | 75.93 | 64.82 | llama.cpp | +17.1% |
| 5 | **79.49** | 64.32 | llama.cpp | +23.6% ← llama.cpp peaks |
| 6 | 69.58 | 75.60 | **vLLM** | +8.7% ← crossover |
| 7 | 67.74 | 87.03 | vLLM | +28.5% |
| 8 | 62.42 | 95.81 | vLLM | +53.5% |

**The table is stored, not a slope.** Neither curve is monotonic and the flip is
sharp: llama.cpp *peaks* at np=5 above its own np=4 value, then drops 12.5% in one
step while vLLM climbs. The margin swings 32 points between np5 and np6. A
straight line from np4 to np8 would put the threshold too early and misprice np5
by ~14%.

Root cause of that shape, one sentence: llama.cpp `-sm layer` is a **serialized
two-GPU pipeline, not data parallelism**, so concurrent requests queue behind the
same pipeline while vLLM's aggregate climbs.

A currency warning rides every vLLM MoE decision: the crossover sweep measured
95.81 at np8 on a newer engine revision where row M7 has 88.7 on an older one
(+8.0%) for the same cell. The cause is a hypothesis, not a measurement — so if
that gap is real, the crossover may sit **below** np=6. That can change an engine
decision, not just a number.

#### Long-document workload → llama.cpp

`--workload longdoc` routes MoE to llama.cpp, which holds the prefill record at
every concurrency measured (1136 / ~1058 / ~1000 vs vLLM 567.6 / 595.8 / 594.4
tok/s, ~1.7–1.9×). The caveat is printed every time: that comparison is
**cross-harness and the prompt lengths were not matched** (2059 vs ~6.4k tokens).
Directionally trusted, not controlled.

### 5. vLLM eligibility is an *image* property

Not a compute capability. An earlier version gated on `MIN_VLLM_CAP = 70` and
routed every sub-sm_70 card to llama.cpp — which made the entire vLLM branch
unreachable on the very hardware every vLLM cell in the table was measured on
(2× P100 sm_60).

The narrow true statement is: **no vLLM decode-or-prefill throughput number on
sm_70 exists at all.** The only sm_70 vLLM figure on record is a 3.76×
shared-prefix win, which is not an instance decision.

So eligibility is *probed*, in this order, and the probe that failed is always
named:

1. `--vllm-image` / `PXA_VLLM_IMAGE` → look the tag up in the launcher's table
2. the image's declared arch set
3. bare-metal: `vllm_pxq4` importable **and** `PXA_PXQ4_LIB` naming an sm tag

If no image resolves, vLLM is not eligible and the reason names the probe — never
a bare "vLLM is sm_70+ only".

**Known image tags:**

| tag | caps | status | notes |
|---|---|---|---|
| `pxa-sm60-dev` | {60} | MEASURED | produced every MoE-crossover vLLM number. **Not self-contained** — see below. |
| `pxa-vllm:sm60` | — | INFERRED | Pascal thin image, torch 2.7.1. `caps` stays empty until a gated boot on a P100 pair passes against *this* tag. |
| `pxa-vllm:sm70` | — | INFERRED | Volta thin image, torch 2.10. Not yet gated. |

An image the launcher does not know has an UNMEASURED arch set and gets no card,
whoever built it.

> **Why there is no single fat image.** One image spanning sm_60+sm_70 was tried:
> 8 boot attempts on a V100, 8 failures. The cause is structural, not
> configuration — `VLLM_SKIP_C_STABLE=1` is required to build against torch 2.7.1
> (the last torch with sm_60 cubins) and it drops `csrc/libtorch_stable/`, where
> an op the V100 serving path calls unconditionally lives. You cannot have sm_60
> cubins and that op in the same build. Hence two thin images.

#### Images whose runtime lives on the host

An image can be a bare CUDA runtime whose python, torch and vllm all live on the
host and are bind-mounted at run time — `pip list` inside it shows pip and nothing
else. Every number attributed to such a tag was really produced by the image
**plus** those host paths.

Those paths are site-local, so **none are hardcoded**. Declare them in a JSON
descriptor and point `PXA_VLLM_HOST_ENV` at it:

```json
{"pxa-sm60-dev": {
   "mounts":   {"/srv/pxa": "/c"},
   "requires": ["/srv/pxa/venv/pyvenv.cfg",
                "/srv/pxa/venv/bin/python",
                "/srv/pxa/venv/lib/python3.12/site-packages/torch",
                "/srv/pxa/vllm-src/vllm/__init__.py",
                "/srv/pxa/kernels/libpxq4_sm60_v10.so"],
   "python":   "/c/venv/bin/python",
   "env":      {"PYTHONPATH": "/c/site",
                "PXQ4_LIB":  "/c/kernels/libpxq4_sm60_v10.so"},
   "editable_source": "/srv/pxa/vllm-src",
   "why": "how you traced it, so the next reader does not have to"}}
```

| key | meaning |
|---|---|
| `mounts` | host dir → container dir. The **model path is translated** through the longest matching prefix before the command is emitted; a model outside every mount is called out by name. |
| `requires` | host entries that must exist or the image is declared ineligible. Checked with `lexists`, not `exists` — `venv/bin/python` is typically a symlink to an interpreter that exists only *inside* the container. |
| `python` | the interpreter to invoke inside the container. `vllm serve` is wrong for this class of image; vllm is not on PATH there. |
| `env` | env the host runtime needs to be importable. Merged with `setdefault`, so a value from a measured arm always wins. |
| `editable_source` | if vllm is an editable install, the tree it resolves to. The launcher prints that tree's branch, sha and dirtiness, so a measurement can be attributed to something. |

Without a declared host environment, a tag marked `needs_host_env` yields **no
eligible card** — because the measurements it carries were produced by the image
plus a runtime the tag alone does not describe.

> **The container contract.** `--vllm-image` decides only *which cards are
> eligible*. The emitted command is a bare `vllm serve` — there is no `docker run`
> in it and the image name appears nowhere. The command **execs where the launcher
> is running.** The launcher says this out loud, because the flag reads like it
> selects a runtime, and a reader who believes that will attribute a measurement
> to an image that was never involved.

### 6. Derive the tensor split from free VRAM

An even split is llama.cpp's default and it is **wrong on any pool where the cards
are not equally free**. On a real five-card pool, one 16 GiB card was already
carrying another instance (7865 of 16384 MiB free) and one was an 11 GiB 1080 Ti also
in use — an even five-way split puts ~9.2 GiB of weights on a card with 7.8 GiB
free.

So `auto_tensor_split()` computes a **capacity-proportional** split over *free*
VRAM. Per card:

```
capacity = free_MiB − compute_buffer − cuda_context(250 MiB) − headroom(1200 MiB)
```

then shares are `round(1000 × capacity / Σcapacity)`.

Two details carry all the weight:

- **The head card is charged more.** llama.cpp places the output head on the last
  device in PCI order, so the last card in the selection is charged a **980 MiB**
  head compute buffer — measured, and flat in context — instead of the ordinary
  card's buffer, which is interpolated linearly in ctx between measured anchors
  (282 MiB @ c8192 → 786 MiB @ c262144, at ub1024).

- **The headroom term is the whole point.** *A config that loads is not a config
  that runs.* At c=262144 a five-card instance loaded, printed every buffer, then died
  on the **first token** with `CUDA error: out of memory` in `llama_decode` —
  the split had left one card with 51 MiB. Decode allocates transient buffers that
  the init-time buffer report does not include. The same instance at c=163840 kept
  807–1803 MiB free per card and ran. Hence the 1200 MiB reserve.

If any card has no capacity left, the launcher **declines to emit a split** rather
than emitting one that cannot work:

```
AUTO -ts DECLINED: card(s) [0, 1] have no capacity left after 1200 MiB headroom
+ compute buffers at ctx=16384. Free a card, drop -c, or pass --ts by hand.
```

> **`-ts` and `-ub` are coupled.** `-ts` partitions **bytes, not layers**, and
> llama.cpp folds a per-device compute allowance into the same walk — so changing
> `-ub` *repacks the layers*, and a split tuned at one `-ub` can OOM at another.
> This was measured: a six-card PXQ4 instance at ub2048 pushed a 16140 MiB V100 over
> with the ub512-tuned split. If you force `--ub`, re-derive `--ts`.

### 7. Pick the micro-batch

**The launcher's default is to pass no `-b`/`-ub` at all.** The engine's
adaptive-ub probes real free VRAM per device at startup and picks per card. One
global `-ub` across a heterogeneous pool is wrong by construction.

The launcher still *prints* the value adaptive-ub should land on, from the
measured card-type table, so you can check it against the server's own
`PXA posture: mode=... fa=... ub=...` line:

| card | expected `-ub` |
|---|---|
| ≥15 GiB | 2048 |
| ≥10 GiB (11 GB 1080 Ti class) | 768 |
| smaller | 512 |

ub2048/1024 compute buffers are measured to OOM next to a ~10 GB model on an
11 GB card; ub768 fits.

**One exception**, scoped on purpose: on arch `qwen4exp` the launcher emits
`-ub 1024`, because both arms were measured on that arch — ub1024 is **+20%
prefill** over the adaptive 512 (337–366 → 430–439 tok/s, six-card PXQ4,
2931-token prompt) with **decode unchanged** (30.6–30.9 vs 30.6–31.1, overlapping
ranges). That is not a trade; it is free. It is not generalised to other
architectures, which have different activation shapes.

### 8. Offload the per-layer embedding table to CPU

Some architectures (`qwen4exp`, `gemma3n`) carry a **per-layer token embedding
(PLE)** table: `per_layer_token_embd.weight`. It is enormous — on one 97 GiB model
it is 160 × 320001536 = 51.2e9 elements, about **51 GiB of the file** — and it is
a pure `GET_ROWS` gather: one lookup per token per head, **no GEMM**.

So it belongs in host RAM. When the launcher sees it, it emits:

```
-ot per_layer_token_embd\.weight=CPU
```

**This is not an optimisation.** Without it the gather table is offloaded with
everything else and the load dies on a `cudaMalloc` of the whole tensor on one
card — measured at 16089.57 MiB on device 2 of a five-card instance.

Two subtleties:

- **The pattern is anchored on the full name deliberately.** A loose `ple` regex
  also matches `blk.N.ple_key`, `ple_conv1d` and the F32 `ple_norm_*` tensors,
  which are tiny and **must stay on the GPU**.

- **The VRAM check subtracts it.** R-17B compares *GPU-resident* bytes, not file
  bytes. A PLE table pinned to host RAM never reaches VRAM, so counting it would
  refuse instances that fit. Measured: a 96.77 GiB file of which 51.15 GiB is PLE has
  a GPU-resident remainder of 46.70 GiB and runs on five cards (75 GiB) with room
  for a 3.75 GiB KV cache at 160k context. Uncorrected, R-17B refused it outright.
  If the PLE's ggml type is not in the block-geometry table, the bytes are **not**
  subtracted and the launcher says so — the figures then overstate what reaches
  the cards, and any refusal should be treated as suspect.

---

## What the fit check may and may not block on

The KV-per-token figure is arithmetic over header fields, unvalidated against a
real allocation on every architecture but one. **So it is never allowed to
block.** It warns, labelled `[INFERRED]`.

Only two facts need no formula, and only those two block:

- **R-17A** — `-c` exceeds `<arch>.context_length` (a KV field, read directly).
- **R-17B** — GPU-resident weights *alone* exceed the total VRAM of the selection
  under full offload. File-byte arithmetic.

Everything else warns:

```
** VRAM estimate [INFERRED, never blocks]: weights 14.64 GiB + KV 4.06 GiB
   (= ctx 16384 x 260.0 KiB/tok) = 18.70 GiB vs 2.26 GiB free / 32.00 GiB total
   across 2 card(s). Compute buffers and fragmentation are NOT in this number.
** HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load.
```

---

## Flags

### Model and hardware

| Flag | Default | Meaning |
|---|---|---|
| `--model PATH` | *required* | GGUF file, or a PXQ4-converted directory. Required even by `--selftest`, which ignores it. |
| `--cards 0,1` | all visible | Card selection. **Never left to the ambient environment** — the launcher always sets `CUDA_VISIBLE_DEVICES` explicitly and refuses to execute if it cannot name the devices. |
| `--engine llama\|vllm` | auto | Force an engine. Blockers are still reported **and still stop the run**. |
| `--tier T` | — | Assert the PXQ tier yourself; the R-03 escape hatch. You own the assertion. |

### Workload shape

| Flag | Default | Meaning |
|---|---|---|
| `--np N` | 1 | Concurrent slots. Drives the MoE engine choice, `-np`, and `--max-num-seqs`. Must be ≥1. |
| `--workload chat\|serve\|longdoc` | `chat` if `--np`≤1, else `serve` | `longdoc` routes MoE to llama.cpp for prefill. |
| `-c`, `--ctx N` | `np × 4096` | **Total** context, not per slot. 4096/slot is the measured envelope. |
| `--threads N` | host core count | `-t`. |

### Placement and memory

| Flag | Default | Meaning |
|---|---|---|
| `--ngl N` | 999 | Layers to offload. Below 99 on a GPU-only tier triggers R-16. |
| `--ts A,B` | auto | Forces the tensor split; the automatic capacity split is then **not** used. Refused on vLLM (R-10). |
| `--sm layer\|graph\|row` | `layer` | Split mode. `graph` on a DeltaNet hybrid is refused (R-12). Refused outright on vLLM (R-11). |
| `--ub N` | 0 = **not passed** | Forces `-b`/`-ub`. Leaving it unset lets adaptive-ub probe each device. |
| `--ctk`, `--ctv` | `f16` | KV cache types. Checked against the compiled FA kernel pairs (R-13L). Refused on vLLM (R-13V). |
| `--no-mmap` | off | Adds `--no-mmap` and sets `PXA_PARALLEL_LOAD=1` (−25..−46% cold load; inert under mmap). |
| `--gmu F` | 0.90 sm_60 / 0.85 sm_70 | vLLM `--gpu-memory-utilization`. These are recipe values, **never swept** — UNMEASURED as a tuning axis. |

### Multimodal

| Flag | Default | Meaning |
|---|---|---|
| `--mmproj PATH` | auto | Vision projector. Auto-resolves **only** if the model carries vision tensors and exactly one candidate sits beside it; two or more is R-24. |
| `--no-mmproj` | off | Suppress projector resolution. The launcher still lists what it *would* have found and states the instance is text-only. |

### Speculation

| Flag | Default | Meaning |
|---|---|---|
| `--spec METHOD[:k=v,...]` | — | e.g. `mtp:n_max=1`, `ngram-mod:n_max=4`. A bare `mtp` expands to `mtp:n_max=1` — **not** the old `n_max=4,n_min=2`, which was a measured loss emitted by default. |
| `--draft-model PATH` | — | External draft-model speculation. Zero coverage in this bench → R-25. |

### vLLM specifics

| Flag | Default | Meaning |
|---|---|---|
| `--vllm-image TAG` | — | Decides **card eligibility only**, not where the command runs. |
| `--cudagraph-mode MODE` | `FULL_DECODE_ONLY` | Anything else is refused (R-08). This is a correctness requirement, not a knob. |

### Serving and control

| Flag | Default | Meaning |
|---|---|---|
| `--host` / `--port` | `0.0.0.0` / `8080` | |
| `--explain` | off | Decide and print; run nothing. Exits 5 if the plan carries known-fatal blockers, 0 if clean. |
| `--selftest` | off | Run the decision table against this machine's real cards. |
| `--accept-unmeasured` | off | Execute a branch the launcher labels `[INFERRED]`/`UNMEASURED`. |
| `--allow-busy` | off | Select a card another process is already resident on. |

---

## Refusals

Refusals are the point of the tool, not an inconvenience. Each has a stable id, so
you can grep for it. Numbering has gaps: **R-04 and R-09 are not implemented as
refusals** — R-09 is the post-boot verification contract, which is printed rather
than enforced, because this process `exec`s the server and cannot observe it.

### Artifact and tier

| id | Refuses |
|---|---|
| **R-01** | **PXQ1 content.** A PXQ1 MoE file loads, clears the composition gate at 80.9% PXQ-family bytes, and generates **incoherent text** — nothing downstream catches it. No dense path, no CPU codec. Detected by tensor type, so it fires on a uniform PXQ1 file *and* on a UNIVERSAL map with PXQ1-mapped experts. |
| **R-02** | Retired quant types 250/251, removed 2026-07-21. No shipped engine reads them. |
| **R-03** | Guessing the tier. `general.file_type` is 38 for every PXQ tier by design and the tensor directory shows no PXQ types either. Without a tier the PXQ1 refusal and the vLLM PXQ4-only gate cannot be enforced. Escape: `--tier`. |
| **R-05** | A foreign-quantized safetensors directory (`compressed-tensors`, `awq`, `gptq`, `fp8`, `bitsandbytes`) — readable by **neither** engine. |
| **R-06** | vLLM requested but the converted artifact does not exist. Names the convert command. |
| **R-18** | An unquantized HF checkpoint. The launcher will not emit `vllm serve --quantization pxq4` against fp16 weights. |
| **R-19** | A config-only stub with no weight files. |
| **R-22** | An existing path that is not a servable artifact — one safetensors shard, a `.tiers` map, an `.imatrix`, a LoRA adapter directory. |
| **R-23** | A file that is not a usable GGUF. *A truncated 5.3 GB file with a zeroed header is a shape that really occurs.* |
| **R-28** | Tensors carrying ggml type ids the current tree does not define (e.g. 246/247, retired clustered PXQ1C/PXQ2C variants). The engine cannot dispatch them. |

### Engine and card selection

| id | Refuses |
|---|---|
| **R-07** | Forced `--engine vllm` with **no eligible card**. Not a warning: with no eligible card the parallel degree collapses to 1, `CUDA_VISIBLE_DEVICES` is never set, and the server inherits **every GPU on the host**. |
| **R-20** | A card with a resident process, or >512 MiB resident with no compute app listed. This may be a shared, live box. Escape: `--allow-busy`. |
| **R-29** | A PXQ4-converted vLLM directory when llama.cpp wins the instance — llama.cpp cannot read safetensors. Names *why* llama.cpp won. |

### Parameters that do not translate

| id | Refuses |
|---|---|
| **R-10** | `-ts` with vLLM. vLLM splits work evenly; a per-card ratio has no equivalent and would be silently ignored. |
| **R-11** | `-sm` with vLLM. No equivalent in its parallelism model. |
| **R-12** | `-sm graph` on a DeltaNet hybrid. Produces **degenerate output** — the cross-device all-reduce never reaches its consumers and each device computes a different router top-8. Not fixable by an env var. |
| **R-13L** | A `-ctk`/`-ctv` pair with no compiled FA vec kernel at head 128. It does **not** fall back — it hard-aborts at request time. Compiled asymmetric pairs: `q8_0/q6_0`, `q8_0/iq4_nl`, `q6_0/q5_0`. |
| **R-13V** | `--ctk`/`--ctv` with vLLM — no equivalent, would be silently dropped. |
| **R-14** | `--spec mtp` with vLLM. No MTP drafter there, and ngram will **not** be substituted: on this model class the two have opposite verdicts (ngram +23.0% code; MTP −8.6%). Substituting a lever's meaning is worse than dropping it. |
| **R-15A** | `mtp:n_max≥2` — a measured loss on both architectures. |
| **R-15B** | `--spec mtp` on a file with **no** nextn/mtp tensors. Two shipped f16 files declare `nextn_predict_layers=1` with zero such tensors — the head was dropped in the pipeline and the flag survived. |
| **R-25** | `--draft-model`. Zero coverage in this bench on any cell. Escape: `--accept-unmeasured` (llama.cpp only; no vLLM draft path is emitted at all). |

### Configuration and envelope

| id | Refuses |
|---|---|
| **R-08** | `cudagraph_mode` other than `FULL_DECODE_ONLY`. `FULL_AND_PIECEWISE` captures **prefill** graphs and returns fluent garbage from character zero on short raw completions. Its best aggregate (88.4) is *below* the correct config's (88.7) — there is no speed argument for it. |
| **R-16** | A partial offload (`--ngl` < 99) on a GPU-only tier (PXQ1, PXQ6). No CPU codec exists; the run would abort. |
| **R-17A** | `-c` beyond the model's trained context. |
| **R-17B** | GPU-resident weights alone exceeding total VRAM under full offload. |
| **R-21** | `--np` above the measured cudagraph capture ladder `[1,2,4,8]`. A too-short ladder has cliffed before — a hardcoded `[1,2]` ladder cliffed at 3+ concurrent. Escape: `--accept-unmeasured`. |
| **R-24** | Guessing among multiple mmproj candidates when nothing ranks them. |
| **R-26** | `--np` < 1. |
| **R-27** | A multimodal/VL checkpoint on vLLM — the emitted command has no multimodal handling at all. Escape: `--engine llama`, or `--accept-unmeasured` to serve text-only. |

### Exit codes

| code | meaning |
|---|---|
| 0 | `--explain` produced a clean plan |
| 2 | no decision / unusable artifact / unusable card selection |
| 3 | a parameter or engine request that does not translate |
| 4 | the environment cannot run the command (no engine binary, no CUDA runtime) |
| 5 | `--explain` produced a plan carrying known-fatal blockers, **or** an unacknowledged UNMEASURED branch |

Exit 5 exists so a CI caller can distinguish "clean plan" from "plan that will not
start here".

---

## The measurement envelope

The entire decision table is keyed to **two cards of one class**. Outside that
envelope the launcher labels the answer and, in several cases, requires
`--accept-unmeasured`.

The evidence that this caution is warranted: **22.3 vs 24.6 tok/s on an identical
config between two different P100 pairs.** Even 2→2 does not transfer cleanly.

Triggers you will see:

- **card count ≠ 2** — the 2-card answer is printed and labelled `[INFERRED]`;
  executing needs an ack.
- **an sm_61 card in the selection** — the np5/np6 thresholds were never measured
  on sm_61, and the BALANCE-mode `PXA_FA_MASK_SKIP_TILE` win explicitly excludes
  all of sm_61.
- **heterogeneous `-ub` expectation** — the card-type table wants different values
  on different cards while the CLI carries one global `-ub`. The launcher passes
  none so adaptive-ub probes per device, but *whether adaptive-ub lands per-card
  correctly in a heterogeneous pool is UNMEASURED*.
- **`PXQ_UNIVERSAL` tier** — a UNIVERSAL MoE is on record loading PASS and
  generating incoherent output. Nothing verifies a coherence check ran on *your*
  file.

Note that sm_61 is deliberately **not** folded into a generic "Pascal" class with
sm_60. The bench has no MoE crossover, no dense pair and no PXQ-tier throughput on
sm_61 — folding it in is how that gap stops being visible.

---

## Worked examples

### A. Single card

```
$ python3 tools/pxa-launch.py \
    --model /models/qwen3.8-27b/Qwen3.8-27B-PXQ4.gguf \
    --cards 0 --explain
```

```
==============================================================================
pxa-launch: ENGINE = llama
  model:  /models/qwen3.8-27b/Qwen3.8-27B-PXQ4.gguf  [gguf, 14.64 GiB]
  class:  dense, arch=qwen35  [read from GGUF header + tensor directory]
  tier:   PXQ4 (provenance KV says PXQ4)  [tensor-type histogram over 866 tensors
          (the signal the loader dispatches on, llama-model-loader.cpp:527)]
  compose:f32x360 PXQ4x325 q8_0x132 MXFP4x48 q6_Kx1
  trained ctx: 262144
  mtp:    4 nextn/mtp tensors; KV nextn_predict_layers=1  [tensor walk]
  deltanet: True  [336 ssm_* tensors - the GGUF spelling of the same
            linear-attention layers]
  serve:  np=1, workload=chat, ctx=4096 (=4096/slot), threads=72
  reason: model is a raw GGUF; vLLM needs a converted artifact
          (tools/vllm-pxq4/src, python -m gguf_to_vllm.convert).
          Cards: 0:Tesla P100-PCIE-16GB sm_60
  ** VRAM estimate [INFERRED, never blocks]: weights 14.64 GiB + KV 1.02 GiB
     (= ctx 4096 x 260.0 KiB/tok) = 15.66 GiB vs 1.36 GiB free / 16.00 GiB total
  ** HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load.
  -b/-ub: NOT PASSED - adaptive-ub probes each device at startup. Card-type table
          expects [2048] on this selection.
  env:     CUDA_VISIBLE_DEVICES=0 PXA_ENHANCE=1 LD_LIBRARY_PATH=...
  command: <ENGINE>/bin/llama-server -m ... --host 0.0.0.0 --port 8080
           -ngl 999 -sm layer -c 4096 -ctk f16 -ctv f16 -np 1 -t 72
           -fa on --jinja --cont-batching
==============================================================================
```

Read it as: raw GGUF → vLLM cannot load it → llama.cpp. Single card → no split, no
`-ts`. `-ub` is not passed; the expected value is printed so you can check the
server's posture line against it.

### B. Two identical cards

```
$ python3 tools/pxa-launch.py --model ...-PXQ4.gguf --cards 2,4 -c 8192 --explain
```

The envelope line now confirms the table applies as measured:

```
  ev:     MEASURED envelope: exactly 2 cards, both sm_60 - the table applies
          as measured (bench D1/D3 and the crossover sweep)
```

and the automatic split appears:

```
  AUTO -ts 697,303 [MEASURED method]: capacity-proportional over FREE VRAM after
  reserving 1200 MiB decode headroom + 282 MiB compute (980 on the head card)
  + 250 MiB context per device. Capacities: 2:1237MiB  4:539MiB. -ts partitions
  BYTES, not layers, and llama.cpp repacks when -ub changes - re-derive if you
  force a different -ub.

  command: ... -c 8192 ... -ts 697,303
```

Both cards are physically identical 16 GiB V100s with identical free memory — yet
the split is **697/303, not 500/500**, because the *head* card (the last in the
selection) is charged the 980 MiB output-head buffer against the ordinary card's
282 MiB. That asymmetry is the difference between an instance that runs and an instance that
OOMs on its first token.

If the cards are too full, the launcher declines rather than emitting a bad split:

```
  AUTO -ts DECLINED: card(s) [0, 1] have no capacity left after 1200 MiB headroom
  + compute buffers at ctx=16384. Free a card, drop -c, or pass --ts by hand.
```

### C. A mixed pool

```
$ python3 tools/pxa-launch.py --model ...-PXQ4.gguf --cards 0,2,3 --np 4 --explain
```

Cards 0 (P100 sm_60), 2 (V100 sm_70), 3 (1080 Ti sm_61) — three classes, three
warnings, then a refusal to *execute*:

```
  ** UNMEASURED card count: the MoE crossover and the dense pair were both
     measured on exactly 2 cards. You selected 3. The 2-card answer is printed
     and labelled [INFERRED]. MEASURED: 22.3 vs 24.6 on an identical config
     between two different P100 PAIRS - even 2->2 does not transfer cleanly.
  ** selection contains an sm_61 card: the np5/np6 thresholds were NEVER measured
     on sm_61, and the BALANCE-mode PXA_FA_MASK_SKIP_TILE win explicitly excludes
     all of sm_61. UNMEASURED.
  ** HETEROGENEOUS -ub expectation across this selection: the card-type table
     wants [768, 2048] on different cards while the CLI carries ONE global -ub.
     This launcher passes NO -ub so the engine's adaptive-ub probes each device
     itself - but whether adaptive-ub lands per-card correctly in a heterogeneous
     pool is UNMEASURED.

  REFUSING to execute an UNMEASURED branch without acknowledgement:
    - 3-card selection (table is 2-card only)
    Re-run with --accept-unmeasured to proceed anyway. The plan above is the plan;
    the refusal is about executing it, not about printing it.
```

That last sentence is the model for every refusal in the tool: **the plan is still
printed in full.** The refusal is about *executing* it, never about hiding it.

Note also that vLLM would be ineligible here regardless — the selection spans
three compute capabilities and the vLLM command carries a single
`--attention-backend`.

---

## What gets emitted

### llama.cpp

```
<ENGINE>/bin/llama-server -m MODEL --host H --port P
  -ngl 999 -sm layer -c CTX -ctk f16 -ctv f16 -np N -t THREADS
  -fa on --jinja --cont-batching
  [-b UB -ub UB]                     only if forced, or qwen4exp
  [-ot per_layer_token_embd\.weight=CPU]   only if a PLE table is present
  [-ts A,B]                          automatic, or forced
  [--spec-type ...] [-md ...] [--mmproj ...]
```

with `PXA_ENHANCE=1` and an `LD_LIBRARY_PATH` pointing into the build tree.

> **The stubs trap.** Any inherited `/stubs` directory is **stripped** from
> `LD_LIBRARY_PATH` and the removal is announced. A 66 KB stub `libcuda.so.1`
> shadows the real driver, ggml logs one line, offloads 0/N layers, and the run is
> numerically **correct** and ~50× slower. A CUDA toolkit install puts that
> directory on the path routinely, so this is not hypothetical.

`GGML_CUDA_NO_PINNED` is deliberately **not** emitted: it appears in no measured
recipe here, and its effect on the anchors is unmeasured. The rule is never to set
a lever the anchor was not measured with.

### vLLM

```
vllm serve MODEL --host H --port P                 (self-contained image)
HOSTPY -m vllm.entrypoints.openai.api_server ...   (host-runtime image)
  --quantization pxq4 --dtype float16
  --max-model-len CTX --max-num-seqs N
  --gpu-memory-utilization 0.90|0.85
  --attention-backend PASCAL_SDPA|FLASH_ATTN_V100
  [--disable-custom-all-reduce]      when topology shows no P2P
  --pipeline-parallel-size D --tensor-parallel-size 1     (MoE)
  --tensor-parallel-size D                                (dense)
  --compilation-config {"custom_ops":["none"],
                        "cudagraph_mode":"FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes":[1,2,4,8]}
```

Three load-bearing details:

- **`custom_ops:["none"]` is mandatory** wherever `FULL_DECODE_ONLY` is emitted on
  sm_60. Without it, PP=2 + FDO is a **hard boot failure** — an illegal memory
  access in `determine_available_memory → profile_run`, on a boot that produced
  zero tokens. Adding the key fixed it with no other change. Its necessity on
  sm_70 is unmeasured; it is emitted anyway, which is the safe direction.

- **MoE goes pipeline-parallel, dense goes tensor-parallel.** PP=2 is the arm that
  holds every MoE number in the table.

- **Custom all-reduce is disabled from *read* topology**, not hardcoded. Measured:
  CAR costs ~18% vs NCCL on MoE without P2P, while the CAR kernel itself is
  exonerated.

- **The parallel degree is truncated to a power of two** if the eligible card count
  is not one — vLLM rejects a non-power-of-two degree at startup — and the dropped
  cards are named.

`VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE` is **never** set: it crash-looped
the container at warmup 3/3 boots and left an instance in a restart loop.

---

## The post-boot contract

The launcher `exec`s the server, so it can observe nothing afterwards — and
therefore **makes no health claim at all**. Instead it prints the checks whoever
owns the instance must run:

1. **Capture mode** (vLLM): grep the log for the *installed* cudagraph mode. If it
   is not `FULL_DECODE_ONLY`, the instance is not healthy — shut it down. *Passing a
   flag is not evidence the flag took effect:* an image can parse
   `FULL_DECODE_ONLY`, boot healthy, and silently override it back from its own
   compile policy. A derived image is the fix, not a flag.
   **Posture** (llama.cpp): the server logs `PXA posture: mode=... fa=... ub=...`
   at startup — compare `ub` against the expectation printed above.
2. **Split / offload**: per-device resident bytes for vLLM; `offloaded N/N layers
   to GPU` for llama.cpp. A stub libcuda yields 0/N, correct output, ~50× slower.
3. **Device scoping**: echo the child's `CUDA_VISIBLE_DEVICES` back.
4. **Short-prompt correctness**: a **raw, non-chat-templated** 1-token and 5-token
   completion before any number is trusted — exactly as all 11 boots of the
   crossover sweep did. Chat-templated traffic pads every prompt past the captured
   sizes, which is precisely why the prefill-graph corruption survived arithmetic
   gating.
5. **Speculation**: if you armed one, the acceptance-rate line must be present and
   non-zero. If it is absent, **drop the claim** and keep serving.

---

## Environment variables

### Read by the launcher

| Variable | Effect |
|---|---|
| `PXA_ENGINE_DIR` (alias `PXQ_ENGINE_DIR`) | Directory containing `bin/llama-server`. Wins over auto-detection — and if that build will not start, that is **reported**, never quietly stepped over. |
| `PXA_VLLM_IMAGE` | Same as `--vllm-image`. |
| `PXA_VLLM_HOST_ENV` | Path to the JSON descriptor of site-local host runtimes. Unreadable or non-object content is reported and treated as **absent**, which refuses rather than proceeding on a half-read file. |
| `PXA_PXQ4_LIB` | Bare-metal fallback probe; the `libpxq4_sm<cc>_v<n>.so` name supplies the arch set. |

Engine auto-detection, when `PXA_ENGINE_DIR` is unset, looks in this order: build
directories inside the repo (`build`, `build-cuda`, `build-unified`,
`build-release`, `build-sm60`, `build-sm70`, `build-all`), the same names as
siblings of the checkout, then install prefixes (`/usr/local`, `/usr`, `/opt/pxa`,
`/opt/pxq_llama`, `~/.local`), then `llama-server` on `PATH`. **Every candidate
that exists but will not start is printed with its reason** — never skipped in
silence. There is no site-specific build list in the file: a hardcoded absolute
path is a machine's private detail.

### Emitted into the child

| Variable | Engine | Why |
|---|---|---|
| `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES` | both | Always set explicitly. |
| `PXA_ENHANCE=1` | llama.cpp | The anchor arm's env, verbatim. |
| `LD_LIBRARY_PATH` | llama.cpp | Build-tree libs, with any `/stubs` directory stripped. |
| `PXA_PARALLEL_LOAD=1` | llama.cpp | Only with `--no-mmap`. |
| `TORCHDYNAMO_DISABLE=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | vLLM | Measured crossover-sweep arm B env. |
| `VLLM_SM70_QUANT_BACKEND=turbomind` | vLLM, sm_70 | From the sm_70 serving recipe. |

`PXA_ALLOW_GRAPH_SPLIT_HYBRID` exists but only **removes the R-12 guard** — it
does not make graph split correct on a DeltaNet hybrid.

---

## `--selftest`

Runs the decision table against this machine's real cards, with no model file
involved. It prints the detected cards, their `-ub` table expectation, the peer
topology, the resolved vLLM eligibility, and then the engine choice for every
combination of card-set × model class × tier × `--np`:

```
=== selftest: decision table against this machine ===
    card 0: Tesla P100-PCIE-16GB  sm_60  16384 MiB total, 14989 MiB used, -ub table -> 2048
    card 3: NVIDIA GeForce GTX 1080 Ti  sm_61  11264 MiB total, 9552 MiB used, -ub table -> 768
    topology: no NVLink/P2P; interconnect NODE/PHB/PIX/PXB/SYS -> custom all-reduce OFF
  --- eligibility as resolved here: image=None, caps=none ---
  all cards   MoE PXQ4 gguf np=6 -> llama   :: model is a raw GGUF; vLLM needs...  <ACK-REQUIRED>
  all cards   MoE PXQ1 gguf np=6 -> REFUSE R-01 :: REFUSING: PXQ1 content
  first card  MoE PXQ3 gguf np=5 -> llama   :: tier is PXQ3: the vLLM backend...
```

Use it after changing images, drivers or card layout to see what the launcher
*would* do before a model is involved.

---

## See also

| Path | What it holds |
|---|---|
| `docs/lab/LEVERS.md` | The supported `PXA_*` levers, defaults and measurements |
| `docs/PXA-SM60-SERVING.md` | Reproduced sm_60 instance numbers and the shipping recipe |
| `docs/PXA-SM70-SERVING.md` | The same for sm_70 |
| `scripts/pxa-serve-sm60.sh`, `scripts/pxa-serve-sm70.sh` | The recipes themselves |
| `tools/vllm-pxq4/` | The vLLM PXQ4 backend and `gguf_to_vllm.convert` |
| `src/llama-quantize.cpp` | Which provenance KVs each PXQ tier writes |
| `src/llama-model-loader.cpp` | How the loader detects PXQ1, by tensor type |
| `ggml/include/ggml.h` | The ggml type ids this launcher dispatches on |
| `docs/RENAME-MAP.md` | The PXQ tier display-name ladder and retired ids |
