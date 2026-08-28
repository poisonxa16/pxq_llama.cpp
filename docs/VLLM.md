# The PXQ4 vLLM backend

PXA Network ships **two** runtimes for one quantization family.

| runtime | what it is | hardware | PXQ tiers |
|---|---|---|---|
| `pxq_llama` (this repo) | the GGUF-native llama.cpp engine | sm_60 Pascal, sm_61, sm_70 Volta, and newer | **all of them** |
| `vllm-pxq4` (`tools/vllm-pxq4/`) | a vLLM quantization plugin | sm_70 Volta, sm_60 Pascal | **PXQ4 only** |

This document covers the second one: what it is, what it will and will not load, how
to convert a model for it, how to start a server, which knobs actually move the
numbers, and the two flags a server needs before it will answer a `tools=` request.

Choosing between the two engines for a given model, card set and concurrency is what
[`tools/pxa-launch.py`](../tools/pxa-launch.py) automates. See
[Choosing an engine](#choosing-an-engine) — you do not have to make this call by hand.

---

## 1. What the backend is

`vllm-pxq4` is an **out-of-tree vLLM quantization backend**. It plugs into stock vLLM
through the documented plugin surface and **patches zero lines of vLLM**:

- a `vllm.general_plugins` entry point, `pxq4 = pxq4_vllm:register`, which calls
  vLLM's `register_quantization_config` at startup;
- a `PXQ4Config` that a checkpoint can self-select via `override_quantization_method`,
  so `--quantization pxq4` is an assertion rather than a discovery;
- a `PXQ4LinearMethod` that owns weight creation, tensor-parallel sharding and the
  forward call for every module the checkpoint declares as PXQ4;
- a standalone torch extension (`libpxq4_*.so`) that registers the `pxq4::*` operators
  through `TORCH_LIBRARY`. It links **libtorch and cudart only** — no vLLM header, no
  vLLM object, no vLLM rebuild.

Because the kernel library is a plain torch extension, upgrading vLLM does not require
rebuilding it; upgrading **torch** does, because the ABI is torch's.

What you get from vLLM that llama.cpp does not offer: tensor and pipeline parallelism,
paged KV, continuous batching, CUDA-graph capture, and real data parallelism.
llama.cpp's `-sm layer` is a *serialized* multi-GPU pipeline, so concurrent requests
queue behind one another — which is the whole shape of the crossover in §3.

### What ships where

| path | contents |
|---|---|
| `tools/vllm-pxq4/src/` | plugin source, CUDA kernels, the GGUF→vLLM converter, the parity harness |
| `tools/vllm-pxq4/docs/` | the format spec, kernel notes, plugin-surface analysis and the design record |
| `pxa/pxq4/kernels/` | prebuilt kernel libraries, one per GPU arch + revision |
| `pxa/pxq4/sidecar/site-sm60/` | the `pxq4_vllm` plugin tree vLLM loads via `PYTHONPATH` |
| `scripts/pxa-serve-sm70.sh` | V100-class serving launcher, measured defaults |
| `scripts/pxa-serve-sm60.sh` | P100-class serving launcher, measured defaults |
| `tools/pxa-launch.py` | picks the engine, prints the evidence and the command |

`pxa/pxq4/MANIFEST.md` lists every shipped binary with size and md5.

---

## 2. Quant tier support — the matrix

**The vLLM backend implements exactly one tier.** Everything else is a llama.cpp job.

| tier | `pxq_llama` (llama.cpp) | `vllm-pxq4` | note |
|---|---|---|---|
| **PXQ4** | yes | **yes** | the only tier with a vLLM kernel |
| PXQ4-HQ | yes | no | bs8 sub-scales; no vLLM kernel |
| PXQ6 | yes | no | GPU-only; no CPU codec |
| PXQ3 | yes | no | |
| PXQ2 | yes | no | |
| PXQ1 | yes (GPU only) | no | no dense path and no CPU codec |
| PXQ_UNIVERSAL | yes | no | a per-tensor mix; a PXQU file is llama.cpp-only by construction |

Two properties of this that matter operationally:

1. **The refusal is clean and early.** A non-PXQ4 tier is rejected at the conversion
   gate, not at load time and never at generation time. There is no silent
   wrong-output path on vLLM: you either get a converted checkpoint or an error.
2. **A PXQ4 file is not uniformly PXQ4.** A real dense 27B artifact parses as
   325 pxq4 + 132 q8_0 + 1 q6_K + 360 f32 + 48 mxfp4 tensors. The backbone
   (attention k/v, the output head, norms) is deliberately held at higher precision by
   the allocation table. Any design that assumes one type throughout is wrong, and the
   converter enforces the consequence: **every fused vLLM module served by
   `PXQ4LinearMethod` is uniformly PXQ4 across all of its output partitions.** There is
   no mixed-precision fused module, ever. That is why the default policy leaves
   `self_attn.qkv_proj` in fp16 even though `attn_q` is already PXQ4 on disk.

`tools/pxa-launch.py` reads the tier from the **per-tensor ggml type histogram**, not
from a metadata key, and refuses a file this backend cannot serve before emitting a
command.

### Why PXQ4 here, honestly stated

PXQ4 is **not smaller** than AWQ. Measured like-for-like on the language-model body:
AWQ g128 asym = 4.156 bpw, PXQ4 = 4.254 bpw — PXQ4 is ~2.3% *larger* per tensor. Since
decode is bytes-read-per-GPU-per-token bound, size is not the argument.

The argument is **quality per bit**: an fp16 row anchor, a per-16-element sub-scale and
a non-uniform 16-entry codebook fit against an importance matrix is a better-conditioned
4 bits than uniform group quantization at essentially the same footprint. A head-to-head
throughput comparison against AWQ has **not** been run.

---

## 3. Choosing an engine

**`tools/pxa-launch.py` automates this decision.** It reads the model file, probes the
cards, applies a measured decision table and then prints the engine, the evidence and
the exact command before running anything:

```bash
tools/pxa-launch.py --model /path/to/model --np 8 --explain   # decide and print, run nothing
tools/pxa-launch.py --model /path/to/model --np 8             # decide and exec
```

It never picks silently, it refuses rather than dropping a parameter that does not
translate between engines, and it labels any branch it is extrapolating as
`[INFERRED]` or `UNMEASURED` instead of guessing quietly. Full behaviour:
[`docs/LAUNCHER.md`](LAUNCHER.md).

The decision is not just about the card. These are the measurements behind it, all on
one 2x Tesla P100-PCIE-16GB pair (sm_60), every boot correctness-gated before its
number was kept:

**Dense 27B PXQ4 — vLLM wins everything measured**

| metric | vLLM | llama.cpp | ratio |
|---|---|---|---|
| single-stream decode | 24.01 | 13.7 | 1.75x |
| aggregate decode @8 | ~70 | 12.4 | 5.6x |
| prefill | ~225 | 156.5 | 1.44x |

*Caveat carried with these: the llama.cpp side is a single boot, below this bench's own
two-boot bar, and the graphs-on dense arm was never launched. The direction is not in
doubt; the exact ratios are single-boot.*

**MoE 35B PXQ4 — the engines swap places at a sharp crossover**

| concurrency | llama.cpp | vLLM | winner |
|---|---|---|---|
| np=1 | 95.6 | 30.4 | llama.cpp 3.14x |
| np=4 | 75.93 | 64.82 | llama.cpp +17.1% |
| np=5 | 79.49 | 64.32 | llama.cpp +23.6% ← llama.cpp peaks |
| np=6 | 69.58 | 75.60 | **vLLM +8.7%** ← crossover |
| np=7 | 67.74 | 87.03 | vLLM +28.5% |
| np=8 | 62.42 | 95.81 | vLLM +53.5% |

Neither curve is monotonic and the flip is abrupt — llama.cpp *peaks* at np=5, above its
own np=4 value, then drops 12.5% in one step while vLLM climbs. The margin swings 32
points between np=5 and np=6. The launcher therefore stores the **table**, never a
fitted slope: a straight line from np=4 to np=8 puts the threshold too early and
misprices np=5 by ~14%.

Long-document prefill on the same MoE model favours llama.cpp by ~1.7–1.9x (1136 /
~1058 / ~1000 vs 567.6 / 595.8 / 594.4 tok/s), but that arm is **cross-harness with
unmatched prompt lengths** — directionally trusted, not controlled.

Rule of thumb, if you are choosing by hand: **dense model or high concurrency → vLLM;
MoE at low concurrency, or long-document prefill → llama.cpp.** Nothing above np=8 was
measured on either engine.

---

## 4. Hardware and images

| arch | cards | attention backend | status |
|---|---|---|---|
| sm_70 | Tesla V100 | `FLASH_ATTN_V100` | measured, serving |
| sm_60 | Tesla P100 | `PASCAL_SDPA` | measured, serving |
| sm_61 | GTX 1080 Ti, P40 | — | **not supported here** — use `pxq_llama` |

**Pascal support is not free.** Stock vLLM compiles for compute capability 7.0 and up,
and the last PyTorch shipping sm_60 cubins is 2.7.1+cu126. Running on P100 therefore
needs its own image, its own torch, the opt-in `tools/vllm-pxq4/tools/patch_sm60_compile.py`,
and `TORCHDYNAMO_DISABLE=1`.

**Two thin images, not one fat one.** A single image spanning sm_60 and sm_70 was tried
and does not work, for a structural reason rather than a configuration one:
`VLLM_SKIP_C_STABLE=1` is required to build against torch 2.7.1, and it drops
`csrc/libtorch_stable/`, where an operator the V100 serving path calls unconditionally
lives. You cannot have sm_60 cubins and that operator in the same build. Each arch gets
an image pinned to the torch its cards need.

**Eligibility is a property of the image, not of the compute capability.** An image only
serves a card if it actually carries PXQ4 kernels for it; `pxa-launch.py` probes this
rather than assuming a capability floor.

### Picking the kernel library

`pxa/pxq4/kernels/` holds one library per arch and revision. Each carries exactly one
CUDA device binary, for the arch in its filename, and **no PTX** — a library will not
run on an architecture it was not built for.

| library | arch | ops added | status |
|---|---|---|---|
| `libpxq4_sm70_v10.so` | sm_70 | `f16_mmv_out` | **shipped** for sm_70 |
| `libpxq4_sm60_v10.so` | sm_60 | `f16_mmv_out` | **shipped** for sm_60 |
| `libpxq4_sm60_v11.so` | sm_60 | `gemm2d_out` | ~+34% prefill behind `PXQ4_GEMM2D`, **default off** — failed first-token quality at 87.5%. Do not enable without re-gating quality. |
| `libpxq4_sm60_v9.so` | sm_60 | `f16_mmv_out` | superseded by v10 |
| `libpxq4_sm70_v9.so` | sm_70 | `f16_mmv_out` | superseded by v10 |
| `libpxq4_sm60_v8.so` | sm_60 | `moe_mmv_out` | superseded |

> **`PXQ4_LIB` must always be set explicitly.** The fallback lookup names are fixed
> (`libpxq4_sm70.so`) regardless of the arch in the file, so with `PXQ4_LIB` unset the
> loader can reach an sm_70 / torch-2.10 library on a Pascal image and die part-way
> through model load with
> `undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv` — not with a clear
> message. Both shipping launchers set it for you.

Identifying a library after the fact — the ops are registered through `TORCH_LIBRARY`
string schemas, so `nm` will not find them:

```bash
grep -aoE '^(moe_mmv_out|f16_mmv_out|gemm2d_out)$' pxa/pxq4/kernels/libpxq4_sm60_v10.so | sort -u
cuobjdump --list-elf pxa/pxq4/kernels/libpxq4_sm60_v10.so   # expect one member: pxq4_kernel.sm_60.cubin
```

---

## 5. Converting a PXQ4 model for vLLM

vLLM cannot read a PXQ GGUF. Two independent blockers, neither patchable without
forking three packages: `gguf.GGMLQuantizationType(252)` raises inside
`GGUFReader._build_tensors`, killing the file open before a single tensor is yielded;
and vLLM's generic GGUF sharder slices rows assuming per-row-contiguous blocks, which
the 64-row panel interleave violates.

So conversion is **offline and explicit**. The converter is pure Python + numpy — no
torch, no CUDA, no vLLM, no GPU:

```bash
cd tools/vllm-pxq4/src
python -m gguf_to_vllm.convert \
  --gguf   /path/to/model-PXQ4.gguf \
  --ref-hf /path/to/reference-hf-checkpoint \
  --out    /path/to/model-PXQ4-vllm \
  --policy p1
```

| flag | meaning |
|---|---|
| `--gguf` | the PXQ4 GGUF to convert. Required. |
| `--ref-hf` | a reference HF checkpoint of the same model: source of `config.json`, tokenizer, the vision tower, and the key-set diff. Effectively mandatory for a servable output. |
| `--out` | output directory. Required unless `--dry-run`. |
| `--policy` | which modules are served as PXQ4 — see below. Default `p1`. |
| `--encoder` | path to `pxq4_encode.so`; required by the `p2*` policies, which re-encode tensors that are not PXQ4 on disk. |
| `--shard-size-gb` | safetensors shard size. Default 4.0. |
| `--dry-run` | plan the entire conversion from the GGUF header alone and run every structural check, without reading tensor data. |
| `--verify` / `--no-verify` | round-trip every native PXQ4 tensor and compare **bytes**. On by default. Leave it on. |
| `--emit-plan` | write the conversion plan as JSON. |

**Run `--dry-run` first.** It exercises everything except the byte-writing and will tell
you, in seconds and off the header alone, whether the artifact is convertible.

### Policies

| policy | serves as PXQ4 | needs `--encoder` |
|---|---|---|
| `p1` | what the dense artifact already carries as PXQ4 on disk | no |
| `p2a` | p1 + the GDN output projection (`ssm_out`, MXFP4 on disk → re-encoded) | yes |
| `p2c` | p2a + a uniformly PXQ4 fused QKV (re-encodes k/v) | yes |
| `m1` | the MoE policy: expert stacks, shared experts, `o_proj`, fused GDN in-projection | no |
| `p2b` | **blocked at the CLI** — it was p2a plus a PXQ4 LM head, and a 4-bit head is not servable by the engine side. Rather than silently emitting a checkpoint byte-identical to p2a under a name that promises more, the converter refuses it by name. |

Start with `p1` (or `m1` for MoE). It needs no encoder and no re-encoding.

### What comes out

For every module the policy serves as PXQ4, **two** tensors and **no** `.weight`:

```
<module>.pxq4_slabs    uint8     [N/64, K/32, 1088]   C-contiguous
<module>.pxq4_anchor   float16   [N/64, 64]           C-contiguous
```

These are derived from the GGUF blob by a **pure split** — the header bytes and the slab
bytes of each panel, reinterpreted, with no value recomputed. That is why the emitted
checkpoint can be proven equal to the GGUF by a byte comparison rather than a numeric
tolerance, and why `--verify` can round-trip every tensor exactly.

Everything else is decoded to fp16 `<module>.weight`. `config.json` is copied from
`--ref-hf` with **only** its `quantization_config` rewritten, so every architectural
field stays byte-identical to what already runs.

There is exactly one place bytes move: **GDN head order.** ggml stores value-heads
repeat-major, HF stores them k-head-major, so every per-v-head axis is gathered into HF
order on the way out. It stays a byte move (a 128-row head block is exactly 2 panels, a
128-column block exactly 4 slabs, so no nibble, sub-scale or anchor value is touched)
and `--verify` undoes the gather before comparing. Both the reorder and its proof
against `--ref-hf` are **fatal if missing**, not warnings: an unpermuted GDN checkpoint
loads, shards, passes every byte gate, and generates fluent garbage.

### Gates before trusting a conversion

The GPU-free suites need no CUDA and no GPU:

```bash
cd tools/vllm-pxq4/src
bash build_hostsim.sh          # compiles the CPU kernel simulator, then runs the kernel suite
python3 test_pxq4_config.py    # quant config / plugin registration
python3 gguf_to_vllm_test.py   # converter, incl. a bit-exact gate against a C oracle
python3 test_pxq4_linear.py    # linear method (skips cleanly if vLLM is absent)
```

`build_hostsim.sh` compiles the **real** `pxq4_kernel.cuh`, unmodified, against a stub
`cuda_fp16.h` and emulates a CUDA launch — so the kernel suite exercises the shipping
kernel source rather than a reimplementation that could drift from it. **A C++ compiler
is required:** without `libpxq4_hostsim.so` the eight simulator-backed tests *fail*
rather than skip. Seeing `9/17` means a missing toolchain, not a kernel defect.

---

## 6. Starting a server

### The easy path — the shipping launchers

Both scripts boot a container, wait for `/health`, and then **read back** the settings
that fail silently if they do not take.

```bash
# Volta / Tesla V100 class
MODEL=/path/to/model-PXQ4-vllm scripts/pxa-serve-sm70.sh

# Pascal / Tesla P100 class
MODEL=/path/to/model-PXQ4-vllm scripts/pxa-serve-sm60.sh

# every parameter and its default
scripts/pxa-serve-sm70.sh --help
```

Only `MODEL` has no usable default. Everything else is an environment variable:
`CARDS` (comma-separated indices; TP size = how many you list), `PORT`, `BIND`
(defaults to `127.0.0.1`), `IMAGE`, `LIB`, `SITE`, `GMU`, `MML`, `MNS`, `LADDER`,
`SPLIT_MAX_BLOCKS`, `TOOL_PARSER`, `EXTRA_ARGS`, `BOOT_TIMEOUT`.

Two optional operator guards, both unset by default: `PXA_REQUIRE_HOST` refuses to run
unless `hostname` matches, and `PXA_RESERVED_CARDS` (sm_60 script) refuses if `CARDS`
intersects a list of indices you have reserved for other workloads.

### The underlying command

```
vllm serve <model> \
  --quantization pxq4 \
  --dtype float16 \
  --attention-backend FLASH_ATTN_V100          # PASCAL_SDPA on sm_60 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --trust-remote-code \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16]}'
```

with `PYTHONPATH` pointing at the `pxq4_vllm` plugin tree and `PXQ4_LIB` pointing at the
kernel library for the card.

> **Bash brace-expands that JSON.** `{"cudagraph_capture_sizes":[1,2,3,4]}` becomes
> `cudagraph_capture_sizes:[1` because of the commas inside the braces. Single-quote it
> for whichever shell finally parses it.

`vllm serve` is correct for a self-contained image. It is **wrong** for an image whose
python, torch and vLLM live on the host and are bind-mounted in — `vllm` is not on
`PATH` there, and the command must be
`<host python> -m vllm.entrypoints.openai.api_server --model ...` instead.
`pxa-launch.py` handles that distinction from a site-local JSON descriptor named by
`PXA_VLLM_HOST_ENV`; nothing site-specific is hardcoded.

---

## 7. Tuning: what moves the number, and by how much

Everything in this section was measured with each arm correctness-gated *before* its
speed number was recorded. A fast wrong answer is not a result.

### The three correctness keys — not tuning knobs

| setting | why |
|---|---|
| `cudagraph_mode: FULL_DECODE_ONLY` | The vLLM default (`FULL_AND_PIECEWISE`) **also captures prefill graphs** at the ladder sizes. A raw `/v1/completions` prompt short enough to fit one then prefills through a captured graph whose input buffer holds stale data and returns fluent garbage from character zero. Chat traffic never shows it, because the chat template pads every prompt past the captured sizes — which is exactly why arithmetic gates stayed green while the bug was live. |
| `custom_ops: ["none"]` | Mandatory wherever `FULL_DECODE_ONLY` is emitted on sm_60. Without it, PP≥2 + FDO is a **hard boot failure** (`CUDA error: an illegal memory access was encountered`, in `profile_run`). On sm_70 its necessity is unmeasured; it is emitted anyway, which is the safe direction. |
| `TORCHDYNAMO_DISABLE=1` (sm_60 only) | Load-bearing. Without it `profile_run` compiles the **language** model through Inductor and dies with `GPUTooOldForTriton` on a capability-6.0 card. This single flag decides whether the server starts at all. |

Do not "fix" the Triton problem by disabling Triton globally on Pascal — that was tried
and reverted. The warmup path then calls `triton.next_power_of_2` on
`TritonPlaceholder`, which does not define it, and shimming that method only moves the
failure to a real Triton kernel launch on the same path.

### The capture ladder — the largest silent loss

**The ladder must be passed explicitly.** When `cudagraph_capture_sizes` is `None` the
sm_70 branch hard-codes `[1, 2]`, so every batch above 2 concurrent runs eager —
roughly **4x slower**. Both launchers read the installed value back out of the startup
log and warn if it collapsed to `[1,2]`; a quoting slip is otherwise invisible.

The ladder should be powers of two covering `--max-num-seqs`. `[1,2,4,8]` is the
measured ladder on sm_60; widening past 8 is inference, not measurement.

### sm_70 (2x Tesla V100-PCIE-16GB, TP=2, dense 27B PXQ4)

| arm | decode | ms/tok | agg@4 | agg@8 |
|---|---|---|---|---|
| baseline | 48.74 | 20.52 | 106.46 | 134.36 |
| `PXQ4_MMV_SPLIT_MAX_BLOCKS=300` | **51.46** | 19.43 | 107.26 | 132.13 |
| `SPLIT=600` | 51.31 | 19.49 | 106.60 | 134.47 |
| `SPLIT=150` | 49.51 | 20.20 | 105.07 | 134.66 |
| `SPLIT=300`, GMU 0.92 | 51.32 | 19.49 | 104.85 | 133.76 |
| `SPLIT=300`, MNS=16, ladder `[1..8,16]` | 50.98 | 19.62 | **107.02** | **135.78** |

Two shipped profiles: `agg` (default; MNS=16) takes both aggregate crowns while still
clearing 50 tok/s single-stream. `PXA_PROFILE=single` (MNS=8) trades 0.5 tok/s of
concurrency headroom for the best single-stream figure.

| knob | value | effect |
|---|---|---|
| `PXQ4_MMV_SPLIT_MAX_BLOCKS` | **300** | routes `gate_up` from mono to split. **+2.7 tok/s** (48.74 → 51.46). 150 is worse, 600 is a wash. |
| `--gpu-memory-utilization` | **0.85** | *not* 0.90+. 0.98 with a pinned 12 GiB KV is a four-card, 32-GiB-per-card setting and **aborts** on a 16 GiB card at TP=2 before the model finishes loading. |
| `--max-num-seqs` | 16 (`agg`) / 8 (`single`) | drives the ladder; see the table above. |
| `f16 mmv hook` | n/a | does **not** arm on sm_70 — it is Pascal-specific. Expected, and costs nothing here. |

### sm_60 (2x Tesla P100-PCIE-16GB, TP=2)

| metric | measured |
|---|---|
| single-stream decode | 24.0 – 26.4 tok/s |
| aggregate @8 | 70.0 – 72.1 tok/s |
| aggregate @4 | 45.0 – 51.5 tok/s |
| long-doc prefill | ~218 tok/s |

| knob | value | effect |
|---|---|---|
| custom all-reduce | **left ON** | the dominant lever: **13.3 → 24.0 tok/s, ~1.8x**. Do *not* pass `--disable-custom-all-reduce` here. Byte-gated 20/20 against an NCCL reference on this exact library + tree + config. It remains unsafe for MoE models on Pascal — a different model class. |
| plugin tree with the fp16 mmv hook | `site-sm60` | **~+12%**, and its absence is *silent*. Count it, do not assume it — the log string is `fp16 mmv fast path armed`; grepping `f16 mmv` matches nothing and makes a working hook look absent. |
| `PXQ4_MMV_SPLIT_MAX_BLOCKS` | **300** | not 150. 150 is **bimodal**: four boots gave 24.58, 24.61, 19.75, 9.27. 300 gave 24.01 / 23.97 / 23.97 across three. A single good 150 sample looks like a win and is not. The mechanism is not understood; 300 ships on stability. |
| `--gpu-memory-utilization` | **0.90** | 0.94 reintroduces the raw-prompt `!!!!` failure. |
| `--max-num-seqs` / ladder | **8** / `[1,2,4,8]` | both required for correctness. With defaults at MNS=4 / ladder `[1,2,4]`, a raw 1-token prompt returns `!!!!` while "Paris" and `17*23` both still pass. MNS=16 is unverified on this arch. |
| `PXQ4_MMV_SLICE_MAX` | 8 | **a dead knob** at serving level (14.43 at 8 vs 14.24 at 16). Pinned only to stop it being re-litigated. |
| `PXQ4_GEMM2D=1` (v11 library) | **off** | reaches **300.1 tok/s prefill (+37%)** and **fails raw-prompt correctness**. Decode and the byte-gate are unaffected. A prefill win that changes what the model says is not a win. Recorded so it is not rediscovered as a fresh idea. |

### Parallelism

- **Dense → tensor parallel.** `--tensor-parallel-size <cards>`.
- **MoE → pipeline parallel.** `--pipeline-parallel-size <cards> --tensor-parallel-size 1`.
  PP=2 + `FULL_DECODE_ONLY` is the arm that produced every MoE number in §3.
- vLLM rejects a non-power-of-two parallel degree at startup.
- **Without P2P between cards, turn custom all-reduce off on MoE** — it costs ~18%
  versus NCCL there. On the sm_60 dense arm above it is the opposite: leaving it on is
  worth 1.8x. `pxa-launch.py` reads the topology rather than hardcoding either.

---

## 8. Tool calling

**A server will not answer a `tools=` request without two flags.** Omit them and every
such request returns **400**.

```
--enable-auto-tool-choice --tool-call-parser qwen3_coder
```

With the shipping launcher:

```bash
MODEL=/path/to/model-PXQ4-vllm TOOL_PARSER=qwen3_coder scripts/pxa-serve-sm70.sh
```

**The parser choice is not cosmetic.** For the Qwen-family coder templates it is
`qwen3_coder`, **not** `hermes`: hermes expects JSON inside `<tool_call>`, while this
template emits XML — `<function=NAME><parameter=K>V`. Choosing hermes does not error;
it returns **empty tool calls**, which is a far more expensive failure to notice. Match
the parser to the chat template your model actually ships.

> `tools/pxa-launch.py` does **not** emit these flags and has no passthrough for extra
> vLLM arguments. If you need tool calling, either run `--explain`, take the printed
> command and append the two flags, or use `scripts/pxa-serve-sm70.sh` with
> `TOOL_PARSER=`.

---

## 9. After the server is up

A launcher that execs the server cannot observe anything afterwards, so neither
launcher makes a health claim. **Passing a flag is not evidence the flag took effect** —
an image can parse `FULL_DECODE_ONLY`, boot healthy, and silently override it back to
`FULL_AND_PIECEWISE` from its own compile policy. Check:

1. **Capture mode.** Grep the server log for the *installed* cudagraph mode. If it is
   not `FULL_DECODE_ONLY`, the server is not healthy — shut it down. A derived image is
   the fix, not a flag.
2. **Capture ladder.** `grep -ao 'cudagraph_capture_sizes[^]]*]'` on the log. If it
   collapsed to `[1, 2]`, the explicit list did not take and you are eager above 2
   concurrent.
3. **The fp16 mmv hook, on sm_60.** `grep -c 'fp16 mmv fast path armed'`. Expect it on
   most layers; a low count costs ~12% and says nothing in the logs by itself.
4. **Per-device resident bytes.** Confirm the split actually happened.
5. **Short-prompt correctness, on a RAW, non-chat-templated prompt** — a 1-token and a
   5-token completion, before any number is trusted. Chat-templated traffic pads every
   prompt past the captured graph sizes, which is precisely how prefill-graph corruption
   survives arithmetic gating.

Both launchers do (2) and (3) for you and warn on failure.

### Failure modes that are silent by default

| symptom | cause |
|---|---|
| `undefined symbol: ..._ZNK3c1010TensorImpl15incref_pyobjectEv` mid-load | `PXQ4_LIB` unset or pointing at the wrong torch ABI |
| `GPUTooOldForTriton` at `profile_run` on P100 | `TORCHDYNAMO_DISABLE=1` missing |
| `CUDA error: an illegal memory access` in `profile_run`, zero tokens | `custom_ops: ["none"]` missing with PP≥2 + FDO |
| fluent garbage from character zero on a raw short prompt | prefill graphs captured — `cudagraph_mode` is not `FULL_DECODE_ONLY` |
| `!!!!` on a raw 1-token prompt, sane answers otherwise (sm_60) | MNS/ladder too small, or GMU ≥ 0.94 |
| ~4x slowdown above 2 concurrent | capture ladder collapsed to `[1,2]` |
| `tools=` requests return 400 | `--enable-auto-tool-choice` / `--tool-call-parser` missing |
| empty tool calls, no error | wrong `--tool-call-parser` for the template |
| ~12% less decode on sm_60, no error | fp16 mmv hook did not arm; check `PYTHONPATH` / `PXQ4_LIB` |

---

## 10. Correctness gates this backend had to pass

1. bit-exact dequant parity against a CPU reference;
2. single-linear-layer GEMM parity;
3. **sharded parity** — each per-rank slice must dequantize to exactly the unsharded
   result;
4. logprob parity against `pxq_llama` on the same prompts at temperature 0;
5. an end-to-end throughput measurement on the target cards.

All five pass on both architectures. `-use_fast_math` is forbidden in any kernel build:
it would change the fp32 fold order, and bit-identity with the llama.cpp kernels is the
whole correctness argument.

---

## Further reading

| document | what it covers |
|---|---|
| [`tools/vllm-pxq4/README.md`](../tools/vllm-pxq4/README.md) | the package itself, attribution, build and test |
| [`docs/PXA-SM70-SERVING.md`](PXA-SM70-SERVING.md) | the full V100 sweep and every trap |
| [`docs/PXA-SM60-SERVING.md`](PXA-SM60-SERVING.md) | the full P100 sweep and every trap |
| [`docs/LAUNCHER.md`](LAUNCHER.md) | `pxa-launch.py`: the decision table, refusals and evidence |
| [`docs/LEVERS.md`](LEVERS.md) | the `PXA_*` levers on the llama.cpp engine |
| [`pxa/pxq4/README.md`](../pxa/pxq4/README.md) | kernel libraries, the `PXQ4_LIB` rule, rebuilding |
| `tools/vllm-pxq4/docs/01-pxq4-format-spec.md` | the PXQ4 on-disk format |
| `tools/vllm-pxq4/docs/09-chosen-design.md` | the design that was built, and its gates |

---

PXQ, PXQ4 and `pxq_llama` are developed by **PXA Network** — <https://pxanetwork.com>.
The vLLM serving engine is Apache 2.0; this backend matches that licence. See
`tools/vllm-pxq4/LICENSE-NOTICE.md` for the full attribution chain.
