# vllm-pxq4 — PXQ4 quantization backend for vLLM

Serve PXQ4-quantized models on vLLM, with tensor parallelism, CUDA-graph capture and
paged KV — on Volta (sm_70) and Pascal (sm_60) datacentre cards.

This ships **alongside** the `pxq_llama` engine, not inside it. Two runtimes, one
quant family:

| runtime | hardware | why |
|---|---|---|
| `pxq_llama` (this repo) | sm_60 Pascal, sm_61, sm_70 Volta, newer | GGUF-native, runs everywhere, the universal engine |
| `vllm-pxq4` (this package) | sm_70 Volta, sm_60 Pascal | tensor parallelism + CUDA graphs + real data parallelism, which llama.cpp's `-sm layer` does not have |

**Pascal support is not free.** Stock vLLM compiles for compute capability 7.0 and up
(`CUDA_SUPPORTED_ARCHS = "7.0;7.5;8.0;..."`), and the last PyTorch shipping sm_60 cubins is
2.7.1+cu126. Running on P100 therefore needs its own image, its own torch, the opt-in
`tools/patch_sm60_compile.py`, and `TORCHDYNAMO_DISABLE=1`. The recipe and every trap are in
[`docs/PXA-SM60-SERVING.md`](../../docs/PXA-SM60-SERVING.md). sm_61 (1080 Ti / P40) is **not**
supported here — use `pxq_llama` for those cards.

Which engine to run on which cards, for which workload, is what `tools/pxa-launch.py`
decides; it prints the evidence and the exact command rather than choosing silently.

## Standing on other people's work

This package exists because two pieces of work happened first, and it would be dishonest
to present it without them:

- **vLLM** (Apache 2.0) — the serving engine, tensor parallelism, paged attention,
  continuous batching, CUDA-graph capture. We patch **zero lines** of it; this plugs in
  through the documented `register_quantization_config` hook.
- **A community sm_70 fork of vLLM** — the Volta port.
  Upstream vLLM dropped sm_70; that fork carries the TurboMind sm_70 W4A16 GEMM, the
  `FLASH_ATTN_V100` attention backend, and the Qwen Gated-DeltaNet kernels
  (`FlashQLA-SM70`) without which none of this runs on a V100. **PXA Network contributed
  to getting that V100 support working.** This package is the continuation of that effort,
  not a fork of it.
- **PXQ4** — the quantization format, its CUDA kernels, and this backend: PXA Network.

If you only want faster inference on Volta and do not need PXQ, use that fork directly.
This package is for people who have PXQ artifacts.

## Honest performance

Do not read a headline number off this without reading the paragraph under it.

**What is measured, on the shipping configuration** (`scripts/pxa-serve-sm70.sh` /
`scripts/pxa-serve-sm60.sh`, dense 27B PXQ4, TP=2, every arm correctness-gated before its
number was kept — full protocol and the full sweep in
[`docs/PXA-SM70-SERVING.md`](../../docs/PXA-SM70-SERVING.md) and
[`docs/PXA-SM60-SERVING.md`](../../docs/PXA-SM60-SERVING.md)):

| cards | single-stream decode | aggregate @8 | prefill |
|---|---|---|---|
| 2x Tesla V100-PCIE-16GB (sm_70) | 50.98–51.46 tok/s | 132–136 tok/s | — |
| 2x Tesla P100-PCIE-16GB (sm_60) | 24.0–26.4 tok/s | 70–72 tok/s | ~218 tok/s |

**What is not measured: a head-to-head against AWQ.** The pre-release projection was
~+9% over AWQ W4A16 on 4x V100-SXM2-32GB (TP=4), against a measured AWQ incumbent of
92.8 peak / 57.4 median decode tok/s. That comparison has **never been run** and the
projection is carried here only so it is not mistaken for a result.

**PXQ4 is not a smaller format than AWQ.** Measured like-for-like on the language-model
body: AWQ g128 asym = 4.156 bpw; PXQ4 = 4.254 bpw (`4.25 + 16/K`). PXQ4 is ~2.3% *larger*
per tensor. Since decode is bytes-read-per-GPU-per-token bound, a naive port that leaves
the non-PXQ4 tensor classes in fp16 is a **~23% regression**, not a win. Reaching +9%
requires re-encoding `lm_head`, `attn_k`/`attn_v` and `ssm_out`.

**The reason to use PXQ4 here is quality per bit, not size.** The format carries an fp16
row anchor, a per-16-element sub-scale and a non-uniform 16-entry codebook fit against an
importance matrix — a better-conditioned 4-bit than uniform group quantization at
essentially the same footprint.

### A free win for any vLLM deployment, PXQ or not

Profiling the incumbent turned up something unrelated to our format: `lm_head` is served
**BF16 (2.37 GiB)** and sits in the 311-entry `ignore` list. It is read on every decode
step on every rank — roughly 12% of all decode traffic. Quantizing it is likely the
cheapest speedup available on that deployment and needs nothing from this package.
Caveat: the output head is the layer most sensitive to quantization error; `q8_0` captures
about half the win at much lower risk, which is why the PXQ backbone table keeps `output`
at `q8_0` rather than 4-bit.

## Status

Serving on real hardware on both architectures; see the measured table above and the two
serving documents it links. The gates that had to pass first, all of which now do:

1. bit-exact dequant parity against a CPU reference
2. single-linear-layer GEMM parity
3. **sharded parity** — per-rank slices must dequantize to exactly the unsharded result
4. logprob parity vs `pxq_llama` on the same prompts at temp 0
5. an end-to-end throughput measurement on the target cards

Still open: the AWQ head-to-head in the previous section, and the re-encoding of
`lm_head` / `attn_k` / `attn_v` / `ssm_out` that the +9% projection assumed.

## The `docs/` set

`docs/01`-`docs/07` are the investigation: the PXQ4 on-disk format, the kernels, the vLLM
plugin surface, MoE and loading, file composition, and the tensor-parallel sharding verdict.
`docs/08-design-*` are three competing designs written against the same constraints -
least-code, max-performance, minimal-risk - and `docs/09-chosen-design.md` is the one that
was built, with the correctness gates it had to pass. `docs/10`-`docs/13` are the kernel
speed work and the two things that stayed blocked.

Source comments in `src/` cite the port's design plan as `plan §N` (`plan §5.3`, `plan §6.3`
and so on). Those section numbers index the design decisions recorded across the `docs/` set
above - `§5` the converter, `§6` the runtime contracts, `§7` the kernel ABI - and are kept in
the comments so a contract can be traced to the reasoning that fixed it.

Where those documents say **"the initial project spec"** they mean the starting assumptions
this work was scoped against, stated once here: target hardware of 4x V100-SXM2-32GB at TP=4;
an incumbent AWQ W4A16 deployment measured at 92.8 peak / 57.4 median decode tok/s; a uniformly
PXQ4 artifact sharding to 3.66 GiB/GPU; and a projected 110-120 tok/s. **Three of those four
turned out to be wrong** - the artifact is five tensor types and not uniformly PXQ4, the
like-for-like footprint is 4.254 bpw against AWQ's 4.156, and the projection collapses to ~+9%
once that is corrected. Each correction is recorded next to the claim it replaces rather than
quietly applied, which is why the phrase appears at all.

## Licence

Apache 2.0, matching vLLM. See `LICENSE-NOTICE.md` for the full attribution chain.

---

## Build and test

Everything under `src/` is a FLAT directory on purpose: `build_hostsim.sh`, `setup.py`,
`CMakeLists.txt` and the test imports all resolve relative to it. An earlier tidy-up into
`csrc/ tests/ vllm_pxq4/` broke every entry point, so the layout that works is the one
that ships.

### GPU-free gates (run these first - they need no CUDA, no GPU)

```bash
cd src
bash build_hostsim.sh          # compiles the CPU simulator, then runs the kernel suite
python3 test_pxq4_config.py    # quant config / plugin registration
python3 gguf_to_vllm_test.py   # converter, incl. a bit-exact gate against a C oracle
python3 test_pxq4_linear.py    # linear method (skips cleanly if vLLM is absent)
```

The converter suite ships every fixture except `src/fixtures/hdr.bin` — a 10.5 MB header
slice of a real artifact, too large to ship. Without it the 40 header-directory tests SKIP
with the command that regenerates them:

```bash
head -c 10997184 <model>-PXQ4.gguf > src/fixtures/hdr.bin   # then set GGUF_TRUE_SIZE
```

`build_hostsim.sh` compiles `pxq4_kernel_hostsim.cpp`, which includes the REAL
`pxq4_kernel.cuh` unmodified against a stub `cuda_fp16.h` and emulates a CUDA launch.
So the kernel suite exercises the shipping kernel source, not a reimplementation that
could drift from it.

**A compiler is required.** Without `libpxq4_hostsim.so` the 8 simulator-backed tests
FAIL rather than skip, and the failure text tells you to build it. On a host with no
`g++` you will see `9/17` — that is a missing toolchain,
not a kernel defect. Build on a dev host or inside the CUDA container.

### CUDA extension (needs the CUDA toolkit; no GPU needed to compile)

```bash
bash src/build.sh              # sm_70; expect a "prior to sm_75" deprecation warning
```

### Measured status

| suite | result | where |
|---|---|---|
| kernel parity (hostsim) | 17/17 | any host with g++ |
| quant config | 27/27 | anywhere |
| config integration | 6/6 | against the real vLLM fork |
| converter | 39/79, 40 skipped (79/79 with the header fixture, below) | anywhere |
| CUDA build | clean sm_70, ops register | CUDA container |
| end-to-end on GPU | serving, correctness- and byte-gated | 2x V100 (sm_70), 2x P100 (sm_60) |

The real artifact parses as 325 pxq4 + 132 q8_0 + 1 q6_K + 360 f32 + 48 mxfp4, and
`data_start + sum(nbytes) == file size` exactly. Any design assuming uniform PXQ4 is wrong.
