<!-- GitHub README for pxq_llama. -->
<p align="center"><img src="banner.png" alt="pxq_llama — PXQ quants and a MoE accelerator for landfill GPUs" width="100%"></p>

# pxq_llama — the codec + kernel pack for cards with no DP4A

> Authored and maintained by **PXA Network** (https://pxanetwork.com) — the creator of pxq_llama and the PXQ/PXA kernel family.

**Community: [Discord — PXA Network](https://discord.gg/BHWmMHHStY)** — support, benchmark wall, dev talk. Release notes post there automatically.

Models: **https://github.com/poisonxa16/pxq_llama** ← you are here · Weights: [huggingface.co/poisonxa](https://huggingface.co/poisonxa)

> 💛 Support: **https://ko-fi.com/shatteredrealms1**

## Where this sits

- **ik_llama.cpp** — best CPU/hybrid/new-quant support on Turing and newer.
- **llama.cpp** — broadest compatibility, the master CUDA backend.
- **p100-patches** — makes llama.cpp master less bad on Pascal.
- **pxq_llama** — the codec + kernel pack for cards with HBM2 and no DP4A: **Pascal (P100)**, and Volta.

The pitch in one sentence: **run real models fully in VRAM on a used Tesla P100** — kernels
written for a chip with no DP4A and no tensor cores, not ported from one that has them. Volta
(V100), the 1080 Ti, and multi-card 122B-class MoE spreads all run on the same engine and are
covered below — they're scaling proof, not the pitch.

## Not MoE-only

The engine loads **83 model architectures** (`src/llama-arch.cpp`) — dense (Qwen, Llama, Gemma,
Mistral, …), GDN hybrids (`qwen3next`), MoE, `gpt-oss`, DeepSeek-V4 (`deepseek4`), GLM
(`glm4moe`, `glm-dsa`), MiniMax (`minimax-m2`), Cohere (`cohere2`, `cohere2_moe`), Laguna,
`gemma4`. **Every one of the engine-level fixes below — the sm_60 fp16-GEMM path,
flash-attention regime routing, the MoE path, `np>1` hybrid concurrency, and the wide f16
GEMV — applies to stock GGUF files (Q4_K, MXFP4, IQ_K) with no PXQ file required.** Point this
engine at a stock Q4_K_M, MXFP4, or IQ_K GGUF you already have and the fixes apply as-is; see
`docs/COOKBOOK.md` → "stock-gguf-on-pxq-engine" for the exact command and the engine-only numbers
that back it.

## Two products, not one

- **The engine** — architecture support + the Pascal/Volta kernel fixes above. Loads and runs
  **any** GGUF a stock llama.cpp/ik_llama.cpp build reads. You are not locked into PXQ to get
  the engine fixes.
- **The PXQ codec** — the quantizer (`llama-quantize … PXQ4/PXQ3/PXQ2/PXQ_UNIVERSAL`) and its
  custom GGUF types, layered on top of the engine.
- **Lock-in, stated plainly:** PXQ tensors are a CUDA-only slab layout with no CPU codec and no
  `PXQ → Q4_K` converter today (`llama-quantize` refuses to requantize a PXQ source —
  `src/llama-quantize.cpp`). Re-quantize from the original F32/BF16 source if you need a
  different format. A converter is on the roadmap; until then, if you only want the engine,
  run a stock GGUF and skip PXQ.

## Fair-battle protocol

Every head-to-head in this repo is one of three shapes — full methodology, raw runs, and every
number below: [`bench/fair-battle.md`](bench/fair-battle.md).

1. **Engine-only** — same GGUF, two engines (upstream vs pxq_llama). Isolates the kernel/arch
   fixes.
2. **Codec-only** — same engine, PXQ4 vs MXFP4 at matched bytes. Isolates the codec.
3. **Product** — best documented recipe per side (own quant, own levers). What you'd actually run.

**Engine-only, the honest number:** the real kernel/scheduler win is **prefill, roughly 1.7×** at
fixed weights — P100 **+59%** in an interactive `-fa on` server, **+88%** in a `-fa off` batch
pass; V100 +12–13%. Same-quant decode is a near no-op: **+2.7–3.3%**, V100 bit-identical output.
(A cold-prefill loss on the 1080 Ti is on the chart too, not hidden.)

<p align="center"><img src="bench/fair-battle.svg" alt="pxq_llama vs upstream ik_llama.cpp benchmark" width="100%"></p>

## Codec-only: PXQ4 vs MXFP4, including the cell we lose

Re-run 2026-09-02 on the candidate engine (the "next engine build" below), same cards, same
protocol as the original table (`llama-server /completion`, temp 0, n=7 median, MTP off both
sides). Full tables, the artifact census, and the raw reps: `bench/fair-battle.md`.

| cell | PXQ4 | MXFP4 | result |
|---|---|---|---|
| **Dense prefill** @3k / @20k, 2×P100 | 227.4 / 203.3 | 178.7 / 163.7 | **+27% / +24%** |
| **Dense decode**, 2×P100 | 18.3 | 14.4 | **+27%** |
| **Dense prefill** @3k / @20k, 2×V100 | 798.1 / 577.0 | 364.0 / 312.9 | **+119% / +84%** |
| **Dense decode**, 2×V100 | 34.5 | 37.7 | **−8% — MXFP4 still wins here** |
| **MoE prefill** @3k / @20k, 2×V100 ‡ | 2,093.8 / 1,467.2 | 1,614.6 / 1,256.8 | **+30% / +17%** |
| **MoE decode**, 2×V100 ‡ | 218.4 | 188.0 | **+16%** |

**‡** the MoE row compares a merge (`PXA-Fusion4-35B`, PXQ4) against the stock 35B-A3B model
(MXFP4) — same architecture and size class, not byte-identical base weights. The PXQ4 artifact's
own codec census shows only the expert stacks are PXQ4 (120 of its tensors); everything else is
other types. Call it an expert-codec delta, not a whole-model one. This replaces the earlier
MoE-decode row this repo withdrew for the same underlying reason — this time the composition is
disclosed up front and the number reproduces.

**The Volta dense-decode loss is unchanged and still understood the same way:** MXFP4's block
layout maps onto DP4A with one scale fixup per 32 values; PXQ4's sub-scale hierarchy costs a
second fixup and a second cache line. At equal bit width against a kernel already near HBM peak,
that's a structural tie-or-lose, not a tuning gap. What you get for the loss on that one cell:
PXQ4 is byte-for-byte the same 4.25 bpw file but ~38% lower reconstruction error and **6.0% lower
perplexity** (6.9704 → 6.5527, paired, same bytes) — unchanged from the original measurement.

**Recipe, unchanged:** Pascal (P100) → **PXQ4**, on both axes. Volta, dense, decode-bound →
**MXFP4**. Volta MoE, or long-prompt/interactive workloads → **PXQ4** (prefill win, better
fidelity).

## The reproducible proof: a 35B MoE on one 16 GB P100

One used P100, one downloadable GGUF (PXQU-16, 14.0 GB), fully GPU-resident:
**~62 t/s decode, 827–843 t/s prefill** — reproduce with `bench/speed-bench.sh`, numbers and
protocol in [`bench/README.md`](bench/README.md). This is the on-ramp, not the ceiling — the
same engine and codec scale to multi-card MoE below.

## Named recipes (`docs/COOKBOOK.md`)

| recipe | hardware | tier | result |
|---|---|---|---|
| 35B MoE, single card | 1× P100 or V100 16 GB | PXQU-16 (q8_0 head) | ~62 t/s (P100) / ~101 t/s (V100) decode |
| 35B MoE, 4-bit flagship | 2× P100 or V100 | PXQ4 (18.7 GB, doesn't fit one 16 GB card) | 55.7 t/s decode |
| 35B MoE, budget card | 1× GTX 1080 Ti 11 GB | PXQ2 (+opt-in int8 prefill) | ~71 t/s decode, 709 t/s prefill |
| 35B MoE, 12 GB card | 1× 12 GB card | PXQU-12 | 58.4 t/s (P100) / 97.6 t/s (V100) decode |
| 4× P100 rig, hybrid MoE (next engine build) | 4× Tesla P100 16 GB | PXQ_UNIVERSAL 4-bit | ~485 t/s prefill @3k, ~19 t/s decode @86k fill |
| 2× V100 rig, 27B dense (vLLM sm_70 line) | 2× Tesla V100 16 GB | PXQ4 | 1,006 t/s prefill @3k, 298 t/s aggregate @16 streams |

Exact commands and expected numbers for each: [`docs/COOKBOOK.md`](docs/COOKBOOK.md).

## The 2026-09-01/09-02 speed campaign: two rigs, before → after

Twenty-four hours of measurement across both rigs this project runs day to day. Naming: the
**4× P100 rig** runs a Qwen3.8 Flash-Next-class hybrid MoE on this llama.cpp-based engine,
PXQ_UNIVERSAL 4-bit; the **2× V100 rig** runs a Qwen3.8-27B dense-hybrid model, PXQ4, on the
separate vLLM-based sm_70 serving line ([`docs/PXA-SM70-SERVING.md`](docs/PXA-SM70-SERVING.md)).
Full protocol, every raw rep, and everything that didn't pan out:
[`RELEASE-NOTES-2026-09-02.md`](RELEASE-NOTES-2026-09-02.md).

**4× P100 rig** — `llama-server /completion`, temp 0, `cache_prompt=false`, n=7 median (1 warmup
discarded), `-c 150016 -b 2048 -ub 2048 -wgt 8 -ts 5079,12612,12612,11897 -sm layer`:

| metric | before | after | change |
|---|---|---|---|
| decode, low fill | 26.1 t/s | 27.8 t/s | +7% |
| decode, 86,401-tok fill | 12.5 t/s | 19.4 t/s | **+55%** |
| prefill @3,121 | 476.8 t/s | 484.9 t/s | +2% |
| prefill @20,801 | 397.6 t/s | 406.8 t/s | +2% |

The deep-fill decode line is the single biggest number in the campaign: `PXA_FA_GQA_PACK=4` reads
each attention key/value once per query group instead of once per head, which only pays off once
the number of re-reads is large — flat at low fill, +40% decode on its own at 86k tokens, stacked
here with five bit-identical host-overhead cuts (below) for the rest of the gap.

**2× V100 rig** — vLLM `/completion`, `--tensor-parallel-size 2 --dtype float16`, n=7, interleaved
boots (protocol and the NCCL finding: [`docs/PXA-SM70-SERVING.md`](docs/PXA-SM70-SERVING.md)):

| metric | before | after | change |
|---|---|---|---|
| prefill @3k | 919 t/s | 1,006 t/s | +9% |
| prefill @20k | 880 t/s | 983 t/s | +12% |
| decode, single stream | 48.5 t/s | 50.3 t/s | +4% |
| decode, aggregate @8 streams | 129 t/s | 187 t/s | +45% |
| decode, aggregate @16 streams | 129 t/s | 298 t/s | **+131%** |

The aggregate line is mostly one kernel — a Volta tensor-core path for the PXQ4 decode GEMV at
batch sizes of 5 and up (`PXQ4_MMV_MMA=1`, kernel v12). The prefill line is mostly not a kernel at
all — see the NCCL finding below.

### Engine: what the next build adds

Not yet in a tagged release. The full ship list, every rejected lever, and the known limits:
[`RELEASE-NOTES-2026-09-02.md`](RELEASE-NOTES-2026-09-02.md). Headline items:

- **Pipelined prefill.** Two scheduler bugs — a full graph re-plan on every prompt chunk instead
  of once per request, and a host-side sync inside every batched MoE layer — were hiding the
  overlap a second CUDA stream should have bought. Fixed, byte-identical: **+21% prefill @20,801**
  at `-c 32768`. It doesn't fit at the rig's real `-c 150016` yet — the second stream's KV and
  mask buffers need roughly 1.5–2 GB more VRAM per card than is free there today.
- **GQA-packed attention**, above: +40% decode at 86k-token fill, flat at low fill,
  output-identical.
- **Host-overhead cuts** — bounded top-k sampling off raw logits, a struct-of-arrays KV-sequence
  mask, a trimmed KQ-mask upload, and four more bit-identical micro-fixes: per-token host time at
  deep fill drops from 6.0 ms to 1.6 ms.
- **Device-side MoE row map.** The expert-routing table used to round-trip through the host once
  per batched MoE layer; it's built on-device now, self-checked bit-identical against the old
  host path.
- **PXQ on CPU.** An AVX2 int8 dot product makes CPU-only and partial-offload PXQ inference real
  instead of a technically-working fallback: **7.7× prefill** on a 12-thread Xeon E5-2699 v3
  (14.3 → 110.6 t/s at 128 tokens, cross-checked to 0 ULP against the CUDA decode).
- **An export tool.** `llama-pxq-export` turns a PXQ GGUF back into plain F16/F32, and
  `llama-quantize --allow-requantize` now accepts a PXQ source — the lock-in objection in
  "Two products, not one" above has an answer: quantize to PXQ, decide later you want something
  else, export and requantize instead of redoing the original conversion.
- **Correctness fixes.** A get_rows grid-overflow bug at large expert counts, an MMQ fusion-chain
  guard for non-MMQ quant types, and a quantized-cpy launch-config fix, all three ported from
  ik_llama.cpp upstream (`docs/DELTA-SINCE-IK.md`). Plus one found here: a per-slot attention
  state window on the hybrid architecture that was never reset between requests, so a second
  request could inherit the first's window.

### Already shipped

Landed in the launcher on 2026-09-01, no rebuild required: `-ub 2048 -wgt 8` on the P100 rig
(zeroes a vocab-sized logits reservation so the larger micro-batch fits, +5–8% prefill); on the
V100 rig, `--max-num-batched-tokens 4096` (+2.5–3.3% prefill, six interleaved boots, no decode or
KV-pool cost) and `--gpu-memory-utilization 0.92` (+44% KV pool, capacity only).

## One switch

```
PXA_ENHANCE=1
```

That's the only env var you need. It auto-selects the measured-good kernel levers per device
(mixed-card boxes get a per-GPU decision, printed at startup so it's auditable, not inferred).
Optionally: `PXA_MODE=balance` (default, fa-on serving) or `PXA_MODE=max` (fa-off, max prefill —
not for GLM/MLA models).

Everything else — every other `PXA_*` var — is the lab: experiment records and measured *losses*
kept for the paper trail, gated per-architecture, and usually **slower** if you set them by hand.
The full reference, including which knobs are dead ends: [`docs/lab/LEVERS.md`](docs/lab/LEVERS.md).

## Origin

**Do they rebase? No.** This is the Pascal/Volta engine, not a fork that tracks `ik_llama.cpp`
main: ikawrakow/ik_llama.cpp @ `1520eda98056` is treated as a parts bin — model graphs and bug
fixes are cherry-picked from it on a case-by-case basis, nothing more.

pxq_llama started as a fork of **ikawrakow/ik_llama.cpp @ `1520eda98056`** (2026-06-04). Since
then, measured against that pinned commit: 20 of 340 `ggml-cuda` files are new PXA kernels and 34
more are modified — the MoE GEMM, MMVQ dispatch, flash-attention regime, and dequant paths; 286
are still byte-identical to upstream. Of the 63 per-architecture graph builders, 7 are new
architectures and 4 more are modified; 52 are untouched. `src/llama-quantize.cpp` grew from
1,785 to 2,791 lines, mostly PXQ. The shared ik/llama.cpp lineage carries the rest of the tree —
tokenizer, GGUF I/O, sampling, and every architecture this project didn't need to touch. The
original work is concentrated exactly where cards with no DP4A and no tensor cores need it: the
MoE and PXQ-codec hot paths. (The upstream base commit is in the history now — `git merge-base`
resolves to it, and `git log --oneline 1520eda98056..HEAD` lists this project's own 501 commits
on top of it — diff against that exact commit if you want to see it yourself.)

## Build (CUDA)

**Full instructions, with every trap and its exact error: [`BUILD-FROM-SOURCE.md`](BUILD-FROM-SOURCE.md).**
The short version, on a machine with the cards in it:

```bash
git clone https://github.com/poisonxa16/pxq_llama && cd pxq_llama

docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$PWD":/src -w /src nvidia/cuda:12.8.1-devel-ubuntu24.04 bash
# (--gpus all is the modern equivalent; it needs the container toolkit out of legacy mode)

# --- inside the container ---
apt-get update && apt-get install -y --no-install-recommends cmake git

cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;70"
cmake --build build --target llama-server llama-cli llama-bench llama-quantize -j"$(nproc)"
```

`"60;70"` is P100 + V100. Use `"60;61;70;86;89"` for the wide list (adds 10-series,
3090/4090-class); `"60"` alone if all you have is a P100. Two build traps that cost people real
time — building on a GPU-less CI host, and leaving the CUDA stub on `LD_LIBRARY_PATH` at runtime
— are documented with their exact errors in `BUILD-FROM-SOURCE.md` §3.

## Quantize your own

```bash
# pure tier:
./build/bin/llama-quantize --imatrix your.imatrix model-bf16.gguf out-PXQ3.gguf PXQ3

# PXQU (mixed tier, sized to fit one card):
./build/bin/llama-quantize --imatrix your.imatrix --pxq-universal my-16gb.tiers model-bf16.gguf out-PXQU-16.gguf PXQ_UNIVERSAL
```

**PXQ models must be FULLY GPU-resident** — the CPU MoE op has no PXQ support, so partial offload
(`-ngl < 99` over PXQ expert layers, or `--n-cpu-moe`) aborts. Tier by VRAM: 16 GB → PXQU-16 or
PXQ3; 12 GB → PXQU-12; 11 GB (1080 Ti) → PXQ2. Recommended: add `--output-tensor-type q8_0`
(+123 MB, +5.2% P100 decode). Quantizing a merged model: recompute the imatrix **on the merge** —
imatrix rows are activation statistics of the anchor model, not the weights, so a parent model's
imatrix is off-distribution exactly on the tensors a merge changed. Full detail, the PXQU tier-map
format, and known traps: [`docs/PXQU-CONVERT.md`](docs/PXQU-CONVERT.md), [`docs/QUANTIZING.md`](docs/QUANTIZING.md), [`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md).

## Changelog

Per-release notes: `RELEASE-NOTES-*.md` in the repo root and `docs/`. Latest:
[`docs/RELEASE-NOTES-2026.08.28-rc3.md`](docs/RELEASE-NOTES-2026.08.28-rc3.md).

## License & credits
**MIT** — this fork inherits the MIT license of its base engines
([ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) / llama.cpp / ggml, © the ggml/llama.cpp/
ik_llama.cpp authors), and the PXQ types + E16-row-scale kernels are contributed under the same MIT terms.
The original LICENSE and AUTHORS are retained unchanged. PXQ quantization and the fused kernels are original
work of the PXA project, built on ikawrakow's ik_llama.cpp.

> Note: the **model weights** published on HuggingFace are a *separate* work under **Apache-2.0** (Qwen3.6
> lineage via Ornith-1.0-35B-AEON / SIQ-1-35B) — see the model card. This repo (code) is MIT; the weights are Apache-2.0.

## Community bug-finders 🏅

Real-hardware testing by the community makes this fork honest. Credits:

- **Last-Guitar-5924** (r/LocalLLM) — found the deepseek2/MLA fa-off context-decay cliff on a Tesla P40 (GLM-4.7-Flash decode collapsing 37 → 3.3 t/s by 36k ctx with flash attention off). His decode curve drove the automatic fa+mla posture for MLA models and the load-time warning shipping in the next release.
- **[bradrlaw](https://github.com/bradrlaw)** — via a rigorous independent benchmark, root-caused the dual-GPU decode collapse to `-sm layer` on a no-NVLink (PHB) topology and showed `-sm graph -ts 1,1` restores full decode; also caught the missing `libnccl.so.2` in the release packaging. Both drove fixes in this release.
