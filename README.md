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

The pitch in one sentence: **run real models fully in VRAM on a $150 Tesla P100** — kernels
written for a chip with no DP4A and no tensor cores, not ported from one that has them. Volta
(V100), the 1080 Ti, and multi-card 122B-class MoE spreads all run on the same engine and are
covered below — they're scaling proof, not the pitch.

## Not MoE-only

The engine loads **83 model architectures** (`src/llama-arch.cpp`) — dense (Qwen, Llama, Gemma,
Mistral, …), GDN hybrids (`qwen3next`), MoE, `gpt-oss`, DeepSeek-V4 (`deepseek4`), GLM
(`glm4moe`, `glm-dsa`), MiniMax (`minimax-m2`), Cohere (`cohere2`, `cohere2_moe`), Laguna,
`gemma4`. The engine-level fixes below (sm_60 fp16-GEMM path, flash-attention regime routing,
the MoE path, `np>1` hybrid concurrency, wide-K f16 GEMV) help **any** quant on these cards, not
just PXQ — a stock Q4_K, MXFP4, or IQ_K GGUF benefits too.

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

Same engine, same cards, `llama-server /completion`, temp 0, n=7 median.

| cell | PXQ4 | MXFP4 | result |
|---|---|---|---|
| **Dense prefill**, 2×P100 | 128.0 | 107.4 | **+19.2%** |
| **Dense decode**, 2×P100 | 15.18 | 14.32 | **+6.0%** |
| **Dense decode**, 2×V100 (MMVQ-armed) | 33.82 | 36.38 | **−7.0% — MXFP4 wins here** |
| **MoE prefill**, 2×V100 ‡ | 1394.0 | 1172.8 | **+18.9%** |

**‡** that MoE row and its paired decode figure trace to one artifact that turned out to be only
27% PXQ4 by tensor count; the decode figure did not reproduce and is **withdrawn**, and the
prefill figure is relabelled as an expert-codec delta rather than a whole-model one. Full
accounting: `docs/LEVERS.md` §0a.

**The Volta dense-decode loss is real and understood:** MXFP4's block layout maps onto DP4A with
one scale fixup per 32 values; PXQ4's sub-scale hierarchy costs a second fixup and a second cache
line. At equal bit width against a kernel already near HBM peak, that's a structural tie-or-lose,
not a tuning gap — eight separate kernel attempts have not closed it. What you get for the 7%:
PXQ4 is byte-for-byte the same 4.25 bpw file but ~38% lower reconstruction error and **6.0% lower
perplexity** (6.9704 → 6.5527, paired, same bytes).

**Recipe:** Pascal (P100) → **PXQ4**, on both axes. Volta, dense, decode-bound → **MXFP4**. Volta
MoE, or long-prompt/interactive workloads → **PXQ4** (prefill win, better fidelity).

## The reproducible proof: a 35B MoE on one 16 GB P100

One $150 card, one downloadable GGUF (PXQU-16, 14.0 GB), fully GPU-resident:
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

Exact commands and expected numbers for each: [`docs/COOKBOOK.md`](docs/COOKBOOK.md).

## Scales up: 4×P100, a 122B-class hybrid MoE

Measured 2026-09-01 on the 2026.08.31 engine — a Qwen3.8-80B-A3B-class GDN-hybrid MoE, PXQU
quantized, on 4× Tesla P100-PCIE-16GB. Protocol: `llama-server /completion`, temp 0,
`cache_prompt=false`, n=7 median (1 warmup discarded), `-c 150016 -b 2048 -ub 2048 -wgt 8
-ts 5079,12612,12612,11897 -sm layer`.

| context fill | prefill | decode |
|---|---|---|
| 3,121 tok | 475–490 tok/s | 27.5 tok/s |
| 20,801 tok | ~400 tok/s | — |
| 86,401 tok | 229.6 tok/s | 13.2 tok/s |

This is not yet a named cookbook recipe — the harness that automates this exact sweep ships in
the next release. Today, reproduce the protocol above by hand against your own PXQU map
(`docs/PXQU-CONVERT.md`).

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
The full reference, including which knobs are dead ends: [`docs/LEVERS.md`](docs/LEVERS.md).

## Origin

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
