<p align="center"><img src="banner.png" alt="pxq_llama — the PXQ quantization family and a MoE accelerator for Pascal and Volta GPUs" width="100%"></p>

# pxq_llama

**A llama.cpp fork by [PXA Network](https://pxanetwork.com), built around the PXQ quantization
family and CUDA kernels tuned for older datacentre cards.**

Three pieces ship in this one tree:

- **the engine** — a fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
  (itself a fork of [llama.cpp](https://github.com/ggerganov/llama.cpp) / ggml), carrying the
  **PXQ** quantization family and fused CUDA kernels written for Pascal (sm_60) and Volta
  (sm_70) silicon;
- **a matching vLLM backend** — [`tools/vllm-pxq4/`](tools/vllm-pxq4), which serves the same
  PXQ4 artifacts through vLLM, for the tensor parallelism, CUDA-graph capture and real data
  parallelism that `-sm layer` does not give you;
- **one launcher over both** — [`tools/pxa-launch.py`](tools/pxa-launch.py), documented in
  [`docs/LAUNCHER.md`](docs/LAUNCHER.md).

Community: **[Discord — PXA Network](https://discord.gg/BHWmMHHStY)**.
Weights: **[huggingface.co/poisonxa](https://huggingface.co/poisonxa)**.

---

## You tell it your cards. It picks the engine and the parameters.

```bash
python3 tools/pxa-launch.py --model <model.gguf> --cards 0,1 --explain
```

`--explain` decides and prints; drop it to run. The launcher reads the cards, reads the
model — architecture, dense or MoE, which PXQ tier is actually inside the file, whether the
MTP tensors it claims are really there — and then picks the engine, the tensor split, the
micro-batch, the CPU offload and the speculative-decoding posture.

It is deliberately **not magic**:

- it prints the decision, the evidence behind it, and the exact command line before running;
- it **refuses** rather than silently dropping a parameter that does not translate between
  the two engines;
- every claim it makes carries one of exactly three tags — **MEASURED** (a number from a
  correctness-gated boot, with the id of the bench row that produced it), **[INFERRED]** (a
  branch taken from an adjacent measurement, never a new number), or **UNMEASURED** (nothing
  was measured — it says the word and stops or asks).

That last part is the point. The engine choice is not a rule of thumb; it is a table. On a
2× P100 pair with a dense 27B PXQ4, the vLLM path wins everything measured (24.0 vs 13.7
tok/s single-stream). On a 35B MoE on the same cards, llama.cpp wins by 3.1× at one
concurrent request, peaks at five, and **loses** from six upward — because `-sm layer` is a
serialized two-GPU pipeline, not data parallelism, so concurrent requests queue behind it
while vLLM's aggregate keeps climbing. The launcher knows where that crossover is because
somebody booted it eleven times and gated each boot on correctness first.

`--selftest` exercises the decision table against the machine you are on
(`--model` is required by the parser even here, and ignored:
`python3 tools/pxa-launch.py --selftest --model /dev/null`).

---

## Quick start

### Serve a PXQ4 model — prebuilt, nothing to compile

Pascal and Volta images with the PXQ4 backend already inside. Pick the tag for your
cards; the container works out which kernel to load from the GPUs it can see.

```bash
docker pull ghcr.io/poisonxa16/pxa-vllm:sm60   # Pascal: P100, GTX 10xx
docker pull ghcr.io/poisonxa16/pxa-vllm:sm70   # Volta:  V100, Titan V

docker run --rm --runtime=nvidia --gpus '"device=0,1"' \
  -v /path/to/models:/models -p 8000:8000 \
  ghcr.io/poisonxa16/pxa-vllm:sm60 \
    --model /models/your-pxq4-model --quantization pxq4 \
    --tensor-parallel-size 2 --host 0.0.0.0 --port 8000
```

An OpenAI-compatible server on `:8000`. Details, environment variables and the kernel
selection table: [`docker/vllm-pxq4/README.md`](docker/vllm-pxq4/README.md).

For prebuilt engine binaries and Docker build recipes, see
[`docker/README.md`](docker/README.md).

### Quantize your own model

Both engines are fed from a GGUF. Full path for each, including Flash-Next:
[`docs/QUANTIZING.md`](docs/QUANTIZING.md).

### Build from source


**Build** (step by step from scratch in [`docs/QUICKSTART.md`](docs/QUICKSTART.md); every
trap and its exact error in [`BUILD-FROM-SOURCE.md`](BUILD-FROM-SOURCE.md)):

```bash
git clone https://github.com/poisonxa16/pxq_llama && cd pxq_llama

docker run --rm -it --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -v "$PWD":/src -w /src nvidia/cuda:12.8.1-devel-ubuntu24.04 bash

# --- inside the container: the stock CUDA image has nvcc but not cmake or git ---
apt-get update && apt-get install -y --no-install-recommends cmake git

cmake -B build -S . -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="60;70"
cmake --build build --target llama-server llama-cli llama-bench llama-quantize -j"$(nproc)"
```

`"60;70"` is P100 + V100. Use `"60;61;70;86;89"` for the wide list (adds 10-series,
Ampere, Ada). There is no Makefile in this tree — CMake only. Prebuilt Linux x86-64 CUDA 12
binaries are attached to each [release](https://github.com/poisonxa16/pxq_llama/releases),
and container images are described in [`docker/README.md`](docker/README.md).

**Serve:**

```bash
PXA_ENHANCE=1 PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1 \
LD_LIBRARY_PATH=build/bin:build/src:build/ggml/src \
./build/bin/llama-server -m your-model-PXQ4.gguf \
  -c 8192 -ngl 99 -sm layer --jinja --host 0.0.0.0 --port 8080
```

`PXA_ENHANCE=1` selects the measured-good kernel levers per card — a mixed-card box gets
per-GPU decisions — and prints the ledger of what it chose and why. The three
`PXA_PXQ6/2/3` switches enable the PXQ format families; set all three for a mixed-tier
model. That is the whole configuration. Everything else is in
[`docs/LEVERS.md`](docs/LEVERS.md).

**Quantize your own:**

```bash
./build/bin/llama-quantize --imatrix your.imatrix model-bf16.gguf out-PXQ3.gguf PXQ3
```

**Or let the launcher do all of it:** `python3 tools/pxa-launch.py --model out-PXQ3.gguf --cards 0,1`

---

## What PXQ is

PXQ quantizes the weights that dominate a MoE — the expert tensors — with a learned codebook
plus **E16-row scales**: a per-row fp16 anchor, amortized to 2 bytes per row over a 64-row
panel, with a 4-bit sub-scale for every 16-element block. The backbone (attention, router,
embeddings) is assigned per tensor class by a separate allocation table rather than flattened
to one type. On top of that sit fused CUDA kernels — grouped-MoE GEMM, K-split decode,
gate/up fusion, DeltaNet decode fusion — written for Pascal and Volta.

| tier | bits | what it is |
|---|---|---|
| **PXQ6** | ~5.27 bpw | 5-bit LM32 × E16-row quality tier |
| **PXQ4** | 4.27 bpw | the 4-bit flagship (PX16 book + E16-row scales); **PXQ4-HQ** adds bs8 sub-scales |
| **PXQ3** | 3.27 bpw | 3-bit, bit-plane packed |
| **PXQ2** | 2.27 bpw | 2-bit, LM4 codebook |
| **PXQ1** | 1.26 bpw | 1-bit sign codes over the same E16-row scales. A stretch tier for mixes, not a whole-model quant |
| **PXQ_UNIVERSAL** | you choose | a per-tensor mix of the tiers above, sized to a specific card |

`PXQ_UNIVERSAL` is the one worth explaining. Instead of picking a bit-width, you hand
`llama-quantize` a tier map — one `regex=type` line per tensor — and each tensor gets the tier
its line names. The published PXQU builds come from a Lagrangian-relaxation knapsack over
measured per-tensor sensitivity: give every expert the cheapest tier such that the total lands
on your VRAM budget with minimum weighted error. The map format is deliberately trivial so you
can hand-author or script one for your own tensor names and your own card
([`docs/PXQU-CONVERT.md`](docs/PXQU-CONVERT.md)).

Measured, on wikitext-2-raw test at the standard llama.cpp perplexity protocol
(`-c 512 --chunks 200`), same corpus and same imatrix for every tier — only the quant type
varies:

| tier | file | perplexity | Δ vs PXQ4 |
|---|---|---|---|
| PXQ4 | 18.7 GB | **7.3563 ± 0.0818** | — |
| PXQ3 | 14.7 GB | **7.4407 ± 0.0830** | +1.1% |
| PXQ2 | 10.7 GB | **8.3906 ± 0.0961** | +14.1% |

The ladder is monotonic. Everything behind those numbers — the exact commands, the checksums
of the files they were run on, the KL-divergence procedure — is in
[`bench/`](bench/README.md).

⚠ **PXQ is a PXA-native format.** Mainline llama.cpp cannot read these GGUFs; build this fork.
And do not read-then-rewrite PXQ tensors with `gguf-py` — no gguf-py size table, mainline's or
this fork's, can express the E16-row per-row anchor, so a read-modify-write silently truncates
them. Re-run `llama-quantize` from the bf16 source instead.

⚠ **PXQ models must be fully GPU-resident.** The CPU fused-MoE op has no PXQ codec at all, so
`-ngl < 99` with PXQ expert layers left on CPU aborts. Pick the tier that fits your card
entirely. Multi-GPU `-sm layer` splits are fine — everything stays on GPUs. For a
partial-offload run, use a standard quant on this engine; the engine-side wins are
format-agnostic.

---

## What else is different

**Hybrid attention and MoE are first-class, not an afterthought.** The tree carries graph
builders and kernels for MoE + linear/recurrent hybrids (Gated-DeltaNet class), MoE + full/SWA
interleave, MLA (`deepseek2`) with a compute-capability-aware flash-attention posture, and
per-layer-embedding architectures where a 51 GiB gather table lives in host RAM while the rest
of the model is GPU-resident. The server's `np>1` concurrency on hybrid-recurrent models is
correct here; the checkpoint/rollback path for sliding-window and recurrent state is the part
that took the longest to get right.

**Tuning is measured, then baked in — not left as advice.** The per-card lever selection
(`PXA_ENHANCE`), the micro-batch table, the flash-attention posture, the tensor split and the
engine choice are all resolved at startup from the device fleet and the loaded model, and the
decision is printed. Where a number exists it is quoted with the cell it was measured on;
where one does not, the code says **UNMEASURED** rather than guessing. Bit-exact kernel
levers ship on by default and each is `memcmp`-proven against the reference path
([`bench/determinism-gates.md`](bench/determinism-gates.md)); levers that change token output
are labelled as such so you can turn them off when you need reproducibility.

**Two runtimes, one artifact.** The same PXQ4 file serves through `llama-server` or through
vLLM. The vLLM backend registers via vLLM's documented `register_quantization_config` hook and
patches **zero lines** of vLLM. On the shipping configurations
([`scripts/pxa-serve-sm70.sh`](scripts/pxa-serve-sm70.sh),
[`scripts/pxa-serve-sm60.sh`](scripts/pxa-serve-sm60.sh)), with every arm correctness-gated
before its number was recorded:

| cards | single-stream decode | aggregate @8 |
|---|---|---|
| 2× Tesla V100-PCIE-16GB (sm_70) | 50.98–51.46 tok/s | 132–136 tok/s |
| 2× Tesla P100-PCIE-16GB (sm_60) | 24.0–26.4 tok/s | 70–72 tok/s |

Full sweeps, the recipes, and the traps that cost real time —
[`docs/PXA-SM70-SERVING.md`](docs/PXA-SM70-SERVING.md),
[`docs/PXA-SM60-SERVING.md`](docs/PXA-SM60-SERVING.md).

---

## Hardware

| architecture | cards | engine | vLLM backend |
|---|---|---|---|
| sm_60 Pascal | Tesla P100 / GP100 | yes | yes — needs its own image and torch 2.7.1; see the sm_60 serving doc |
| sm_61 Pascal | GTX 1080 Ti, P40 | yes | no |
| sm_70 Volta | Tesla V100 | yes | yes |
| sm_75 and newer | Turing, Ampere, Ada, … | yes, builds and runs normally | use upstream vLLM |

The fork exists for the first three rows. It builds and runs on modern cards, but nothing in
it is tuned for them and we publish no numbers there.

`./install.sh` reads the card with `nvidia-smi`, names the supported path, and refuses to
guess when it does not recognise the capability.

---

## Honest positioning

**This is a fork, and the base is not ours.** llama.cpp and ggml are the foundation;
ik_llama.cpp is the immediate upstream. This tree forked from `ikawrakow/ik_llama.cpp` at
commit `1520eda98056` (2026-06-04) and has been developed independently since. The repository
history is flattened, so there is **no git merge-base** with upstream — to diff or cherry-pick,
compare against that exact commit. Individual changes ported from upstream after the fork point
are marked inline in the source with the originating pull request number, so their provenance
survives without the history. Upstream's README is preserved at
[`docs/README-upstream-ik_llama.md`](docs/README-upstream-ik_llama.md).

**What is ours:** the PXQ quantization family and its CPU/CUDA codecs, the quantizer's
tier-selection policy and backbone allocation table, the `PXA_ENHANCE` / `PXA_MODE`
per-architecture acceleration system, the fused PXQ kernels, the vLLM PXQ4 backend, and the
launcher.

**What we do not claim.** We publish no speed comparison we cannot reproduce from
[`bench/`](bench/README.md), and where a published number failed to reproduce we withdrew it
rather than relabelling it — the withdrawn rows are still in `bench/` with the reason. All
published numbers come from one workstation of salvaged datacentre cards (Tesla V100 16 GB,
Tesla P100 16 GB, GTX 1080 Ti); match that configuration when comparing, or report your own.
Community numbers on other hardware are welcome and credited.

**And the loss.** On Volta, *dense-model decode* is about 7% slower on PXQ4 than on MXFP4 at
the same bit width. The cause is understood — MXFP4's block layout maps onto DP4A with a
single scale fixup per 32-value block, while PXQ4's sub-scale hierarchy costs a second fixup
chain and a second cache sector — and it has survived roughly eight distinct kernel-side
attacks, including one rewrite that we built, measured, and reverted because it came in
slightly worse. What you buy for those 7% is fidelity: at identical file size, PXQ4's
effective width is 4.25 bpw against MXFP4's 3.64, reconstruction error is 38% lower, and
paired perplexity is 6.5527 against 6.9704. Whether that trade is worth it is your call, which
is why the number is here.

---

## Documentation

| | |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | clone, build, first run — start here if this is your first time |
| [`BUILD-FROM-SOURCE.md`](BUILD-FROM-SOURCE.md) | the complete build path, and the exact error for every trap |
| [`docs/COOKBOOK.md`](docs/COOKBOOK.md) | copy-paste command lines per card, with the numbers they produce |
| [`docs/LAUNCHER.md`](docs/LAUNCHER.md) | the launcher in full — how it picks the engine, the split and the micro-batch, and what every refusal means |
| [`docs/LEVERS.md`](docs/LEVERS.md) | the supported `PXA_*` levers, their defaults and their measurements |
| [`docs/KNOWN-ISSUES.md`](docs/KNOWN-ISSUES.md) | standing traps and their workarounds |
| [`docs/PXQU-CONVERT.md`](docs/PXQU-CONVERT.md) | the `PXQ_UNIVERSAL` tier-map format |
| [`docs/RENAME-MAP.md`](docs/RENAME-MAP.md) | the PXQ tier re-ladder, and which old names still work |
| [`docs/VLLM.md`](docs/VLLM.md) | the PXQ4 vLLM backend: which tiers it serves, converting, serving, tuning, tool calling |
| [`docs/PXA-SM60-SERVING.md`](docs/PXA-SM60-SERVING.md) · [`docs/PXA-SM70-SERVING.md`](docs/PXA-SM70-SERVING.md) | the vLLM serving recipes, measured |
| [`docs/parameters.md`](docs/parameters.md) | the CLI surface |
| [`docs/build.md`](docs/build.md) · [`docs/docker.md`](docs/docker.md) · [`docs/install.md`](docs/install.md) | other build and install routes |
| [`docs/function-calling.md`](docs/function-calling.md) · [`docs/speculative.md`](docs/speculative.md) · [`docs/autoparser.md`](docs/autoparser.md) | server features |
| [`bench/README.md`](bench/README.md) | the reproduction pack for every published number |
| [`pxa-bench/README.md`](pxa-bench/README.md) | the codec gate harnesses and the `PXQ_UNIVERSAL` tier maps |
| [`tools/vllm-pxq4/README.md`](tools/vllm-pxq4/README.md) | the vLLM backend, its gates and its status |
| [`pxa/pxq4/README.md`](pxa/pxq4/README.md) | the prebuilt PXQ4 kernel libraries and how to rebuild them |

Release notes for each cycle are in [`docs/`](docs/), newest first — the current cycle is
[`docs/RELEASE-NOTES-2026.08.28-rc3.md`](docs/RELEASE-NOTES-2026.08.28-rc3.md).

---

## Licence and attribution

Two engines, two permissive licences, mapped in [`LICENSING.md`](LICENSING.md):

| path | what | licence |
|---|---|---|
| repository root | the `pxq_llama` inference engine | **MIT** — [`LICENSE`](LICENSE) |
| [`tools/vllm-pxq4/`](tools/vllm-pxq4) | the PXQ4 backend for vLLM | **Apache-2.0** — [`tools/vllm-pxq4/LICENSE-NOTICE.md`](tools/vllm-pxq4/LICENSE-NOTICE.md) |

The engine inherits the MIT licence of its base: llama.cpp and ggml (© the llama.cpp and ggml
authors) and ik_llama.cpp (© the ik_llama.cpp authors). Their copyright notices are preserved
in full in [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and [`AUTHORS`](AUTHORS), and the lineage
is set out in `NOTICE`. The PXQ types and the E16-row-scale kernels are contributed under the
same MIT terms.

The vLLM backend is Apache-2.0 to match vLLM, and its notice records the community sm_70 vLLM
work it depends on. Apache-2.0 section 4 requires that notice be carried into redistributions;
do not drop it.

> The **model weights** published on HuggingFace are a separate work under a separate licence —
> see the model card.

pxq_llama is developed and maintained by **PXA Network** — https://pxanetwork.com
