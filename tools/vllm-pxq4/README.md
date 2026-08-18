# vllm-pxq4 — PXQ4 quantization backend for vLLM on Volta (sm_70)

Serve PXQ-quantized models on vLLM, with tensor parallelism, CUDA-graph capture and
paged KV — on V100-class hardware.

This ships **alongside** the `pxq_llama` engine, not inside it. Two runtimes, one
quant family:

| runtime | hardware | why |
|---|---|---|
| `pxq_llama` (this repo) | sm_60 Pascal, sm_61, sm_70 Volta, newer | GGUF-native, runs everywhere, the universal engine |
| `vllm-pxq4` (this package) | **sm_70 only** | tensor parallelism + CUDA graphs, which llama.cpp does not have on Volta |

**Pascal cannot run this.** vLLM's compiled kernels target compute capability 7.0 and up
(`CUDA_SUPPORTED_ARCHS = "7.0;7.5;8.0;..."`). P100 (sm_60) and 1080 Ti (sm_61) are out.
For those cards use `pxq_llama`, which is the reason it stays the primary engine.

## Standing on other people's work

This package exists because two pieces of work happened first, and it would be dishonest
to present it without them:

- **vLLM** (Apache 2.0) — the serving engine, tensor parallelism, paged attention,
  continuous batching, CUDA-graph capture. We patch **zero lines** of it; this plugs in
  through the documented `register_quantization_config` hook.
- **[KewaiiGamer/1Cat-vLLM](https://github.com/KewaiiGamer/1Cat-vLLM)** — the Volta port.
  Upstream vLLM dropped sm_70; that fork carries the TurboMind sm_70 W4A16 GEMM, the
  `FLASH_ATTN_V100` attention backend, and the Qwen Gated-DeltaNet kernels
  (`FlashQLA-SM70`) without which none of this runs on a V100. **PXA Network contributed
  to getting that V100 support working.** This package is the continuation of that effort,
  not a fork of it.
- **PXQ4** — the quantization format, its CUDA kernels, and this backend: PXA Network.

If you only want faster inference on Volta and do not need PXQ, use 1Cat-vLLM directly.
This package is for people who have PXQ artifacts.

## Honest performance

Do not read a headline number off this without reading the paragraph under it.

Measured on 4x V100-SXM2-32GB (TP=4), Qwen3.8-27B:

| | decode tok/s |
|---|---|
| `pxq_llama`, `-sm layer` + `ngram_mod` | 47.96 prose / 63.76 code (**measured**) |
| vLLM + AWQ W4A16 (incumbent) | 92.8 peak / 57.4 median (**measured**) |
| vLLM + PXQ4, this package | **projected ~+9% over AWQ** — *not yet measured* |

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

Pre-release. The implementation is complete and reviewed; it has **not** been executed
against real weights on a GPU. Nothing here is validated hardware-in-the-loop yet.
Ship gates before anyone should trust it:

1. bit-exact dequant parity against a CPU reference
2. single-linear-layer GEMM parity
3. **sharded parity** — per-rank slices must dequantize to exactly the unsharded result
4. logprob parity vs `pxq_llama` on the same prompts at temp 0
5. an end-to-end throughput measurement to replace the projection above

## Licence

Apache 2.0, matching vLLM. See `LICENSE-NOTICE.md` for the full attribution chain.
