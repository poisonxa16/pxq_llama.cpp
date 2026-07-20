# Known issues

## `--pxq-universal` quantize: harmless CUDA-driver noise + argument order

**Two things trip people up when building a PXQU (`--pxq-universal`) quant. Neither is a real bug.**

**1. Harmless container noise.** Running `llama-quantize` inside an `nvidia/cuda` container prints,
before any real output:

```
ERROR: driverInitFileInfo 578 result=11
ERROR: init 664 result=11
ERROR: init 250 result=11
```

This is emitted by the **NVIDIA container runtime's own driver probe**, not by `llama-quantize` —
the string does not exist anywhere in our source or binary. Quantization proceeds and completes
normally. Ignore it. (It also appears on a plain `--help`, and on any other quant type; it is not
specific to `--pxq-universal`.)

**2. `--pxq-universal` is a flag — put it before the positional filenames.** `llama-quantize`
parses option flags only until the first positional argument. If `--pxq-universal 16g` comes
*after* the input/output paths, `--pxq-universal` lands in the positional `type` slot and you get:

```
main: invalid ftype '--pxq-universal'
```

Correct order (flag first, then `in out PXQ_UNIVERSAL`):

```bash
llama-quantize --imatrix model.imatrix --pxq-universal 16g \
  model-bf16.gguf model-PXQU16.gguf PXQ_UNIVERSAL
```

Presets: `12g`, `16g`, `16g-hq`, or a path to a `.tiers` map. See `docs/COOKBOOK.md`.

## llama-imatrix crashes on CPU / partial-offload configs (pre-existing; fix pending)

**Symptom:** an imatrix capture run (`llama-imatrix`, or any `cb_eval`-based activation capture)
on a configuration that keeps some expert tensors on CPU — `-ngl < 99`, `--n-cpu-moe N`, or a pure
CPU run — hangs in a futex wait or dies with SIGSEGV / malloc corruption partway through the pass.
A control run *without* the capture callback on the same config is fine, and the same capture run
**fully GPU-resident (`-ngl 99`, all experts on GPU) works correctly**.

- **Scope:** the activation-collection path in the CPU backend for MoE expert tensors; inherited,
  not introduced by the PXQ kernels (reproduces with the PXQ envs unset). Plain generation and
  perplexity on partial-offload configs are unaffected.
- **Workaround:** run imatrix captures full-GPU-resident. For models too big for your card(s) at
  full residency, capture on a smaller-tier quant of the same model (importance statistics are
  approximately quant-independent) or on a multi-GPU `-sm layer` split — just keep `-ngl 99`.
- **Status:** upstream fix pending; tracked here so nobody burns a day rediscovering it.

## PXQ models require full GPU residency (no CPU / partial offload)

The CPU backend's fused MoE op (`MOE_FUSED_UP_GATE`) has **no PXQ support**: a PXQ-quantized
model with expert layers left on the CPU — `-ngl < 99`, or `--n-cpu-moe N` — **aborts** at the
first expert op. This is a standing property of the format (PXQ has no CPU codec at all; the
slab layout is CUDA-consumer-only), not a regression.

- **Consequence:** pick the tier that fits your card entirely (see the README tier table);
  multi-GPU `-sm layer` splits are fine (everything stays on GPUs).
- **Also affects imatrix capture** (see the entry above — capture must be full-GPU-resident).

## PXQ tensors and mainline gguf-py

Not a bug in this fork, but a standing trap: no gguf-py size table (mainline's or this fork's) can
express the E16-row per-row anchor, so a gguf-py read-modify-write **silently truncates** PXQ
tensors. Re-run `llama-quantize` from the bf16/f16 source instead. (See README.)
