# Known issues

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
