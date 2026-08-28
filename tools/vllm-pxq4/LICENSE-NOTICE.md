# Attribution

This package is licensed Apache 2.0, matching vLLM.

## vLLM
Copyright the vLLM contributors. Apache License 2.0.
This package patches no vLLM source; it registers through the documented
`register_quantization_config` plugin hook.

## PyTorch — one optional patch, and it is not vLLM

Be precise about the claim above: we patch no *vLLM* line, but `tools/patch_sm60_compile.py`
does edit **PyTorch** (`torch/utils/_triton.py`) to lift a hard-coded `device major >= 7` gate
in `has_triton()`. That gate is not a real capability limit — Triton 3.3 compiles and runs
ordinary pointwise/reduction kernels on sm_60; only `tl.dot` (MMA) is genuinely unavailable,
and inductor does not lower matmuls to Triton unless max_autotune is on.

It is OPT-IN and only needed to get torch.compile / CUDA graphs working on Pascal (P100).
Nothing on sm_70 or newer requires it, and PXQ4 itself does not depend on it.

## Pascal (sm_60) support — PXA Network

Upstream vLLM does not target compute capability 6.0, and neither did the sm_70
fork below. The Pascal port is PXA Network's own work: the P100 support, the
Pascal SDPA path, the Triton paged decode-attention backend and the
`csrc/pascal_compat` shims.

## Volta (sm_70) support — joint work

Upstream vLLM does not target compute capability 7.0 either. The Volta support
this package runs on was developed jointly by PXA Network and the **1Cat-vLLM**
project (https://github.com/KewaiiGamer/1Cat-vLLM), Apache License 2.0:
  - TurboMind sm_70 W4A16 GEMM (csrc/sm70_turbomind/)
  - the FLASH_ATTN_V100 attention backend
  - the Qwen Gated-DeltaNet kernels (FlashQLA-SM70)

This package is a continuation of that shared effort, not a fork of it.

### Container images

The published container images (`ghcr.io/poisonxa16/pxa-vllm`) contain vLLM and the
jointly-developed sm_70 support above, both Apache-2.0, built from source and
combined with PXA Network's Pascal port and PXQ4 backend. Per Apache-2.0 section 4
that attribution travels with the binaries, which is what this file is for.

Changes made relative to those upstreams, stated as section 4(b) requires:
  - the distribution is renamed `pxa-vllm`, and internal environment variables are
    named `VLLM_PXA_*`
  - the PXQ4 quantization backend and its CUDA kernels are added
  - build trees and development scripts are removed from the shipped image

No warranty is offered by any upstream for these modifications; see the licence.

## PXQ4
The PXQ quantization format, its CUDA kernels, and this backend: PXA Network.
