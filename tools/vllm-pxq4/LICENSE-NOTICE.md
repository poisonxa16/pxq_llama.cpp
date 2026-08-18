# Attribution

This package is licensed Apache 2.0, matching vLLM.

## vLLM
Copyright the vLLM contributors. Apache License 2.0.
This package patches no vLLM source; it registers through the documented
`register_quantization_config` plugin hook.

## 1Cat-vLLM (Volta / sm_70 support)
https://github.com/KewaiiGamer/1Cat-vLLM

Upstream vLLM does not target compute capability 7.0. That fork carries the
sm_70 work this package depends on entirely:
  - TurboMind sm_70 W4A16 GEMM (csrc/sm70_turbomind/)
  - the FLASH_ATTN_V100 attention backend
  - the Qwen Gated-DeltaNet kernels (FlashQLA-SM70)
PXA Network contributed to getting that V100 support working. This package is a
continuation of that effort, not a fork of it.

## PXQ4
The PXQ quantization format, its CUDA kernels, and this backend: PXA Network.
