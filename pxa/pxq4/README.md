# PXA PXQ4 Sidecar Package

The complete PXQ4 quantized-inference sidecar for vLLM, as built and shipped by
PXA Network. Everything needed to serve PXQ4 lives here; nothing in this
directory depends on a path outside the repository.

    kernels/       prebuilt PXQ4 kernel libraries, one per arch + revision.
                   Selected at runtime with the PXQ4_LIB environment variable.
    kernels-src/   the CUDA/C++/Python sources those libraries are built from.
    sidecar/       the pxq4_vllm plugin trees vLLM loads via PYTHONPATH.
                   site-union  - both arches, carries the fp16 mmv hook.
                   site-sm60   - Pascal-targeted variant.

## Why this package exists

Until 2026-08-27 none of this was under version control. The kernel libraries,
the plugin sites and the kernel sources lived only in scratch working
directories on a single machine, referenced by absolute path from launcher
scripts. Losing that machine would have meant losing the ability to rebuild the
kernels at all. This package is the fix: the artifacts, their sources and their
provenance in one branded, self-contained tree.

## Choosing a kernel

| library | arch | status |
|---|---|---|
| `libpxq4_sm70_v10.so` | sm_70 | **SHIPPED.** 51.46 tok/s single-stream, 2x V100. |
| `libpxq4_sm70_v9.so`  | sm_70 | fp16 hook build, superseded by v10. |
| `libpxq4_sm60_v10.so` | sm_60 | survival-gated final; fp16 smem tile kernel. |
| `libpxq4_sm60_v11.so` | sm_60 | adds `gemm2d_out` behind `PXQ4_GEMM2D`, **default off**: ~+34% prefill but it FAILED first-token quality at 87.5%. Do not enable without re-gating quality. |
| `libpxq4_sm60_v9.so`  | sm_60 | adds `f16_mmv_out`, ~+9.4% single-stream on P100. |
| `libpxq4_sm60_v8.so`  | sm_60 | adds expert-indexed MoE (`moe_mmv_out`). |

`PXQ4_LIB` must always be set explicitly. `sidecar/site-union` bundles an
sm_70-only `.so` built against the torch 2.10 ABI; if `PXQ4_LIB` is unset the
loader can reach that one on a Pascal image and die with
`undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv` part-way through
model load, rather than with a clear message.

## Identifying a kernel revision

The ops are registered through TORCH_LIBRARY string schemas, not exported C
symbols, so `nm` will not find them — `nm -D | grep linear_out` returns nothing
even for a library that has it. Scan printable strings instead:

    strings -a kernels/libpxq4_sm60_v10.so | grep -E '^(moe_mmv_out|f16_mmv_out|gemm2d_out)$'

    moe_mmv_out present -> v8 or later
    f16_mmv_out present -> v9 or later
    gemm2d_out  present -> v11

See `MANIFEST.md` for sizes and md5s of every library in this package.
