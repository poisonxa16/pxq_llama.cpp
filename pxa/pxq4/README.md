# PXA PXQ4 Sidecar Package

The prebuilt PXQ4 kernel libraries and the vLLM plugin tree that uses them, as
built and shipped by PXA Network. Nothing in this directory depends on a path
outside the repository.

    kernels/       prebuilt PXQ4 kernel libraries, one per GPU arch + revision.
                   Selected at runtime with the PXQ4_LIB environment variable.
    sidecar/       the pxq4_vllm plugin tree vLLM loads via PYTHONPATH.
                   site-sm60  - Pascal-targeted variant (torch 2.7 ABI),
                                carries the fp16 mmv and tiled-SDPA hooks.

The CUDA/C++ sources these libraries are built from are **not** duplicated here;
they live once, at `tools/vllm-pxq4/src/` (`pxq4_kernel.cu`, `pxq4_kernel.cuh`,
`pxq4_kernel_torch.cpp`, `pxq4_kernel_launch.h`, `pxq4_kernel_tables.h`).

`MANIFEST.md` lists every file in this package with its size and md5.

## What each library is

Each `.so` is an x86-64 Linux shared object: a standalone torch extension that
registers the `pxq4::*` operators through `TORCH_LIBRARY` and is loaded at
runtime with `torch.ops.load_library()`. It links against libtorch and
`cudart` only — it includes no vLLM header, links no vLLM object, and needs no
vLLM rebuild.

Each library carries exactly one CUDA device binary, for the architecture in its
filename, and no PTX. A library will not run on an architecture it was not built
for.

| library | arch | GPU class | torch ABI | ops added | status |
|---|---|---|---|---|---|
| `libpxq4_sm70_v10.so` | sm_70 | Tesla V100 | 2.10 | `f16_mmv_out` | **SHIPPED** for sm_70. 51.46 tok/s single-stream on 2x Tesla V100. |
| `libpxq4_sm70_v9.so`  | sm_70 | Tesla V100 | 2.10 | `f16_mmv_out` | fp16 hook build, superseded by v10. |
| `libpxq4_sm60_v10.so` | sm_60 | Tesla P100 | 2.7 | `f16_mmv_out` | **SHIPPED** for sm_60. Survival-gated final; fp16 smem tile kernel. |
| `libpxq4_sm60_v11.so` | sm_60 | Tesla P100 | 2.7 | `gemm2d_out` | Adds `gemm2d_out` behind `PXQ4_GEMM2D`, **default off**: ~+34% prefill but it FAILED first-token quality at 87.5%. Do not enable without re-gating quality. |
| `libpxq4_sm60_v9.so`  | sm_60 | Tesla P100 | 2.7 | `f16_mmv_out` | Adds `f16_mmv_out`, ~+9.4% single-stream on P100. |
| `libpxq4_sm60_v8.so`  | sm_60 | Tesla P100 | 2.7 | `moe_mmv_out` | Adds expert-indexed MoE (`moe_mmv_out`). |

All six export the same base op set — `dequant_out`, `linear_out`, `version`,
`mmv_max_m`, `mmv_supported`, `set_tables`, `moe_mmv_out`, `moe_mmv_out_mono` —
plus the revision-specific ops in the table. "torch ABI" is the libtorch the
library was compiled against; the sm_70 builds reference
`c10::TensorImpl::incref_pyobject`, which does not exist in torch 2.7, and the
sm_60 builds do not.

## Choosing a kernel: the PXQ4_LIB rule

`pxq4_vllm.ops` resolves the kernel library in this order, taking the first path
that exists:

1. `$PXQ4_LIB`, used verbatim, whatever the file is named;
2. `<pxq4_vllm package dir>/_lib/libpxq4_sm70.so`;
3. `<pxq4_vllm package dir>/libpxq4_sm70.so`.

**`PXQ4_LIB` must always be set explicitly.** The two fallback names are fixed
regardless of the architecture in the file, so with `PXQ4_LIB` unset the loader
can reach a bundled sm_70 / torch-2.10 library on a Pascal image and die
part-way through model load with

    undefined symbol: _ZNK3c1010TensorImpl15incref_pyobjectEv

rather than with a clear message. The `sidecar/site-sm60` tree in this package
bundles no `.so` of its own, so on a stock image the fallbacks simply miss and
the plugin logs `pxq4: could not find libpxq4_sm70.so` — but an image that ships
its own `pxq4_vllm` can still supply one.

The launchers in `scripts/` set `PXQ4_LIB` for you: `pxa-serve-sm60.sh` pins
`libpxq4_sm60_v10.so`, `pxa-serve-sm70.sh` pins `libpxq4_sm70_v10.so`. Override
either with `LIB=`.

`sidecar/site-sm60` is the only plugin tree in this package, and both launchers
in `scripts/` default to it. The name is historical: the tree is pure Python and
bundles no `.so`, so it serves sm_60 and sm_70 alike — the architecture is decided
by the kernel library in `PXQ4_LIB`, not by the tree. An image that carries its own
`pxq4_vllm` should override `SITE=`. Either launcher refuses to start, with the path
in the message, if the tree it was pointed at is missing.

## Identifying a kernel revision

The ops are registered through TORCH_LIBRARY string schemas, not exported C
symbols, so `nm` will not find them — `nm -D | grep linear_out` returns nothing
even for a library that has it. Scan printable strings instead:

    strings -a kernels/libpxq4_sm60_v10.so | grep -E '^(moe_mmv_out|f16_mmv_out|gemm2d_out)$'

    moe_mmv_out present -> v8 or later
    f16_mmv_out present -> v9 or later
    gemm2d_out  present -> v11

Confirm the architecture the same way, from the embedded CUDA binary:

    cuobjdump --list-elf kernels/libpxq4_sm60_v10.so
    # exactly one member: pxq4_kernel.sm_60.cubin

## Rebuilding

Both builds compile the same sources with the same CMake project; they differ
only in the toolchain and the target architecture, because one libtorch cannot
cover both. `-use_fast_math` is forbidden in either: it would change the fp32
fold order, and bit-identity with the llama.cpp kernels is the whole
correctness argument.

**sm_70 (Tesla V100), torch 2.10** — `tools/vllm-pxq4/src/build.sh` runs the
build inside a throwaway container started from your vLLM image, so the
extension links against exactly the torch, nvcc and gcc that will later dlopen
it. Nothing is written into the image; the build tree stays on a host directory
you choose.

    PXQ4_IMAGE=<your-vllm-image> bash tools/vllm-pxq4/src/build.sh
    # or PXQ4_REF_CONTAINER=<a running vLLM container> to read the image name off it

**sm_60 (Tesla P100), torch 2.7.1** — `tools/vllm-pxq4/tools/buildv60.sh`. A
stock sm_70 vLLM image cannot be used: its torch 2.10+cu128 ships no sm_60
kernel image, so nothing built against it can even allocate on a P100. The last
official torch with Pascal support is 2.7.1+cu126, so this build needs its own
venv and its own CUDA devel builder image:

    python3 -m venv "$SM60_ROOT/venv"
    "$SM60_ROOT/venv/bin/pip" install torch==2.7.1 numpy \
        --index-url https://download.pytorch.org/whl/cu126
    SM60_ROOT=<workdir> SM60_IMAGE=<cuda-devel-image> bash tools/vllm-pxq4/tools/buildv60.sh

`TORCH_CUDA_ARCH_LIST=6.0` is required — the script sets it. Without it torch's
CMake overrides `CMAKE_CUDA_ARCHITECTURES` with its no-GPU default list and
silently produces a library with no sm_60 cubin at all. Verify the result with
`cuobjdump --list-elf` before trusting it.

A rebuilt library will not reproduce the md5 in `MANIFEST.md` — the checksums
identify the artifacts shipped here, not a byte-reproducible build.
