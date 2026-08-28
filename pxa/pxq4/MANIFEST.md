# PXA PXQ4 Package Manifest

Every file shipped in `pxa/pxq4/`, with its size in bytes and md5, so a
deployment can be verified against the artifacts that produced the recorded
numbers. See `README.md` for what each library is and how to rebuild it.

## Kernel libraries — `kernels/`

Six libraries: four sm_60 (Tesla P100 class, torch 2.7 ABI) and two sm_70
(Tesla V100 class, torch 2.10 ABI). Each carries exactly one CUDA device
binary, for the architecture in its filename, and no PTX.

| library | arch | bytes | md5 |
|---|---|---|---|
| `libpxq4_sm60_v8.so`  | sm_60 | 1314768 | c602c7077e96c1d3d4e967f5a8f598d6 |
| `libpxq4_sm60_v9.so`  | sm_60 | 1491856 | 9a1c50670e33fedd60bc115a71c4d0f7 |
| `libpxq4_sm60_v10.so` | sm_60 | 2135960 | 6e17fdbbbdc8fab7cec6ba777efa59eb |
| `libpxq4_sm60_v11.so` | sm_60 | 2201888 | 5b856f449c3b4cfefbb600c284e0c61c |
| `libpxq4_sm70_v9.so`  | sm_70 | 1563712 | 33ec9a86e6736a0ccb0ebeb96afb2f7e |
| `libpxq4_sm70_v10.so` | sm_70 | 2228200 | de087888e1eee5dbe41ae0bdbfb1b63c |

Registered `pxq4::*` operators per library:

| library | dequant_out | linear_out | moe_mmv_out | f16_mmv_out | gemm2d_out |
|---|---|---|---|---|---|
| `libpxq4_sm60_v8.so`  | yes | yes | yes | -   | -   |
| `libpxq4_sm60_v9.so`  | yes | yes | yes | yes | -   |
| `libpxq4_sm60_v10.so` | yes | yes | yes | yes | -   |
| `libpxq4_sm60_v11.so` | yes | yes | yes | yes | yes |
| `libpxq4_sm70_v9.so`  | yes | yes | yes | yes | -   |
| `libpxq4_sm70_v10.so` | yes | yes | yes | yes | -   |

All six also register `moe_mmv_out_mono`, `version`, `mmv_max_m`,
`mmv_supported` and `set_tables`.

## Plugin tree — `sidecar/`

One site tree, `site-sm60`, placed on `PYTHONPATH`. It bundles no `.so`; the
kernel library comes from `PXQ4_LIB` (see `README.md`).

| file | bytes | md5 |
|---|---|---|
| `sidecar/site-sm60/sitecustomize.py` | 1950 | 6f7c66ffdd653481712eb3b40ad7c84d |
| `sidecar/site-sm60/pxq4_vllm-0.1.0.dist-info/METADATA` | 53 | 6410d6ec55924956d2dc1ee479e4d521 |
| `sidecar/site-sm60/pxq4_vllm-0.1.0.dist-info/entry_points.txt` | 49 | 505c390452ecf725885225ec5e9ab9cf |
| `sidecar/site-sm60/pxq4_vllm/__init__.py` | 3305 | 3424f396bffc4b8ea0e36bcb89532c31 |
| `sidecar/site-sm60/pxq4_vllm/config.py` | 32362 | 3ce8a7c6f249ec46c45d24f97b78fb67 |
| `sidecar/site-sm60/pxq4_vllm/linear.py` | 42297 | 3d3a5b81e395d888391b40f670dd44b6 |
| `sidecar/site-sm60/pxq4_vllm/moe.py` | 16337 | 2e41b83caf65ae9e8bcd5f19321e13c1 |
| `sidecar/site-sm60/pxq4_vllm/ops.py` | 19333 | 27c1a55d9ffafc6f05a8637bbb3c2f53 |
| `sidecar/site-sm60/pxq4_vllm/parameters.py` | 9134 | 70c07c3a6464ce1eb844311921cf058a |
| `sidecar/site-sm60/pxq4_vllm/pxa_sdpa_tiled.py` | 3643 | 64433f081a1a70fd0a3b24ef08ec9b8a |
| `sidecar/site-sm60/pxq4_vllm/pxa_sm60_f16.py` | 6300 | bd670f239d623c63f10c522b96676771 |

## Verifying

    cd pxa/pxq4 && md5sum kernels/*.so $(find sidecar -type f | sort)

A rebuilt kernel library will not reproduce these md5s; the checksums identify
the artifacts shipped here, not a byte-reproducible build.
