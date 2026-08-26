# SM70 AWQ Long-Prefill Exact-Dense Path

## Scope

This ledger covers the Qwen3.6-27B-AWQ TP4 `M=4096` prefill projection path
on V100. Decode, partial prefill chunks, TP2, and unknown AWQ shapes stay on
TurboMind AWQ. The bounded workspace has not yet had a model-level 16 GB gate.

The path is enabled by default through
`VLLM_SM70_AWQ_PREFILL_EXACT_DENSE=1`. Set it to `0` to remove its 85 MiB
workspace and use TurboMind for every prefill projection.

## Reason For The Path

NCU on the dominant `M4096,N8704,K5120` gate/up kernel showed:

- 248 registers/thread and 65.55 KB shared memory;
- one CTA/SM and 12.5% achieved occupancy;
- 62.1% of scheduler cycles with no eligible warp;
- 9.36% DRAM throughput, so HBM was not the limit;
- dependency, barrier, math, and shared-load stalls around online AWQ unpack.

Existing registry autotuning selected the same `CTA128x256x16` kernel. Shape
tuning was therefore closed. The structural replacement keeps compact AWQ
weights for decode, expands one selected projection at a time into a shared
FP16 workspace, and lets cuBLAS run the compute-dense `M=4096` projection.

## Numerical Contract

TurboMind forms each dequantized weight as:

```text
bias = fp16(-zero * scale)
weight = fp16_fma(q, scale, bias)
```

Using `(q - zero) * scale` is not equivalent and produced up to `0.008789`
output error in the first probe. The production helper preserves the FP16
bias rounding and single-FMA order. All accepted `M=4096` projection outputs
were bitwise equal to TurboMind AWQ.

The route is limited to AWQ group size 128 and the following validated TP4
`(K,N)` shapes:

| projection | K | N | layers | AWQ | exact dense | saved/chunk | former resident |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP gate/up | 5120 | 8704 | 63 | 7.089 ms | 4.781 ms | 145.40 ms | 5.23 GiB |
| MLP down | 4352 | 5120 | 63 | 3.371 ms | 2.296 ms | 67.76 ms | 2.61 GiB |
| linear-attention QKVZ | 5120 | 4096 | 47 | 3.376 ms | 2.377 ms | 46.96 ms | 1.84 GiB |
| linear-attention out | 1536 | 5120 | 47 | 1.279 ms | 0.899 ms | 17.86 ms | 0.69 GiB |
| full-attention out | 1536 | 5120 | 16 | 1.276 ms | 0.900 ms | 6.01 ms | 0.23 GiB |

The original implementation modeled `283.99 ms` saving per full chunk but
kept `10.60 GiB` of expanded weights per rank. The current implementation
reuses only the largest row in this table, an 85 MiB `K x N` workspace.

## Initial Resident-Weight Acceptance

The comparison includes the default gather-to-exact-dense and dense split-KV3
attention paths, CUDA graph decode, FlashQLA, and custom TP all-reduce.

| Qwen3.6-27B-AWQ TP4, input 64K/output 16 | prefill | TPOT | token result |
| --- | ---: | ---: | --- |
| latest attention baseline | 25.514792 s | 21.351 ms | reference |
| all selected exact-dense projections | 20.988843 s | 21.390 ms | identical |
| delta | -4.525949 s (-17.74%) | noise | identical IDs/text/hash |

Prefill throughput improves by 21.56%. Model residency rises from 6.50 GiB to
17.36 GiB per rank. This result established the speed target, but the
per-layer resident weights were not acceptable as the final storage design.

## Direct KxN Storage Refinement

The initial resident implementation stored each expanded weight as contiguous
`N x K` and transposed that tensor at every `torch.mm` call. Its accepted
refinement materialized contiguous `K x N` and called the same FP16 cuBLAS
GEMM directly. It changed neither dequantization nor the GEMM numerical
contract.

- Twenty real TP4 rank/shape cases are bitwise equal to the original layout.
- Across all five projections, operator latency improves by 2.39%-3.01%
  (2.58% geometric mean).
- Nsight Systems records the 3,824 selected projection calls per rank falling
  from 38.511 to 36.959 seconds in aggregate (`-4.03%`).
- A fixed-clock 64K full-model A-B-A gate gives a conservative endpoint range
  of 0.85%-3.07%, with an A-B-A center gain of 1.96%. All nine runs produce
  the same token hash.

The route retains the existing `VLLM_SM70_AWQ_PREFILL_EXACT_DENSE=0` rollback;
there is no separate layout switch because both representations have the same
memory footprint and the direct layout is exact on every selected TP shard.

## Bounded Workspace Acceptance

The production path reverses the TurboMind K8/N32 packed layout directly into
one process-local FP16 workspace. Every eligible layer holds a reference to
the same allocation. The next layer reuses it only after the previous GEMM on
the same CUDA stream, so no per-layer dense copy remains. Allocation failure
logs a warning and leaves the layer on TurboMind.

All 20 real TP4 rank/shape cases produce bitwise-identical dequantized weights
and final GEMM outputs. Across the four rank slices, the weighted
dequant-plus-cuBLAS path is `1.47x-1.54x` faster than TurboMind. Dequantization
itself costs about 24-25 ms over all selected projections in one 4096-token
chunk.

The same-source memory gate uses `gpu_memory_utilization=0.9`, 131K maximum
length, decode CUDA graph, and all accepted Flash-V100 prefill routes:

| mode | model residency/rank | available KV/rank | KV tokens | 131K concurrency |
| --- | ---: | ---: | ---: | ---: |
| TurboMind control | 6.50 GiB | 20.49 GiB | 1,312,253 | 10.01x |
| bounded workspace | 6.59 GiB | 20.41 GiB | 1,306,887 | 9.97x |
| former resident weights | 16.77 GiB | 10.34 GiB | - | 5.05x |

The current optimization therefore costs only 0.09 GiB/rank and reduces the
historical 10.86 GiB overhead by about 99.2%. It no longer needs a 30 GiB
physical-memory gate.

At fixed 1530 MHz clocks, 64K input/64-token output, and the model's official
`temperature=1.0`, `top_p=0.95`, `top_k=20` sampling:

| mode | prefill samples | mean | TPOT | token result |
| --- | --- | ---: | ---: | --- |
| TurboMind control | 24.774, 23.321 s | 24.048 s | 20.879 ms | reference |
| bounded workspace | 20.900, 19.775 s | 20.338 s | 20.922 ms | identical |
| delta | - | -15.43% | noise | identical IDs/text/hash |

An A-B-A first-run check (`20.900 -> 24.774 -> 21.241 s`) gives a conservative
14.95% latency reduction. The current workspace latency also remains within
run-to-run variance of the former resident-weight acceptance, so the memory
reduction did not trade away the endpoint speedup.

## Rejected Or Deferred Variants

- Fused gate/up plus SiLU epilogue was exact but saved only 0.096 ms/layer,
  below 0.4% modeled end-to-end. Do not prioritize it over exact-dense.
- Ordinary `(q-zero)*scale` pre-expansion is numerically wrong for this route.
- Dense weights are not used for decode or partial chunks because those shapes
  lack a bitwise and performance gate.
- Extending the shape list requires per-rank bitwise operator evidence and a
  matching full-model token gate; suffix-only expansion is not accepted.

## Evidence

- `awq_prefill_exact_dense_gateup_tp4_allranks.json`
- `awq_prefill_exact_dense_all_shapes_tp4_allranks_exactness.json`
- `awq_prefill_exact_dense_all_shapes_m4096_tp4_rank0.json`
- `awq-prefill-dense-fullmodel/candidate_i8192.json`
- `awq-prefill-dense-fullmodel/candidate_i65536.json`
- `awq-prefill-dense-fullmodel/candidate_all_i65536.json`
- `ncu_awq_gateup_m4096_tp4.ncu-rep`
- `awq_exact_dense_kn_layout_tp4_allranks.json`
- `awq-kn-fullmodel-i65536-clock1530-repeat3.json`
- `awq-nk-control-fullmodel-i65536-clock1530-repeat3.json`
- `awq-kn-fullmodel-i65536-clock1530-repeat3-aba.json`
- `awq-kn-nsys-i65536-clock1530.nsys-rep`
- `workspace-production/micro/tp4_rank{0,1,2,3}.json`
- `workspace-production/micro/t210_tp4_rank0.json`
- `workspace-production/fullmodel/candidate-t210/i65536_o64.json`
- `workspace-production/fullmodel/control-t210/i65536_o64.json`
- `workspace-production/fullmodel/candidate-aba-t210/i65536_o64.json`

The artifacts are under
`/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/`.
