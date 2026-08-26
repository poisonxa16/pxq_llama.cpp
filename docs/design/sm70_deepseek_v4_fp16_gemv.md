# SM70 DeepSeek V4 FP16 GEMV

## Trace Evidence

The 1024-input TP8 decode trace attributes 3.235 ms/token to dense GEMV,
split-K reducers, and compressor work. The fixed FP16 GEMV shapes are:

| role | shape N x K | calls/token | traced main-kernel mean |
|---|---:|---:|---:|
| MoE router | 256 x 4096 | 43 | 9.73 us plus reducer/cast |
| Indexer weights | 64 x 4096 | 21 | 20.07 us plus reducer |
| C4 indexer compressor | 512 x 4096 | 21 | 16.02 us plus reducer |
| C4 main compressor | 2048 x 4096 | 21 | 38.90 us |
| C128 main compressor | 1024 x 4096 | 20 | 22.15 us plus reducer |

The C4 input projections run on separate CUDA streams, so summed service-time
savings are not an endpoint projection. A candidate must also shorten their
joined graph envelope.

## Candidate

The screening kernel assigns one program to one output row, accumulates FP16
products in FP32, and performs no cross-program split-K reduction. It sweeps
the K tile and warp count using real checkpoint weights.

Acceptance requires:

1. lower graph replay latency for each material shape;
2. lower joined C4 multi-stream envelope before production integration;
3. stable router top-6 IDs across real-weight seeds;
4. bounded compressor error and a later full-model quality gate.

No production dispatch is changed until these gates pass.

## Microbenchmark Result

Real checkpoint weights, CUDA Graph, 500 replays and five repeats selected
`BLOCK_K=1024`, `num_warps=4` for every accepted shape.

| role | cuBLAS median | candidate median | speedup | projected service saving |
|---|---:|---:|---:|---:|
| Router | 10.047 us | 5.624 us | 1.79x | 0.190 ms/token |
| Indexer weights | 6.834 us | 4.418 us | 1.55x | 0.051 ms/token |
| C4 indexer compressor | 8.772 us | 5.415 us | 1.62x | 0.070 ms/token |
| C4 main compressor | 26.792 us | 22.737 us | 1.18x | 0.085 ms/token |
| C128 main compressor | 15.901 us | 12.597 us | 1.26x | 0.066 ms/token |

The summed operator projection is 0.463 ms/token. The three C4 operations in a
joined multi-stream graph improve from 42.004 to 31.908 us, projecting 0.212
ms/token across 21 C4 layers before accounting for overlap with the FP8 default
stream.

Sixteen real-weight router seeds all preserve top-6 IDs after the production
`sqrt(softplus(logit)) + correction_bias` selection. Maximum normalized routed
weight error is 3.61e-5. Compressor maximum absolute error is at most 2.39e-6.

The production route is default-off behind
`VLLM_SM70_DSV4_FP16_GEMV=1`. A combined endpoint quality gate is still
required before enabling it by default.

Artifact:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-sm70-fp16-gemv-micro-20260802/validated.json
```

## Full-Graph Transfer

The latest stacked TP8 graph run selected the production gate on every rank.
Its low-overhead 1024-input/32-output request measured 20.276 ms/token; the
node-traced 64-output request measured 22.708 ms/token and attributed 2.392
ms/token to 125.9 fixed-shape GEMV launches. The trace is composition evidence,
not a same-source route-off/route-on endpoint A/B.

Both requests used the model's official `temperature=1.0`, `top_p=1.0`
sampling and produced complete, non-repetitive text. This is a route and text
health gate only. It does not replace the required long-output quality test,
so `VLLM_SM70_DSV4_FP16_GEMV` remains default-off.

The corrected V100 test resets the cached environment values around every
case. All five supported CUDA Graph shapes plus the default-off test pass on a
V100 (`6 passed`).

Trace artifact:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-tp8-latest-graphtrace-20260803/
  graph_node_latest_v5_i1024_o64.sqlite
  per_token_latest_v5_i1024_o64.json
  warmup_i1024_o32.json
  trace_request_i1024_o64.json
```

## Rejected Router Fusion

A last-CTA Triton prototype fused the FP16 router GEMV with
`sqrt(softplus)`, correction bias, top-6 selection, and normalization. The
single-layer microbenchmark overestimated the saving because the 2 MiB gate
weight stayed hot in L2. A 40-layer CUDA Graph using every regular router
weight (`layers.3..42`) measured the actual per-token chain:

| route | 40-layer median | delta |
|---|---:|---:|
| Separate FP16 GEMV + C++ top-k | 0.5201 ms | baseline |
| Fused, 128 CTAs | 0.4551 ms | -0.0650 ms |
| Fused, 256 CTAs | 0.3861 ms | -0.1340 ms |

All 40 top-6 ID sets matched the separate route, with maximum normalized
weight difference `1.05e-7`. The best saving is only 0.66% of the 20.276 ms
endpoint baseline, below the 0.2 ms/token admission threshold. The prototype
is retained as a benchmark but is not integrated into production dispatch.

Artifact:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-sm70-fused-router-20260803/
  v5-all40.json
```
