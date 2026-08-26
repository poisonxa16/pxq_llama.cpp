# SM70 DeepSeek V4 MXFP4 Active-Expert Optimization

## Scope

- Base: `dd462e37f2552f3e038f1ed7128e62bd7b4ab0d7`
- Dependency: DeepSeek V4 SM70 bring-up PR #159
- Model: `deepseek-ai/DeepSeek-V4-Flash`, MXFP4 experts
- Runtime: 8 x V100-SXM2-32GB, TP8, FP16 activations, `fp8_ds_mla` KV
- Quantization backend: TurboMind only; Marlin is out of scope
- Decode mode: CUDA Graph enabled, no MTP, no eager execution

The first optimization target is the graph-safe batch-one MXFP4 MoE path. The
change must preserve expert routing, accumulation order, output dtype, graph
replay stability, and official-sampling output quality.

## Baseline Evidence

The accepted trace request used exactly 1024 prompt tokens and 256 generated
tokens with `temperature=1.0`, `top_p=1.0`, and natural EOS behavior. It
completed all 256 tokens without malformed output.

Raw artifacts are retained on the profiling host at:

```text
/home/fudanwl/v100-worktrees/runs/
  dsv4-tp8-nsys-i1024-o256-retry1-20260802/
```

The Nsight Systems report contains 255 decode replays per rank: the first
emitted token comes from prefill, followed by 255 decode forwards. The parser
drops four fill/drain steps at each edge and aggregates 247 steady steps.

| Item | Mean per token | Notes |
|---|---:|---|
| Node-trace TPOT | 149.687 ms | Composition only; CUPTI adds overhead |
| TP rank interval max | 150.183 ms | Replay-to-next-replay |
| Rank-average GPU service | 130.080 ms | Categories sum exactly to this value |
| TurboMind MXFP4 MoE | 54.594 ms | 22,016 launches/rank/token |
| SM70 sparse MLA attention | 46.880 ms | 43 launches/rank/token |
| TP all-reduce | 10.154 ms | Overlaps other streams; not additive wall |
| TurboMind FP8 dense | 6.064 ms | 279 launches/rank/token |

The prior unprofiled artifact was labeled 1024 tokens but the serving
tokenizer reports 1020. It is excluded from the accepted A/B below.

## Root Cause

The graph-safe path launches both MXFP4 stages for every local expert:

```text
43 layers * 256 experts * 2 stages = 22,016 launches/token/rank
```

On rank 0, 5,476,780 of 5,614,080 MXFP4 decode launches (97.55%) finish in
less than 2.5 microseconds. Those empty or near-empty launches consume
42.738 ms/token of traced GPU service. Approximately 538 launches/token do
material work, close to the `43 * top_k(6) * 2 = 516` active-expert bound.

This makes active-expert device-side dispatch or a persistent grouped stage a
higher-value target than tuning the existing per-expert GEMM tile.

The first microbenchmark avoids a new arithmetic kernel. It reuses the exact
existing per-expert TurboMind GEMM but captures only six fixed route slots.
Expert IDs stay in a device tensor, so their values can change at replay time.

| Exact TP8 stage | 256-expert graph | 6-expert graph | Speedup | Numeric gate |
|---|---:|---:|---:|---|
| W13, K4096/N512 | 0.7369 ms | 0.1268 ms | 5.81x | Bitwise exact |
| W2, K256/N4096 | 0.5255 ms | 0.0461 ms | 11.41x | Bitwise exact |

Changing all six expert IDs after graph capture also remains bitwise equal to
the 256-expert reference and changes the output as expected. A second gate
uses the production `moe_permute_with_scratch` output directly; its sorted
expert IDs match the six selected experts and both stages remain bitwise equal
to the 256-expert reference. The retained artifact is
`dsv4-mxfp4-active-expert-micro/baseline_vs_active_permute.json`.

The active-expert route remains explicit opt-in until the final deterministic
full-model gate passes. It also falls back to dense dispatch for expert-parallel
layouts and when either legacy single-token permute fastpath is enabled. Those
paths do not guarantee the six valid local runtime expert IDs required by
active-expert dispatch.

## Unprofiled A/B

Both sides use exact 1024-token input, 256-token output, TP8, FP8 MLA KV,
official `temperature=1.0` / `top_p=1.0`, no MTP, and CUDA Graph. Three requests
per side completed with `finish_reason=length`.

| Route | Mean TPOT | Throughput | Mean TTFT |
|---|---:|---:|---:|
| Dense 256-expert control | 133.480 ms | 7.492 tok/s | 1783.624 ms |
| Active six-slot candidate | 76.268 ms | 13.112 tok/s | 1789.254 ms |
| Delta | -57.212 ms (-42.86%) | +75.01% | +5.630 ms |

Artifacts are retained in
`dsv4-tp8-active-expert-b1-{control,candidate}`. Official-sampling outputs are
structurally healthy, but random sampling is not reproducible within one
service on this runtime, so cross-service text equality is not a valid quality
oracle. Greedy text is also not reproducible across two requests in the same
candidate service; one repeat entered a repetitive continuation. A repeated
dense-control run is still required to separate a baseline determinism problem
from an active-expert regression. Until that comparison is complete, the route
remains opt-in and the quality gate is explicitly open.

## Optimization Gate

1. Reproduce the exact DeepSeek V4 shapes in an operator microbenchmark.
2. Compare against the current graph-safe dense-expert path with fixed routing.
3. Verify numerical output against the current FP16 output using the real
   permute metadata and all six distinct routed slots; do not change
   quantization or precision.
4. Prove CUDA Graph capture/replay with routing IDs changed between replays.
5. Require fewer expert-stage launches and lower CUDA-event wall time.
6. Run the full-model 1024/256 official-sampling quality gate.
7. Accept an end-to-end claim only from an unprofiled same-contract A/B.

If active-expert dispatch cannot remain graph-safe or its numerical result
changes, reject it and move to the next trace hotspot rather than weakening the
quality contract.

## Rejected Profiling Paths

- The old token parser grouped TP ranks by an 8 ms time window and recognized
  only `cudagraph.FULL.replay`; that is invalid for this TP8 Breakable Graph
  trace. The profiling skill now aligns each worker's Nth replay by ordinal.
- `nsys stats cuda_gpu_kern_sum` reached 77 GB RSS on the 49-million-kernel
  trace and was stopped before OOM. SQLite streaming aggregation produced the
  accepted table with bounded memory.
