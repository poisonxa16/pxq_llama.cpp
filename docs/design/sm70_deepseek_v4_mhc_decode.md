# SM70 DeepSeek V4 mHC Decode Optimization

## Scope

This work targets the 85 fused mHC calls in one DeepSeek V4 Flash decode
token: layer 0 FFN plus attention and FFN in layers 1 through 42. The guarded
route is limited to CUDA SM70, FP16 activations, `M=1`, `H=4096`, `hc=4`,
`tile_n=2`, and split-K 8. Other shapes retain the upstream TileLang path.

The integration base is `f868e6b56053ca54f6a2aa32acefae66caedb74e` and
the task branch is `agent/v100-dsv4-mhc-hotspot-20260803-013355`.

## Trace Baseline

The low-overhead TP8 baseline is 20.276 ms/token. The matching graph-node
trace reports 2.862 ms/token of mHC service across 173.9 launches per rank:

| Kernel | Calls/token | Service/token | Mean/call |
|---|---:|---:|---:|
| `mhc_fused_tilelang_kernel` | 85 | 1.851 ms | 21.781 us |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 85 | 0.970 ms | 11.418 us |

The service values overlap other streams and are not an additive endpoint
decomposition.

NCU on the first kernel found a 96-CTA grid on 80 SMs, 70 registers/thread,
14.91% achieved occupancy, 83.09% cycles with no eligible warp, 6.43% DRAM
throughput, and 15.24% SM throughput. Long-scoreboard stalls account for
31.08% of issue spacing. The source recomputes the same post mapping once for
each of 12 output tiles; those repeated FP32 operations dominate the small,
latency-limited grid.

## Exact FP32 Stage

The accepted microbenchmark candidate splits the old fused kernel into:

1. Eight split-K CTAs compute the post mapping once, store the normal FP16
   residual, retain an FP32 staging copy, and preserve the original sqrsum
   reduction order.
2. The original 96-CTA output grid reads the FP32 stage and preserves the
   original dot-product and warp-reduction order.
3. The existing pre, Sinkhorn, and fused RMSNorm kernel is unchanged.

The FP32 stage is required for exactness. Reusing the existing separate post
plus prenorm GEMV path rounds the dot input to FP16 and changes split-K from 8
to 1.

| Exact-shape CUDA Graph case | Baseline | Candidate | Saving | Exactness |
|---|---:|---:|---:|---|
| one L2-hot layer | 27.881 us/call | 14.668 us/call | 13.213 us/call | bitwise |
| all 85 real checkpoint calls | 2.84065 ms/token | 1.52876 ms/token | 1.31189 ms/token | all four outputs bitwise for all calls |

The all-layer sequence prevents one layer's 1.5 MB mHC weight from remaining
artificially hot in L2.

## Endpoint Result

TP8 uses eight V100-SXM2-32GB GPUs, FP8 MLA KV, Breakable CUDA Graph,
TurboMind FP8/MXFP4, hierarchical custom all-reduce, sparse MLA split-K/QK
split, exact 1024-token input, 256-token maximum output, and official
`temperature=1.0`, `top_p=1.0` sampling without `ignore_eos`.

| Seed | Baseline TPOT | Candidate TPOT | Saving |
|---:|---:|---:|---:|
| 4201 | 20.772 ms | 19.284 ms | 1.488 ms |
| 4202 | 20.764 ms | 19.377 ms | 1.387 ms |
| 4203 | 20.716 ms | 19.366 ms | 1.350 ms |
| median | 20.764 ms | 19.366 ms | 1.398 ms |

The matched median TPOT reduction is 6.73%; decode throughput rises from
48.16 to 51.64 token/s, or 7.22%. The corrected candidate graph-node trace
reports 1.650 ms/token of mHC service versus 2.864 ms/token before the change,
a 1.214 ms/token service reduction. The node-trace request itself measures
21.884 ms/token and is used only for composition because profiler overhead
raises it above the accepted low-overhead endpoint result.

An earlier endpoint table was discarded after its remote runtime assembly was
found to omit the accepted FP16 GEMV dispatch sites. The table above uses the
same combined source, flags, model, sampling contract, and three seeds on both
sides; only `VLLM_SM70_DSV4_MHC_FP32_STAGE` changes. Both official-sampling
sides produce coherent Chinese technical answers. Two identical greedy
requests within one service can produce different text under the existing TP8
multi-stream/custom-all-reduce stack, so quality acceptance uses bitwise
operator outputs across all 85 real calls plus text-health checks rather than
requiring text identity across service restarts.

## Rejected Paths

| Path | Result | Decision |
|---|---|---|
| `tile_n=1` only | About 0.035 ms/token projected | Stop below threshold |
| separate post plus FP16 prenorm GEMV | 27.881 to 24.668 us/call; post mix max difference 2.20e-5 | Reject numerical change |
| TP sharding of 24 mHC rows | Requires 85 latency-sensitive all-gathers per token | Defer; exact local staging already removes the dominant repeat work |

## Dispatch And Artifacts

`VLLM_SM70_DSV4_MHC_FP32_STAGE=1` enables the accepted default. Set it to `0`
to restore the original fused TileLang kernel. The route adds about 5.3 MB of
graph-pool FP32 staging per rank for the 85 captured calls.

Raw evidence is retained under:

```text
/home/fudanwl/v100-worktrees/runs/dsv4-mhc-hotspot-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-mhc-endpoint-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-mhc-combined-endpoint-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-mhc-combined-matched-baseline-20260803/
/home/fudanwl/v100-worktrees/runs/dsv4-combined-latest-graphtrace-20260803/
```
