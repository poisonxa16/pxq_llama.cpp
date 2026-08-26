# SM70 Gemma Long-Prefill Residual/RMSNorm Fusion

## Scope

This path targets the Qwen3.5/Qwen3.6 Gemma-style normalization boundary after
TP all-reduce. It does not change the collective or its reduction order. The
default dispatch is deliberately limited to SM70, two-dimensional FP16 input,
FP32 residual, hidden size 5120, at least 256 tokens, and contiguous tensors.
FP16, BF16, and FP32 norm weights are supported. Decode and small tails retain
the existing path. Set `VLLM_SM70_GEMMA_LONG_PREFILL_FUSED=0` to roll back.

## Trace Motivation

The accepted 64K Qwen3.6-27B-AWQ TP4 trace exposed a repeated post-collective
chain for every residual boundary:

| Baseline kernel family, rank 0 | Calls | GPU sum |
|---|---:|---:|
| FP32 residual add | 2048 | 512.352 ms |
| FP32 RMSNorm | 2048 | 517.251 ms |
| Float copies/conversions | 5199 | 412.633 ms |
| FP32-to-FP16 copies | 2576 | 271.487 ms |

The mixed-dtype fusion consumes the unchanged FP16 all-reduce output and FP32
residual, writes the required FP32 residual, performs the exact 256-thread CUB
variance reduction, applies `(weight.float() + 1)`, and writes FP16 normalized
output in one kernel. The first production trace reduced the four baseline
families by 1.661 seconds and spent 668.714 ms in 2032 fused calls, a net
recovery of 0.992 seconds. Sixteen first/final norm calls remain on the generic
path because they do not have the matching residual contract.

## Exactness And Microbenchmark

The kernel matches `rms_norm_kernel<float, 4, 2>` accumulation and CUB reduction
order. Tests compare both outputs with `torch.equal`, not a tolerance. The
production operator passed all six combinations of `M={256,4096}` and
weight dtype `{FP16,BF16,FP32}` bitwise.

The final kernel keeps each thread's five `float4` residual vectors in
registers across the variance reduction, avoiding a second global read of the
just-written residual. It uses 48 registers/thread, 168 bytes shared memory,
and no local-memory spill. Fixed-clock V100 measurements are:

| M | Baseline local chain | Initial fused | Register-resident fused | Final speedup |
|---:|---:|---:|---:|---:|
| 4096 | 0.8131-0.8141 ms | 0.3574 ms | 0.2806-0.2826 ms | 2.88-2.90x |

The checkpoint stores these norm weights as BF16, but the matched
`--dtype half` runtime casts them to FP16; Nsight therefore records the
`WeightT=half` specialization. Final FP16 and BF16 microbenchmarks have the
same timing range and are both bitwise exact.

## Full-Model Gate

The matched workload is Qwen3.6-27B-AWQ, TP4, four 32 GB V100s fixed at
1530 MHz, FP16 KV, chunk size 4096, Flash-V100 exact gather plus split-KV3,
exact-dense AWQ projections, 65,536 input tokens, and 16 deterministic output
tokens. A three-repeat baseline immediately followed by the fused candidate
measured:

| Route | Prefill values | Mean | Decode | Output |
|---|---|---:|---:|---|
| Fusion disabled | 21.938 / 22.022 / 22.349 s | 22.103 s | 51.320 tok/s | exact hash |
| Fusion enabled | 21.111 / 21.369 / 21.510 s | 21.330 s | 51.254 tok/s | exact hash |

The conservative paired result is 0.773 seconds lower latency, or 3.50%, with
all six measured output hashes identical. Decode differs by 0.13%, within run
noise, and cannot hit the `M>=256` dispatch. A final post-refinement route-hit
run measured 20.192 seconds, retained the same 16-token hash, and decoded at
51.742 tok/s; this single run is an implementation gate, not the headline A/B.

## Remaining Bottlenecks

After fusion, the 64K rank-0 trace is led by exact FP16 projection GEMMs,
D256 full attention, and TP all-reduce. The captured collective sum varied
from 1.412 seconds in the control trace to 3.117 seconds in a later capture;
three ranks waited while rank 1's attention kernels were transiently slower.
An isolated fixed-clock `Q4096/KV64K` attention test measured
44.231/44.301/44.176/44.329 ms on physical GPUs 0-3, a maximum spread of only
0.35%, with identical output sums. The rank skew is not reproducible outside
that capture. Do not optimize communication from this anomaly or reopen the
small-M fused custom all-reduce route: it changes the communication substrate
and previously lost to Inductor in decode.

## Evidence

Artifacts are under
`/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/`:

- `sm70-gemma-local-fused-src/production_op_summary.json`
- `sm70-gemma-local-fused-src/production_bfloat16_4096_reg_*.json`
- `sm70-gemma-local-fused-src/production_float16_4096_reg_*.json`
- `sm70-gemma-fullmodel/baseline/i65536_repeat3.json`
- `sm70-gemma-fullmodel/candidate/i65536_after_baseline_repeat3.json`
- `sm70-gemma-fullmodel/candidate/i65536_register_resident.json`
- `sm70-gemma-fullmodel/nsys/qwen36_awq_tp4_i64k_gemma_fused.nsys-rep`
- `awq-prefill-dense-fullmodel/nsys/qwen36_awq_tp4_i64k_exact_dense.nsys-rep`
- `rank-skew/physical_gpu*_q4096_kv65536.json`
