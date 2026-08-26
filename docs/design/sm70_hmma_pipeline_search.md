# SM70 HMMA Pipeline Search

## Scope

This record covers the structural P0-P3 search performed on the accepted
SM70 D256 exact-dense split-KV3 prefill kernel. The gate shape is `Q=4096`,
`Hq=6`, `Hkv=1`, `D=256`, FP16, causal attention on one V100 at a fixed
1530 MHz graphics clock. Arithmetic order and FP32 accumulation are fixed.

The search has two purposes:

- determine whether QK FP32 accumulators can feed PV without cross-lane data
  movement;
- use measured Volta latency and register-lifetime constraints to schedule
  the remaining LDG/STS/LDS/HMMA pipeline.

Every source candidate must retain zero spill, pass the operator quality gate,
and improve same-build wall time. A model-only cycle reduction is insufficient.

## P0: Native QK-to-PV Ownership

`benchmarks/tools/sm70_hmma_c2a_layout_solver.py` enumerates the SM70 TN, TT,
NT, and NN operand layouts for both P and transposed-P QK outputs. It also
permits every fixed permutation of the eight lanes in a Volta quadpair. This
8! search is a strict superset of the permutations directly expressible by
the four MMA layout variants.

No zero-transfer mapping exists. The best mapping keeps 32 of 64 FP16 values
on their source lane and requires the other 32 values to cross a lane. A
transposed-P source is worse. P0 therefore rejects a native zero-movement
QK-to-PV mapping for this `m8n8k4` ownership domain.

Artifact:

- `p0_hmma_c2a_layout_solver.json`

## P1: Minimum-Communication Mapping

The exact transfer graph was edge-colored rather than manually packed. The
result cuts the transfer sequence from 33 to 17 static SHFL instructions and
uses no local memory. The standalone microkernel is exact and reduces its
register count from 40 to 32.

The full D256 integration does not improve wall time. After packing the eight
source half values into four 32-bit registers, the split-KV3 kernel changes as
follows:

| Static SASS item | Accepted | Minimum communication |
|---|---:|---:|
| Total instructions | 2,184 | 2,160 |
| SHFL | 52 | 36 |
| F2F | 80 | 72 |
| HADD2 | 8 | 0 |
| SEL | 0 | 20 |

Dynamic instructions fall by 1.79%, but the issue dependency chain becomes
longer: long-scoreboard cost rises from 0.096 to 0.311 cycles per issue. At
64K, wall time regresses by about 1.4%. P1 establishes the mathematical
minimum communication count, but that minimum is not a minimum critical path
on Volta. This mapping is closed for production.

## P2: Measured Latency And Register-Lifetime Scheduling

`benchmarks/benchmark_sm70_instruction_latency.py` measures compile-time
unrolled dependency chains, verifies their static SASS, and emits the costs
consumed by `benchmarks/tools/sm70_pipeline_schedule_solver.py`.

Measured at 1530 MHz with 32 unrolled operations:

| Operation | Median cost |
|---|---:|
| Dependent `LDG.CG`, L2 hit | 210.6875 cycles |
| Dependent LDS | 26.4375 cycles |
| Dependent PTX `m8n8k4` (four HMMA steps) | 8.65625 cycles |
| Four-chain PTX `m8n8k4` issue | 8.382812 cycles |
| STS + LDS + warp-sync round trip | 33.5625 cycles |

The scheduler models operation dependencies, MIO/LG/Tensor pipe occupancy,
and the interval from fragment production to its final consumer. It admits a
two-fragment PV pipeline and rejects schedules that exceed the register
budget. Its cycle estimates are lower bounds; NCU and wall time remain the
acceptance authority.

### Accepted PV pipeline

The accepted source loads phase `i+1` of the V operand into a second register
fragment before issuing the phase `i` PV HMMA stream. It alternates the two
fragments recursively at compile time. The arithmetic and accumulator order
do not change.

PTXAS remains at 253 registers/thread, zero spills, and 45.568 KiB shared
memory. Static HMMA, LDG, LDS, and STS counts are unchanged. At 64K, NCU shows:

| Metric | Reference | PV pipeline | Change |
|---|---:|---:|---:|
| Kernel duration | 42.785 ms | 42.085 ms | -1.64% |
| Tensor active | 39.83% | 40.19% | +0.36 pp |
| Eligible warps/scheduler | 0.5489 | 0.5562 | +1.3% |
| MIO throttle/issue | 1.4013 | 1.3528 | -3.46% |
| Short scoreboard/issue | 0.2948 | 0.2652 | -10.1% |
| Registers/thread | 253 | 253 | unchanged |

Three alternating process pairs give the conservative wall-time result:

| KV length | Reference p50 | PV pipeline p50 | Gain |
|---:|---:|---:|---:|
| 8K | 3.8984 ms | 3.8492 ms | 1.28% |
| 64K | 37.7313 ms | 37.4948 ms | 0.63% |
| 256K | 153.0276 ms | 152.6062 ms | 0.28% |

The 8K row is the same-build 50-sample gate. The 64K and 256K rows are the
median of three alternating 20-sample process medians. Output hashes and
quality metrics match the reference at 8K, 64K, 128K, and 256K.

### Rejected K-panel lookahead

A second K register fragment moves panel 2 ahead of QK phase 0 and refills
the freed fragment with panel 3. It keeps 253 registers and the same dynamic
instruction count, but does not advance the first critical K publication.
NCU duration changes from 42.085 to 42.123 ms and long-scoreboard cost rises
from 0.2681 to 0.2721 cycles per issue. This path is closed.

## P3: Conditional Rescale And Softmax Overlap

Two exact conditional-rescale forms were tested:

| Candidate | 8K change | 64K change | Result |
|---|---:|---:|---|
| Per-row condition | +6.1% | +4.7% | reject |
| Warp-uniform `VOTE` condition | +7.2% | +5.4% | reject |

Both retain the exact output hashes. The per-row form introduces divergent
control flow and one extra register. The warp-uniform form stays at 253
registers but adds one VOTE, four comparisons, and one branch. Removing FP32
multiply-by-one work does not shorten the critical path because those FMULs
were already hidden; the added issue/control work is exposed.

The current body already issues V global loads before online softmax and
issues the second V pair before the first PV D128 half. Further softmax/HMMA
overlap requires warp specialization, independent group barriers, and
additional V/P shared stages. At 253 registers and one CTA/SM, that is a new
kernel topology rather than a local P3 supplement. It is not admitted by this
search without a separate register and instruction-count proof.

## Production Decision

Only the P2 PV double-buffer pipeline is applied to
`cmake/patches/sm70_flash_attn_d256_pipeline.patch`. Replaying that patch and
`sm70_flash_attn_d256_splitkv3.patch` on the pinned FlashAttention source
produces a source file byte-identical to the accepted candidate.

At 256K, the accepted split-KV3 operator sustains 43.03 causal TFLOP/s and
passes the finite/error gate (`max_abs=7.6294e-6`, `relative_l2=3.1674e-4`).
The 47-50 TFLOP/s objective is not reached by P0-P3. Remaining NCU evidence is
dominated by MIO and dependency issue pressure, while HBM is not saturated.

Do not repeat these closed forms:

- shuffle-based minimum-communication C-to-A repacking;
- a second K lookahead fragment that leaves the first publication unchanged;
- per-row or warp-uniform conditional rescale around the existing FP32 output;
- manual TN/TT/NT layout guesses within the already enumerated ownership set.

Primary artifacts are under
`/data/minimax-h3/task-cache/1cat-fa2-sm70-long-attn-20260812/`:

- `p2_sm70_instruction_latency_unrolled32.json`;
- `p2_sm70_pipeline_schedule_solver.json`;
- `p2_pv_lds_pipeline_gate_8k_64k.json`;
- `p2_pv_lds_pipeline_gate_128k_256k.json`;
- `p2_pv_pipeline_alternating_64k_256k.json`;
- `ncu_p2_reference_samebuild_q4096_kv65536.ncu-rep`;
- `ncu_p2_pv_lds_pipeline_q4096_kv65536.ncu-rep`.
