# SM70 DeepSeek V4 Performance Integration

## Source Composition

This branch provides one reproducible Git tree for the current DeepSeek V4
Flash SM70 work. It starts from `onecat/main` at
`3a12f4cef3ee3df911a24bf84c74f26a711f27a3`, which already contains PRs
`#159`, `#160`, `#162`, and `#165`, then integrates:

| PR | Scope |
|---|---|
| #170 | Compact MXFP4 decode and skew-safe TP8 graph all-reduce |
| #171 | Sparse MLA split-K and QK dimension split |
| #175 | Exact SM70 FP16 GEMV and mHC FP32 staging |

The component PRs remain the review and rollback boundaries. This branch is
the exact build and endpoint validation target.

## Endpoint Evidence

The exact `6f946b603a281c0e7ea6008108e7d25d04b8df5f` tree was built and run with
TP8 on 8 x V100-SXM2-32GB, 1024 input tokens, 256 output tokens, FP8 MLA KV,
no MTP, Breakable CUDA Graph, `temperature=1.0`, and `top_p=1.0`. Only
`VLLM_SM70_DSV4_MHC_FP32_STAGE` changed between the matched sides.

| Route | Median TPOT | Decode throughput |
|---|---:|---:|
| Exact stack, mHC route off | 20.401 ms/token | 49.02 token/s |
| Exact stack, mHC route on | 19.457 ms/token | 51.40 token/s |

The candidate reduces median TPOT by 0.944 ms (4.63%) and raises decode
throughput by 4.85%. The full SM70 extension linked after internalizing the
header-defined TP8 reduce kernel. Ten TP8 collective tests and 24 FP16 GEMV,
mHC, and DeepSeek V4 route tests passed. Runtime logs selected every requested
SM70 route. The official-sampling three-prompt text-health suite passed; a
separate concise chat request also stopped naturally. Long HTML requests hit
their token limit without malformed tag prefixes, replacement characters, or
code/natural-language contamination before truncation.

## Excluded Drafts

PR #178 compressor state-save fusion and PR #179 MXFP4 grouped prefill remain
default-off WIP screens and are intentionally excluded. Negative benchmark
PRs remain separate documentation and do not alter this runtime tree. Sparse
MLA, grouped MXFP4, TP8 hierarchical all-reduce, and FP16 GEMV stay opt-in
until long-context, concurrency, and second-host promotion gates pass.
