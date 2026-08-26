# DeepSeek V4 Flash DSpark on SM70

## Scope

Optimize single-request DSpark speculative decode for
`deepseek-ai/DeepSeek-V4-Flash` on
eight V100-SXM2-32GB GPUs. Correct output and a measured end-to-end gain over
the same-contract no-speculation route are mandatory. The checkpoint calls
this mechanism DSpark; ordinary MTP is not weight-compatible. The production
candidate is the official model-card configuration, DSpark with seven proposed
tokens (`N=7`, verifier `M=8`). This work does not use eager
execution, Marlin, altered model weights, or reduced-precision shortcuts.

## Source And Runtime Contract

- Integration base: `agent/v100-dsv4-quality-rootcause-20260802`
- Base SHA: `a089aa6c22b9421f529dcaa27a2b59c769f9465f`
- Branch: `agent/v100-dsv4-mtp4-20260802-132909`
- Worktree: `worktrees/v100-dsv4-mtp4-20260802-132909`
- Model: `/home/fudanwl/Desktop/dir`, TP8 on eight V100-SXM2-32GB GPUs
- Weights: MXFP4 routed experts and FP8 dense layers
- KV cache: `fp8_ds_mla`
- Decode: CUDA Graph enabled, `max_num_seqs=1`, no prefix cache, no eager
- Speculator: three DSpark blocks, one non-causal block forward, sequential
  Markov samples, verifier width `N + 1`
- Drafter attention: DeepSeek V4 SM70 sparse SWA with DSpark non-causal indices
- Initial workload: exactly 1024 prompt tokens and at most 256 output tokens
- Sampling: target `temperature=1.0`, `top_p=1.0` for non-agentic workloads;
  the public V4 model-card route uses greedy drafts, while paper-comparable
  acceptance uses probabilistic drafts and standard rejection sampling;
  natural EOS is preserved
- Bring-up context limit and token budget: 2048, sufficient for the exact
  1024-token prompt plus 256-token decode workload
- Active-expert candidate: disabled during the first MTP/no-MTP comparison

The checkpoint index contains 4,705 `mtp.*` tensors across stages 0, 1 and 2.
Stage 0 owns `main_proj/main_norm`; stage 2 owns the final norm, mHC head,
Markov head and unused confidence head. The `mtp` prefix is a checkpoint
namespace, not evidence that the ordinary vLLM MTP architecture applies.

## Acceptance Gates

1. Worker logs must prove DSpark, non-causal SM70 sparse SWA, FP8 DS MLA KV,
   TurboMind dense/MXFP4 dispatch, TP8, and non-eager CUDA Graph execution.
2. A deterministic request must be repeatable after the KV RoPE race fix.
3. Official-sampling output must be coherent and stop naturally. Report mean
   acceptance length, per-position acceptance, and rejection distribution.
4. Report TTFT, steady TPOT, and output throughput separately for no-speculation
   and DSpark7 under the exact same contract.
5. A candidate is accepted only when output quality remains valid and its
   unprofiled end-to-end TPOT improves. Profile-only service-time reductions
   are not performance claims.

## Speculative Width

The checkpoint stores `dspark_block_size=5`, and the DSpark paper's V4
production route uses a maximum width of five with confidence scheduling. The
public DeepSeek model-card vLLM command instead uses `N=7` with greedy draft
sampling. Keep `N=7` as the current model-card deployment baseline, but do not
treat it as the paper's V4 production contract. See
`sm70_deepseek_v4_dspark_acceptance.md` before running another width or
acceptance experiment.

## Measurement Order

1. Establish unprofiled no-speculation and DSpark7 quality and speed baselines.
2. Split target verifier, target logits/rejection sampling, target auxiliary
   state capture, context-KV projection/insertion, three-layer non-causal draft
   forward, seven Markov head/sample steps, bookkeeping, and host wall time.
3. Capture a focused Nsight Systems CUDA Graph node trace for the critical TP
   rank and keep an explicit unattributed residual.
4. Use Nsight Compute only on a confirmed hot kernel. Record its exact shape,
   duration, occupancy, registers, shared memory, memory/SM throughput, and
   dominant stalls.
5. Reject or admit each optimization with the smallest exact-shape
   microbenchmark, then rerun the full quality and endpoint benchmark.

## Baseline

The table below uses the same source snapshot, model, TP8 topology, FP8 DS MLA
KV cache, 1024-token raw prompt, 256 generated tokens, seed 4201, and official
`temperature=1.0, top_p=1.0` sampling. Decode TPOT is
`decode_wall / (completion_tokens - 1)`; streamed text chunks are not treated
as individual tokens.

| Route | TTFT | Decode TPOT | Decode throughput | Acceptance length | Quality |
| --- | ---: | ---: | ---: | ---: | --- |
| TP8 no speculation, current source, warm | 1814.293 ms | 130.063 ms | 7.689 tok/s | N/A | Same raw sample is repetitive/mixed-language; short chat stops naturally and is correct |
| TP8 DSpark7, static-anchor fix | 2062.147 ms | 123.913 ms | 8.070 tok/s | 1.555 | Same raw sample is repetitive/mixed-language; no DSpark-only corruption established |
| DSpark7 delta | +247.853 ms | -6.150 ms (-4.73%) | +4.96% | - | Short natural-stop smoke remains mandatory after integration |
| TP8 DSpark4 before SM70 range fix | Diagnostic only | Diagnostic only | Diagnostic only | 1.00-1.04 | Target verifier preserved coherent text; drafter invalid |
| TP8 DSpark4 after SM70 range fix | Diagnostic only | Diagnostic only | Diagnostic only | 2.06-2.50 | Coherent; more root-cause work required |
| TP8 DSpark5 | Diagnostic only | Diagnostic only | Diagnostic only | 2.06 | Coherent short smoke; suffix acceptance remained weak |

For the official-sampling N=7 request, the four metrics windows contained 91
accepted draft tokens from 1,148 proposals, or 164 proposal rounds. This gives
an aggregate mean emitted length of `1 + 91 / 164 = 1.555`. A clearly labelled
target-greedy diagnostic reached 3.514, proving the chain is not wholly
misaligned while also showing that official random sampling materially lowers
acceptance.

The separate no-speculation result near 39 tok/s came from source
`1.2.3.dev3+gea4b4da78`, which includes later FP16 GEMV work absent from this
branch. It is not a valid comparison for this PR. DSpark must be rebased onto
that integration source and rerun before release-level speed claims are made.

## Microbenchmark Gates

| Gate | Exact shape or contract | Result |
| --- | --- | --- |
| FP8 dense CUDA Graph | Real `mtp.0.main_proj`, M8xK12288xN4096, first uncached M8 capture after M1 warmup | Finite; 20 replays bit-stable; graph and eager max error 0, cosine 1.0 |
| Sequential Markov chain | Real full-vocabulary W1/W2, N7, vocab 129280, rank 256 | 0.901 ms per graph replay; tokens and logits exactly match eager; replay stable |
| Query/SWA metadata | Prefix 200, N7, window 128, block 128, 20 graph replays | Anchor/noise IDs, positions, slots, sample indices, all 135-entry non-causal rows and lengths match the oracle |
| SM70 packed-FP8 attention | Real layer-43 dump expanded to q=7 | Finite and replay-stable; shape/capture gate only |
| Independent attention oracle | Live N5 dumps from layers 43/44/45, byte-level FP8 dequantization and FP32 softmax | Relative L2 0.000217-0.000230 and cosine above 0.999999; causality, slot indices, packed-cache decode and sink are correct |

The q=7 attention replay deliberately repeats two saved q rows, so its equality
against the saved kernel output is not an independent numerical oracle. The
live N5 byte-level check is the semantic attention proof; the q=7 test proves
only N=7 shape and CUDA Graph stability.

## Experiment Log

| Date | Change or test | Result | Decision |
| --- | --- | --- | --- |
| 2026-08-02 | First TP8 ordinary-MTP startup | The Qwen-only dynamic-vocabulary default was incorrectly applied at TP8 and was scoped back to its validated Qwen architecture/TP sizes. | DeepSeek V4 uses the full target vocabulary. |
| 2026-08-02 | Exact ordinary-MTP weight load | Failed on missing `mtp_block.main_norm.weight`; the checkpoint has three DSpark stages and a Markov head, not the ordinary MTP schema. | Reject ordinary MTP rather than skipping weights or weakening strict loading. |
| 2026-08-02 | Checkpoint and official-vLLM audit | Confirmed 1,568/1,565/1,572 tensors in DSpark stages 0/1/2 and official `method=dspark`. | Backport DSpark through the mature local DFlash non-causal execution path; start at four speculative tokens and compare official seven later. |
| 2026-08-02 | Local and remote static tests | Exact model config resolves `DSparkDraftModel`, target layers 40/41/42 and noise token 128799; DSpark mapping/Markov tests plus draft-vocab regression: 23 passed. | Proceed to TP8 non-eager load and quality smoke when the owned GPUs are free. |
| 2026-08-02 | Official vLLM main `0601850791` audit | The newly landed NVIDIA DeepSeek V4 DSpark implementation confirms anchor-first non-causal queries, three draft blocks, sequential Markov sampling, and replicated Markov weights. | Keep the local backport aligned with the official algorithm while retaining the existing SM70 attention and mHC fast paths. |
| 2026-08-02 | First TP8 DSpark4 profile at `max_model_len=4096` | Model and draft weights loaded, but 1.03 GiB available KV memory was below the 1.72 GiB requirement. | Use 2048 for the 1024+256 bring-up gate; tune the production context-memory contract separately. |
| 2026-08-02 | First live request | Request-time draft combine failed because this branch's `ReplicatedLinear(return_bias=False)` returns a Tensor, while the backport unpacked a tuple in both `main_proj` and Markov `w2`. | Match the branch API directly and cover both calls with a CPU regression test. |
| 2026-08-02 | First finite/logit pipeline dump | Target aux states were finite, but unscaled SM70 W8A16 `main_proj` produced NaN `main_x`; all three draft context-KV tensors were consequently NaN. Acceptance was 0/76 drafted tokens in the first window. | Stop attention tuning: the first correctness blocker is FP16 output-range overflow in `main_proj`. |
| 2026-08-02 | Exact-shape FP32 `main_proj` oracle | Real aux input had absmax 49,056. FP32 projected absmax was 146,502, causing 293/57,344 Inf values on FP16 conversion. Power-of-two input scaling by `2^-6` reduced projected absmax to 2,290; normalized output matched the FP32 oracle with cosine 1.0 and MAE about `1.0e-5`. | Apply the SM70-only scale before `main_proj`; the immediately following RMSNorm removes it, so this restores range without changing weights or lowering precision. |
| 2026-08-02 | DSpark4 after range fix | `main_x` and all three context-KV tensors became finite. Deterministic smoke remained coherent; observed acceptance length improved to 2.06-2.50 with first-position acceptance 66.7%-83.3%, but suffix position 4 remained 0% in the short sample. | Range fix is necessary and accepted, but DSpark4 is not yet a performance baseline. Test the checkpoint-native width 5 and official deployment width 7 before deeper kernel work. |
| 2026-08-02 | Checkpoint-native DSpark5 smoke | The same deterministic prompt produced coherent text, acceptance length 2.06, average draft acceptance 21.1%, and per-position rates 50.0%/33.3%/11.1%/5.6%/5.6%. Diagnostic PIECEWISE throughput was about 3.7 tokens/s and is not a production benchmark. | Width 5 does not by itself repair suffix acceptance; do not infer a kernel bug from one short prompt. |
| 2026-08-02 | Exact live SM70 non-causal attention oracle | For draft layers 43/44/45, every query row attended to all 14 context plus 5 query slots. Packed-FP8 kernel output matched a byte-level dequantized FP32 softmax reference with cosine about 0.999999 and MAE `8.2e-5` to `1.13e-4`. | SM70 attention causality, indices, packed-cache decoding and softmax are correct. Remove temporary dumps and move the investigation above attention. |
| 2026-08-03 | N7 first FULL CUDA Graph capture | Capture hung on the first uncached M8 FP8 dense shape. `select_dense_dispatch_policy_impl` and its MoE counterpart held `tune_mutex`, then recursively called `has_imported_cache`, which attempted to lock the same non-recursive mutex. | Check `imported_cache_devices` directly while the caller owns the mutex. Rebuilt `_C` captures N7 FULL M8 in about 4 seconds. |
| 2026-08-03 | Exact N7 microbenchmark suite | M8 FP8 graph, N7 Markov, query/SWA metadata, and q=7 packed-FP8 attention all passed finite/replay/eager gates. | Admit one same-source end-to-end comparison; do not profile kernels before this gate. |
| 2026-08-03 | Persistent anchor parity with official vLLM | The local proposer retained an external `next_token_ids` tensor reference across asynchronous scheduling and graph replay. Official vLLM reads the anchor from the persistent expanded input buffer. | Read `input_ids[request_index * N]`. Keep the fix for graph correctness; it did not by itself restore high official-sampling acceptance. |
| 2026-08-03 | Exact-source N7 versus no-spec | N7 measured 8.070 tok/s versus 7.689 tok/s no-spec, a 4.96% decode-throughput gain. Official-sampling acceptance length was 1.555; target-greedy diagnostic was 3.514. | N7 is functional but its current gain is small. Rebase onto the FP16 GEMV integration branch before deciding whether it remains net-positive. |

## Artifacts And Handoff

- Remote source: `/home/fudanwl/v100-worktrees/deepseek-v4-mtp4-a089aa6c22`
- Compiler caches: `/home/fudanwl/v100-worktrees/cache/dsv4-mtp4-a089`
- Run artifacts: `/home/fudanwl/v100-worktrees/runs/dsv4-mtp4-baseline-20260802`
- M8 core extension SHA256:
  `e1c5e083488c7c81d2a0478c92b30682cfc11d4db1ce250afc4492fe86f26b9d`
- Range-fix diagnostic: `dspark4_scaled_context.pt` is the retained post-fix
  context dump; the server logs record the NaN baseline and exact acceptance
  counters. Temporary full-cache attention dumps are not release artifacts.
- Exact result artifacts:
  `dspark7_anchor_fixed_stream_i1024_o256_seed4201.json` and
  `nospec_exact_source_warm_stream_i1024_o256_seed4201.json`
- Microbenchmark artifacts: `micro_m8_fp8_graph.json`,
  `micro_dspark7_markov.json`, `micro_dspark7_metadata.json`, and
  `micro_dspark7_attention_l43.json`
- API port `18082` is stopped; all eight task GPUs were released after testing.

Record every launch command, source hash, output token IDs, metrics snapshot,
profile path, failed experiment, and active process in the run artifact tree.
