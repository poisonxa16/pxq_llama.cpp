# DeepSeek V4 DSpark Acceptance on SM70

## Scope

This investigation fixes the low DSpark acceptance observed with
`DeepSeek-V4-Flash` on eight V100-SXM2-32GB GPUs. The acceptance contract uses
the target model's official `temperature=1.0, top_p=1.0` sampling and standard
lossless rejection sampling. Output quality must match the target model; draft
acceptance may not be improved by changing target sampling or accepting biased
tokens.

- Integration base: `onecat/main`
- Base SHA: `48e89751b4b98c18e1be6506dca15f015155d068`
- Branch: `agent/v100-dsv4-dspark-acceptance-20260803-133931`
- Model: `/home/fudanwl/Desktop/dir`
- Runtime: TP8, FP8 DS MLA KV, CUDA Graph enabled, non-eager
- Workload: raw 1024-token report prompts plus natural chat, math, and code

## Sampling Contracts

Three configurations must not be conflated:

1. The public [DeepSeek V4 model card][v4-model-card] command uses seven greedy
   draft tokens. In standard rejection sampling this makes the draft
   distribution one-hot.
2. The [DSpark paper][dspark-paper] and [DeepSpec][deepspec] evaluation code
   sample from the complete draft distribution at `temperature=1.0` and verify
   with the `min(1, p/q)` probability-ratio test. The paper's offline N=7 table
   covers Qwen3 and Gemma4 checkpoints, not DeepSeek V4.
3. The paper's production section uses DeepSeek V4 with a maximum block size
   of five and a confidence scheduler. The released V4 checkpoint agrees:
   `dspark_block_size=5` and a `[1, 4352]` confidence projection for a 4096-wide
   hidden state plus the 256-wide Markov embedding.

The old 1Cat proposer rejected probabilistic draft sampling and always returned
`draft_probs=None`. Under a stochastic target this treated every proposal as a
one-hot distribution and made acceptance depend on the target probability of
the greedy token. It was not comparable to the paper's probabilistic results.

## Fix

The proposer now applies the trained Markov correction and samples each
position sequentially through the existing probabilistic sampling helper. It
returns each full draft distribution in request-major position order so the
standard rejection sampler can perform exact probability-ratio verification.
Greedy requests and the default greedy configuration retain the previous
one-hot path.

Use the paper-consistent route explicitly:

```text
--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}'
```

## Acceptance Results

The matched raw-report suite uses seeds 4210-4219, 1024 prompt tokens, at most
256 output tokens, and identical target sampling.

| Route | Rounds | Accepted / proposed | Mean emitted length | Per-position unconditional acceptance |
| --- | ---: | ---: | ---: | --- |
| N=5 greedy | 1420 | 1139 / 7100 (16.04%) | 1.802 | 38.73%, 18.80%, 11.06%, 6.62%, 5.00% |
| N=5 probabilistic | 1071 | 1493 / 5355 (27.88%) | 2.394 | 64.43%, 37.16%, 21.10%, 11.30%, 5.42% |
| N=7 probabilistic | 1098 | 1459 / 7686 (18.98%) | 2.329 | 62.57%, 34.88%, 19.13%, 9.56%, 4.74%, 1.46%, 0.55% |

Acceptance is workload-dependent. A separate N=5 domain smoke produced:

| Domain | Mean emitted length | Accepted draft tokens |
| --- | ---: | ---: |
| Math prompt 1 / 2 | 4.394 / 4.691 | 67.88% / 73.82% |
| Python prompt 1 / 2 | 3.794 / 4.655 | 55.88% / 73.09% |
| Open chat prompt 1 / 2 | 2.500 / 2.590 | 30.00% / 31.80% |

An N=7 open-chat rerun generated coherent output and measured 2.771 mean
emitted length over 35 rounds. Its unconditional acceptance was 74.29%,
48.57%, 28.57%, 11.43%, 8.57%, 2.86%, and 2.86% by position. The raw report
prompt is therefore a high-entropy stress case, not a universal acceptance
baseline.

## Root-Cause Checks

- Position cross-correlation found no token-row or verifier-logit shift. Draft
  row `i` aligns best with target row `i`.
- A byte-level FP8 attention oracle from the earlier bring-up already measured
  cosine above 0.999999 for all three draft layers. Attention metadata and
  non-causal visibility are not the acceptance limiter.
- Markov bias is beneficial and correctly scaled. On 12 captured open-chat
  rounds, removing it reduced expected emitted length from 2.542 to 1.513.
  Scaling it by 0.75 or 1.25 produced 2.514 and 2.426 respectively; 1.0 was
  best.
- Global draft-temperature scaling cannot repair the suffix. The best tested
  scale, 0.9, changed predicted length by only +0.004 token.
- The checkpoint confidence head correlates with exact target/draft overlap
  (Pearson 0.712 in the diagnostic sample). Fixed N=7 spends two verifier rows
  on a suffix with little expected return. Confidence scheduling is a separate
  change because this runner still needs efficient variable M=2-6 verifier
  graphs; merely loading the unused head would not improve acceptance or speed.

## Quality And Performance Gates

- A natural-stop Chinese quality prompt remained coherent and completed in 45
  tokens.
- A deterministic 256-token control exactly matched the prior token SHA256
  `63254475cd7fef9b04e338dc709f8eee4cc51803ac0d1632f178598362d93a52`.
- Probabilistic sampling currently materializes and samples seven full-vocab
  distributions. It fixes the acceptance contract but is not yet a speed win
  on SM70. The next performance change must fuse or port the Gumbel draft
  sampler before making an end-to-end acceleration claim.

## Decisions

1. Keep target sampling unchanged and retain standard lossless rejection.
2. Land probabilistic DSpark support without changing the global greedy
   default.
3. Do not tune Markov scale, draft temperature, or attention again without new
   contradictory evidence.
4. Treat five as the checkpoint-native V4 width. Before changing the release
   default from model-card N=7, optimize and benchmark the M=6 verifier path.
5. Implement confidence scheduling only together with variable-width verifier
   dispatch and measure accepted tokens, verifier rows, output quality, and
   end-to-end TPOT.

## Artifacts

- Remote run root:
  `/home/fudanwl/v100-worktrees/runs/dsv4-dspark-acceptance-20260803`
- N=5/N=7 acceptance runs: `n5-prob`, `n7-prob`
- Confidence and Markov diagnostics: `n7-confidence-calibration`
- Main analyses: `confidence-analysis.json`, `markov-scale-analysis.json`, and
  `temperature-overlap.json`

[deepspec]: https://github.com/deepseek-ai/DeepSpec
[dspark-paper]: https://arxiv.org/abs/2607.05147
[v4-model-card]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark
