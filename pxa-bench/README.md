# pxa-bench/ — PXQ codec gate harnesses and tier-map recipes

Not to be confused with [`bench/`](../bench/README.md). The split is by purpose:

| directory | what it holds |
|---|---|
| [`bench/`](../bench/README.md) | the **reproduction pack** for every published number — perplexity, speed, KLD, gauntlets |
| `pxa-bench/` (here) | the **codec-level gate harnesses** the quantizer must pass, and the `PXQ_UNIVERSAL` tier maps |

Nothing here is needed to build or run the engine. These are the tools that prove a PXQ
codec change is byte-correct before it ships, plus the tier maps `llama-quantize
--pxq-universal` consumes.

## Gate harnesses

Each file carries its build line in a header comment. The gates they implement are
described in [`bench/determinism-gates.md`](../bench/determinism-gates.md).

| file | gate |
|---|---|
| `pxq6_ref.cpp` | Q-G1 byte-parity: standalone build of the production converter |
| `pxqu_golden.py` | Q-G1 golden numpy reference the above must byte-match |
| `pxqu_wrel.py` | Q-G2 relative-weight-error reproduction (frozen 36-slice, seed 42) |
| `pxqu_ref.cpp` | CPU decode reference / wrel reproduction tool |
| `pxq6_test.cu` | device-side memcmp battery for the fast kernels |
| `pxq1_selftest.cpp` | PXQ1 byte-parity gate against the shipped codec |
| `pxq4_bench.cu` | PXQ4 dequant/GEMM proof-of-concept microbenchmark |
| `pxq5_verify_file.py` | file-level verification of a PXQ5 GGUF (retired type; historical) |
| `pxq5_quantize.py` | **RETIRED** — PXQ5 (type id 251) was removed; kept as reference only |
| `pxq4_repack.py` | lossless MXFP4 → `PXQ4-LEGACY` (type id 250) repack |
| `dump_types.py` | list the tensor-type histogram of a GGUF |
| `make_glm_fixture.py` | build a small synthetic `deepseek2`-arch GGUF fixture |

## `pxq-universal/` — tier maps

`.tiers` files are the input to `llama-quantize --pxq-universal`; the format is documented in
[`docs/PXQU-CONVERT.md`](../docs/PXQU-CONVERT.md). A bare name resolves against
`$PXA_PXQU_DIR` (default: this directory).

- `12g.tiers`, `16g.tiers`, `16g-hq.tiers` — the card-sized presets, also baked into
  `llama-quantize` as a fallback so they work from a bare clone.
- `recipes/` — complete per-model maps, each with its target hardware and launch flags in
  the header.
- `preflight.py` — check a map against real tensor geometry before starting a multi-hour run.
- `gen_pxqu_flashnext.py` — generate a map for `qwen4exp` experts under a byte budget.
