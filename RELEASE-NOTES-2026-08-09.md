# Release notes — 2026-08-09

Covers everything since the previous public push (2026-07-31).

Headline: **122B-A10B decode on 4× P100 went 27.69 → 33.20 t/s (+19.9%)**, prefill unchanged.
That is a landfill-Pascal fleet running a 122B at 33 t/s.

---

## ⚠ Breaking change — `--pxq-universal` now takes a map path

The three bundled tier-map presets (`12g`, `16g`, `16g-hq`) and the reference `.tiers` recipe files
have been **removed**, along with the copies compiled into `llama-quantize`. They were per-tensor bit
allocations computed for specific model layouts — a recipe, not part of the codec.

```
# before
llama-quantize --imatrix m.imatrix --pxq-universal 16g in.gguf out.gguf PXQ_UNIVERSAL

# now
llama-quantize --imatrix m.imatrix --pxq-universal my-16gb.tiers in.gguf out.gguf PXQ_UNIVERSAL
```

The mechanism is unchanged and the map format is deliberately trivial — `#`-commented lines of
`regex=type`, one per expert tensor, types `pxq1|pxq2|pxq3|pxq4|pxq6`. `docs/PXQU-CONVERT.md` now
documents the format in full, plus the budget/composition splits behind the published 122B-A5B builds,
so you can author or generate a map for your own tensor names and VRAM budget. A bare name still
resolves against `$PXA_PXQU_DIR`. Already-quantized PXQU GGUFs are unaffected — this is a
quantize-time flag only.

## Performance

**The 122B/P100 decode campaign** — 122B-A10B PXQU48-core, 4× P100, `-sm layer -c 32768 -np 1
-b 512 -ub 512 -fa on`, f16 KV, `PXA_ENHANCE=1 PXA_P100_FP16_GEMM=1 PXA_ROUTER_FUSE=3`, temp 0,
fill 8881, n=3 medians, both binaries alternating in one session, engagement asserted from the banner
in every cell:

| cell | levers | deep decode | shallow | prefill |
|---|---|---:|---:|---:|
| baseline | none | 27.69 | 31.72 | 300.70 |
| + FA GQA pack | `PXA_FA_GQA_PACK=4` + `QSMEM` + split target 160 | 30.67 | 31.94 | 300.76 |
| + canonical chunk cap | `PXQ6_CANON_CMAX=4` | 30.56 | 35.62 | 300.59 |
| **both** | | **33.22** | 35.61 | 301.11 |
| both, repeat | | 33.19 | 35.56 | 300.70 |

**33.20 t/s from 27.69 — +19.9%, 36.12 → 30.12 ms/token. Prefill unchanged at ~300 t/s.** The two
levers overlap (both remove memory traffic, so they do not add): the chunk cap alone is +10.4%, the FA
pack + split retune adds +8.6% on top of it.

Both reorder fp32 accumulation, so neither is bit-exact by construction and both were gated:
perplexity **on the decode path** (`-b 1 -ub 1`, 6 chunks) 3.8109 → 3.8027, inside the stderr and
slightly lower; 256-token greedy generation from an 8881-token prompt **byte-identical**, plus four
greedy chat probes byte-identical. The first version of that gate was vacuous — `llama-perplexity` at
`-b 512` never executes the `ncols == 1` vector kernel the FA lever replaces, so both arms were
trivially identical. Gate decode levers at `-b 1 -ub 1`.

**Codec decode paths** (all bit-exact, verified by temp-0 hash):

- `PXA_PXQ6_SHFL` — moves the codebook off the LSU onto the shuffle unit: **+4.1%**.
- `PXA_PXQ3_PAIRLUT` — paired-nibble LUT for the PXQ3 decode and the MoE down split: **+3.7%**.
- `PXQ_CANON_v2` — chained-FFMA accumulation: **+5.7%** on drafted decode (122B/4×P100).

**sm_60 attention** — `flash_attn_vec_ext_f32_gqa<D,NH>`: one block takes NH Q heads sharing a KV head,
so the K row is fetched once and dotted against NH query vectors. Under GQA the stock kernel re-reads
the whole K/V cache once per Q head — on a 32-Q/2-KV model that was 13.4% of the token, issuing ~291 MB
of loads per layer against 18.2 MB of unique cache. Off by default pending a wider coherence gate.

**Self-speculation** — `PXA_AUTO_SPEC` arms the measured-best drafter per model family under
`PXA_ENHANCE=1`. One row ships (`qwen35moe` → `ngram-mod:n_max=4,n_min=2`): **first-request decode
+23.0% on code traffic, +4.6% on prose, prefill neutral.** No `mtp` row ships — on the same model MTP
measured −8.6% at `n_max=1` and −29.8% at `n_max=2` *despite* 0.800 acceptance, because a drafter
costing a transformer forward per cycle cannot pay on bandwidth-bound sparse-MoE decode, where a
k-token verify batch routes to up to k×8 distinct experts. An explicit `--spec-type` or `-md` always
wins; only absence is filled.

Also: router GEMV mode 3 (128-thread float4 form for pre-Volta), sm_60 MXFP4 dmmv, D=256 tile-f16
attention, and a partial-unroll of the fattn vec f32 ILP path (the full unroll spilled and cost deep
decode 26.8 → 18.6, so it is not the shipped form).

## Correctness and robustness

- `ggml_validate_row_data` now accepts the PXA slab types, so `--check-tensors` works on PXQ files.
- Out-of-range `PXA_ROUTER_FUSE` values are reported and the lever explicitly forced OFF instead of
  being silently folded to 0. A build predating mode 3 resolved `=3` to `0`, which made an A/B run two
  identical arms and report a result.
- Fired/declined engagement counters for every router-fuse mode (`PXA_ROUTER_FUSE_DBG=1`); the ratio
  is the assertion, since a one-shot bool cannot distinguish "fired once" from "fires every token".
- Upstream fixes folded in: sampling OOB on newline-less vocabs, a jinja for-loop scope leak, and a
  boolean-flag argv swallow.

## Documented dead ends (kept, default off, so nobody re-runs them)

`PXA_PXQ_REDUCE_BLK` (+0.22% against controls differing by 0.35% — below the noise floor),
`PXQ_GU_MINBLK` (+0.14% median), `PXA_PXQ6_ROWX2` (null in both regimes), and the
`__launch_bounds__` min-blocks forcing experiment (forced registers 10–45 below natural, spilled,
−27.7%).

## Documentation

`docs/LEVERS.md` gains a row for every gated flag added in this release, each with its measured
effect and the exact cell it was measured on, including the ones that measured null. Two build-time
constants (`PXQ6_CANON_CMAX`, `PXQ_GU_MINBLK`) are documented as constants rather than env flags,
because flipping them changes the canonical fp32 fold and is a re-baselining event.

⚠ `PXQ6_CANON_CMAX` default stays **16** even though **4** is worth +10.4% on the cell above: the
evidence is sm_60 only, and on sm_70 the split target is 16×nsm so the selected S differs. If you run
Pascal, build with `-DPXQ6_CANON_CMAX=4` and re-baseline.
