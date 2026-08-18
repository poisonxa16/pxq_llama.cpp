# 10 — Decode kernel speed: diagnosis, K-chunk-split fix, measurements

Date: 2026-08-18. Hardware: DGX-1V, V100-SXM2-32GB (sm_70, 80 SMs, ~900 GB/s HBM2).
Artifact for serving numbers: `Qwen3.8-27B-PXQ4-vllm-p2a-nf` (norm-fixed, policy p2a,
4.018 GiB/GPU/token at TP=4). Kernel change: commit `9916ea69`.

## The question

TP=4 serving decoded at ~44 tok/s (~200 GiB/s of weight traffic per GPU, ~22% of peak)
vs AWQ/TurboMind's ~80 tok/s at ~31% of peak. Candidate explanations: (a) the unfused
dequant+GEMM path fires during decode; (b) the mmv kernel converts bytes to work badly.

## Diagnosis 1 — dispatch counting kills hypothesis (a)

`site-instr` (instrumented `linear.py`) serving TP=1, enforce-eager, single stream,
after warmup + 3x(1+512)-token runs + one 128-token run:

    mmv=398080  gemm=1920
    m_hist: M=1 x397840   (decode  -> mmv, 100%)
            M=8 x240      (warmup  -> mmv)
            M=11 x1440    (prefill -> dequant+GEMM)
            M=256/2048 x240 each (profile runs -> dequant+GEMM)

Decode-time linear calls are 100% mmv (240 PXQ4 modules per rank = 60 layers x 4).
The 340 MiB dequant workspace is touched only at prefill. Hypothesis (a) is dead.

## Diagnosis 2 — the monolithic mmv is grid-starved (hypothesis (b) confirmed)

`k_pxq4_mmv` launches one 256-thread block per (panel, token). At M=1 on this model's
TP=4 shapes that is 64-136 blocks for 80 SMs: <= 2 blocks/SM, 256-512 threads/SM of a
2048-thread budget, 12.5-25% occupancy — far too few outstanding loads to hide HBM2
latency. Weight-byte throughput, measured with CUDA events (300 iters, M=1):

    shape            slab bytes   mono GB/s   cuBLAS fp16 GEMV GB/s (same card)
    tp4 gate_up      23.7 MB      276         690
    tp4 down         11.8 MB      180         663
    tp4 o_proj        4.2 MB      200         453
    tp4 qkvz         11.1 MB      145         631
    tp2 gate_up      47.4 MB      460         567
    tp2 down         23.7 MB      218         714
    tp2 o_proj        8.4 MB      175         670
    tp2 qkvz         22.3 MB      291         704
    tp1 ffn_down     47.4 MB      253         732

The old kernel runs at 16-32% of peak while dense cuBLAS GEMV reaches 50-81% — so the
4x byte advantage of PXQ4 bought only ~1.2x in time (TP4 layer set: 0.249 ms PXQ4 vs
0.298 ms fp16). Note the shape trend: 272 panels (tp2 gate_up) already reaches 460 GB/s;
64-80 panels sit at 145-200. It is a parallelism problem, not a coalescing problem —
all weight reads are coalesced by construction (64 consecutive sub-scale bytes/warp-pair,
16 B/thread contiguous code rows = 512 B/warp, 128 B contiguous anchor header).
Consistent with this, cutting bandwidth 11.4% (p2a vs p1 at TP=2) gave zero speedup:
decode is latency-bound, so only more parallelism (or fewer launches) helps.

## The fix — K-chunk-split mmv (`k_pxq4_mmv_part` + `k_pxq4_mmv_reduce`)

The canonical-chunk fold already partitions K into nfix (8-16) per-lane partial sums.
The split kernel gives each chunk its own block (grid = panels x nfix x M), stores each
per-lane chunk partial t_c unsummed (fp32, one coalesced 1024 B store per block), and a
64-thread reduce kernel replays the monolithic kernel's exact fold: chunk order, then
kseg order, one final `__float2half_rn`. No addition is reassociated and no rounding
point moves, so the result is bit-identical — not approximately, but by construction,
and gated:

  * hostsim (REAL kernel source on CPU): 10/10 configs bit-exact (vecx 0/1, nfix 8/16,
    M 1-8) — `test_pxq4_mmv_split.py`, now run by `build_hostsim.sh`;
  * GPU: 27/27 shape/M combos bit-exact vs `mmv_out_mono`;
  * the 17 pre-existing hostsim gates still pass.

Cost: an fp32 partials tensor, M * panels * nfix * 256 floats (1.3-2.2 MB at M=1 —
mostly L2-resident between the two kernels; worst case +22% traffic on tp4 down, in
exchange for 8-16x the blocks). Partials come from `at::empty`, the same
graph-pool-backed allocator as the `out = torch.empty(...)` the caller already does
per `apply()`, so CUDA-graph capture safety is unchanged.

Measured (same protocol), mono -> split:

    tp4 gate_up   0.086 -> 0.056 ms   276 -> 422 GB/s
    tp4 down      0.066 -> 0.034 ms   180 -> 346 GB/s
    tp4 o_proj    0.021 -> 0.032 ms   REGRESSION -> stays mono (threshold below)
    tp4 qkvz      0.077 -> 0.034 ms   145 -> 329 GB/s
    tp2 gate_up   0.103 -> 0.104 ms   tie (was already 272 panels)
    tp2 down      0.109 -> 0.051 ms   218 -> 460 GB/s
    tp2 o_proj    0.048 -> 0.032 ms   175 -> 263 GB/s
    tp2 qkvz      0.077 -> 0.052 ms   291 -> 428 GB/s
    tp1 ffn_down  0.187 -> 0.085 ms   253 -> 555 GB/s

    TP4 per-layer linear time: 0.249 ms -> 0.145 ms (best-of dispatch), 1.72x.

## The dispatch threshold, and why 8 MB

`mmv_out` takes the split path when `nfix > 1 && slab_bytes >= PXQ4_MMV_SPLIT_MIN_BYTES`
(default 8 MB, env-overridable). Mechanism: the split pays a fixed floor — two launches
plus the partials round-trip — of roughly 15-25 us; the monolithic kernel, even starved,
moves B bytes at >= ~200 GB/s when panels >= 64. Crossover where
B/450GB/s + floor < B/200GB/s gives B ~ 7 MB, matching the two measured sides of the
boundary (tp4 o_proj 4.2 MB: mono 21 us vs split 32 us; tp2 o_proj 8.4 MB: mono 48 us
vs split 32 us). A panels-aware predicate was considered and rejected: above ~160
panels mono and split converge anyway (tp2 gate_up ties), so bytes capture the only
regime that matters. Known conservatism: the floor amortises over M, so at M=8 the
split would likely win even at 4 MB; M is not in the predicate because decode is M=1
and the difference is ~10 us on the smallest tensor.

## End-to-end

  * TP=1, enforce-eager (NOT the shipping config; dominated by per-op launch/Python
    overhead — 240 pxq4 calls + attention/mamba ops per token, no graphs):
    old 5.23/5.23/5.28 -> new 5.34 tok/s. Confirms eager TP1 is overhead-bound, not
    kernel-bound; useless as a kernel vehicle, recorded for completeness.
  * TP=4, CUDA graphs, p2a-nf (the shipping config): OLD kernel 43.40/41.57/41.70
    tok/s. NEW kernel measurement pending at the time of this commit; predicted from
    the per-layer delta (0.104 ms x 60 layers = 6.2 ms/token off a ~23 ms step):
    mid-50s tok/s. This file will be amended when the A/B lands.

## Remaining headroom, and what is NOT worth doing

  * The split kernel sits at 330-460 GB/s on TP4 shapes vs cuBLAS GEMV's 630-730.
    Occupancy is no longer grid-limited (1280-2176 blocks). The next suspects are the
    per-byte ALU cost of MODE_TAB (two smem table lookups + shifts per weight pair;
    random-nibble smem reads carry bank conflicts) and the fixed two-launch floor.
    Candidate next steps, in order of expected value: warp-shuffle table lookup
    (registers instead of smem, values identical hence still bit-exact), fusing the
    reduce into the part kernel via a cooperative last-block-reduces scheme (one
    launch), and only then ILP restructuring (changes the fold -> re-baselining event).
  * Bandwidth-side work (smaller formats) is currently worthless for speed: measured
    11.4% fewer bytes -> 0% faster. Fix parallelism first; bytes may matter again
    after the kernel reaches ~600 GB/s.
  * The fp16-vs-PXQ4 serving control adds nothing the GB/s table does not already say.

## site-v2 promotion gate

`site-v2` (op version 2) becomes the default `site/` when: (1) a CUDA-graphs TP=4 boot
serves sane text (proves capture-time `at::empty` in the op), and (2) the A/B shows no
regression. Bit-exactness is already proven, so no quality gate is needed — outputs are
byte-identical by construction. Until promotion, the old lib remains the default and v2
is opt-in via PYTHONPATH.
