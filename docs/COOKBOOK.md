# Config cookbook — per-card recommended command lines

Copy-paste starting points for the cards this fork is tuned for. Every number is a measured
median (protocol: `bench/speed-bench.sh` — server-reported `timings.predicted_per_second`,
200-token temp-0 generations, median of ≥3, model fully GPU-resident; prefill = cold prompt at
the stated `-ub`). Weights: `huggingface.co/poisonxa/PXA-Fusion2-35B-GGUF`.

**The one hard rule: PXQ models must be FULLY GPU-resident.** The CPU MoE op has no PXQ
support — `-ngl < 99` over PXQ expert layers (or `--n-cpu-moe`) aborts. Pick the tier that fits
your VRAM with ~2.6 GB headroom for compute buffer + KV (see `docs/KNOWN-ISSUES.md`).

The recommended env used by every recipe below:

```bash
export PXA_ENHANCE=1
export LD_LIBRARY_PATH=build/bin:build/src:build/ggml/src:build/examples/mtmd
```

`PXA_ENHANCE=1` auto-selects the measured-good kernel levers per device and prints the decision
at startup (mixed-card boxes get a per-GPU line). Optionally add `PXA_MODE=balance` (default,
fa-on serving) or `PXA_MODE=max` (fa-off, max prefill). Every recipe below is written against
just these two vars — no other `PXA_*` env is needed to reproduce the published numbers.

> **Lab footnote:** everything `PXA_ENHANCE=1` arms for you — the PXQ6 kernel family
> (KSPLIT/VECX/GUFUSE/SCATFUSE), `PXA_FUSE_DELTANET=3`, `PXA_G2_ADDFUSE=1` (2026-07-19, +1.9% V100
> / +1.2% P100 decode, bit-exact), and the sm_61 `PXA_PXQ_INT8_PREFILL` carrier used in the 1080 Ti
> recipe below — is a hand-settable lab knob in its own right, each with its own measurement and
> gate class, in [`docs/lab/LEVERS.md`](lab/LEVERS.md). Setting any of them by hand
> bypasses the per-arch gate `PXA_ENHANCE` applies and is usually **slower**, not faster.

## Two FA regimes — pick by workload (read this before quoting a prefill number)

On these pre-Turing cards (P100/V100/1080 Ti), flash-attention is a **decode win but a
cold-prefill loss** — for this engine *and* for upstream ik_llama. You run **one** setting per
server, so choose by what you're doing. Measured (35B, cold 5.8k-token prompt, `-b 2048`, median
of 3; full sweep in `bench/fair-battle.md`):

| card | `-fa on` (interactive: chat/agent) | `-fa off` (batch: ingest/summarize/embed) |
|---|---|---|
| P100 | prefill **817** · decode **56.7** | prefill **1,213** · decode 41.1 |
| V100 | prefill **1,589** · decode **94.1** | prefill **1,700** · decode 76.6 |
| 1080 Ti | prefill **667** · decode **65.4** | prefill **1,001** · decode 34.2 |

- **Interactive serving → `-fa on`** (what the recipes below use). You get the full decode speed
  *and* a solid prefill in the same server — e.g. P100 gets **+59% prefill** vs upstream (the engine
  win). The accompanying **+30% decode** in that comparison comes from the smaller PXQ quant tier
  (PXQU-16 + q8_0 head vs upstream IQ3_KS) **plus MTP speculative decode**, not the kernel — the
  same-quant engine control is decode +2.7–3.3% (see `bench/fair-battle.md`).
- **Prefill-heavy batch → `-fa off`.** Prefill jumps 26–56% (this is where the "+88% P100
  prefill" headline comes from) but decode drops 16–48%. Use it for one-shot ingest/summarize
  passes where you barely decode.
- The recipes below are the interactive (`-fa on`) defaults. For a batch job, add `-fa off` and
  read the prefill from the right column above.

## 1× Tesla P100 16 GB — PXQU-16 (q8_0 head)

```bash
./build/bin/llama-server -m fusion2-35b-U16-q8head.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
Expected: **~62–63 t/s decode** (62.4 published; 63.0 with ADDFUSE), **827–843 t/s prefill**
@ ub2048. Decode is ub-insensitive — drop to `-b/-ub 512` if you want a smaller compute buffer.

## 1× Tesla V100 16 GB — PXQU-16 (q8_0 head)

Same command as the P100. Expected: **~101–102 t/s decode** (101.3 published; 102.0 with
ADDFUSE), **~1800–1900 t/s prefill** @ ub2048.

## 2× Tesla P100 (or V100) — PXQ4 flagship (18.7 GB, the `*-PXQ6.gguf` file)

```bash
./build/bin/llama-server -m PXA-Fusion2-35B-PXQ6.gguf \
  -c 8192 -np 1 -ngl 99 -sm layer -ts 1,1 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
Expected: **55.7 t/s decode** (2×P100), **~843 t/s prefill**. The 4-bit flagship does NOT fit
one 16 GB card — single-card 16 GB users want PXQU-16 instead. For the MTP variant
(`*-PXQ6-MTP.gguf`) add `--spec-type mtp:n_max=3,p_min=0.5`.

## 1× GTX 1080 Ti 11 GB — PXQ2 + int8 prefill tile

```bash
./build/bin/llama-server -m PXA-Fusion2-35B-PXQ2.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 768 -ub 768 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
Expected: **~71 t/s decode**, prefill **248 t/s** stock → **709 t/s** with `PXA_ENHANCE=1`
(+182%; `PXA_ENHANCE=1` auto-arms the sm_61 int8-prefill tile on the 1080 Ti — a G3-class lever,
see [`docs/lab/LEVERS.md`](lab/LEVERS.md) §4 for the lab-lever name and its own measurement). Use
`-ub 768`: a ub2048 compute buffer (~1.9 GiB) cannot allocate next to the resident model on 11 GB.
⚠ PXQU-12 (11.6 GB) does NOT fit an 11 GB card — it's a 12 GB tier; PXQ2 is the 1080 Ti tier.

## stock-gguf-on-pxq-engine — a stock quant on this engine, no PXQ file required

The engine fixes above — sm_60 fp16-GEMM, flash-attention regime routing, the MoE path, `np>1`
hybrid concurrency, wide f16 GEMV — apply to any stock GGUF you already have (Q4_K, MXFP4,
IQ_K, …). You do not need a PXQ file to get them:

```bash
./build/bin/llama-server -m your-model-Q4_K_M.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 2048 -ub 2048 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```

That's the whole recipe: point `-m` at a stock quant, keep `PXA_ENHANCE=1` exported from the
shared env above, run as normal. **Engine-only numbers** (same GGUF, two engines — upstream
ik_llama.cpp vs this engine, matched config, isolating the kernel/arch fixes from any codec):
from the "Same-quant control" table in [`bench/fair-battle.md`](../bench/fair-battle.md) —
V100 decode 84.5 → 87.2 t/s (**+3.2%**, bit-identical output, same temp-0 sha), P100 decode 44.0
→ 45.2 t/s (**+2.7%**), 1080 Ti decode 52.2 → 53.9 t/s (**+3.3%**) — all on upstream's own IQ_K
ggufs, no PXQ tensor involved. The bigger engine-only win is prefill — see the README's
"Engine-only, the honest number" and `bench/fair-battle.md`'s regime tables for the fa-on/fa-off
split on the same stock files.

## 1× 12 GB card — PXQU-12

Same command shape as PXQU-16 with `fusion2-35b-U12.gguf`. Measured on the 16 GB Teslas:
58.4 t/s decode P100 / 97.6 V100 (see `bench/HEAD-TO-HEAD.md` §12 GB tier).

## Vision / MTP extras

- Vision: add `--mmproj mmproj-fusion2-f16.gguf` (projector loads on the first CUDA device).
- MTP speculative decode (flagship-MTP file only): `--spec-type mtp:n_max=3,p_min=0.5`.

## Quantizing your own model

See the README "Quantize your own" section — pure tiers (`PXQ4`, `PXQ3`, `PXQ2`) or a
mixed-tier PXQU map (`--pxq-universal <map>.tiers`, `docs/PXQU-CONVERT.md`), plus:
- **`--output-tensor-type q8_0`** (recommended): +5.2% decode on P100 for +123 MB.
- **Imatrix doctrine:** quantizing a merged model? Recompute the imatrix ON the merge
  (activation statistics are anchor-specific), full-GPU-resident (the CPU/partial-offload
  capture path crashes — `docs/KNOWN-ISSUES.md`).
