# Config cookbook — per-card recommended command lines

Copy-paste starting points for the cards this fork is tuned for. Every number is a measured
median (protocol: `bench/speed-bench.sh` — server-reported `timings.predicted_per_second`,
200-token temp-0 generations, median of ≥3, model fully GPU-resident; prefill = cold prompt at
the stated `-ub`). Weights: `huggingface.co/poisonxa/PXA-Fusion2-35B-GGUF`.

**The one hard rule: PXQ models must be FULLY GPU-resident.** The CPU MoE op has no PXQ
support — `-ngl < 99` over PXQ expert layers (or `--n-cpu-moe`) aborts. Pick the tier that fits
your VRAM with ~2.6 GB headroom for compute buffer + KV (see `docs/KNOWN-ISSUES.md`).

The recommended env used by every recipe below (bit-exact kernel set):

```bash
export PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1
export PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1
export PXA_FUSE_DELTANET=3 PXA_G2_ADDFUSE=1
export LD_LIBRARY_PATH=build/bin:build/src:build/ggml/src:build/examples/mtmd
```

(`PXA_G2_ADDFUSE=1` is the 2026-07-19 late addition: +1.9% V100 / +1.2% P100 decode on top of
the published rows, bit-exact. What each var does: `docs/LEVERS.md`.)

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
PXA_PXQ_INT8_PREFILL=1 \
./build/bin/llama-server -m PXA-Fusion2-35B-PXQ2.gguf \
  -c 8192 -np 1 -ngl 99 -fa on -ctk f16 -ctv f16 -b 768 -ub 768 \
  --jinja --temp 1.0 --top-p 0.95 --top-k 20 --host 0.0.0.0 --port 8080
```
Expected: **~71 t/s decode**, prefill **248 t/s** stock → **709 t/s** with
`PXA_PXQ_INT8_PREFILL=1` (+182%; opt-in, G3-class — see `docs/LEVERS.md` §4). Use `-ub 768`:
a ub2048 compute buffer (~1.9 GiB) cannot allocate next to the resident model on 11 GB.
⚠ PXQU-12 (11.6 GB) does NOT fit an 11 GB card — it's a 12 GB tier; PXQ2 is the 1080 Ti tier.

## 1× 12 GB card — PXQU-12

Same command shape as PXQU-16 with `fusion2-35b-U12.gguf`. Measured on the 16 GB Teslas:
58.4 t/s decode P100 / 97.6 V100 (see `bench/HEAD-TO-HEAD.md` §12 GB tier).

## Vision / MTP extras

- Vision: add `--mmproj mmproj-fusion2-f16.gguf` (projector loads on the first CUDA device).
- MTP speculative decode (flagship-MTP file only): `--spec-type mtp:n_max=3,p_min=0.5`.

## Quantizing your own model

See the README "Quantize your own" section — pure tiers (`PXQ4`, `PXQ3`, `PXQ2`) or the
PXQU knapsack presets (`--pxq-universal {12g|16g|16g-hq}`), plus:
- **`--output-tensor-type q8_0`** (recommended): +5.2% decode on P100 for +123 MB.
- **Imatrix doctrine:** quantizing a merged model? Recompute the imatrix ON the merge
  (activation statistics are anchor-specific), full-GPU-resident (the CPU/partial-offload
  capture path crashes — `docs/KNOWN-ISSUES.md`).
