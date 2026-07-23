# Converting to PXQU (universal mixed-tier) quants

`PXQ_UNIVERSAL` (a.k.a. PXQU) is a per-tensor mixed quant: instead of one bit-width for the whole model, a
**tier map** assigns a quant type to each expert tensor by name. This lets a large MoE land in a target VRAM
budget by spending bits where they matter (high-importance experts) and squeezing the rest (down to the 1-bit
`pxq1` tier). The result is a model that fits fully GPU-resident on a card it otherwise couldn't.

## The flag

```
llama-quantize --allow-requantize \
  --imatrix <model>.imatrix \
  --override-kv <arch>.expert_used_count=int:<top_k> \
  --pxq-universal <map>.tiers \
  <source-q8>.gguf  <out>.gguf  PXQ_UNIVERSAL  <threads>
```

- `--pxq-universal <map>.tiers` — the tier map (see below). Also accepts the presets `12g` / `16g` / `16g-hq`.
- `--imatrix` — importance matrix for the source model; PXQU leans on it to place bits well.
- `--override-kv …expert_used_count=int:N` — pin the routing to the model's real top-k so the calibration
  matches how the model actually runs.
- Source should be a near-lossless **Q8_0** gguf.

Files containing the `pxq1` (1-bit) tier require a build with the PXQ1 codec (this release); other tiers
(`pxq2/pxq3/pxq4/pxq6`) run on any current build.

## The tier maps (example: 122B-A5B, in `pxa-bench/pxq-universal/recipes/`)

A tier map is `#`-commented lines of `regex=type`, one per expert tensor. Three reference budgets ship:

| Map | Target card | Composition (experts) | ~Resident |
|---|---|---|---|
| `pxqu24-122b-a5b.tiers` | 24 GB | 126× pxq1 · 18× pxq2 | ~23.5 GiB |
| `pxqu32-122b-a5b.tiers` | 32 GB | 61× pxq1 · 57× pxq2 · 26× pxq3 | ~31.7 GiB |
| `pxqu48-122b-a5b.tiers` | 48 GB | 61× pxq2 · 57× pxq3 · 26× pxq6 | ~45.8 GiB |

These are keyed to the 122B-A5B tensor names (48 layers × 256 experts) — use them as templates. For a
different model, generate a map for your own tensor names and VRAM budget; the tighter the budget, the more
experts drop to `pxq1`, and the more the aggressive tiers benefit from a no-think serving posture.

> The 24 GB map is the aggressive edge (mostly 1-bit). Always live-validate after quantizing — coherence is
> the gate, and heavily-1-bit maps do best served no-think.

## Building for 30-series / 40-series (Ampere / Ada)

The release binary already targets `sm_86`/`sm_89`. To build from source for these cards:

```
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86;89" -DCMAKE_BUILD_TYPE=Release
```

For a mixed fleet, list every arch you run: `-DCMAKE_CUDA_ARCHITECTURES="60;61;70;86;89"` (Pascal → Ada).
The per-arch performance levers auto-select at runtime under `PXA_ENHANCE=1`.
