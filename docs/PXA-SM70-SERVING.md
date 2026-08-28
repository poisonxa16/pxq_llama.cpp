# PXA Network — PXQ4 sm_70 (Tesla V100) serving

Measured 2026-08-27 on a pair of Tesla V100-PCIE-16GB cards (sm_70, PHB, no P2P).
Every arm below was correctness-gated ("The capital of France is" -> Paris)
BEFORE its speed number was recorded. A fast wrong answer is not a result.

## Shipping configuration

    scripts/pxa-serve-sm70.sh          # default profile: agg
    PXA_PROFILE=single scripts/pxa-serve-sm70.sh

| profile | MNS | ladder          | decode | ms/tok | agg@4  | agg@8  |
|---------|-----|-----------------|--------|--------|--------|--------|
| agg     | 16  | [1..8,16]       | 50.98  | 19.62  | 107.02 | 135.78 |
| single  | 8   | [1..8]          | 51.46  | 19.43  | 107.26 | 132.13 |

`agg` ships by default: it clears the 50 tok/s bar on single-stream and takes
both aggregate crowns. `single` trades 0.5 tok/s of concurrency headroom for the
best single-stream figure.

## The full sweep (all correctness-passed)

| arm                                   | decode | ms/tok | agg@4  | agg@8  |
|---------------------------------------|--------|--------|--------|--------|
| reproduce final-v10 (recorded 48.40)  | 48.74  | 20.52  | 106.46 | 134.36 |
| + PXQ4_MMV_SPLIT_MAX_BLOCKS=300       | 51.46  | 19.43  | 107.26 | 132.13 |
| + SPLIT=600                           | 51.31  | 19.49  | 106.60 | 134.47 |
| + SPLIT=150                           | 49.51  | 20.20  | 105.07 | 134.66 |
| + SPLIT=300, GMU 0.92                 | 51.32  | 19.49  | 104.85 | 133.76 |
| + SPLIT=300, MNS=16, ladder[1..8,16]  | 50.98  | 19.62  | 107.02 | 135.78 |

Prior recorded best on this pair: 50.76 (median). We now exceed it at 51.46.

## Why the thin image measured 34.1 before this

Nothing was broken in the build. The gate shipped a conservative recipe while
the later tuning never landed in it. The whole 34.1 -> 51.46 delta is config:

  GMU 0.90 -> 0.85          0.98 + a pinned 12 GiB KV is a large-VRAM 4-card
                            setting and ABORTS on a 16 GiB card at TP=2 before
                            the model loads
  MNS 4 -> 8/16
  ladder [1,2,3,4] -> [1..8(,16)]
  PXQ4_LIB pinned to the v10 kernel
  PXQ4_MMV_SPLIT_MAX_BLOCKS=300   (gate_up mono -> split; 48.74 -> 51.46)

## Things that will silently cost you, if changed

- **The capture ladder must be passed explicitly.** When `cudagraph_capture_sizes`
  is None the sm70 branch hard-codes `[1,2]`, so every batch above 2 runs eager,
  roughly 4x slower. The launcher reads the value back from the startup log and
  warns if it collapsed to `[1,2]` -- a quoting slip there is otherwise invisible.
- **Bash brace-expands the JSON.** `{"cudagraph_capture_sizes":[1,2,3,4]}` becomes
  `cudagraph_capture_sizes:[1` because of the commas inside the braces. Single-quote
  it for whichever shell finally parses it.
- **PXQ4_LIB must be explicit.** The site tree bundles an sm70-only `.so` built
  against the torch 2.10 ABI; leaving it unset on the wrong image dies with an
  undefined-symbol crash part-way through model load, not a clear message.
- **The f16 mmv hook does not arm on sm_70** (`f16 hook armed: 0` in every arm
  above). It is Pascal-specific. It is worth ~12% on sm_60, so its absence there
  is a real loss; here it is expected and costs nothing.

## The one genuine build bug that had to be fixed first

`vllm/triton_utils/importing.py` looks up its own distribution under a short list
of hard-coded package names. Ours is named `pxa-vllm`, which is not one of them,
so every lookup raised, and
`PackageNotFoundError` subclasses `ImportError` -- so it was swallowed by the
enclosing `except ImportError` and misreported as "triton.backends could not be
imported. Disabling Triton." Triton was silently disabled on a machine where it
works, vLLM installed a placeholder module, and a kernel later called
`triton.next_power_of_2` on that placeholder and killed both workers.

Before the fix the image did not serve at all. After it, sm70 gates clean.

The same fix re-enabled Triton on Pascal, where Triton raises fatally at kernel
compile time ("Triton only supports devices of CUDA Capability >= 7.0"). A
capability guard in the same file now disables Triton below sm_70. It reads the
capability via NVML rather than `torch.cuda`, deliberately: NVML does not create
a CUDA context, so the check stays safe at import time in a process that later
forks its workers. Verified in both directions -- V100 keeps Triton, P100 does not.

## Known build-label defect (not yet fixed)

`pxa-vllm:sm60` carries `CUDA_ARCH=sm_70` in its image environment.
`TORCH_CUDA_ARCH_LIST=6.0;7.0` is correct, but `CUDA_ARCH` is inherited wrong, so
it cannot be used to identify the target arch. That is why the capability guard
above reads NVML instead of trusting the image label.
