# bench/ — the PXQ repro pack

Every number we publish for **PXA Fusion2-35B** + **pxq_llama** is reproducible from this
directory. If a number in the model card or a launch post isn't reproducible from here, treat it
as a bug and open an issue.

| claim | where to reproduce |
|---|---|
| Perplexity ladder (PXQ6 7.36 / PXQ3 7.44 / PXQ2 8.39, wikitext-2) | `ppl-ladder.sh` |
| KL-divergence vs the bf16 reference | `kld.sh` |
| Decode/prefill speed per card (V100 99.1 t/s, P100 55.8, 1080 Ti 71.4) | `speed-bench.sh` |
| 0 refusals / 57 adversarial prompts | `refusal-probes/` |
| 9/10 hard reasoning gauntlet, 11/12 factual, ~19/24 coding | `gauntlets/` |
| SWE-bench Lite 19.6% (11/56), official docker grading | `swe/` |
| Bit-exact kernel determinism gates (G1/G2/G3) | `determinism-gates.md` |
| File integrity | `checksums.sha256` |

## Reference hardware (what OUR numbers were measured on)

All published numbers were measured on one homelab box of salvaged cards. Match the config
below when comparing; report your own hardware when they differ (we'll add community numbers
to the compatibility table with credit).

- **GPUs:** Tesla V100-PCIE-16GB (sm_70), Tesla P100-PCIE-16GB (sm_60), GTX 1080 Ti 11GB (sm_61)
- **Driver:** NVIDIA 580.142 · **CUDA:** 12.8.1 (built + run inside `nvidia/cuda:12.8.1-devel-ubuntu24.04`)
- **Build:** `cmake -DCMAKE_CUDA_ARCHITECTURES="60;61;70" -DGGML_CUDA=ON`, targets
  `llama-server llama-cli llama-perplexity llama-quantize`
- **PCIe:** x4 per card (bifurcation risers) — decode is unaffected (single-card residency);
  only multi-card `-sm layer` hand-off touches the bus.
- **Runtime env (the bit-exact fast kernels):**
  `PXA_PXQ6=1 PXA_PXQ2=1 PXA_PXQ3=1 PXA_PXQ6_KSPLIT=1 PXA_PXQ6_VECX=1 PXA_PXQ6_GUFUSE=1 PXA_PXQ6_SCATFUSE=1 PXA_PXQ6_RAGTAIL=1`

## The published numbers, exactly

Perplexity: wikitext-2-raw **test**, `-c 512 --chunks 200 -b 512 -ub 512 -fa on -ctk f16 -ctv f16`
(the standard llama.cpp ppl protocol — comparable to other GGUF ppl tables at n_ctx=512):

| tier | file | wikitext-2 ppl (200 chunks) | Δ vs PXQ6 |
|---|---|---|---|
| PXQ6 (flagship, 4.27 bpw) | 18.7 GB | **7.3563 ± 0.0818** | — |
| PXQ3 (3.27 bpw) | 14.7 GB | **7.4407 ± 0.0830** | **+1.1%** |
| PXQ2 (2.27 bpw) | 10.7 GB | **8.3906 ± 0.0961** | **+14.1%** |

The ladder is monotonic. Same corpus, same protocol, same imatrix for every tier — only the
quant type varies. (KLD vs the bf16 merge source: `kld.sh` is the exact procedure; our own run
is queued behind production GPU occupancy and will be published as a follow-up — run it on
your card and beat us to it.)

Speed (measured via `llama-server` `timings.predicted_per_second`, 200-token generations,
median of 3, model fully GPU-resident — see `speed-bench.sh`):

| card | tier | decode | prefill |
|---|---|---|---|
| 1× Tesla V100 16 GB | PXQ6 | **99.1 t/s** | ~1920–1960 t/s @ `-ub 2048` |
| 1× Tesla P100 16 GB | PXQ3 | **55.8 t/s** | — |
| 2× Tesla P100 | PXQ6 | 55.7 t/s | — |
| 1× GTX 1080 Ti 11 GB | PXQ2 | **71.4 t/s** | — |

Cells we didn't measure are absent, not implied. PRs with measured numbers welcome.

## Model files

The tiers are published at `huggingface.co/poisonxa/PXA-Fusion2-35B-GGUF`
(`PXA-Fusion2-35B-{PXQ2,PXQ3,PXQ6,PXQ6-MTP}.gguf` + `mmproj-fusion2-f16.gguf`).
Verify integrity against `checksums.sha256` in this directory.

⚠ PXQ is a PXA-native format: **mainline llama.cpp cannot read these GGUFs** — build this fork.
⚠ Never round-trip PXQ tensors through mainline `gguf-py` (its size table silently truncates
the E16-row anchors).
