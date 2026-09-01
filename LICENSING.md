# Licensing map

This repository publishes the inference engine under a single permissive licence.

| path | what | licence | upstream |
|---|---|---|---|
| repository root | `pxq_llama` inference engine (C/C++) | **MIT** — see `LICENSE` | llama.cpp -> ik_llama.cpp |
| `pxa/pxq4/` | PXQ4 kernels and the vLLM sidecar (Python) | **MIT** — see `LICENSE` | PXA Network |

Attribution lives in `NOTICE` at the repository root.

The PXQ4 vLLM sidecar published under `pxa/pxq4/sidecar/` is the integration
layer that teaches vLLM to read PXQ4; it is licensed with the rest of this
repository. The `pxa-vllm` serving stack it plugs into is a separate
Apache-2.0 fork of vLLM (via 1Cat-vLLM) and is not distributed here — see
`docs/PXA-SM70-SERVING.md` and the prebuilt images referenced in
`docker/vllm-pxq4/README.md`.

The upstream README is preserved rather than deleted:
`docs/README-upstream-ik_llama.md`.

The engine sits at the repository root rather than under `engine/` because it
is the original tree and moving it would break every documented path, every
build recipe and every issue link written to date. The split is by licence,
not by symmetry of directory names.
