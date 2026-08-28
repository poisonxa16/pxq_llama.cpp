# Licensing map

This repository is one clone containing two engines under two different
permissive licences. They are compatible, but they are not the same licence,
and each keeps its own notices.

| path | what | licence | upstream |
|---|---|---|---|
| repository root | `pxq_llama` inference engine (C/C++) | **MIT** — see `LICENSE` | llama.cpp -> ik_llama.cpp |
| `tools/vllm-pxq4/` | PXQ4 quantization backend for vLLM (Python + CUDA) | **Apache-2.0** — see `tools/vllm-pxq4/LICENSE-NOTICE.md` | vLLM |

Attribution for each side lives with that side: `NOTICE` at the root,
`tools/vllm-pxq4/LICENSE-NOTICE.md` for the vLLM backend. Apache-2.0 section 4
requires that notice be carried into redistributions; do not drop it.

The upstream README is preserved rather than deleted:
`docs/README-upstream-ik_llama.md`.

The engine sits at the repository root rather than under `engine/` because it
is the original tree and moving it would break every documented path, every
build recipe and every issue link written to date. The split is by licence,
not by symmetry of directory names.
