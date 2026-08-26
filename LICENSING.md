# Licensing map

This repository is one clone containing two engines under two different
permissive licences. They are compatible, but they are not the same licence,
and each keeps its own notices.

| path | what | licence | upstream |
|---|---|---|---|
| repository root | `pxq_llama` inference engine (C/C++) | **MIT** — see `LICENSE` | llama.cpp -> ik_llama.cpp |
| `serving/` | `pxa-vllm` serving stack (Python) | **Apache-2.0** — see `serving/LICENSE` | vLLM -> 1Cat-vLLM |

Attribution for each side lives with that side: `NOTICE` at the root,
`serving/NOTICE` for the serving stack. Apache-2.0 section 4 requires the
serving NOTICE be carried into redistributions; do not drop it.

Upstream READMEs are preserved rather than deleted:
`docs/README-upstream-ik_llama.md` and `serving/docs/README-upstream-1cat-vllm.md`.

The engine sits at the repository root rather than under `engine/` because it
is the original tree and moving it would break every documented path, every
build recipe and every issue link written to date. The split is by licence,
not by symmetry of directory names.
