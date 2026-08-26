# PXA vLLM

**A vLLM serving fork for the cards everyone else dropped: Tesla P100 (sm_60) and
Tesla V100 (sm_70).** It serves PXQ4, PXA Network's 4-bit quantization format, on
hardware that current releases of torch and vLLM no longer build kernels for.

Built on [1Cat-vLLM](https://github.com/KewaiiGamer/1Cat-vLLM), whose Volta/SM70
engineering this fork depends on. See [NOTICE](NOTICE) for the full attribution
chain and what each project contributes — the short version is that the V100 work
is theirs and the Pascal work and PXQ4 are ours.

---

## Why this exists

torch 2.7.1 is the last torch whose official wheels carry **sm_60** cubins. Anything
newer passes every extension gate and then dies on a P100 with `no kernel image is
available` from torch's own kernels. So Pascal support is not a flag — it pins the
whole stack to an older torch.

But the SM70 paths were written against torch 2.10, and forcing them onto 2.7.1
breaks them for reasons that have nothing to do with Volta. **So this repo builds two
images rather than one**, and each architecture gets the torch it actually needs:

```bash
scripts/build-images.sh sm60     # Tesla P100 / Pascal  -> pxa-vllm:sm60
scripts/build-images.sh sm70     # Tesla V100 / Volta   -> pxa-vllm:sm70
scripts/build-images.sh both
```

Both are built from this tree. Nothing here depends on a third-party image.

## What you get

`pxa-launch.py` (in the engine repo) reads the model and the cards and picks the
configuration from measured numbers rather than defaults. Two things it enforces that
are easy to get wrong by hand:

- **`FULL_DECODE_ONLY` is a correctness requirement on sm_60**, not a tuning knob. Without
  it, short raw (non-chat-templated) prompts return fluent garbage from character zero.
  Chat templates hide this by padding every prompt past the captured graph sizes.
- **`custom_ops:["none"]` is mandatory wherever FDO is emitted on sm_60**, or the server
  dies with a CUDA illegal memory access during `profile_run`. It makes no difference on
  sm_70 — that one is measured, both arms failed identically without it.

## Status, stated honestly

| | sm_60 (P100) | sm_70 (V100) |
|---|---|---|
| builds | yes | yes |
| boots and serves | **gated** | **see RELEASE-GATE** |
| PXQ4 | yes | yes |

The sm_70 half has a documented history of boot failures caused by the torch downgrade
described above; the two-image split exists precisely to remove that class of problem.
Do not treat a `200` on `/health` as a passing gate — the gate is a **raw,
non-chat-templated** one-token and five-token completion, because that is the probe that
catches the failure mode chat traffic hides.

## Build from source

`scripts/build-images.sh` is the reproducible recipe. It emits wheels
(`pxa_vllm-*.whl`, `flash_attn_v100-*.whl`) — those are the artifact; Docker is a
convenience, not a requirement. A wheel is tied to its Python version **and to the exact
torch it was built against**, so the matching constraints file ships with it. Install the
wrong pairing and you land straight in `no kernel image is available`.

## License

Apache-2.0, as are both upstream projects. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
The upstream project's own README is preserved at
[docs/README-upstream-1cat-vllm.md](docs/README-upstream-1cat-vllm.md).
