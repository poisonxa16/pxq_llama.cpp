# PXQ4 for vLLM — Docker

Run PXQ4-quantized models under vLLM on Pascal (P100, sm_60) and Volta (V100, sm_70).

PXQ4 is a **vLLM plugin**, not a vLLM fork. The image here is a thin overlay: it adds the
plugin, the prebuilt CUDA kernels and an entrypoint that picks the right kernel for your
cards. It adds roughly 15 MB to whatever vLLM image you point it at, and builds in
seconds.

---

## 1. Pick a base image

The overlay does not provide vLLM — you choose one, and it has to actually run on your
GPUs.

| Your cards | Base image |
|---|---|
| **Volta** — V100, Titan V (sm_70) | Upstream vLLM images generally work. Start with `vllm/vllm-openai:latest`; if it refuses to start, pin an older tag. |
| **Pascal** — P100, GTX 10xx (sm_60) | Upstream vLLM **dropped pre-Volta support**, so stock images will not start. You need a Pascal-capable vLLM build (see below). |

The build fails immediately if the base has no importable `vllm`, so a wrong base is
caught at build time rather than at load time.

### Pascal notes

Getting vLLM itself onto sm_60 is a separate problem from PXQ4, and it is the harder
half. It needs, at minimum, a torch build with `6.0` in `TORCH_CUDA_ARCH_LIST`, a vLLM
compiled with `TORCH_CUDA_ARCH_LIST="6.0"`, and replacements for the kernels that assume
Volta-or-newer (notably the attention path — Pascal has no tensor cores and no `bf16`).

If you have solved that, point `BASE_IMAGE` at your image and PXQ4 will layer onto it.
The plugin ships torch-version shims (`sitecustomize.py`) for bases built against
torch 2.7 where the vLLM code expects torch 2.10 APIs.

---

## 2. Build

```bash
git clone https://github.com/poisonxa16/pxq_llama.cpp && cd pxq_llama.cpp

docker build -f docker/vllm-pxq4/Dockerfile \
  --build-arg BASE_IMAGE=vllm/vllm-openai:latest \
  -t pxq4-vllm:local .
```

## 3. Run

```bash
docker run --rm --runtime=nvidia --gpus '"device=0,1"' \
  -v /path/to/models:/models \
  -p 8000:8000 \
  pxq4-vllm:local \
    --model /models/your-pxq4-model \
    --quantization pxq4 \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 --port 8000
```

Or use the wrapper, which only asks which cards you have:

```bash
./docker/vllm-pxq4/pxq4-serve.sh --cards 0,1 --model /path/to/models/your-pxq4-model
```

Everything after the known flags is passed straight through to vLLM.

---

## How the kernel is chosen

You do not pick one. The entrypoint reads the compute capability of the visible GPUs and
selects:

| Compute capability | Kernel |
|---|---|
| 6.0, 6.1 (Pascal) | `libpxq4_sm60_v10.so` |
| 7.0, 7.2 (Volta)  | `libpxq4_sm70_v10.so` |

Two rules it will not bend:

- **Mixed architectures abort.** If the visible GPUs report more than one compute
  capability, one kernel library cannot serve them, so it stops and tells you to restrict
  the container to one architecture. It does not silently pick one and emit garbage from
  the other cards.
- **Unsupported architectures abort.** PXQ4 kernels exist for Pascal and Volta. On newer
  cards, use a standard vLLM quantization — they are better served by it anyway.

Override with `PXQ4_LIB=/opt/pxq4/kernels/<file>` to pin a specific revision.

Available kernels: four sm_60 revisions (`v8`–`v11`) and two sm_70 (`v9`, `v10`).
`v10` is the default for both and is the revision the published numbers were measured on.

---

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `PXQ4_LIB` | auto-detected | Kernel library. Set to pin a revision. |
| `PXQ4_ROOT` | `/opt/pxq4` | Where the kernels and plugin live in the image. |
| `PXQ4_MMV_SPLIT_MAX_BLOCKS` | unset | Splits the `gate_up` mono-kernel. `300` measured +5.6% decode on 2× V100. |
| `PXA_SDPA_TILED` | enabled | Tiled prefill SDPA. `0` disables. |

---

## Verifying it loaded

```bash
docker run --rm --runtime=nvidia --gpus '"device=0"' pxq4-vllm:local bash -c \
  'python3 -c "import pxq4_vllm; pxq4_vllm.register(); \
   from vllm.model_executor.layers.quantization import get_quantization_config; \
   print(get_quantization_config(\"pxq4\"))"'
```

On startup the server logs the two lines that matter:

```
pxq4: kernel  /opt/pxq4/kernels/libpxq4_sm60_v10.so
pxq4: plugin  /opt/pxq4/site
```

If you do not see them, the entrypoint was overridden and PXQ4 is not active.

---

## Troubleshooting

**`no PXQ4 kernel for compute capability X`** — your GPU is not Pascal or Volta. PXQ4
kernels are written specifically for those two.

**`visible GPUs report more than one compute capability`** — restrict the container to
one architecture with `--gpus '"device=0,1"'` or `NVIDIA_VISIBLE_DEVICES`.

**`BASE_IMAGE does not provide an importable vllm`** — the base is not a vLLM image, or
its vLLM is broken. Check with
`docker run --rm <base> python3 -c "import vllm"`.

**Server starts but the model fails to load on a P100** — almost always the base image,
not PXQ4. Confirm the base can serve an unquantized model on that card first; that
isolates vLLM-on-Pascal from PXQ4.

**`No module named 'triton.language.target_info'`** — comes from the base image's Triton,
not from PXQ4. It is logged as an error and execution continues; PXQ4 does not use those
Triton kernels.

---

## What is in the image

```
/opt/pxq4/kernels/     six prebuilt kernel libraries (4× sm_60, 2× sm_70)
/opt/pxq4/site/        the pxq4_vllm plugin tree, placed on PYTHONPATH
/usr/local/bin/pxq4-entrypoint
```

The plugin registers through the standard vLLM entry point
`vllm.general_plugins: pxq4 = pxq4_vllm:register`, which is how `--quantization pxq4`
becomes available. It is on `PYTHONPATH` rather than pip-installed into the base's
site-packages, so the same overlay works across vLLM versions and is removed by unsetting
one variable.

Kernel sources are in [`tools/vllm-pxq4/src/`](../../tools/vllm-pxq4/src); rebuild
instructions and per-library checksums are in
[`pxa/pxq4/README.md`](../../pxa/pxq4/README.md) and `pxa/pxq4/MANIFEST.md`.

Measured serving numbers: [`docs/PXA-SM70-SERVING.md`](../../docs/PXA-SM70-SERVING.md) and
[`docs/PXA-SM60-SERVING.md`](../../docs/PXA-SM60-SERVING.md).
