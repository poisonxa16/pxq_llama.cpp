# Build and use pxq_llama with CPU or CPU+CUDA

Built on top of [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and [llama-swap](https://github.com/mostlygeek/llama-swap)

Commands are provided for Podman and Docker.

CPU or CUDA sections under [Prebuilt](#Prebuilt)/[Build](#Build) and [Run](#Run) are enough to get up and running.

## Overview

- [Prebuilt](#Prebuilt)
- [Build](#Build)
- [Run](#Run)
- [Troubleshooting](#Troubleshooting)
- [Extra Features](#Extra)
- [Credits](#Credits)

## Prebuilt binaries and images

**Engine binaries** — `llama-cli`, `llama-server`, `llama-quantize` for Linux x86-64,
CUDA 12, built for Pascal (sm_60) and Volta (sm_70), with the bundled libraries they
need. Attached to the release:

<https://github.com/poisonxa16/pxq_llama.cpp/releases/latest>

```bash
tar xzf pxq_llama-*-linux-x64-cuda12.tar.gz
cd pxq_llama-*-linux-x64-cuda12
./llama-cli --version
```

Requires an NVIDIA driver and the CUDA 12 runtime (`libcudart.so.12`, `libcublas.so.12`,
`libcublasLt.so.12`). Everything else is bundled.

**vLLM images with the PXQ4 backend** — for serving PXQ4 models through vLLM:

```bash
docker pull ghcr.io/poisonxa16/pxa-vllm:sm60   # Pascal: P100, GTX 10xx
docker pull ghcr.io/poisonxa16/pxa-vllm:sm70   # Volta:  V100, Titan V
```

See [`vllm-pxq4/README.md`](vllm-pxq4/README.md).

There is no prebuilt image of the llama.cpp engine itself yet. Build one with the
Containerfiles in this directory, as below.

## Build

The project uses Docker Bake for building multiple targets efficiently.

Clone the repository: `git clone https://github.com/poisonxa16/pxq_llama`

Use `docker-bake`.

```bash
docker buildx create --name pxq-llama-builder --use
```

### CPU Variant

```bash
VARIANT=cpu docker buildx bake --builder pxq-llama-builder --load full swap
```

Or with custom tags:

```bash
REPO_OWNER=yourname VARIANT=cpu docker buildx bake --builder pxq-llama-builder --load \
  -f ./docker-bake.hcl \
  full swap
```

### CUDA Variant

Set the CUDA version and GPU architecture as bake variables (they are build
args on `docker/pxq_llama-cuda.Containerfile`):
- `CUDA_DOCKER_ARCH`: your GPU's compute capability (e.g., `86` for RTX 30*, `89` for RTX 40*, `120` for RTX 50*)
- `CUDA_VERSION`: CUDA Toolkit version (e.g., `12.6.2`, `13.1.1`)

```bash
VARIANT=cu12 CUDA_VERSION=12.6.2 CUDA_DOCKER_ARCH=86 \
  docker buildx bake --builder pxq-llama-builder --load full swap
```

Any `VARIANT` beginning with `cu` selects the CUDA Containerfile; anything else
selects the CPU one. Set `CONTAINERFILE=<path>` to override that choice.

### Build Targets

Three targets per variant:

- **`server`**: `llama-server` and the shared libraries only - the smallest image.
- **`full`**: adds `llama-quantize`, the other `llama-*` utilities and the Python conversion scripts.
- **`swap`**: `llama-swap` in front of `llama-server`.

## Run

- Download `.gguf` model files to your favorite directory (e.g., `/my_local_files/gguf`).
- Map it to `/models` inside the container.
- Open browser `http://localhost:9292` and enjoy the features.
- API endpoints are available at `http://localhost:9292/v1` for use in other applications.

### CPU

```bash
podman run -it --name pxq_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/pxq_llama-cpu:swap
```

```bash
docker run -it --name pxq_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro localhost/pxq_llama-cpu:swap
```

### CUDA

- Install Nvidia Drivers and CUDA on the host.
- For Docker, install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For Podman, install [CDI Container Device Interface](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)
- Identify your GPU:
  - [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda/gpus) (e.g., `8.6` for RTX30*, `8.9` for RTX40*, `12.0` for RTX50*)
  - [CUDA Toolkit supported version](https://developer.nvidia.com/cuda-toolkit-archive)

```bash
podman run -it --name pxq_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --device nvidia.com/gpu=all --security-opt=label=disable localhost/pxq_llama-cuda:swap
```

```bash
docker run -it --name pxq_llama --rm -p 9292:8080 -v /my_local_files/gguf:/models:ro --runtime nvidia localhost/pxq_llama-cuda:swap
```

## Troubleshooting

- If CUDA is not available, use the `pxq_llama-cpu` image instead.
- If models are not found, ensure you mount the correct directory: `-v /my_local_files/gguf:/models:ro`
- If you need to install `podman` or `docker` follow the [Podman Installation](https://podman.io/docs/installation) or [Install Docker Engine](https://docs.docker.com/engine/install) for your OS.

## Extra

- **Custom commit**: the build copies the working tree (`COPY . /app`), so check out the commit
you want to build before running bake. Use `BUILD_NUMBER` to tag the resulting images:

```bash
BUILD_NUMBER=396 VARIANT=cpu docker buildx bake --builder pxq-llama-builder --load full swap
```

- **Using the tools in the `full` image**:

```bash
$ podman run -it --name pxq_llama_full --rm -v /my_local_files/gguf:/models:ro --entrypoint bash localhost/pxq_llama-cpu:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

```bash
docker run -it --name pxq_llama_full --rm -v /my_local_files/gguf:/models:ro --runtime nvidia --entrypoint bash localhost/pxq_llama-cuda:full
# ./llama-quantize ...
# python3 gguf-py/scripts/gguf_dump.py ...
# ./llama-perplexity ...
# ./llama-sweep-bench ...
```

- **Customize `llama-swap` config**: Save the `./docker/pxq_llama-cpu-swap.config.yaml` or `./docker/pxq_llama-cuda-swap.config.yaml` locally (e.g., under `/my_local_files/`) then map it to `/app/config.yaml` inside the container appending `-v /my_local_files/pxq_llama-cpu-swap.config.yaml:/app/config.yaml:ro` to your `podman run ...` or `docker run ...`.

- **Run in background**: Replace `-it` with `-d`: `podman run -d ...` or `docker run -d ...`. To stop it: `podman stop pxq_llama` or `docker stop pxq_llama`.

- **GGML_NATIVE**: If you build the image on a different machine, change `-DGGML_NATIVE=ON` to `-DGGML_NATIVE=OFF` in the `.Containerfile`.

- **KV quantization types**: To use more KV quantization types, build with `-DGGML_IQK_FA_ALL_QUANTS=ON`.

- **Cleanup unused CUDA images**: If you experiment with several `CUDA_VERSION`, delete unused images (they are several GB):
  ```bash
  podman image rm docker.io/nvidia/cuda:12.4.0-runtime-ubuntu22.04 && \
    podman image rm docker.io/nvidia/cuda:12.4.0-devel-ubuntu22.04
  ```

- **Build without `llama-swap`**: Change `--target swap` to `--target server` in docker-bake or Containerfiles.

- **Pre-made quants**: Look for premade quants from [ubergarm](https://huggingface.co/ubergarm/models).

- **GGUF tools**: Build custom quants with [Thireus](https://github.com/Thireus/GGUF-Tool-Suite)'s tools.

- **Download prebuilt binaries**: prebuilt `pxq_llama` binaries are attached to each release on the [releases page](https://github.com/poisonxa16/pxq_llama/releases).

- **KoboldCPP experience**: [Croco.Cpp is a fork of KoboldCPP inferring GGUF/GGML models on CPU/Cuda with KoboldAI's UI. It's powered partly by IK_LLama.cpp, and compatible with most of Ikawrakow's quants except Bitnet.](https://github.com/Nexesenex/croco.cpp)

## Credits

All credits to the awesome community:

[llama-swap](https://github.com/mostlygeek/llama-swap)
