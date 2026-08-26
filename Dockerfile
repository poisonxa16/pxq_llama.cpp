FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

# =========================================================================================
# ONE DOCKERFILE, TWO IMAGES — both built from THIS tree. No third-party image anywhere.
#
#   scripts/build-images.sh sm60      -> pxa-vllm:sm60   (Tesla P100, Pascal)
#   scripts/build-images.sh sm70      -> pxa-vllm:sm70   (Tesla V100, Volta)
#   scripts/build-images.sh both
#
# Use the script; it sets the six ARGs below as a matched set. Setting them by hand is
# possible and is how you get a mismatched pair.
#
# WHY TWO IMAGES AND NOT ONE FAT ONE. torch 2.7.1 is the last torch whose wheels carry
# sm_60 cubins, so Pascal support forces that version. The fork's sm_70 paths, however,
# were written against torch 2.10, and running them on 2.7.1 fails for reasons that have
# nothing to do with Volta:
#   * VLLM_SKIP_C_STABLE=1 (needed to build against 2.7.1) drops csrc/libtorch_stable/,
#     which is where sm70_gemma_long_prefill_fused_add_rms_norm lives
#   * torch._dynamo.aot_compile does not exist before 2.10
#   * torch 2.7.1's dynamo cannot handle PEP 604 (X | None) annotations on traced fns
# NONE of those exist on a 2.10 build. Volta never needed the downgrade — only Pascal did
# — so the sm70 variant simply does not take it, and the whole class of bug disappears
# rather than being patched around one boot at a time.
# =========================================================================================
ARG VARIANT=sm60
ARG TORCH_SPEC="torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cu126"
ARG TORCH_ASSERT="2.7.1+cu126"
ARG ARCH_LIST="6.0;7.0"
ARG SKIP_C_STABLE="1"
ARG EXTRA_CMAKE="-DVLLM_TORCH27_COMPAT=ON"

ENV DEBIAN_FRONTEND=noninteractive

# CUDA/compiler environment
ENV CUDA_ARCH=sm_70
ENV CC=gcc-12
ENV CXX=g++-12

# vLLM / V100 configuration
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ENV VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0
# Both Pascal (P100) and Volta (V100) cubins in ONE image. This env var at
# CONFIGURE time is the knob that works: torch's cmake silently discards
# CMAKE_CUDA_ARCHITECTURES and substitutes its own list. Verified at the end of
# this Dockerfile with cuobjdump --list-elf (the .so filename proves nothing).
ARG ARCH_LIST
# sm60 image carries BOTH (a Pascal host often also holds a Volta card).
# sm70 image is 7.0 only.
ENV TORCH_CUDA_ARCH_LIST="${ARCH_LIST}"
ENV VLLM_SM70_QUANT_BACKEND=marlin
ENV PYTHONUNBUFFERED=1

# Install Python 3.12, GCC 12 and build tools
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        wget \
        ca-certificates
RUN add-apt-repository "ppa:deadsnakes/ppa"
RUN apt-get install -y \
        git \
        cmake \
        ninja-build \
        python3-pip \
        gcc-12 \
        g++-12 \
        protobuf-compiler \
        python3.12 \
        python3.12-venv \
        python3.12-dev \
        && \
    rm -rf /var/lib/apt/lists/*

# Make GCC 12 the default compiler
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120

# Rust toolchain, required by setuptools-rust (vllm-rs frontend binary)
ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH
RUN wget --quiet --output-document=/tmp/rustup-init https://sh.rustup.rs && \
    sh /tmp/rustup-init -y --profile minimal --default-toolchain stable && \
    rm /tmp/rustup-init

RUN python3.12 -m venv /opt/vllm-venv

# Put the venv, cargo and CUDA toolkit first in PATH (README "Source Build")
ENV PATH="/opt/vllm-venv/bin:/usr/local/cargo/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda-12.8 \
    LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# Packaging tools
RUN python -m pip install --upgrade --no-cache-dir \
        pip \
        setuptools \
        wheel

# Torch 2.7.1+cu126: the LAST torch whose official wheels ship sm_60 (Pascal)
# cubins (2.8+/cu128 wheels are sm_70-and-up: a 2.10-based image passes every
# extension gate and still dies on a P100 with "no kernel image is available"
# from torch's OWN kernels — observed 2026-08-24). The whole image therefore
# runs on 2.7.1; the tree builds against it via -DVLLM_TORCH27_COMPAT=ON +
# VLLM_SKIP_C_STABLE=1 (the pascal_compat shims exist for exactly this).
# The constraints file keeps every later pip resolve from upgrading torch back.
ARG TORCH_SPEC
ARG TORCH_INDEX
ARG TORCH_ASSERT
ARG VARIANT
ENV TORCH_ASSERT=${TORCH_ASSERT}
ENV VARIANT=${VARIANT}
RUN python -m pip install --no-cache-dir ${TORCH_SPEC} --index-url ${TORCH_INDEX} && \
    if [ "$VARIANT" = "sm60" ]; then \
        printf 'torch==2.7.1+cu126\ntorchvision==0.22.1+cu126\ntorchaudio==2.7.1+cu126\n' \
            > /tmp/torch-constraints.txt ; \
    else \
        : > /tmp/torch-constraints.txt ; \
    fi
# ^ the constraints file exists to stop later pip resolves upgrading torch back OFF the
#   Pascal-capable pin. The sm70 build wants the tree's own torch pins, so its constraints
#   file is deliberately EMPTY rather than absent - every later -c reference stays valid.

# Build from THIS checkout (docker build context = repo root), NOT from a clone
# of the 1CatAI org repo. The previous clone pulled a tree without the Pascal
# port, so the image silently built with zero sm_60 support and still succeeded.
# .git is included in the context so setuptools-scm derives the real version.
WORKDIR /opt/1Cat-vLLM
COPY . .
RUN test -d csrc/sm70_turbomind/lmdeploy && \
    test -d flash-attention-v100 && \
    test -d csrc/pascal_compat

# Install build dependencies (README "Install build dependencies")
# The tree pins torch==2.10.0 / torchaudio==2.10.0 / torchvision==0.25.0 directly.
# A constraints file cannot override a DIRECT requirement pin (ResolutionImpossible),
# so the torch-trio lines are stripped here and the constraints file governs the
# resolution instead — the same shape as the proven P100 venv (deps.log), which was
# assembled around those pins, not through them.
# (filtered copies live NEXT TO the originals: requirements/cuda.txt says
#  "-r common.txt", and pip resolves that relative to the requirements file's
#  own directory — a copy in /tmp breaks the include.)
# sm60 STRIPS the tree's direct torch pins (a constraints file cannot override a direct
# requirement pin - that is a ResolutionImpossible, not a downgrade). sm70 KEEPS them,
# because the tree pins exactly the torch that variant wants.
RUN if [ "$VARIANT" = "sm60" ]; then \
        sed -E "/^(torch|torchvision|torchaudio)[=<>!~ ]/d" requirements/build/cuda.txt > requirements/build/cuda-notorch.txt ; \
        sed -E "/^(torch|torchvision|torchaudio)[=<>!~ ]/d" requirements/cuda.txt > requirements/cuda-notorch.txt ; \
    else \
        cp requirements/build/cuda.txt requirements/build/cuda-notorch.txt ; \
        cp requirements/cuda.txt requirements/cuda-notorch.txt ; \
    fi && \
    python -m pip install --no-cache-dir -c /tmp/torch-constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r requirements/build/cuda-notorch.txt && \
    python -m pip install --no-cache-dir -c /tmp/torch-constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r requirements/cuda-notorch.txt && \
    python -m pip install --no-cache-dir -c /tmp/torch-constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu126 \
        -r requirements/common.txt && \
    python -m pip install --no-cache-dir cmake build && \
    python -c "import torch,os,sys; want=os.environ['TORCH_ASSERT']; got=torch.__version__; sys.exit('torch pin BROKEN: want %s got %s' % (want,got)) if not got.startswith(want) else print('torch pin held:', got)"

# Build the flash-attention-v100 and 1cat-vllm wheels (README "Build wheels")
# FLASH_ATTN_V100 stays 7.0-only: it is a wmma extension (sm_70+); on sm_60 the
# fork serves attention through the PASCAL_SDPA / pascal_decode_attn backends.
ARG SKIP_C_STABLE
ARG EXTRA_CMAKE
ENV FLASH_ATTN_V100_CUDA_ARCH_LIST=7.0 \
    MAX_JOBS=24 \
    NVCC_THREADS=1 \
    CMAKE_ARGS="${EXTRA_CMAKE}" \
    VLLM_SKIP_C_STABLE=${SKIP_C_STABLE}
# sm70 sets SKIP_C_STABLE=0, so csrc/libtorch_stable/ IS built and
# sm70_gemma_long_prefill_fused_add_rms_norm exists. That single difference removes the
# first and loudest V100 boot failure at its source instead of guarding around it.
RUN bash -c '\
    rm -rf build vllm.egg-info && \
    rm -rf .deps/*-build .deps/*-subbuild && \
    pushd flash-attention-v100 && \
    python -m build --wheel --no-isolation --skip-dependency-check --outdir ../dist-cu128-sm70 && \
    popd && \
    python -m build --wheel --no-isolation --skip-dependency-check --outdir dist-cu128-sm70 && \
    ls -l dist-cu128-sm70'
# ^ --skip-dependency-check: pyproject [build-system] requires torch==2.10.0, and this
#   image DELIBERATELY builds against the pinned 2.7.1+cu126 quadruple (the last torch
#   with sm_60 cubins). The check is advisory under --no-isolation; the real dependency
#   discipline lives in /tmp/torch-constraints.txt + the post-install version asserts.

# Install the built wheel
RUN WHEEL="$(find dist-cu128-sm70 -maxdepth 1 -type f -name '1cat_vllm-*.whl' -print -quit)" && \
    echo "Installing: $WHEEL" && \
    python -m pip install --no-deps --no-cache-dir "$WHEEL" && \
    python -c "import torch; assert torch.__version__.startswith('2.7.1'), torch.__version__; import vllm; print('import OK', vllm.__version__)"

# FAT-BUILD GATE: prove both cubins are actually in the shipped extensions.
# TORCH_CUDA_ARCH_LIST regressions are silent (the build succeeds, the .so is
# named the same, and the P100 dies at runtime with "no kernel image").
RUN set -e; \
    TORCH_SO=$(find /opt/vllm-venv/lib -path "*/torch/lib/libtorch_cuda.so" | head -1); \
    test -n "$TORCH_SO" || { echo "FAT GATE: libtorch_cuda.so not found"; exit 1; }; \
    cuobjdump --list-elf "$TORCH_SO" | grep -q "sm_60" \
      || { echo "FAT GATE FAILED: torch itself has no sm_60 cubins ($TORCH_SO) — a P100 dies on torch ops regardless of our extensions"; exit 1; }; \
    echo "FAT GATE OK: torch carries sm_60"; \
    for name in _C _moe_C; do \
      SO=$(find /opt/vllm-venv/lib -path "*/site-packages/vllm/*" -name "${name}*.so" | head -1); \
      test -n "$SO" || { echo "FAT GATE: $name .so not found"; exit 1; }; \
      cuobjdump --list-elf "$SO" > /tmp/elfs.txt; \
      grep -q "sm_60" /tmp/elfs.txt || { echo "FAT GATE FAILED: no sm_60 cubin in $SO"; exit 1; }; \
      grep -q "sm_70" /tmp/elfs.txt || { echo "FAT GATE FAILED: no sm_70 cubin in $SO"; exit 1; }; \
      echo "FAT GATE OK: $SO carries sm_60 + sm_70"; \
    done; \
    FA=$(find /opt/vllm-venv/lib -path "*flash_attn_v100*" -name "*.so" | head -1); \
    if [ -n "$FA" ]; then cuobjdump --list-elf "$FA" | grep -q "sm_70" \
      && echo "FAT GATE OK: flash_attn_v100 carries sm_70 (sm_70-only by design)" \
      || { echo "FAT GATE FAILED: no sm_70 cubin in $FA"; exit 1; }; fi

WORKDIR /mnt/models/vllm

# Verify toolchain
RUN gcc --version && \
    g++ --version && \
    python3.12 --version && \
    nvcc --version

CMD ["/bin/bash"]