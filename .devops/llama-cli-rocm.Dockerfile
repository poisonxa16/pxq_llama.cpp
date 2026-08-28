ARG UBUNTU_VERSION=22.04

# This needs to generally match the container host's environment.
ARG ROCM_VERSION=5.6

# Target the ROCm build image
ARG BASE_ROCM_DEV_CONTAINER=rocm/dev-ubuntu-${UBUNTU_VERSION}:${ROCM_VERSION}-complete

FROM ${BASE_ROCM_DEV_CONTAINER} AS build

# Unless otherwise specified, we make a fat build.
# List from https://github.com/ggerganov/llama.cpp/pull/1087#issuecomment-1682807878
# This is mostly tied to rocBLAS supported archs. Semicolon separated: the value is
# passed straight to CMake as AMDGPU_TARGETS.
ARG ROCM_DOCKER_ARCH="gfx803;gfx900;gfx906;gfx908;gfx90a;gfx1010;gfx1030;gfx1100;gfx1101;gfx1102"

RUN apt-get update && \
    apt-get install -y build-essential cmake git libcurl4-openssl-dev curl python3 python3-pip

COPY requirements.txt   requirements.txt
COPY requirements       requirements

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

WORKDIR /app

COPY . .

ENV CC=/opt/rocm/llvm/bin/clang
ENV CXX=/opt/rocm/llvm/bin/clang++

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
        -DGGML_HIPBLAS=ON \
        -DAMDGPU_TARGETS="${ROCM_DOCKER_ARCH}" && \
    cmake --build build --config Release --target llama-cli -j$(nproc) && \
    cp build/bin/llama-cli /app/llama-cli && \
    find build -name "*.so" -exec cp {} /app/ \;

ENV LD_LIBRARY_PATH=/app

ENTRYPOINT [ "/app/llama-cli" ]
