# PXQ4 for vLLM — overlay image.
#
# This does NOT rebuild vLLM. PXQ4 is a vLLM *plugin* (entry point
# `vllm.general_plugins: pxq4`) plus a prebuilt CUDA kernel library, so it layers onto
# an existing vLLM image in seconds and adds ~15 MB. Rebuilding vLLM from source to ship
# a plugin would cost hours and ~26 GB for no benefit.
#
# You supply the base. It must be a vLLM that actually runs on your cards:
#
#   Volta  (V100, sm_70)  — stock upstream vLLM images generally work.
#   Pascal (P100, sm_60)  — upstream vLLM dropped pre-Volta support, so a stock image
#                           will not start. You need a Pascal-capable vLLM build.
#                           See docker/vllm-pxq4/README.md for what that requires.
#
# Build:
#   docker build -f docker/vllm-pxq4/Dockerfile --build-arg BASE_IMAGE=<your-vllm-image> \
#                -t pxq4-vllm:local .
#
# The build refuses a base without a working `import vllm`, so a wrong base fails here
# rather than at 3am under load.

ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="PXQ4 for vLLM"
LABEL org.opencontainers.image.description="PXQ4 low-bit MoE quantization plugin and CUDA kernels for Pascal and Volta GPUs"
LABEL org.opencontainers.image.source="https://github.com/poisonxa16/pxq_llama.cpp"
LABEL org.opencontainers.image.licenses="Apache-2.0"

USER root

# The plugin tree and the prebuilt kernels. `sidecar/site-sm60` is the shared Python
# package for BOTH architectures despite the name -- only the kernel .so differs, and
# that is chosen at runtime from the actual compute capability.
COPY pxa/pxq4/kernels/            /opt/pxq4/kernels/
COPY pxa/pxq4/sidecar/site-sm60/  /opt/pxq4/site/
COPY docker/vllm-pxq4/pxq4-entrypoint.sh    /usr/local/bin/pxq4-entrypoint

RUN chmod +x /usr/local/bin/pxq4-entrypoint

# PYTHONPATH (not pip install) is deliberate: it keeps the plugin out of the base image's
# site-packages, so the same overlay works across vLLM versions and can be removed by
# unsetting one variable.
ENV PYTHONPATH=/opt/pxq4/site \
    PXQ4_ROOT=/opt/pxq4

# Fail the build now if the base is not a usable vLLM.
RUN python3 -c "import vllm, sys; print('base vLLM', vllm.__version__)" \
 || (echo "BASE_IMAGE does not provide an importable vllm -- see docker/vllm-pxq4/README.md" && exit 1)

ENTRYPOINT ["/usr/local/bin/pxq4-entrypoint"]
