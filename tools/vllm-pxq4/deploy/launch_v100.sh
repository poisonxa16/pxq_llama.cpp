#!/usr/bin/env bash
# $1 = container name, $2 = port; rest = extra api_server args
set -eu
NAME=$1; PORT=$2; shift 2
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
  --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=2,4 \
  --shm-size 16g --tmpfs /root/home:exec,size=24g \
  -e HOME=/root/home -e TMPDIR=/root/home \
  -e PYTHONPATH=/plugin/site -e PYTHONUNBUFFERED=1 \
  -e VLLM_SM70_FLASH_ATTN_V100=1 -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0 \
  -e PXQ4_MMV_SLICE_MAX=8 \
  ${EXTRA_ENV:-} \
  -v <local-path>:/models -v <local-path>:/plugin \
  -p 127.0.0.1:${PORT}:${PORT} --ipc=host \
  --entrypoint python kewaii/vllm:latest \
  -m vllm.entrypoints.openai.api_server \
    --model ${MODEL:-/models/qwen38-27b-unc-vllm-p2cf} \
    --quantization pxq4 --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size 2 --host 0.0.0.0 --port ${PORT} \
    --served-model-name m --trust-remote-code \
    --gpu-memory-utilization ${GMU:-0.94} --max-model-len ${MML:-32768} \
    --max-num-seqs ${MNS:-4} --enable-prefix-caching "$@"
