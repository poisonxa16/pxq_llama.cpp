#!/usr/bin/env bash
# PXQ4 on 4x V100 (sm_70), tensor-parallel 4, cards 4-7.
#
# These flags are the measured-best configuration, not a guess. Every one of them
# was A/B'd against AWQ W4A16 on the same cards with the same client. Notes on the
# non-obvious ones are inline - please read them before changing anything, because
# three of these look like they should be improvements and are not.
set -euo pipefail

MODEL=${MODEL:-/mnt/models/pxa-models/Qwen3.8-27B-PXQ4-vllm-p2a-nf}
PORT=${PORT:-8421}
NAME=${NAME:-pxq4-27b}

docker run -d --rm --name pxq4-serve \
  --gpus '"device=4,5,6,7"' \
  --shm-size 16g \
  --tmpfs /root/home:exec,size=24g \
  -v /mnt/models:/mnt/models \
  -p ${PORT}:${PORT} \
  -e HOME=/root/home -e TMPDIR=/root/home \
  \
  `# --- the PXQ4 plugin. No pip install: the container root filesystem is full, so` \
  `# the plugin is registered by PYTHONPATH plus a hand-written entry_points.txt. ---` \
  -e PYTHONPATH=/mnt/models/pxa-int-v6/site \
  \
  `# --- sm_70 attention. The stock backend does not support V100. ---` \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_SM70_FLASH_ATTN_V100=1 \
  \
  `# --- MTP defaults off. The auto-applied SM70 MTP defaults set max_num_seqs=4,` \
  `# which contradicts the engine's own requirement of max_num_seqs=1 for the` \
  `# dynamic GPU LRU path, and the boot then fails on its own assertion. ---` \
  -e VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS=0 \
  \
  -e PYTHONUNBUFFERED=1 \
  --entrypoint python \
  kewaii/vllm:latest \
  -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization pxq4 \
    --attention-backend FLASH_ATTN_V100 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 --port ${PORT} \
    --served-model-name "$NAME" \
    --trust-remote-code \
    --gpu-memory-utilization 0.93 \
    --max-model-len 200000 \
    --enable-prefix-caching

echo "starting; first boot takes 13-18 min (torch.compile + graph capture)."
echo "  health:  curl -s localhost:${PORT}/health"
echo "  logs:    docker logs -f pxq4-serve"
