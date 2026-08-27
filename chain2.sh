#!/bin/bash
# Fire the quantize+gate once the Q8_0-PLE conversion lands.
if [ "$(hostname)" != "the box" ]; then echo "WRONG HOST: $(hostname)"; exit 1; fi
BF16=<local-path>
while ps -eo args --no-headers | grep -q "[c]onvert_qwen4exp"; do sleep 60; done
echo "=== [$(date -u +%H:%M:%S)] converter exited ==="
sz=$(stat -c %s "$BF16" 2>/dev/null || echo 0)
echo "size: $(awk -v s=$sz 'BEGIN{printf "%.2f GiB", s/1073741824}')  (plan was 306 GB)"
if [ "$sz" -lt 280000000000 ]; then
  echo "LOOKS TRUNCATED -- refusing to quantize a partial model"
  tail -15 <local-path> | tr '\r' '\n' | tail -8
  exit 1
fi
echo "=== [$(date -u +%H:%M:%S)] starting pipeline3.sh ==="
exec <local-path>
