#!/bin/bash
# Wait for the conversion to finish, then run the corrected continuation.
# pipeline.sh is expected to exit non-zero at its BF16 GPU gate (that gate can
# never pass -- see pipeline2.sh); the conversion itself is what matters.
if [ "$(hostname)" != "the box" ]; then echo "WRONG HOST: $(hostname)"; exit 1; fi
BF16=<local-path>

while ps -eo args --no-headers | grep -q "[c]onvert_qwen4exp"; do sleep 60; done
echo "=== [$(date -u +%H:%M:%S)] converter exited ==="
sz=$(stat -c %s "$BF16" 2>/dev/null || echo 0)
echo "BF16 size: $(awk -v s=$sz 'BEGIN{printf "%.2f GiB", s/1073741824}')"

# The writer reports a 354 GB plan; refuse to continue on a short file rather
# than quantize a truncated model.
if [ "$sz" -lt 330000000000 ]; then
  echo "BF16 LOOKS TRUNCATED -- refusing to continue"
  tail -20 <local-path>
  exit 1
fi
# let pipeline.sh finish its own (failing) gate step before starting
while ps -eo args --no-headers | grep -q "[p]ipeline.sh"; do sleep 20; done
echo "=== [$(date -u +%H:%M:%S)] starting pipeline2.sh ==="
exec <local-path>
