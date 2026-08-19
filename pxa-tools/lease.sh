#!/usr/bin/env bash
# GPU lease for DGX cards 4-7 (ours; 0-3 are Kewaii's - NEVER touch).
#   lease.sh acquire <name> <max_minutes>   - blocks until free, then holds
#   lease.sh release <name>
#   lease.sh status
L=/mnt/models/pxa-lease/holder
case "${1:-status}" in
  acquire)
    who="${2:?name}"; mx="${3:-120}"
    for i in $(seq 1 720); do
      if mkdir /mnt/models/pxa-lease/lock 2>/dev/null; then
        echo "$who $(date -u +%s) $mx" > $L; echo "ACQUIRED by $who"; exit 0
      fi
      if [ -f "$L" ]; then
        set -- $(cat $L); h=$1; t=$2; m=$3
        now=$(date -u +%s)
        if [ $(( (now - t) / 60 )) -gt "${m:-120}" ]; then
          echo "stale lease from $h ($(( (now-t)/60 ))min > ${m}min) - breaking"
          rm -rf /mnt/models/pxa-lease/lock; continue
        fi
      fi
      sleep 10
    done
    echo "TIMEOUT waiting for lease"; exit 1 ;;
  release)
    rm -rf /mnt/models/pxa-lease/lock; rm -f $L; echo "RELEASED by ${2:-?}" ;;
  status)
    if [ -d /mnt/models/pxa-lease/lock ]; then echo "HELD: $(cat $L 2>/dev/null)"; else echo "FREE"; fi ;;
esac
