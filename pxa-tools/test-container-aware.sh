#!/bin/bash
# Verification for PXA_CONTAINER_AWARE_v1 / PXA_PORT_GUARD_v1 / PXA_UTF8_FINAL_v1 /
# PXA_TYPEARG_STRICT_v1 / PXA_BT_NOFORK_v1 — CPU-only, no brain GPUs touched.
# Run INSIDE the build container: docker run --rm --runtime=nvidia --network host \
#   -v .:/src -v ./models:/models ...
set -u
B=/src/build-unified/bin
export LD_LIBRARY_PATH=/src/build-unified/bin:/src/build-unified/src:/src/build-unified/ggml/src:/src/build-unified/examples/mtmd
MODEL=/models/draft/Qwen3-0.6B-Q8_0.gguf
PASS=0; FAIL=0
ok()   { echo "PASS: $1"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== T1: container detection (inside docker => container) ==="
OUT=$($B/llama-server -m /nonexistent-model.gguf --port 18999 2>&1 | head -30)
echo "$OUT" | grep -q "PXA_CONTAINER_AWARE_v1: runtime=container" && ok "T1 verdict container" || { bad "T1"; echo "$OUT" | head -5; }

echo "=== T2: PXA_IN_CONTAINER=0 override => bare-metal ==="
OUT=$(PXA_IN_CONTAINER=0 $B/llama-server -m /nonexistent-model.gguf --port 18999 2>&1 | head -30)
echo "$OUT" | grep -q "runtime=bare-metal (PXA_IN_CONTAINER=0 (operator override))" && ok "T2 override bare" || { bad "T2"; echo "$OUT" | head -5; }

echo "=== T3: PXA_IN_CONTAINER=1 override => container ==="
OUT=$(PXA_IN_CONTAINER=1 $B/llama-server -m /nonexistent-model.gguf --port 18999 2>&1 | head -30)
echo "$OUT" | grep -q "runtime=container (PXA_IN_CONTAINER=1 (operator override))" && ok "T3 override container" || { bad "T3"; }

echo "=== T4: port guard refuses a live listener ==="
python3 -m http.server 18997 --bind 127.0.0.1 >/dev/null 2>&1 &
HTTPPID=$!
sleep 1
OUT=$($B/llama-server -m /nonexistent-model.gguf --host 127.0.0.1 --port 18997 2>&1 | head -40)
echo "$OUT" | grep -q "PXA_PORT_GUARD_v1: a LIVE listener already answers" && ok "T4 guard refuses" || { bad "T4"; echo "$OUT" | head -8; }

echo "=== T5: PXA_PORT_GUARD=0 bypasses (reaches bind failure or model load) ==="
OUT=$(PXA_PORT_GUARD=0 $B/llama-server -m /nonexistent-model.gguf --host 127.0.0.1 --port 18997 2>&1 | head -40)
if echo "$OUT" | grep -q "PXA_PORT_GUARD_v1"; then bad "T5 guard should be off"; else ok "T5 bypass"; fi
kill $HTTPPID 2>/dev/null

echo "=== T6: quantize strict type arg — bad name FAILS LOUD ==="
OUT=$($B/llama-quantize --token-embedding-type q6_kk /nonexistent-in.gguf /tmp/out.gguf Q8_0 2>&1); RC=$?
if [ $RC -ne 0 ] && echo "$OUT" | grep -q "invalid/unknown ggml type name 'q6_kk'"; then ok "T6 bad type loud-fails rc=$RC"; else bad "T6 rc=$RC"; echo "$OUT" | head -5; fi

echo "=== T7: quantize case-fold — q6_k now parses (proceeds past the flag) ==="
OUT=$($B/llama-quantize --token-embedding-type q6_k /nonexistent-in.gguf /tmp/out.gguf Q8_0 2>&1); RC=$?
if echo "$OUT" | grep -q "invalid/unknown ggml type name"; then bad "T7 q6_k rejected"; else ok "T7 q6_k accepted (fails later on missing file, rc=$RC)"; fi

echo "=== T8: ggml_abort with no debugger: no orphan child, prompt termination ==="
cat > /tmp/abt.c <<'CEOF'
extern void ggml_abort(const char *file, int line, const char *fmt, ...);
int main(void) { ggml_abort("abt.c", 1, "deliberate test abort"); return 0; }
CEOF
gcc /tmp/abt.c -o /tmp/abt -L/src/build-unified/ggml/src -lggml 2>/tmp/abt-cc.log || gcc /tmp/abt.c -o /tmp/abt -L/src/build-unified/ggml/src -lggml -lggml-base 2>>/tmp/abt-cc.log
if [ -x /tmp/abt ]; then
  ( PATH=/usr/bin-nonexistent timeout 30 /tmp/abt >/tmp/abt.out 2>&1; echo "abt rc=$?" )
  sleep 1
  ORPHANS=$(ps -eo pid,ppid,args | grep "/tmp/abt" | grep -v grep | wc -l)
  [ "$ORPHANS" -eq 0 ] && ok "T8 no orphan after abort" || { bad "T8 orphans=$ORPHANS"; ps -eo pid,ppid,args | grep abt | grep -v grep; }
else
  echo "SKIP T8: could not link test binary"; cat /tmp/abt-cc.log
fi

echo "=== T9: CPU server + UTF-8 final truncation + wedge armed line ==="
if [ -f "$MODEL" ]; then
  PXA_WEDGE_EXIT_MS=120000 $B/llama-server -m $MODEL --host 127.0.0.1 --port 18996 -ngl 0 -c 2048 -np 1 > /tmp/utf8srv.log 2>&1 &
  SRV=$!
  for i in $(seq 1 60); do curl -sf http://127.0.0.1:18996/health >/dev/null 2>&1 && break; sleep 1; done
  if curl -sf http://127.0.0.1:18996/health >/dev/null 2>&1; then
    grep -q "PXA_WEDGE_EXIT_v1 armed" /tmp/utf8srv.log && ok "T9a wedge armed line present" || bad "T9a no armed line"
    grep -q "PXA_CONTAINER_AWARE_v1" /tmp/utf8srv.log && ok "T9b verdict logged on real server" || bad "T9b"
    # force a mid-codepoint cut: n_predict sweep over an emoji-continuation prompt
    HIT=""
    for NP in 1 2 3 4 5 6; do
      R=$(curl -s http://127.0.0.1:18996/completion -d "{\"prompt\":\"Repeat this emoji forever: \\ud83d\\ude00\\ud83d\\ude00\\ud83d\\ude00\",\"n_predict\":$NP,\"temperature\":0}")
      echo "$R" | python3 -c "import sys,json; json.load(sys.stdin)" 2>/dev/null || { bad "T9c response not valid JSON at n_predict=$NP"; HIT=fail; break; }
      if grep -q "dropping incomplete trailing UTF-8" /tmp/utf8srv.log; then HIT=yes; break; fi
    done
    if [ "$HIT" = "yes" ]; then ok "T9c mid-codepoint cut handled (truncation warning fired, response valid JSON)";
    elif [ "$HIT" = "fail" ]; then :;
    else echo "NOTE T9c: no mid-codepoint cut produced in sweep (all responses valid JSON) — fix path not exercised"; fi
    # 500-regression probe: the canonical reproducer class (ignore_eos runaway)
    R=$(curl -s http://127.0.0.1:18996/completion -d '{"prompt":"Explain the determinant of a 4x4 matrix.","n_predict":128,"temperature":1.0,"seed":42,"ignore_eos":true}')
    echo "$R" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'content' in d" 2>/dev/null && ok "T9d ignore_eos runaway returns valid JSON" || { bad "T9d"; echo "$R" | head -c 300; }
  else
    bad "T9 server never became healthy"; tail -5 /tmp/utf8srv.log
  fi
  kill $SRV 2>/dev/null; sleep 1; kill -9 $SRV 2>/dev/null
else
  echo "SKIP T9: model not mounted"
fi

echo "=== SUMMARY: PASS=$PASS FAIL=$FAIL ==="
