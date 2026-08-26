#!/usr/bin/env bash
# thin-image-gate.sh — the gate for the SHIPPING SET: pxa-vllm:sm60 and pxa-vllm:sm70.
#
# The fat image is withdrawn (RELEASE-GATE 3.7). What ships is two thin images, each
# pinned to the torch its cards need. fat-image-smoke.sh gated the withdrawn artifact
# and is kept only as the record of how the sm_60 half was proven; it hardcodes a
# fat-image profile (SKIP_C_STABLE ops forced off, one model, one site) and cannot
# express the sm_70 image, whose whole reason to exist is that those ops are BUILT.
#
# WHAT A PASS MEANS, AND WHAT IT DOES NOT
#   Building an image verifies nothing. A /health 200 verifies nothing. What this
#   script is willing to call evidence:
#     - torch itself dispatches on the target card         (phase 1)
#     - the ops that justify this variant are actually in it (phase 2)
#     - a RAW, non-chat-templated 1-token prompt comes back non-garbage (phase 3)
#       -- the failure this catches is silent: chat completions look fine while short
#          raw prompts return "" or "!!!!" (ENGINE-VERDICT 2026-08-24)
#     - a decode rate measured on THIS image, not inherited from another one (phase 4)
#   Only after all four may an image's caps move from set() to {NN} in pxa-launch.py.
#
# CAPTURE MODE: FULL_DECODE_ONLY, deliberately. On the breakable-cudagraph stack ANY
# captured-prefill config corrupts prompts short enough to fit a captured graph. That is
# a serving-config defect, not an image defect, and the gate must test the config we
# actually ship. Do not drop the mode from this file: without it a PASSING battery does
# not prove short raw prompts.
#
# CARD 3 IS PRODUCTION. It carries the VLM + embedding seats the email archiver depends
# on. This script resolves every requested index to a GPU UUID and refuses to run if the
# card-3 UUID is in the set, then passes UUIDs -- not indices -- to the runtime, so no
# enumeration-order surprise can reach it.
#
# Usage:
#   ./scripts/thin-image-gate.sh sm60            # P100 pair, cards 0+6
#   ./scripts/thin-image-gate.sh sm70            # V100 pair, cards 2+4
#   CARDS=2 TP=1 ./scripts/thin-image-gate.sh sm70    # single-card run
#   PHASES=0,1,2 ./scripts/thin-image-gate.sh sm70    # preflight only, no serve
set -uo pipefail

# NOTE: no braces in the :? message. ${1:?word} ends the expansion at the FIRST }
# inside word, so a message containing {sm60|sm70} left a literal } appended to the
# value -- VARIANT became "sm70}" and every variant looked unknown.
VARIANT=${1:?usage: $0 sm60   OR   $0 sm70}
PROTECTED_UUID=${PROTECTED_UUID:-GPU-2e2e834b-604d-ae66-dae6-1dbbcf5506bf}   # the 1080 Ti
PORT=${PORT:-8433}
PHASES=${PHASES:-0,1,2,3,4,5}
RESULTS=${RESULTS:-/tmp/thin-gate-$VARIANT-$$}
mkdir -p "$RESULTS"

# ---- per-variant profile ---------------------------------------------------------
# The two variants differ in torch, in the plugin site, and in which sm_70 ops exist.
# Nothing here is a preference; each line is forced by what the hardware or the build
# can do. See scripts/build-images.sh for why the torch pins differ.
case "$VARIANT" in
  sm60)
    IMAGE=${IMAGE:-pxa-vllm:sm60}
    CARDS=${CARDS:-0,6}
    ATTN_BACKEND=${ATTN_BACKEND:-PASCAL_SDPA}
    MODEL=${MODEL:-/c/models/qwen38-27b-unc-vllm-p1f}
    SITE=${SITE:-/c/pxq4-sm60/site}
    PXQ4_LIB=${PXQ4_LIB-/c/pxq4-sm60/libpxq4_sm60.so}
    WANT_TORCH=2.7.1
    WANT_ARCH=60
    # torch 2.7.1 => VLLM_SKIP_C_STABLE=1 => csrc/libtorch_stable/ is NOT in this image,
    # so the one op that lives only there is absent and the unfused path must be used.
    FUSED_LONG_PREFILL=0
    ;;
  sm70)
    IMAGE=${IMAGE:-pxa-vllm:sm70}
    CARDS=${CARDS:-2,4}
    ATTN_BACKEND=${ATTN_BACKEND:-FLASH_ATTN_V100}
    MODEL=${MODEL:-/c/models/qwen38-27b-unc-vllm-p2cf}
    SITE=${SITE:-/c/moe-branch/site}
    PXQ4_LIB=${PXQ4_LIB-}          # empty: the site's bundled sm_70 .so loads
    WANT_TORCH=2.10
    WANT_ARCH=70
    # torch 2.10 builds libtorch_stable, so the fused op EXISTS here. Baseline still
    # runs with it OFF -- one variable at a time; phase 5 turns it on by itself.
    FUSED_LONG_PREFILL=0
    ;;
  *) echo "unknown variant: $VARIANT (want sm60 or sm70)" >&2; exit 2 ;;
esac
TP=${TP:-$(awk -F, '{print NF}' <<<"$CARDS")}
GMU=${GMU:-0.90}
NAME=thin-gate-$VARIANT-$$
phase_on() { [[ ",$PHASES," == *",$1,"* ]]; }
say() { printf '\n== %s\n' "$*"; }
FAILED=()

# ---- phase 0: preflight ----------------------------------------------------------
if phase_on 0; then
say "phase 0: preflight"
[ "$(hostname)" = "the box" ] || { echo "WRONG HOST: $(hostname) -- this gate touches GPUs; ABORT"; exit 1; }

# CARD SAFETY FIRST, before anything else can exit early. This check previously sat
# behind the image-exists check, so a run against an unbuilt image never reached it and
# the guard was never actually exercised. Refusing to touch the production card is the
# one thing that must not be contingent on any other check passing.
UUIDS=""
for idx in ${CARDS//,/ }; do
  u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$idx" 2>/dev/null | tr -d ' ')
  [ -n "$u" ] || { echo "FAIL: no GPU at index $idx"; exit 1; }
  [ "$u" = "$PROTECTED_UUID" ] && { echo "REFUSING: index $idx is the PROTECTED card ($u)."; echo "  It serves the production VLM + embedding seats. This gate will not touch it."; exit 1; }
  n=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$idx")
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$idx")
  echo "  card $idx    $n  ${used} MiB in use  $u"
  [ "$used" -gt 512 ] && echo "  WARNING: card $idx is not idle (${used} MiB); a co-tenant makes phase 4 meaningless"
  UUIDS="${UUIDS:+$UUIDS,}$u"
done

docker image inspect "$IMAGE" >/dev/null 2>&1 \
  || { echo "FAIL: image $IMAGE does not exist. Build it first: ./scripts/build-images.sh $VARIANT"; exit 1; }
echo "  image     $IMAGE  ($(docker images --format '{{.Size}}' "$IMAGE" | head -1))"
echo "  tp        $TP   gmu $GMU   backend $ATTN_BACKEND"

hostmodel=${MODEL/#\/c/<local-path>}
[ -d "$hostmodel" ] || { echo "FAIL: model $hostmodel not present on host"; exit 1; }
hostsite=${SITE/#\/c/<local-path>}
[ -d "$hostsite" ] || { echo "FAIL: plugin site $hostsite not present on host"; exit 1; }
echo "  model     $MODEL"
echo "  site      $SITE   lib=${PXQ4_LIB:-<bundled>}"
fi
: "${UUIDS:=$CARDS}"

# ---- phase 1: torch dispatches ---------------------------------------------------
# 5 seconds, and it is the single highest-value check in the file: on 2026-08-24 an
# image passed every byte-level cubin check in the Dockerfile and died on a P100
# inside torch's OWN kernels, because torch 2.10/cu128 wheels ship no sm_60.
if phase_on 1; then
say "phase 1: torch dispatches on the target card(s)"
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$UUIDS" --entrypoint python "$IMAGE" -c '
import torch
x = torch.ones(8, device="cuda"); assert float(x.sum()) == 8.0
y = (torch.randn(256,256,device="cuda",dtype=torch.float16) @ torch.randn(256,256,device="cuda",dtype=torch.float16))
assert torch.isfinite(y).all(), "fp16 matmul produced non-finite values"
cc = torch.cuda.get_device_capability(0)
print("  torch", torch.__version__, "|", torch.cuda.get_device_name(0), "| sm_%d%d" % cc, "| devices", torch.cuda.device_count())
' 2>&1 | tee "$RESULTS/phase1.log" \
  || { echo "PHASE 1 FAILED: torch cannot execute here -- the image is dead on this card regardless of what cubins it contains"; exit 1; }

grep -q "torch $WANT_TORCH" "$RESULTS/phase1.log" \
  || { echo "PHASE 1 FAILED: expected torch $WANT_TORCH in $IMAGE; see $RESULTS/phase1.log"; exit 1; }
grep -q "sm_$WANT_ARCH" "$RESULTS/phase1.log" \
  || { echo "PHASE 1 FAILED: card is not sm_$WANT_ARCH -- wrong cards for this variant"; exit 1; }
fi

# ---- phase 2: the ops that justify this variant are present ----------------------
# An image is not its build args. sm_70's entire reason to exist is that
# csrc/libtorch_stable/ builds against torch 2.10; if that target silently dropped,
# the image is just a slower sm60 and the fused path will fall back at runtime
# without saying so.
if phase_on 2; then
say "phase 2: variant-defining ops are actually in the image"
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES="$UUIDS" \
  -e PYTHONPATH="$SITE" ${PXQ4_LIB:+-e PXQ4_LIB="$PXQ4_LIB"} \
  -v <local-path>:/c --entrypoint python "$IMAGE" -c "
import torch, vllm  # noqa
want_stable = ${FUSED_LONG_PREFILL_EXPECT:-$([ "$VARIANT" = sm70 ] && echo 1 || echo 0)}
ops = set(dir(torch.ops._C)) if hasattr(torch.ops, '_C') else set()
try:
    import vllm._C          # noqa
    ops |= set(dir(torch.ops._C))
except Exception as e:
    print('  note: vllm._C import:', e)
fused = [o for o in ops if 'fused_add_rms_norm' in o]
print('  fused_add_rms_norm variants visible:', sorted(fused) or '<none>')
stable = hasattr(torch.ops, '_C_stable_libtorch')
print('  _C_stable_libtorch target present:', stable)
import pxq4_vllm
print('  pxq4_vllm plugin loaded from', pxq4_vllm.__file__)
if want_stable and not stable:
    raise SystemExit('PHASE 2 FAILED: sm70 image lacks _C_stable_libtorch -- '
                     'the target that justifies torch 2.10 did not build')
if (not want_stable) and stable:
    print('  note: stable target present in an sm60 image (unexpected but not fatal)')
print('  phase2 OK')
" 2>&1 | tee "$RESULTS/phase2.log"
grep -q "phase2 OK" "$RESULTS/phase2.log" || { echo "PHASE 2 FAILED (see $RESULTS/phase2.log)"; exit 1; }
fi

# ---- serve helper ----------------------------------------------------------------
serve() {   # serve <extra-env-as-repeated -e args...>
  docker rm -f "$NAME" >/dev/null 2>&1
  docker run -d --name "$NAME" --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES="$UUIDS" \
    -e TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}" -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    -e VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=1 \
    -e VLLM_SM70_GDN_DECODE_FLASHQLA="${GDN_FLASHQLA:-0}" \
    -e VLLM_SM70_FUSED_SIGMOID_MIXED_QKV="${FUSED_SIGMOID:-0}" \
    -e VLLM_SM70_GEMMA_LONG_PREFILL_FUSED="$FUSED_LONG_PREFILL" \
    -e PXQ4_MMV_SLICE_MAX=8 -e PXQ4_MMV_SPLIT_MAX_BLOCKS=300 \
    -e HOME=/tmp -e TMPDIR=/tmp -e PYTHONUNBUFFERED=1 \
    -e PYTHONPATH="$SITE" ${PXQ4_LIB:+-e PXQ4_LIB="$PXQ4_LIB"} \
    "$@" \
    -v <local-path>:/c -p 127.0.0.1:$PORT:$PORT --shm-size=16g --ipc=host \
    "$IMAGE" python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --quantization pxq4 \
      --attention-backend "$ATTN_BACKEND" --tensor-parallel-size $TP --dtype float16 \
      --host 0.0.0.0 --port $PORT --served-model-name m --trust-remote-code \
      --gpu-memory-utilization $GMU --max-model-len 16384 --max-num-seqs 4 \
      --disable-custom-all-reduce \
      --compilation-config '{"custom_ops": ["none"], "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1,2,4]}' \
    >/dev/null || return 1
  local code=000
  for _ in $(seq 1 75); do
    [ "$(docker inspect "$NAME" --format '{{.State.Status}}' 2>/dev/null)" = "exited" ] && {
      docker logs "$NAME" > "$RESULTS/death.log" 2>&1
      echo "  server DIED. cause:"
      grep -aE "Error|Traceback|no kernel image|undefined symbol|RuntimeError" "$RESULTS/death.log" \
        | grep -vE "min_frames|max_frames" | head -8 | sed 's/^/    /'
      echo "  full log: $RESULTS/death.log"
      return 1; }
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health" 2>/dev/null)
    [ "$code" = "200" ] && return 0
    sleep 20
  done
  echo "  never became healthy (last code $code)"; docker logs "$NAME" > "$RESULTS/nohealth.log" 2>&1; return 1
}
gen() { # gen <prompt> <max_tokens>
  curl -s -m 180 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
    -d "$(python3 -c 'import json,sys;print(json.dumps({"model":"m","prompt":sys.argv[1],"max_tokens":int(sys.argv[2]),"temperature":0}))' "$1" "$2")" \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["choices"][0]["text"])' 2>/dev/null
}

# ---- phase 3: correctness battery ------------------------------------------------
if phase_on 3; then
say "phase 3: serve + correctness battery (the shipping config)"
serve || { echo "PHASE 3 FAILED: server did not come up"; FAILED+=("phase3-boot"); exit 1; }
echo "  healthy."

p5=$(gen "The capital of France is" 6)
echo "  ptok=5  -> ${p5:-<none>}"
case "$p5" in *Paris*) ;; *) echo "  FAIL: expected Paris"; FAILED+=("paris");; esac

# The one that catches the silent bug: RAW, no chat template, prompt short enough to
# fit a captured graph. Chat completions can look perfect while this returns "" or "!!!!".
p1=$(gen "Hi" 12)
echo "  ptok=1  -> ${p1:-<none>}"
case "$p1" in ""|*"!!!!"*) echo "  FAIL: 1-token raw prompt empty or garbage"; FAILED+=("raw1");; esac

ans=$(curl -s -m 180 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"What is 17*23? Reply with only the number."}],"max_tokens":200,"temperature":0}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip()[-10:])' 2>/dev/null)
echo "  17*23   -> ${ans:-<none>}"
case "$ans" in *391*) ;; *) echo "  FAIL: expected 391"; FAILED+=("mul");; esac

# Long prefill: 16k ctx is configured; a prefill that never gets exercised is a
# configuration claim, not a tested one.
long=$(python3 -c 'print("The quick brown fox jumps over the lazy dog. " * 200 + "\nThe animal in the sentence above is a")')
lp=$(gen "$long" 8)
echo "  prefill~2k -> ${lp:-<none>}"
case "$lp" in ""|*"!!!!"*) echo "  FAIL: long prefill empty or garbage"; FAILED+=("prefill");; esac
fi

# ---- phase 4: decode rate, measured on THIS image --------------------------------
# Two-point method: time a short and a long generation on the same prompt and take the
# slope. Subtracting cancels prefill, queueing and HTTP overhead, so what is left is the
# decode rate itself. A single timed request would report a number that is mostly prefill
# at low max_tokens -- which is how a stack gets described as "4 tok/s" when it is not.
if phase_on 4; then
say "phase 4: decode rate (two-point, prefill cancelled)"
curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/health" || serve || { echo "  no server"; FAILED+=("phase4-boot"); }
if curl -s -m 5 -o /dev/null "http://127.0.0.1:$PORT/health"; then
  python3 - "$PORT" <<'MEASURE' | tee "$RESULTS/phase4.txt"
import json, sys, time, urllib.request
port = sys.argv[1]
P = "Write a detailed technical explanation of how a GPU executes a matrix multiplication."

def gen(n):
    body = json.dumps({"model":"m","prompt":P,"max_tokens":n,"temperature":0,
                       "ignore_eos":True}).encode()
    req = urllib.request.Request("http://127.0.0.1:%s/v1/completions" % port,
                                 body, {"content-type":"application/json"})
    t0 = time.perf_counter()
    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
    return time.perf_counter() - t0, d["usage"]["completion_tokens"]

gen(8)                     # warm: the first call pays graph capture and allocator growth
(ta, na), (tb, nb) = gen(16), gen(144)
dt, dn = tb - ta, nb - na
if dn <= 0 or dt <= 0:
    print("  INCONCLUSIVE: t(%d)=%.2fs t(%d)=%.2fs -- noise dominates; rerun idle"
          % (na, ta, nb, tb))
else:
    print("  t(%d tok)=%.2fs   t(%d tok)=%.2fs" % (na, ta, nb, tb))
    print("  -> decode %.1f tok/s  (%.1f ms/tok), single-stream bs=1" % (dn/dt, 1000*dt/dn))
    print("  implied prefill+overhead: %.2fs" % (ta - na*dt/dn))
    print("  NOTE: a decode rate, not throughput. max-num-seqs is 4 here and nothing")
    print("  in this gate measures concurrency.")
MEASURE
fi
fi

# ---- phase 5: the levers ---------------------------------------------------------
# Each lever is flipped ALONE against the passing baseline. A lever that changes the
# answer is a correctness bug, not a speed knob, and gets recorded as such.
if phase_on 5; then
say "phase 5: lever probes (one at a time, correctness must hold)"
# BASELINE = the values phase 3 passed with. Every probe starts from these and changes
# exactly one. This reset is not decoration: in bash a `VAR=x func` prefix on a SHELL
# FUNCTION persists after the call returns, so without it probe 2 would silently still
# carry probe 1's flag and the "one at a time" claim would be false.
BASE_FUSED_LONG_PREFILL=$FUSED_LONG_PREFILL
reset_levers() {
  FUSED_LONG_PREFILL=$BASE_FUSED_LONG_PREFILL
  GDN_FLASHQLA=0
  FUSED_SIGMOID=0
  TORCHDYNAMO_DISABLE=1
}
lever() { # lever <label>  -- caller runs reset_levers, sets the ONE var, then calls this
  local label=$1
  echo "  --- $label"
  echo "      env: LONG_PREFILL=$FUSED_LONG_PREFILL GDN_FLASHQLA=$GDN_FLASHQLA SIGMOID=$FUSED_SIGMOID DYNAMO_DISABLE=$TORCHDYNAMO_DISABLE"
  if ! serve; then echo "      BOOT FAIL"; FAILED+=("lever:$label"); reset_levers; return; fi
  local a b
  a=$(gen "The capital of France is" 6); b=$(gen "Hi" 12)
  case "$a" in *Paris*) ;; *) echo "      CORRECTNESS FAIL (Paris): ${a:-<none>}"; FAILED+=("lever:$label"); reset_levers; return;; esac
  case "$b" in ""|*"!!!!"*) echo "      CORRECTNESS FAIL (raw1): ${b:-<none>}"; FAILED+=("lever:$label"); reset_levers; return;; esac
  echo "      OK (Paris + raw1 hold)"
  reset_levers
}
if [ "$VARIANT" = sm70 ]; then
  reset_levers; FUSED_LONG_PREFILL=1;  lever "fused long prefill ON (the op torch 2.10 exists for)"
  reset_levers; GDN_FLASHQLA=1;        lever "GDN decode FlashQLA ON"
  reset_levers; FUSED_SIGMOID=1;       lever "fused sigmoid mixed-QKV ON"
  reset_levers; TORCHDYNAMO_DISABLE=0; lever "dynamo ENABLED (2.7.1 could not trace this tree; 2.10 may)"
else
  reset_levers; TORCHDYNAMO_DISABLE=0; lever "dynamo ENABLED"
fi
fi

# ---- verdict ---------------------------------------------------------------------
docker rm -f "$NAME" >/dev/null 2>&1
say "verdict"
# A PARTIAL RUN IS NOT A GATE RESULT. PHASES exists for debugging, and a debugging run
# that prints "GATE PASS" is worse than no gate at all: it manufactures exactly the
# unearned claim this file was written to prevent. The gate is phases 1-4; 5 is
# informative. Only a run that executed 1,2,3,4 may use the word PASS.
GATE_PHASES="1 2 3 4"
missing=""
for p in $GATE_PHASES; do phase_on "$p" || missing="$missing $p"; done

if [ -n "$missing" ]; then
  echo "  PARTIAL RUN - NOT A GATE RESULT (phases skipped:$missing)"
  [ ${#FAILED[@]} -eq 0 ] && echo "  what ran did not fail, which is not the same as passing." \
                          || echo "  and what ran DID fail: ${FAILED[*]}"
  echo "  Nothing here licenses a caps change. Re-run without PHASES."
  echo "  artifacts: $RESULTS"
  [ ${#FAILED[@]} -eq 0 ] && exit 0 || exit 1
fi

if [ ${#FAILED[@]} -eq 0 ]; then
  echo "  GATE PASS: $IMAGE on cards $CARDS (sm_$WANT_ARCH)"
  echo "  torch dispatches, the variant-defining ops are present, raw short prompts are"
  echo "  clean, and the decode rate was measured on THIS image."
  echo "  This is what licenses caps set() -> {$WANT_ARCH} and INFERRED -> MEASURED in"
  echo "  tools/pxa-launch.py. Nothing else does."
  [ -s "$RESULTS/phase4.txt" ] && { echo "  measured:"; sed 's/^/  /' "$RESULTS/phase4.txt"; }
  echo "  artifacts: $RESULTS"
  exit 0
else
  echo "  GATE FAIL: ${FAILED[*]}"
  echo "  caps for $IMAGE must stay set(). artifacts: $RESULTS"
  exit 1
fi
