#!/usr/bin/env python3
"""Discriminating probe for the fab-fab-kaskal 961x loop on the 122B.

Four confounds are separated, each on a FRESH server with the target prompt as the
FIRST request, and --ctx-checkpoints 0:

  1. attention codec        (published MXFP4 attn  vs  requant PXQ4 attn)
  2. slot contamination     (fresh + first-request + ctx-checkpoints 0 kills it)
  3. PXA_PXQ4_2D_SPLIT      (default ON, documented G3 / not bit-exact)
  4. PXA_PXQ_MMVQ           (off by default; the q8_1 path)

If the loop only appears mid-sequence in the 60-prompt run and NOT here, the cause is
the open recurrent-checkpoint contamination bug, not the codec.

NOTE: finish_reason is documented to lie on truncation (runaways report 'stop' at exactly
max_tokens), so completion_tokens is compared against the cap explicitly.
"""
import json, os, sys, time, subprocess, urllib.request, collections, re

BUILD = "<local-path>"
IMG = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PORT = 8489
NAME = "kaskalfix"
GPUS = "0,1,2,4,5"
MAXTOK = 2048

PUB = "<local-path>"
FIX = "<local-path>"

CELLS = [
    ("fix_PXQ4attn_newbin_default", FIX, {}),
    ("fix_PXQ4attn_newbin_SPLIT0",  FIX, {"PXA_PXQ4_2D_SPLIT": "0"}),
]


def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)


def serve(model, env):
    sh("docker rm -f %s" % NAME)
    e = " ".join("-e %s=%s" % (k, v) for k, v in env.items())
    cmd = ("docker run -d --name %s --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=%s %s "
           "-e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd "
           "-p 127.0.0.1:%d:8080 -v %s:/build:ro -v %s:/mdir:ro "
           "%s /build/bin/llama-server -m /mdir/%s "
           "-c 8192 -np 1 -ngl 99 -sm layer -fa on -ctk f16 -ctv f16 "
           "--ctx-checkpoints 0 --host 0.0.0.0 --port 8080 -t 8 "
           "--jinja --chat-template-kwargs '{\"enable_thinking\":false}'"
           % (NAME, GPUS, e, PORT, BUILD, os.path.dirname(model), IMG, os.path.basename(model)))
    if sh(cmd).returncode != 0:
        return "docker_fail"
    for _ in range(220):
        time.sleep(6)
        if sh("docker inspect -f '{{.State.Running}}' %s" % NAME).stdout.strip() != "true":
            return "died:" + sh("docker logs --tail 20 %s" % NAME).stdout[-700:]
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=4)
            return "ok"
        except Exception:
            pass
    return "timeout"


def ask(prompt):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "top_k": 1, "max_tokens": MAXTOK,
                       "cache_prompt": False}).encode()
    r = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % PORT, data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.load(resp)
    ch = d["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason"), d.get("usage", {})


def g8(txt):
    w = txt.split()
    c = collections.Counter(tuple(w[i:i + 8]) for i in range(max(0, len(w) - 7)))
    return c.most_common(1)[0][1] if c else 0


PROMPT = None
for f in ["<local-path>"]:
    for p in json.load(open(f)):
        if "kaskal" in p["qid"]:
            PROMPT = p["prompt"]
if PROMPT is None:
    print("kaskal prompt not found")
    sys.exit(2)
print("PROMPT: %s\n" % PROMPT[:160])

for tag, model, env in CELLS:
    st = serve(model, env)
    if st != "ok":
        print("%-24s SERVE FAILED %s" % (tag, st[:160]))
        continue
    txt, fr, u = ask(PROMPT)
    ct = u.get("completion_tokens")
    truncated = (ct == MAXTOK)
    print("%-24s g8=%4d chars=%5d finish=%-6s comp_tok=%-5s %s"
          % (tag, g8(txt), len(txt), fr, ct, "<-- HIT CAP (runaway)" if truncated else ""))
    sh("docker rm -f %s" % NAME)

print("KASKAL_PROBE_COMPLETE")
