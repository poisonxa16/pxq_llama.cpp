#!/usr/bin/env python3
"""Interleaved speed A/B driver. Usage: speed_drv.py <out.json> <gpus> <ts> <rounds> arm=tag:path[:ENV=V,...] ...
Protocol: per round, per arm: fresh server, coherence gate, 1 discarded warm rep, then N_REPS
reps of fill-5739 cold prefill + 256-token temp-0 decode (cache_prompt false). Rotates arm
order each round. Reports per-rep prefill/decode from server timings."""
import json, os, sys, time, subprocess, urllib.request, itertools

OUTJ = sys.argv[1]
GPUS = sys.argv[2]
TS = sys.argv[3]
ROUNDS = int(sys.argv[4])
ARMS = []
for a in sys.argv[5:]:
    assert a.startswith("arm=")
    body = a[4:]
    parts = body.split(":")
    tag, path = parts[0], parts[1]
    env = dict(kv.split("=", 1) for kv in parts[2].split(",")) if len(parts) > 2 and parts[2] else {}
    ARMS.append((tag, path, env))

BUILD = os.environ.get("SPEED_BUILD", "<local-path>")
IMG = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PORT = 8461
NAME = "pxq-speeddrv"
N_REPS = 3   # per round per arm; rounds*reps >= 8 total
PROMPT = open("<local-path>").read()

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)

def serve(path, env):
    sh("docker rm -f %s" % NAME)
    envs = " ".join("-e %s=%s" % kv for kv in env.items())
    cmd = ("docker run -d --name %s --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=%s "
           "-e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src %s "
           "-p 127.0.0.1:%d:8080 -v %s:/build:ro -v %s:/mdir:ro %s "
           "/build/bin/llama-server -m /mdir/%s -c 8192 -np 1 -ngl 99 -sm layer -ts %s "
           "-b 2048 -ub 2048 -fa on -ctk f16 -ctv f16 --ctx-checkpoints 0 "
           "--host 0.0.0.0 --port 8080 -t 8"
           % (NAME, GPUS, envs, PORT, BUILD, os.path.dirname(path), IMG, os.path.basename(path), TS))
    if sh(cmd).returncode != 0:
        return "docker_fail"
    for _ in range(150):
        time.sleep(4)
        if sh("docker inspect -f '{{.State.Running}}' %s" % NAME).stdout.strip() != "true":
            return "died:" + sh("docker logs --tail 20 %s" % NAME).stdout[-500:]
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=4)
            return "ok"
        except Exception:
            pass
    return "timeout"

def comp(prompt, n, cache=False):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1,
                       "cache_prompt": cache, "ignore_eos": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        return json.load(resp)

rows = []
order = list(range(len(ARMS)))
for rnd in range(ROUNDS):
    rot = order[rnd % len(ARMS):] + order[:rnd % len(ARMS)]
    for ai in rot:
        tag, path, env = ARMS[ai]
        st = serve(path, env)
        if st != "ok":
            print("[%s r%d] SERVE FAIL %s" % (tag, rnd, st[:300]), flush=True)
            rows.append({"arm": tag, "round": rnd, "error": st[:300]})
            continue
        # coherence gate
        g = comp("Q: What is 2+2? A:", 8)
        gtxt = g.get("content", "")
        if "4" not in gtxt:
            print("[%s r%d] COHERENCE FAIL %r" % (tag, rnd, gtxt[:80]), flush=True)
            rows.append({"arm": tag, "round": rnd, "error": "coherence:" + gtxt[:80]})
            sh("docker rm -f %s" % NAME)
            continue
        comp(PROMPT, 16)   # warm rep, discarded
        for rep in range(N_REPS):
            d = comp(PROMPT, 256)
            t = d.get("timings", {})
            rows.append({"arm": tag, "round": rnd, "rep": rep,
                         "prefill": t.get("prompt_per_second"),
                         "decode": t.get("predicted_per_second"),
                         "n_prompt": t.get("prompt_n"), "content_head": d.get("content", "")[:60]})
            print("[%s r%d rep%d] prefill=%.1f decode=%.2f n_prompt=%s" %
                  (tag, rnd, rep, t.get("prompt_per_second") or -1, t.get("predicted_per_second") or -1,
                   t.get("prompt_n")), flush=True)
        sh("docker rm -f %s" % NAME)
        json.dump(rows, open(OUTJ, "w"), indent=1)
json.dump(rows, open(OUTJ, "w"), indent=1)
print("SPEED_DRV_COMPLETE")
