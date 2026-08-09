#!/usr/bin/env python3
"""Recurrent-checkpoint contamination repro/verify (hybrid qwen35moe).

Test: on ONE slot, ask P1 (long generation, creates in-generation checkpoints), then ask P2
(shares a short prefix with P1, then diverges), then ask P1 and P2 again. Compare every answer
against the same question asked on a FRESH server (ground truth). A correct engine gives
byte-identical temp-0 answers regardless of what the slot served before. The contamination bug
makes post-P1 answers differ (recurrent state restored from the wrong generation).

Usage: ckpt_repro.py <model> <gpus> <ts> <build> <ckpts_n> <tag>
"""
import json, sys, time, subprocess, urllib.request, hashlib, os

MODEL, GPUS, TS, BUILD, CKPTS, TAG = sys.argv[1:7]
IMG = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PORT = 8463
NAME = "pxq-ckptrepro"

SYS = "You are a terse assistant. " * 40   # shared prefix ~ 360 tokens
P1 = SYS + "\nList the numbers from 1 to 400, comma-separated, no spaces."
P2 = SYS + "\nWhat is the capital of France? Answer with just the city name."
P3 = SYS + "\nCompute 17*23. Reply with just the number."

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)

def serve():
    sh("docker rm -f %s" % NAME)
    cmd = ("docker run -d --name %s --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=%s "
           "-e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd "
           "-p 127.0.0.1:%d:8080 -v %s:/build:ro -v %s:/mdir:ro %s "
           "/build/bin/llama-server -m /mdir/%s -c 8192 -np 1 -ngl 99 -sm layer -ts %s "
           "-fa on -ctk f16 -ctv f16 --ctx-checkpoints %s --host 0.0.0.0 --port 8080 -t 8"
           % (NAME, GPUS, PORT, BUILD, os.path.dirname(MODEL), IMG, os.path.basename(MODEL), TS, CKPTS))
    assert sh(cmd).returncode == 0, "docker fail"
    for _ in range(150):
        time.sleep(4)
        assert sh("docker inspect -f '{{.State.Running}}' %s" % NAME).stdout.strip() == "true", \
            "died: " + sh("docker logs --tail 20 %s" % NAME).stdout[-500:]
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=4)
            return
        except Exception:
            pass
    raise RuntimeError("serve timeout")

def ask(prompt, n):
    body = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0, "top_k": 1,
                       "cache_prompt": True}).encode()
    r = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        d = json.load(resp)
    return d.get("content", "")

def h(s):
    return hashlib.sha256(s.encode()).hexdigest()[:12]

# ground truth: each question on a fresh server
truth = {}
for tagq, (p, n) in {"P1": (P1, 700), "P2": (P2, 32), "P3": (P3, 32)}.items():
    serve()
    truth[tagq] = ask(p, n)
    print("[truth %s] %s %r" % (tagq, h(truth[tagq]), truth[tagq][:60]), flush=True)
sh("docker rm -f %s" % NAME)

# contamination sequence on ONE server / one slot
serve()
seq = [("P1", P1, 700), ("P2", P2, 32), ("P1", P1, 700), ("P3", P3, 32), ("P2", P2, 32)]
results = []
ok_all = True
for i, (tq, p, n) in enumerate(seq):
    out = ask(p, n)
    match = out == truth[tq]
    ok_all &= match
    results.append({"step": i, "q": tq, "match": match, "hash": h(out), "truth_hash": h(truth[tq]),
                    "head": out[:60]})
    print("[seq %d %s] match=%s %s vs truth %s %r" % (i, tq, match, h(out), h(truth[tq]), out[:60]), flush=True)
sh("docker rm -f %s" % NAME)
json.dump({"tag": TAG, "ckpts": CKPTS, "ok": ok_all, "truth": {k: h(v) for k, v in truth.items()},
           "seq": results}, open(os.environ.get("CKPT_OUT", "ckpt_repro_%s.json") % TAG, "w"), indent=1)
print("CKPT_REPRO %s: %s" % (TAG, "CLEAN (all match)" if ok_all else "CONTAMINATED (mismatch)"))
