#!/usr/bin/env python3
"""Recurrent-checkpoint contamination repro/verify v2 (hybrid qwen35moe) — WITHIN-INSTANCE.

On ONE server/slot: ask P2 (clean, its own per-instance truth), P3, then P1 (a long listing
generation that advances the recurrent state far past the shared prefix), then P2/P3 again.
On a correct engine the repeats are byte-identical to their first ask (temp 0). With the
contamination bug the recurrent conv/SSM state left by P1 leaks into the P2/P3 re-asks
(prefix reuse rolls attention back but not the DeltaNet state).

Usage: ckpt_repro2.py <model> <gpus> <ts> <build> <ckpts_n> <tag>
"""
import os
import json, sys, time, subprocess, urllib.request, hashlib, os

MODEL, GPUS, TS, BUILD, CKPTS, TAG = sys.argv[1:7]
IMG = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PORT = 8463
NAME = "pxq-ckptrepro"

SYS = ""
P1 = SYS + "\nList the numbers from 1 to 400, comma-separated, no spaces."
P2 = SYS + "\nDescribe the water cycle in detail, covering evaporation, condensation, precipitation and collection."
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

serve()
seq = [("P2", P2, 300), ("P3", P3, 32), ("P1", P1, 700),
       ("P2", P2, 300), ("P3", P3, 32), ("P1", P1, 700), ("P2", P2, 300)]
first = {}
results = []
ok_all = True
for i, (tq, p, n) in enumerate(seq):
    out = ask(p, n)
    if tq not in first:
        first[tq] = out
        match = None
    else:
        match = out == first[tq]
        ok_all &= match
    results.append({"step": i, "q": tq, "match": match, "hash": h(out), "head": out[:60]})
    print("[seq %d %s] match=%s %s %r" % (i, tq, match, h(out), out[:60]), flush=True)
roll = sh("docker logs %s 2>&1 | grep -cE 'restored HYBRID context checkpoint'" % NAME).stdout.strip()
reset = sh("docker logs %s 2>&1 | grep -cE 'forcing full prompt re-processing'" % NAME).stdout.strip()
sh("docker rm -f %s" % NAME)
json.dump({"tag": TAG, "ckpts": CKPTS, "ok": ok_all, "hybrid_restores": roll, "full_resets": reset,
           "seq": results}, open((os.environ.get("PXQ_WORK", "./work") + "/ckpt_repro3_%s.json") % TAG, "w"), indent=1)
print("hybrid_restores=%s full_resets=%s" % (roll, reset))
print("CKPT_REPRO2 %s: %s" % (TAG, "CLEAN (repeats byte-identical)" if ok_all else "CONTAMINATED (repeat mismatch)"))
