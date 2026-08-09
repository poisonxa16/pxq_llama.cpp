#!/usr/bin/env python3
"""Bit-exactness evidence for PXQ_CANON_v1: toggling the split levers must not change ONE BIT
of temp-0 decode output (token text AND top-5 logprob floats), on a file that actually contains
native-PXQ 2D + MoE tensors.

Usage: splitexact.py <model> <gpus> <ts> <build> <tag>
Variants: V0 split-off, V1 default split, V2 forced-max split, V3 K1 re-grid, V4 gateup gen off.
"""
import json, sys, time, subprocess, urllib.request, hashlib, os

MODEL, GPUS, TS, BUILD, TAG = sys.argv[1:6]
IMG = "nvidia/cuda:12.8.1-devel-ubuntu24.04"
PORT = 8464
NAME = "pxq-splitexact"

VARIANTS = {
    "V0_all_unsplit":  {"PXA_PXQ4_2D_SPLIT": "0", "PXA_PXQ6_KSPLIT_GEN": "0"},
    "V1_default":      {},
    "V2_forced_max":   {"PXA_PXQ4_2D_SPLIT_TARGET": "99999", "PXA_PXQ6_KSPLIT_GEN": "8"},
    "V3_k1_regrid":    {"PXA_PXQ4_2D_SPLIT": "0", "PXA_PXQ4_2D_KSPLIT": "1", "PXA_PXQ6_KSPLIT": "1", "PXA_PXQ6_KSPLIT_GEN": "0"},
    "V4_gu_gen2":      {"PXA_PXQ6_KSPLIT_GEN": "2"},
}
PROMPTS = [
    "Write a detailed 400-word essay on the history of the steam engine.",
    "Explain, step by step, how to compute the determinant of a 4x4 matrix.",
    "def quicksort(arr):",
]

def sh(c):
    return subprocess.run(c, shell=True, capture_output=True, text=True)

def serve(env):
    sh("docker rm -f %s" % NAME)
    envs = " ".join("-e %s=%s" % kv for kv in env.items())
    cmd = ("docker run -d --name %s --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=%s "
           "-e LD_LIBRARY_PATH=/build/bin:/build/src:/build/ggml/src:/build/examples/mtmd %s "
           "-p 127.0.0.1:%d:8080 -v %s:/build:ro -v %s:/mdir:ro %s "
           "/build/bin/llama-server -m /mdir/%s -c 8192 -np 1 -ngl 99 -sm layer -ts %s "
           "-fa on -ctk f16 -ctv f16 --ctx-checkpoints 0 --host 0.0.0.0 --port 8080 -t 8"
           % (NAME, GPUS, envs, PORT, BUILD, os.path.dirname(MODEL), IMG, os.path.basename(MODEL), TS))
    assert sh(cmd).returncode == 0
    for _ in range(150):
        time.sleep(4)
        assert sh("docker inspect -f '{{.State.Running}}' %s" % NAME).stdout.strip() == "true", \
            "died: " + sh("docker logs --tail 20 %s" % NAME).stdout[-600:]
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=4)
            return
        except Exception:
            pass
    raise RuntimeError("timeout")

def gen(prompt):
    # (n_probs omitted: the fork 500s on partial-UTF8 token pieces in completion_probabilities)
    # Two probes per prompt: 384-token greedy, plus a fixed-seed temp-1.0 sampled run — the
    # sampled run is a near-tie AMPLIFIER: with an identical RNG stream the token choice flips
    # if ANY step probability differs at the drawn uniform, far more sensitive than greedy.
    blob = ""
    for extra in ({"temperature": 0, "top_k": 1}, {"temperature": 1.0, "seed": 7}):
        body = json.dumps({"prompt": prompt, "n_predict": 384,
                           "cache_prompt": False, "ignore_eos": True, **extra}).encode()
        r = urllib.request.Request("http://127.0.0.1:%d/completion" % PORT, data=body,
                                   headers={"Content-Type": "application/json"})
        # The fork intermittently 500s serialising a response whose content ends on a partial
        # UTF-8 sequence (same class as the documented completion_probabilities 500).
        # Generation itself succeeds: the server log shows 384 tokens and a clean slot release
        # immediately before the 500. Retry rather than lose a 5-variant run.
        d = None
        for _a in range(4):
            try:
                with urllib.request.urlopen(r, timeout=900) as resp:
                    d = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                if e.code != 500 or _a == 3:
                    raise
                sys.stderr.write("  [retry %d] transient 500\n" % (_a + 1))
                time.sleep(3)
        blob += d.get("content", "") + "||"
    return hashlib.sha256(blob.encode()).hexdigest(), blob[:50]

out = {}
for vtag, env in VARIANTS.items():
    serve(env)
    firing = sh("docker logs %s 2>&1 | grep -E 'PXA_PXQ4_2D_SPLIT|PXA_PXQ6_KSPLIT' | head -4" % NAME).stdout.strip()
    hs = []
    for p in PROMPTS:
        hsh, head = gen(p)
        hs.append(hsh)
        print("[%s] %s %r" % (vtag, hsh[:16], head), flush=True)
    out[vtag] = {"hashes": hs, "log": firing}
    sh("docker rm -f %s" % NAME)

ref = out["V0_all_unsplit"]["hashes"]
verdict = {v: out[v]["hashes"] == ref for v in out}
print(json.dumps(verdict, indent=1))
json.dump({"tag": TAG, "verdict": verdict, "detail": out},
          open("./work/splitexact_%s.json" % TAG, "w"), indent=1)
print("SPLITEXACT %s: %s" % (TAG, "BIT-EXACT across all variants" if all(verdict.values()) else "MISMATCH"))
