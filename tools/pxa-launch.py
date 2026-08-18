#!/usr/bin/env python3
"""pxa-launch — one entry point; picks the right engine for your hardware and model.

WHY
  Two runtimes serve one quant family. pxq_llama runs on everything we own
  (sm_60 Pascal upward). vllm-pxq4 runs ONLY on sm_70+ but brings tensor
  parallelism and CUDA-graph capture that llama.cpp lacks on Volta. Choosing by
  hand means remembering which card is which, every time.

DESIGN RULE - NEVER MAGIC
  Prints the decision, the evidence for it, and the exact command before running.
  REFUSES rather than silently dropping a parameter that does not translate.
  A launcher that quietly picks differently turns every perf question into a
  debugging session about the launcher.

  --explain  decide and print, run nothing
  --engine   force llama|vllm (blockers are still reported)
  --selftest exercise the decision table against this machine
"""
import argparse, json, os, shutil, subprocess, sys

PASCAL = {60, 61}
MIN_VLLM_CAP = 70
BYTES_PER_GIB = 1024 ** 3

def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None

def gpu_table():
    """[(index, name, cc_int, mem_total_MiB, mem_used_MiB)] or (None, error)."""
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found - cannot detect GPUs. Use --engine to force."
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,memory.used",
                "--format=csv,noheader,nounits"])
    if out is None:
        return None, "nvidia-smi failed (driver not loaded?). Use --engine to force."
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 5:
            continue
        try:
            rows.append((int(p[0]), p[1], int(round(float(p[2]) * 10)), int(p[3]), int(p[4])))
        except ValueError:
            continue
    if not rows:
        return None, "nvidia-smi returned no usable rows"
    return rows, None

def model_kind(path):
    """gguf | hf_dir | vllm_dir | not_a_model | missing"""
    if not os.path.exists(path):
        return "missing"
    if os.path.isfile(path) and path.endswith(".gguf"):
        return "gguf"
    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "config.json")):
            try:
                cfg = json.load(open(os.path.join(path, "config.json")))
                q = (cfg.get("quantization_config") or {}).get("quant_method")
                if q == "pxq4":
                    return "vllm_dir"
            except Exception:
                pass
            return "hf_dir"
        return "not_a_model"          # exists, but no config.json - a real path, wrong contents
    return "missing"

def model_bytes(path, kind):
    if kind == "gguf":
        return os.path.getsize(path)
    if kind in ("hf_dir", "vllm_dir"):
        t = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".bin")):
                    try: t += os.path.getsize(os.path.join(root, f))
                    except OSError: pass
        return t
    return 0

def has_vllm_pxq4():
    try:
        import vllm_pxq4  # noqa
        return True
    except Exception:
        return False

def decide(sel, kind, model, forced, pkg):
    """-> (engine, reason, blockers[])"""
    caps = sorted({g[2] for g in sel})
    names = ", ".join(f"{g[0]}:{g[1].replace('NVIDIA ','')} sm_{g[2]}" for g in sel)

    if forced:
        # A forced engine must work even with no GPU table (that is what the
        # "use --engine to force" advice promises when nvidia-smi is unavailable).
        bl = []
        if not sel:
            bl.append("no GPUs visible - proceeding on your say-so; the engine may fail to start")
        if forced == "vllm":
            bad = sorted({f"sm_{c}" for c in caps if c < MIN_VLLM_CAP})
            if bad:
                bl.append(f"FORCED vllm but {'/'.join(bad)} present - vLLM kernels are sm_{MIN_VLLM_CAP}+; "
                          "it will fail to load on these cards")
            if not pkg:
                bl.append("FORCED vllm but vllm_pxq4 is not importable")
            if kind == "gguf":
                bl.append("FORCED vllm but the model is a raw .gguf - vLLM needs the converted form")
        if kind in ("missing", "not_a_model"):
            bl.append(f"FORCED engine but the model path is unusable ({kind}): {model}")
        return forced, f"forced by --engine ({names or 'no GPUs visible'})", bl

    if not sel:
        return None, "no GPUs selected/visible (use --engine to force anyway)", []
    if kind == "missing":
        return None, f"model path does not exist: {model}", []
    if kind == "not_a_model":
        return None, f"path exists but has no config.json and is not a .gguf: {model}", []
    if any(c < MIN_VLLM_CAP for c in caps):
        bad = sorted({f"sm_{c}" for c in caps if c < MIN_VLLM_CAP})
        return "llama", (f"{'/'.join(bad)} present - vLLM is compiled sm_{MIN_VLLM_CAP}+ only and "
                         f"cannot use these cards ({names})"), []
    if kind == "gguf":
        return "llama", (f"model is raw GGUF ({os.path.basename(model)}); vLLM needs a converted "
                         f"artifact (tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: {names}"), []
    if len(sel) < 2:
        return "llama", (f"single GPU - no tensor parallelism to gain and llama.cpp has lower "
                         f"single-stream overhead ({names})"), []
    if not pkg:
        return "llama", (f"all cards sm_{MIN_VLLM_CAP}+ ({names}) but vllm_pxq4 is not installed - "
                         f"install it to enable the TP path"), []
    return "vllm", (f"{len(sel)} x sm_{MIN_VLLM_CAP}+ ({names}) with vllm_pxq4 present - "
                    f"TP + CUDA-graph capture is the win here"), []

def fit_check(sel, mbytes, ctx, engine):
    """Warn (do not block) if the model plausibly will not fit."""
    if not mbytes:
        return []
    free = sum((g[3] - g[4]) for g in sel) * 1024 * 1024
    kv = ctx * 64 * 1024          # ~64 KiB/token measured on the qwen35 hybrid
    need = mbytes + kv + 2 * BYTES_PER_GIB * max(1, len(sel)) * 0.5
    if need > free:
        return [f"TIGHT FIT: model {mbytes/BYTES_PER_GIB:.1f} GiB + KV ~{kv/BYTES_PER_GIB:.1f} GiB "
                f"vs {free/BYTES_PER_GIB:.1f} GiB free across {len(sel)} card(s). "
                f"Reduce -c or add cards."]
    return []

UNTRANSLATABLE = {
    "ts":  "vLLM splits tensor-parallel work EVENLY; a per-card ratio has no equivalent and "
           "would be silently ignored.",
    "sm":  "vLLM's parallelism model has no -sm equivalent (layer/attn/graph).",
}

def build_cmd(engine, a, sel, kind):
    if engine == "llama":
        E = os.environ.get("PXA_ENGINE_DIR", "/mnt/models/pxa-sky-build/build70")
        cmd = [f"{E}/bin/llama-server", "-m", a.model, "--host", a.host, "--port", str(a.port),
               "-ngl", "99", "-sm", a.sm, "-c", str(a.ctx), "-b", str(a.ub), "-ub", str(a.ub),
               "-ctk", "f16", "-ctv", "f16", "-np", str(a.np), "-t", str(a.threads),
               "-fa", "on", "--jinja"]
        if a.ts:   cmd += ["-ts", a.ts]
        if a.spec: cmd += ["--spec-type", a.spec if ":" in a.spec else f"{a.spec}:n_max=4,n_min=2"]
        if a.mmproj: cmd += ["--mmproj", a.mmproj]
        env = {"PXA_ENHANCE": "1", "GGML_CUDA_NO_PINNED": "1"}
    else:
        model = a.model
        if kind == "gguf":
            model = a.model[:-5] + "-vllm"
            if not os.path.isdir(model):
                print(f"  REFUSING - vLLM needs a converted artifact and {model} does not exist.")
                print(f"    Convert first:  python3 src/gguf_to_vllm/... {a.model} {model}")
                print(f"    Or use --engine llama to serve the .gguf directly.")
                sys.exit(3)
        cmd = ["vllm", "serve", model, "--host", a.host, "--port", str(a.port),
               "--quantization", "pxq4", "--dtype", "float16",
               "--tensor-parallel-size", str(len(sel)),
               "--max-model-len", str(a.ctx), "--max-num-seqs", str(a.np)]
        if a.spec:
            cmd += ["--speculative-config",
                    json.dumps({"method": "ngram", "num_speculative_tokens": 4})]
        env = {"VLLM_SM70_QUANT_BACKEND": "turbomind"}
    return cmd, env

def selftest(gpus):
    print("=== selftest: decision table against this machine ===")
    pkg = has_vllm_pxq4()
    idx = [g[0] for g in gpus]
    cases = [("all cards", idx), ("first card", idx[:1])]
    for cc in (60, 61, 70):
        sub = [g[0] for g in gpus if g[2] == cc]
        if sub: cases.append((f"all sm_{cc}", sub))
    mixed = [g[0] for g in gpus if g[2] in PASCAL][:1] + [g[0] for g in gpus if g[2] >= 70][:1]
    if len(mixed) == 2: cases.append(("mixed pascal+volta", mixed))
    for label, ids in cases:
        sel = [g for g in gpus if g[0] in set(ids)]
        for kind in ("gguf", "vllm_dir"):
            e, r, _ = decide(sel, kind, "/x/m.gguf" if kind == "gguf" else "/x/m", None, pkg)
            print(f"  {label:22} {kind:9} -> {str(e):6} :: {r[:78]}")
    print(f"  vllm_pxq4 importable: {pkg}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cards", default="")
    ap.add_argument("--engine", choices=["llama", "vllm"])
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--spec", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--sm", default="layer")
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    gpus, err = gpu_table()
    if err:
        print(f"pxa-launch: {err}", file=sys.stderr)
        if not a.engine:
            sys.exit(2)
        gpus = []
    if a.selftest:
        selftest(gpus or []); return

    cards = {int(x) for x in a.cards.split(",") if x.strip()} if a.cards else set()
    sel = [g for g in (gpus or []) if g[0] in cards] if cards else (gpus or [])
    if cards and len(sel) != len(cards):
        missing = sorted(cards - {g[0] for g in sel})
        print(f"pxa-launch: --cards asked for {missing} which are not visible", file=sys.stderr)
        sys.exit(2)

    kind = model_kind(a.model)
    mbytes = model_bytes(a.model, kind)
    engine, reason, blockers = decide(sel, kind, a.model, a.engine, has_vllm_pxq4())

    print("=" * 78)
    print(f"pxa-launch: ENGINE = {engine}")
    print(f"  model:  {a.model}  [{kind}, {mbytes/BYTES_PER_GIB:.2f} GiB]" if mbytes
          else f"  model:  {a.model}  [{kind}]")
    print(f"  reason: {reason}")
    if engine is None:
        print("=" * 78); sys.exit(2)
    for b in blockers + fit_check(sel, mbytes, a.ctx, engine):
        print(f"  !! {b}")

    refusals = []
    if engine == "vllm":
        if a.ts: refusals.append(("-ts", UNTRANSLATABLE["ts"]))
        if a.sm and a.sm != "layer": refusals.append(("-sm", UNTRANSLATABLE["sm"]))
    if refusals:
        print("  REFUSING - these parameters do not translate to this engine:")
        for f, why in refusals:
            print(f"    {f}: {why}")
        print("  Drop them, or use --engine llama to keep them.")
        print("=" * 78); sys.exit(3)

    cmd, env = build_cmd(engine, a, sel, kind)
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    cv = a.cards or "all"
    print(f"  env:     CUDA_VISIBLE_DEVICES={cv} {envs}")
    print(f"  command: {' '.join(cmd)}")
    print("=" * 78)
    if a.explain:
        # exit 5 => a plan was produced but carries known-fatal blockers, so a CI
        # caller can tell "clean plan" from "plan that will not start"
        sys.exit(5 if blockers else 0)
    if not shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
        print(f"pxa-launch: {cmd[0]} not found", file=sys.stderr); sys.exit(4)
    e = dict(os.environ); e.update(env)
    if a.cards: e["CUDA_VISIBLE_DEVICES"] = a.cards
    os.execvpe(cmd[0], cmd, e)

if __name__ == "__main__":
    main()
