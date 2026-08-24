#!/usr/bin/env python3
"""pxa-launch — one entry point; picks the right engine for your hardware, your
model AND your workload.

WHY
  Two runtimes serve one quant family. pxq_llama runs on everything we own
  (sm_60 Pascal upward). vllm-pxq4 runs ONLY on sm_70+ but brings real data
  parallelism that llama.cpp's `-sm layer` does not have. Choosing by hand means
  remembering which card is which, whether the model is dense or MoE, and which
  of the two engines wins at the concurrency you actually serve at.

WHAT CHANGED (2026-08-24) — THE DECISION IS NO LONGER JUST ABOUT THE HARDWARE
  The old table was one rule: sm_60 -> llama, sm_70+ -> vllm. Measurement says
  the model class and the concurrency matter more than the card does:

    DENSE 27B PXQ4, 2x P100 -- vLLM wins everything, by a lot
      single decode   vLLM 24.01  vs llama.cpp 13.7   (1.75x)
      agg decode @8   vLLM ~70    vs llama.cpp 12.4   (5.6x)
      prefill         vLLM ~225   vs llama.cpp 156.5  (1.44x)

    MoE 35B PXQ4, 2x P100 -- SPLIT SEAT, this is the important branch
      single decode   llama.cpp 95.6   vs vLLM 30.4    (llama.cpp 3.14x)
      agg decode @4   llama.cpp 75.93  vs vLLM 65.8    (llama.cpp +14%)
      agg decode @8   vLLM 88.7        vs llama.cpp 64 (vLLM +39%)
      long-doc prefill llama.cpp ~1136 vs vLLM ~568    (llama.cpp ~1.9x)

  Root cause of that whole shape, one sentence: llama.cpp `-sm layer` is a
  SERIALIZED 2-GPU PIPELINE, not data parallelism. Concurrent requests queue
  behind the same pipeline instead of adding throughput, so its aggregate goes
  flat-to-negative as concurrency rises while vLLM's climbs. It buys a
  spectacular single-stream number it cannot convert into aggregate.

  Evidence: <local-path> (2026-08-24).

DESIGN RULE - NEVER MAGIC
  Prints the decision, the evidence for it, and the exact command before running.
  REFUSES rather than silently dropping a parameter that does not translate.
  Says UNMEASURED out loud instead of guessing quietly.
  A launcher that quietly picks differently turns every perf question into a
  debugging session about the launcher.

    --explain   decide and print, run nothing
    --engine    force llama|vllm (blockers are still reported)
    --workload  chat | serve | longdoc  (default: inferred from --np)
    --selftest  exercise the decision table against this machine
"""
import argparse, json, os, shutil, struct, subprocess, sys

PASCAL = {60, 61}
MIN_VLLM_CAP = 70
BYTES_PER_GIB = 1024 ** 3

# ---------------------------------------------------------------------------
# THE MEASURED DECISION TABLE
# ---------------------------------------------------------------------------
# The MoE seat changes hands somewhere in this band. llama.cpp trends DOWN with
# concurrency (95.6 np1 -> 75.93 np4 -> 64 np8); vLLM trends UP (30.4 -> 65.8 ->
# 88.7). They cross in np 5-7 and as of 2026-08-24 not one point in that range
# has been measured. We pick the np4 winner through the gap and SAY SO, rather
# than interpolating and pretending. When MOE-CROSSOVER.md lands, set
# MOE_CROSSOVER_NP to the measured value and flip MOE_CROSSOVER_MEASURED — that
# is the only edit required.
MOE_LLAMA_MAX_NP       = 4      # llama.cpp measured winner at and below this
MOE_VLLM_MIN_NP        = 8      # vLLM measured winner at and above this
MOE_CROSSOVER_NP       = None   # set to the measured crossover when known
MOE_CROSSOVER_MEASURED = False

# vLLM's PXQ4 backend implements exactly one PXQ type. Everything else in the
# family must go to llama.cpp. Silently serving the wrong kernel would be worse
# than refusing, so we refuse.
VLLM_SUPPORTED_PXQ = {"PXQ4"}

# LLAMA_FTYPE_MOSTLY_PXQ* -> display name (include/llama.h).
# 250 and 251 are RESERVED, not valid: retired 2026-07-21 (PXQ4_LEGACY and PXQ5).
PXQ_FTYPE = {
    248: "PXQ1", 252: "PXQ4", 253: "PXQ4-HQ", 254: "PXQ2",
    255: "PXQ3", 256: "PXQ_UNIVERSAL", 257: "PXQ6",
}
RETIRED_FTYPE = {250: "PXQ4_LEGACY (MXFP4-repack)", 251: "PXQ5 (learned book + SE8)"}

# '-sm graph' is hard-guarded off for the DeltaNet hybrid architectures: it
# produces degenerate output there because the cross-device reduce is never
# delivered. Where it DOES work it is a phase trade, not a win: measured +64%
# prefill and -17% decode on 4x P100. Never recommend it for decode.
GRAPH_SPLIT_GUARDED_ARCHES = {"qwen35moe", "qwen3next", "qwen35"}


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


# ---------------------------------------------------------------------------
# Model introspection. The engine choice now depends on what the model IS, so we
# have to actually look inside it rather than trust the filename.
# ---------------------------------------------------------------------------
def _gguf_kv(path, want):
    """Read only the KV header of a GGUF. Returns {} on any malformed input --
    a launcher must not die on a file it was merely trying to describe."""
    got = {}
    try:
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return {}
            struct.unpack("<I", f.read(4))              # version
            struct.unpack("<Q", f.read(8))              # tensor count
            nkv, = struct.unpack("<Q", f.read(8))

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rval(t):
                fixed = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                if t == 8:
                    return rstr()
                if t == 9:                              # array: skip it, cheaply
                    et, = struct.unpack("<I", f.read(4))
                    ln, = struct.unpack("<Q", f.read(8))
                    for _ in range(ln):
                        rval(et)
                    return None
                b = f.read(fixed[t])
                if t == 4:                              # uint32 - the only one we read back
                    return struct.unpack("<I", b)[0]
                if t == 5:
                    return struct.unpack("<i", b)[0]
                return None

            for _ in range(nkv):
                k = rstr()
                t, = struct.unpack("<I", f.read(4))
                v = rval(t)
                if k in want or k.endswith(".expert_count") or k.endswith(".block_count"):
                    got[k] = v
    except Exception:
        return got
    return got


def model_kind(path):
    """gguf | hf_dir | vllm_dir | not_a_model | missing"""
    if not os.path.exists(path):
        return "missing"
    if os.path.isfile(path) and path.endswith(".gguf"):
        return "gguf"
    if os.path.isdir(path):
        cfgp = os.path.join(path, "config.json")
        if os.path.exists(cfgp):
            try:
                cfg = json.load(open(cfgp))
                q = (cfg.get("quantization_config") or {}).get("quant_method")
                if q == "pxq4":
                    return "vllm_dir"
            except Exception:
                pass
            return "hf_dir"
        return "not_a_model"          # exists, but no config.json - a real path, wrong contents
    return "missing"


def model_profile(path, kind):
    """-> dict(arch, n_expert, is_moe, pxq, ftype, why). Best effort, never raises."""
    p = {"arch": None, "n_expert": 0, "is_moe": False, "pxq": None, "ftype": None,
         "why": "not inspected"}
    if kind == "gguf":
        kv = _gguf_kv(path, {"general.architecture", "general.file_type"})
        p["arch"] = kv.get("general.architecture")
        p["ftype"] = kv.get("general.file_type")
        for k, v in kv.items():
            if k.endswith(".expert_count") and isinstance(v, int):
                p["n_expert"] = v
        p["is_moe"] = p["n_expert"] > 0
        if isinstance(p["ftype"], int):
            if p["ftype"] in PXQ_FTYPE:
                p["pxq"] = PXQ_FTYPE[p["ftype"]]
            elif p["ftype"] in RETIRED_FTYPE:
                p["pxq"] = "RETIRED:" + RETIRED_FTYPE[p["ftype"]]
        p["why"] = "read from GGUF header"
        return p
    if kind in ("hf_dir", "vllm_dir"):
        try:
            cfg = json.load(open(os.path.join(path, "config.json")))
            p["arch"] = cfg.get("model_type")
            for key in ("num_experts", "n_routed_experts", "num_local_experts",
                        "moe_num_experts", "num_experts_per_tok"):
                v = cfg.get(key)
                if isinstance(v, int) and v > 0 and key != "num_experts_per_tok":
                    p["n_expert"] = max(p["n_expert"], v)
            # a text_config / sub-config is common on hybrid + VL models
            for sub in ("text_config", "llm_config"):
                s = cfg.get(sub) or {}
                for key in ("num_experts", "n_routed_experts", "num_local_experts"):
                    v = s.get(key)
                    if isinstance(v, int) and v > 0:
                        p["n_expert"] = max(p["n_expert"], v)
            p["is_moe"] = p["n_expert"] > 0
            qm = (cfg.get("quantization_config") or {}).get("quant_method")
            if qm:
                p["pxq"] = qm.upper()
            p["why"] = "read from config.json"
        except Exception as e:
            p["why"] = f"config.json unreadable ({e.__class__.__name__})"
    return p


def model_bytes(path, kind):
    if kind == "gguf":
        return os.path.getsize(path)
    if kind in ("hf_dir", "vllm_dir"):
        t = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".bin")):
                    try:
                        t += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        return t
    return 0


def has_vllm_pxq4():
    try:
        import vllm_pxq4  # noqa
        return True
    except Exception:
        return False


def infer_workload(np_, explicit):
    if explicit:
        return explicit
    return "chat" if np_ <= 1 else "serve"


# ---------------------------------------------------------------------------
# THE DECISION
# ---------------------------------------------------------------------------
def decide(sel, kind, model, forced, pkg, prof=None, np_=1, workload="chat"):
    """-> (engine, reason, blockers[], notes[])

    notes[] is for things the operator must KNOW but which do not block: an
    unmeasured band, a phase trade, a cross-harness caveat."""
    prof = prof or {"is_moe": False, "n_expert": 0, "pxq": None, "arch": None}
    notes = []
    caps = sorted({g[2] for g in sel})
    names = ", ".join(f"{g[0]}:{g[1].replace('NVIDIA ','')} sm_{g[2]}" for g in sel)
    cls = "MoE" if prof["is_moe"] else "dense"

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
            if prof["pxq"] and prof["pxq"] not in VLLM_SUPPORTED_PXQ \
                    and not prof["pxq"].startswith("RETIRED"):
                bl.append(f"FORCED vllm but this model is {prof['pxq']} and the vLLM backend "
                          f"implements {'/'.join(sorted(VLLM_SUPPORTED_PXQ))} only")
        if kind in ("missing", "not_a_model"):
            bl.append(f"FORCED engine but the model path is unusable ({kind}): {model}")
        return forced, f"forced by --engine ({names or 'no GPUs visible'})", bl, notes

    if not sel:
        return None, "no GPUs selected/visible (use --engine to force anyway)", [], notes
    if kind == "missing":
        return None, f"model path does not exist: {model}", [], notes
    if kind == "not_a_model":
        return None, f"path exists but has no config.json and is not a .gguf: {model}", [], notes

    # ---- Hard eligibility gates. These are about what CAN run, not what is fastest.
    if any(c < MIN_VLLM_CAP for c in caps):
        bad = sorted({f"sm_{c}" for c in caps if c < MIN_VLLM_CAP})
        return "llama", (f"{'/'.join(bad)} present - vLLM is compiled sm_{MIN_VLLM_CAP}+ only and "
                         f"cannot use these cards ({names})"), [], notes
    if kind == "gguf":
        return "llama", (f"model is raw GGUF ({os.path.basename(model)}); vLLM needs a converted "
                         f"artifact (tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: {names}"), [], notes
    if prof["pxq"] and prof["pxq"].startswith("RETIRED"):
        return None, (f"this file declares a RETIRED quant type ({prof['pxq'][8:]}). Types 250/251 "
                      f"were removed 2026-07-21 and no engine we ship can read it. Requantize."), [], notes
    if prof["pxq"] and prof["pxq"] not in VLLM_SUPPORTED_PXQ:
        return "llama", (f"model is {prof['pxq']}; the vLLM backend implements "
                         f"{'/'.join(sorted(VLLM_SUPPORTED_PXQ))} only, llama.cpp reads the whole "
                         f"PXQ family ({names})"), [], notes
    if len(sel) < 2:
        return "llama", (f"single GPU - no parallelism to gain and llama.cpp has lower "
                         f"single-stream overhead ({names})"), [], notes
    if not pkg:
        return "llama", (f"all cards sm_{MIN_VLLM_CAP}+ ({names}) but vllm_pxq4 is not installed - "
                         f"install it to enable the vLLM path"), [], notes

    # ---- Both engines can run it. Now pick on MEASURED performance.
    if not prof["is_moe"]:
        return "vllm", (f"DENSE model on {len(sel)}x sm_{MIN_VLLM_CAP}+ ({names}). vLLM wins dense at "
                        f"every workload measured: 24.01 vs 13.7 tok/s single (1.75x), ~70 vs 12.4 "
                        f"agg@8 (5.6x), ~225 vs 156.5 prefill (1.44x). [SCOREBOARD 2a]"), [], notes

    # MoE: the split seat.
    if workload == "longdoc":
        notes.append("long-doc prefill: llama.cpp measured ~1.7-1.9x faster at every concurrency "
                     "(1136 vs 568 tok/s). CAVEAT: cross-harness, prompt lengths were not matched "
                     "(2059 vs ~6.4k tok). Directionally trusted, not yet controlled. [SCOREBOARD 0.2]")
        return "llama", (f"MoE model, long-document workload ({names}). llama.cpp -sm layer holds the "
                         f"prefill record at every concurrency level measured."), [], notes

    if np_ <= MOE_LLAMA_MAX_NP:
        why = ("95.6 vs 30.4 tok/s single-stream (3.14x)" if np_ <= 1
               else f"75.93 vs 65.8 tok/s agg@4 (+14%)")
        return "llama", (f"MoE model at np={np_} on {names}. llama.cpp -sm layer wins at and below "
                         f"np={MOE_LLAMA_MAX_NP}: {why}. [SCOREBOARD 2b]"), [], notes

    if np_ >= MOE_VLLM_MIN_NP:
        return "vllm", (f"MoE model at np={np_} on {names}. vLLM PP=2 + FULL_DECODE_ONLY wins from "
                        f"np={MOE_VLLM_MIN_NP}: 88.7 vs 64 tok/s agg@8 (+39%). [SCOREBOARD 2b]"), [], notes

    # np 5-7: the unmeasured band.
    if MOE_CROSSOVER_MEASURED and MOE_CROSSOVER_NP is not None:
        if np_ <= MOE_CROSSOVER_NP:
            return "llama", (f"MoE at np={np_}; measured crossover is np={MOE_CROSSOVER_NP}, "
                             f"llama.cpp still ahead here."), [], notes
        return "vllm", (f"MoE at np={np_}; measured crossover is np={MOE_CROSSOVER_NP}, "
                        f"vLLM ahead from here."), [], notes
    notes.append(f"np={np_} IS IN THE UNMEASURED BAND (np {MOE_LLAMA_MAX_NP+1}-{MOE_VLLM_MIN_NP-1}). "
                 f"llama.cpp trends DOWN with concurrency (95.6->75.9->64), vLLM trends UP "
                 f"(30.4->65.8->88.7); they cross somewhere in here and no point in the range has "
                 f"been measured. Picking the np={MOE_LLAMA_MAX_NP} winner. This could be wrong by "
                 f"up to ~39%. Measure np={np_} on your traffic before trusting it.")
    return "llama", (f"MoE model at np={np_} on {names}. UNMEASURED BAND - defaulting to the "
                     f"np={MOE_LLAMA_MAX_NP} winner, llama.cpp -sm layer."), [], notes


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
    "ts":  "vLLM splits parallel work EVENLY; a per-card ratio has no equivalent and "
           "would be silently ignored.",
    "sm":  "vLLM's parallelism model has no -sm equivalent (layer/attn/graph).",
}

ENGINE_DIR_CANDIDATES = [
    "/mnt/models/pxa-sky-build/build70",   # DGX, sm_70
    "/mnt/models/pxq_llama/build70",       # DGX, sm_70
    "<local-path>",   # Unraid, sm_60;70 - serves PXQ4 in production
    "<local-path>",      # Unraid, second live seat
    "<local-path>",
    "<local-path>",
]


def resolve_engine_dir():
    """-> (dir, note). Honour PXA_ENGINE_DIR, else first candidate that exists."""
    env = os.environ.get("PXA_ENGINE_DIR")
    if env:
        return env, ("PXA_ENGINE_DIR" if os.path.isfile(f"{env}/bin/llama-server")
                     else "PXA_ENGINE_DIR (no llama-server there)")
    for d in ENGINE_DIR_CANDIDATES:
        if os.path.isfile(f"{d}/bin/llama-server"):
            return d, "auto-detected"
    onpath = shutil.which("llama-server")
    if onpath:
        return os.path.dirname(os.path.dirname(onpath)), "found on PATH"
    return None, "no llama-server found"


def build_cmd(engine, a, sel, kind, prof):
    if engine == "llama":
        E, note = resolve_engine_dir()
        if E is None:
            print("  REFUSING - no llama-server binary found. Looked at "
                  f"{', '.join(ENGINE_DIR_CANDIDATES)} and $PATH.")
            print("    Set PXA_ENGINE_DIR=/path/to/build (the dir containing bin/llama-server).")
            sys.exit(4)
        if note != "auto-detected":
            print(f"  engine dir: {E}  [{note}]")
        cmd = [f"{E}/bin/llama-server", "-m", a.model, "--host", a.host, "--port", str(a.port),
               "-ngl", "99", "-sm", a.sm, "-c", str(a.ctx), "-b", str(a.ub), "-ub", str(a.ub),
               "-ctk", a.ctk, "-ctv", a.ctv, "-np", str(a.np), "-t", str(a.threads),
               "-fa", "on", "--jinja"]
        if a.ts:
            cmd += ["-ts", a.ts]
        if a.spec:
            cmd += ["--spec-type", a.spec if ":" in a.spec else f"{a.spec}:n_max=4,n_min=2"]
        if a.mmproj:
            cmd += ["--mmproj", a.mmproj]
        env = {"PXA_ENHANCE": "1", "GGML_CUDA_NO_PINNED": "1"}
        return cmd, env, ",".join(str(g[0]) for g in sel) if sel else ""

    # ---- vLLM
    model = a.model
    if kind == "gguf":
        model = a.model[:-5] + "-vllm"
        if not os.path.isdir(model):
            print(f"  REFUSING - vLLM needs a converted artifact and {model} does not exist.")
            print(f"    Convert first:  python3 src/gguf_to_vllm/... {a.model} {model}")
            print(f"    Or use --engine llama to serve the .gguf directly.")
            sys.exit(3)
    # Only sm_70+ cards can run vLLM at all, and the parallel degree must be a
    # power of two - an odd degree is rejected by vLLM at startup. Take the
    # largest power of two among the ELIGIBLE cards, and say what we dropped.
    elig = [g for g in sel if g[2] >= MIN_VLLM_CAP]
    deg = 1
    while deg * 2 <= len(elig):
        deg *= 2
    used = elig[:deg]
    if used and len(used) != len(sel):
        dropped = [f"{g[0]}:sm_{g[2]}" for g in sel if g not in used]
        print(f"  parallel degree {deg} on card(s) {[g[0] for g in used]}; not using {dropped} "
              f"(sm_<{MIN_VLLM_CAP} and/or the degree must be a power of two)")

    cmd = ["vllm", "serve", model, "--host", a.host, "--port", str(a.port),
           "--quantization", "pxq4", "--dtype", "float16",
           "--max-model-len", str(a.ctx), "--max-num-seqs", str(a.np)]

    # MoE goes PIPELINE parallel; dense goes TENSOR parallel. PP=2 is what holds
    # the shippable MoE aggregate record (88.7 agg@8).
    if prof.get("is_moe") and deg >= 2:
        cmd += ["--pipeline-parallel-size", str(deg), "--tensor-parallel-size", "1"]
        print(f"  MoE -> PIPELINE parallel (PP={deg}). PP=2 + FULL_DECODE_ONLY holds the shippable "
              f"MoE record (88.7 tok/s agg@8).")
    else:
        cmd += ["--tensor-parallel-size", str(max(1, deg))]

    # FULL_DECODE_ONLY IS NOT A TUNING KNOB - IT IS A CORRECTNESS REQUIREMENT.
    # The default (FULL_AND_PIECEWISE) also captures PREFILL graphs at the ladder
    # sizes. A raw /v1/completions prompt short enough to fit one prefills through
    # a captured graph whose input buffer holds stale data and returns fluent
    # garbage from character zero. Chat traffic never shows it because the chat
    # template pads every prompt past the captured sizes - which is exactly why
    # arithmetic gates stayed green while the bug was live. Decode-only capture
    # removes the poisoned path, and measured FASTER on the P100 TP=2 dense A/B
    # (22.3 -> 24.0 tok/s single). There is no speed argument for the broken one:
    # its best aggregate (88.4) is BELOW the correct config's (88.7).
    cmd += ["--compilation-config",
            json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY",
                        "cudagraph_capture_sizes": [1, 2, 4, 8]})]
    if a.spec:
        cmd += ["--speculative-config",
                json.dumps({"method": "ngram", "num_speculative_tokens": 4})]
    env = {"VLLM_SM70_QUANT_BACKEND": "turbomind"}
    return cmd, env, ",".join(str(g[0]) for g in used) if used else ""


def selftest(gpus):
    print("=== selftest: decision table against this machine ===")
    pkg = has_vllm_pxq4()
    idx = [g[0] for g in gpus]
    cases = [("all cards", idx), ("first card", idx[:1])]
    for cc in (60, 61, 70):
        sub = [g[0] for g in gpus if g[2] == cc]
        if sub:
            cases.append((f"all sm_{cc}", sub))
    mixed = [g[0] for g in gpus if g[2] in PASCAL][:1] + [g[0] for g in gpus if g[2] >= 70][:1]
    if len(mixed) == 2:
        cases.append(("mixed pascal+volta", mixed))

    profiles = [
        ("dense pxq4", {"is_moe": False, "n_expert": 0, "pxq": "PXQ4", "arch": "qwen35"}),
        ("moe   pxq4", {"is_moe": True, "n_expert": 256, "pxq": "PXQ4", "arch": "qwen35moe"}),
        ("moe   pxq3", {"is_moe": True, "n_expert": 256, "pxq": "PXQ3", "arch": "qwen35moe"}),
    ]
    # Run the table TWICE. On a box where vllm_pxq4 is not importable, that
    # eligibility gate short-circuits every branch and the selftest would show
    # nothing but "not installed" - hiding the actual decision table, which is
    # the thing worth testing. The second pass asks what we WOULD pick.
    for pkg_mode, banner in ((pkg, f"as configured here (vllm_pxq4 importable: {pkg})"),
                             (True, "hypothetical: vllm_pxq4 present")):
        if pkg_mode == pkg and pkg:
            banner = f"as configured here (vllm_pxq4 importable: {pkg})"
        elif pkg_mode is True and pkg:
            continue                      # identical to the first pass, do not print it twice
        print(f"  --- {banner} ---")
        for label, ids in cases:
            sel = [g for g in gpus if g[0] in set(ids)]
            for plabel, prof in profiles:
                for np_ in (1, 4, 6, 8):
                    e, r, _, notes = decide(sel, "vllm_dir", "/x/m", None, pkg_mode, prof, np_,
                                            infer_workload(np_, None))
                    flag = "  <-- UNMEASURED BAND" if any("UNMEASURED BAND" in n for n in notes) else ""
                    print(f"  {label:20} {plabel:11} np={np_:<2} -> {str(e):6} :: {r[:62]}{flag}")
    print(f"  vllm_pxq4 importable: {pkg}")
    print(f"  MoE crossover: {'np=' + str(MOE_CROSSOVER_NP) if MOE_CROSSOVER_MEASURED else 'UNMEASURED (np 5-7)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cards", default="")
    ap.add_argument("--engine", choices=["llama", "vllm"])
    ap.add_argument("--workload", choices=["chat", "serve", "longdoc"], default=None,
                    help="default: chat when --np<=1, else serve")
    ap.add_argument("-c", "--ctx", type=int, default=32768)
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--ub", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--spec", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--sm", default="layer")
    ap.add_argument("--ctk", default="f16")
    ap.add_argument("--ctv", default="f16")
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
        selftest(gpus or [])
        return

    cards = {int(x) for x in a.cards.split(",") if x.strip()} if a.cards else set()
    sel = [g for g in (gpus or []) if g[0] in cards] if cards else (gpus or [])
    if cards and len(sel) != len(cards):
        missing = sorted(cards - {g[0] for g in sel})
        print(f"pxa-launch: --cards asked for {missing} which are not visible", file=sys.stderr)
        sys.exit(2)

    kind = model_kind(a.model)
    prof = model_profile(a.model, kind)
    mbytes = model_bytes(a.model, kind)
    workload = infer_workload(a.np, a.workload)
    engine, reason, blockers, notes = decide(sel, kind, a.model, a.engine, has_vllm_pxq4(),
                                             prof, a.np, workload)

    print("=" * 78)
    print(f"pxa-launch: ENGINE = {engine}")
    size = f", {mbytes/BYTES_PER_GIB:.2f} GiB" if mbytes else ""
    print(f"  model:  {a.model}  [{kind}{size}]")
    # Only claim a class if we actually read the file. Printing "dense" for a
    # path that does not exist is a confident statement about nothing, which is
    # the exact failure mode this launcher exists to prevent.
    if prof["arch"] or prof["n_expert"] or prof["pxq"]:
        cls = "MoE" if prof["is_moe"] else "dense"
        print(f"  class:  {cls}" + (f" ({prof['n_expert']} experts)" if prof["is_moe"] else "")
              + (f", arch={prof['arch']}" if prof["arch"] else "")
              + (f", quant={prof['pxq']}" if prof["pxq"] else "")
              + f"  [{prof['why']}]")
    else:
        print(f"  class:  UNKNOWN  [{prof['why']}]")
    print(f"  serve:  np={a.np}, workload={workload}, ctx={a.ctx}")
    print(f"  reason: {reason}")
    if engine is None:
        print("=" * 78)
        sys.exit(2)
    for n in notes:
        print(f"  ** {n}")
    for b in blockers + fit_check(sel, mbytes, a.ctx, engine):
        print(f"  !! {b}")

    if kind in ("missing", "not_a_model"):
        print(f"  REFUSING - no command can be built for a {kind} path: {a.model}")
        print("=" * 78)
        sys.exit(2)

    refusals = []
    if engine == "vllm":
        if a.ts:
            refusals.append(("-ts", UNTRANSLATABLE["ts"]))
        if a.sm and a.sm != "layer":
            refusals.append(("-sm", UNTRANSLATABLE["sm"]))
    if engine == "llama" and a.sm == "graph" and prof.get("arch") in GRAPH_SPLIT_GUARDED_ARCHES:
        refusals.append(("-sm graph",
                         f"'{prof['arch']}' is a DeltaNet hybrid: graph split is hard-guarded off for it "
                         f"(measured degenerate output - the cross-device reduce is never delivered). "
                         f"Use -sm layer. Even where graph split works it is a phase trade, not a win: "
                         f"+64% prefill / -17% decode on 4x P100."))
    if refusals:
        print("  REFUSING - these parameters do not translate to this engine/model:")
        for f, why in refusals:
            print(f"    {f}: {why}")
        print("  Drop them, or use --engine llama to keep them.")
        print("=" * 78)
        sys.exit(3)

    cmd, env, cv = build_cmd(engine, a, sel, kind, prof)
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    cv = cv or "all"
    print(f"  env:     CUDA_VISIBLE_DEVICES={cv} {envs}")
    print(f"  command: {' '.join(cmd)}")
    print("=" * 78)
    if a.explain:
        # exit 5 => a plan was produced but carries known-fatal blockers, so a CI
        # caller can tell "clean plan" from "plan that will not start"
        sys.exit(5 if blockers else 0)
    if not shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
        print(f"pxa-launch: {cmd[0]} not found", file=sys.stderr)
        sys.exit(4)
    e = dict(os.environ)
    e.update(env)
    if cv and cv != "all":
        e["CUDA_VISIBLE_DEVICES"] = cv
    os.execvpe(cmd[0], cmd, e)


if __name__ == "__main__":
    main()
