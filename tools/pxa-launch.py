#!/usr/bin/env python3
"""pxa-launch - one entry point; picks the right engine for your hardware, your
model AND your workload.

WHY
  Two runtimes serve one quant family. pxq_llama runs on everything we own.
  vllm-pxq4 runs wherever its IMAGE has kernels for the card (see "sm_70 ONLY WAS
  A LIE", below) and brings real data parallelism that llama.cpp's `-sm layer`
  does not have. Choosing by hand means remembering which card is which, whether
  the model is dense or MoE, which PXQ tier is actually inside the file, and
  which of the two engines wins at the concurrency you actually serve at.

DESIGN RULE - NEVER MAGIC
  Prints the decision, the evidence for it, and the exact command before running.
  REFUSES rather than silently dropping a parameter that does not translate.
  Says UNMEASURED out loud instead of guessing quietly.
  A launcher that quietly picks differently turns every perf question into a
  debugging session about the launcher.

    --explain            decide and print, run nothing
    --engine             force llama|vllm (blockers are still reported AND STOP)
    --workload           chat | serve | longdoc  (default: inferred from --np)
    --selftest           exercise the decision table against this machine
    --accept-unmeasured  execute a branch this file labels [INFERRED]/UNMEASURED
    --allow-busy         select a card another process is already resident on

HOW TO READ EVERY CLAIM IN THIS FILE
  Exactly three tags, no fourth category. They appear in the comments AND in the
  printed output, so a reader can tell a measured branch from a guess without
  leaving the file:

    MEASURED     a number from a gated boot on this box, with its source doc:line.
    [INFERRED]   a branch taken from an ADJACENT measurement. Never a new number.
    UNMEASURED   nothing was measured. The launcher says the word and stops or asks.

  Source corpus (all committed 2026-08-24, tree <local-path>):
    pxa-stack/docs/baselines-20260824/{SCOREBOARD,MOE-CROSSOVER,PXQ-TYPE-MATRIX,
                                       PERPLEXITY-RESULTS,RELEASE-GATE}.md
    pxa-stack/docs/run-20260824/{ENGINE-VERDICT,PORT-ASSESSMENT,BUILD-RECIPE,
                                   DGX-FDO-FAILURE}.md
    docs/LEVERS.md ; src/llama-quantize.cpp ; src/llama-model-loader.cpp ;
    ggml/include/ggml.h
  Spec this file is measured against: <local-path>
  (md5 86115482841eaa371e6050bc58734ca8). Where this file DIFFERS from that spec,
  the comment says so and why - see "SPEC CORRECTIONS", below.

THE DECISION IS NOT JUST ABOUT THE HARDWARE

    DENSE 27B PXQ4, 2x P100 sm_60 -- vLLM wins everything measured
      single decode   vLLM 24.01  vs llama.cpp 13.7   (1.75x)   SCOREBOARD D3/D1
      agg decode @8   vLLM ~70    vs llama.cpp 12.4   (5.6x)    SCOREBOARD D3/D1
      prefill         vLLM ~225   vs llama.cpp 156.5  (1.44x)   SCOREBOARD D3/D1
      agg decode @4   vLLM UNMEASURED vs llama.cpp 12.0         SCOREBOARD D3
      ^ the llama.cpp side (D1) is ONE boot, below this corpus's own 2-boot bar,
        and the graphs-ON dense arm (D2) was never launched at all.

    MoE 35B PXQ4, 2x P100 sm_60 -- SPLIT SEAT, this is the important branch
      np1  llama.cpp 95.6  vs vLLM 30.4   llama.cpp 3.14x   SCOREBOARD M1/M7
      np4  llama.cpp 75.93 vs vLLM 64.82  llama.cpp +17.1%  MOE-CROSSOVER
      np5  llama.cpp 79.49 vs vLLM 64.32  llama.cpp +23.6%  <- llama.cpp PEAKS
      np6  llama.cpp 69.58 vs vLLM 75.60  vLLM +8.7%        <- crossover
      np7  llama.cpp 67.74 vs vLLM 87.03  vLLM +28.5%
      np8  llama.cpp 62.42 vs vLLM 95.81  vLLM +53.5%

  Root cause of that shape, one sentence: llama.cpp `-sm layer` is a SERIALIZED
  2-GPU PIPELINE, not data parallelism, so concurrent requests queue behind the
  same pipeline while vLLM's aggregate climbs.

SPEC CORRECTIONS - things this file does NOT do the way the spec says, with why
  (each was verified against the tree/box before being written here)

  C1. THE TIER IS READ FROM THE TENSOR DIRECTORY, NOT FROM `pxa.pxq.tier`.
      LAUNCHER-SPEC M3 says to read the tier from "custom pxa.pxq.* KV keys".
      There is no such key. `grep -n "gguf_set_val" src/llama-quantize.cpp` shows
      the only tier-bearing provenance KVs written are pxa.pxq6.{version,tier}
      (:1760-1767), pxa.pxq2.version (:1775), pxa.pxq3.version (:1780) and
      pxa.pxqu.version (:1790) - and PXQ1 writes NONE of them (":1506 comment:
      Fixed {-1,+1} book + the shared SUB16 -> no provenance KVs needed"). So the
      spec's own fix would still leave PXQ1 - the one tier it exists to refuse -
      invisible. The engine itself does not use KV either: it detects PXQ1 by
      TENSOR TYPE (src/llama-model-loader.cpp:527). This file does the same.
      Ground truth = the per-tensor ggml type histogram; the provenance KV is
      read too and any CONFLICT between the two is printed, never resolved
      silently. Verified on the box: 5/5 PXQ GGUFs yield a tier this way, 0/5
      through general.file_type (all report 38).

  C2. I-2's "22.3 -> 24.0 dense" IS A MoE NUMBER. Adversarial review caught this
      and it is right. SCOREBOARD rows M8/M9 sit in Table 1a "MoE 35B PXQ4", not
      the dense table; ENGINE-VERDICT.md:14-22 calls the same pair "the MoE seat".
      Dense TP=2 single is D3 = 24.01, a different measurement, and there is NO
      dense FAP-vs-FDO pair in the corpus at all (D2 = 0 boots). The OLD version
      of this file carried the same mislabel in its FULL_DECODE_ONLY comment.
      Fixed in both places.

  C3. "EVERY vLLM NUMBER IS sm_60" IS OVERSTATED. Also caught by review, also
      right. PORT-ASSESSMENT.md:82 records a vLLM shared-prefix win of 3.76x
      measured on the DGX (V100, sm_70). What is true, and what the eligibility
      fix actually rests on, is narrower: NO vLLM DECODE-OR-PREFILL THROUGHPUT
      NUMBER on sm_70 exists anywhere in this corpus. The comment at
      vllm_eligibility() states the narrow version.

  C4. THE FIT CHECK STILL CANNOT BLOCK ON A KV ESTIMATE - so it does not pretend
      to. LAUNCHER-SPEC lists R-17 as "ctx > n_ctx_train, OR fit-check impossible"
      while M8 forbids the KV formula from blocking anything until Q9 validates
      it. Review flagged the contradiction. Resolution here: R-17 blocks ONLY on
      facts that need no formula - ctx > n_ctx_train (a KV field), and weights
      alone > total VRAM of the selection under full offload (arithmetic on file
      bytes). Everything that needs the KV-per-token estimate warns and says
      [INFERRED]. A real 42.93 GB PXQ4 file on this box (Fusion-Coder-80) is
      refused by the weights-only clause on 2 cards, which was the case review
      raised.

  C5. THE `-sm graph` GUARD IS STRUCTURAL, NOT A NAME LIST. I-9 keys on three
      arch strings. The hazard is a property of the DeltaNet tensors, so this
      file refuses on `linear_attn.*` presence OR the arch allowlist, whichever
      fires. That closes the muse-glimmer/dflash hole review raised - and note
      the box's muse-glimmer PXQ4 has NO linear_attn tensors (verified), so it is
      still allowed, on evidence rather than on a name.

sm_70 ONLY WAS A LIE, AND IT MADE THE MoE BRANCH UNREACHABLE
  The old file set MIN_VLLM_CAP=70 and routed any selection containing a
  sub-sm_70 card to llama.cpp. Every vLLM decode/prefill number in the decision
  table below was produced on 2x P100 sm_60 (SCOREBOARD.md:6; MOE-CROSSOVER.md:3,
  image pxa-sm60-dev, libpxq4_sm60_v10.so, --attention-backend PASCAL_SDPA,
  MOE-CROSSOVER.md:77-82). So the old table could never reproduce a single one of
  its own vLLM cells, and the np>=6 vLLM branch was dead on the only hardware
  where the crossover was measured. Eligibility is a property of the resolved
  IMAGE, probed. See vllm_eligibility().
"""
import argparse, collections, json, os, re, shutil, struct, subprocess, sys

SPEC_MD5 = "86115482841eaa371e6050bc58734ca8"       # <local-path>
BYTES_PER_GIB = 1024 ** 3

# ---------------------------------------------------------------------------
# HARDWARE FACTS
# ---------------------------------------------------------------------------
# cc -> human card class. sm_61 is NOT folded into "Pascal" here: the corpus has
# no MoE crossover, no dense pair and no PXQ-tier throughput on sm_61, and the
# BALANCE-mode PXA_FA_MASK_SKIP_TILE win explicitly excludes all of sm_61
# (LEVERS.md:85). Folding it into a Pascal set is how it stops being visible.
CARD_CLASS = {60: "P100-class sm_60", 61: "GTX 1080 Ti-class sm_61", 70: "V100-class sm_70"}

# MEASURED, hardware-verified (LEVERS.md:99-103, ADAPTIVE-UB fallback table):
#   >=15 GiB card -> 2048 ; 11 GB 1080Ti class -> 768 ; else 512.
# ub2048/1024 compute buffers OOM next to a ~10 GB model on 11 GB; ub768 fits.
# The engine picks this itself at startup when -ub is UNSET, probing real free
# VRAM per device. So the correct launcher behaviour is to pass NO -ub and print
# the value adaptive-ub should land on. Emitting a single global -ub across a
# heterogeneous pool is the bug this replaces (old file: -ub 2048 for every card,
# including card 3's 11 GB).
def ub_for_card(mem_total_mib):
    if mem_total_mib >= 15 * 1024:
        return 2048
    if mem_total_mib >= 10 * 1024:
        return 768
    return 512

# ---------------------------------------------------------------------------
# THE MEASURED DECISION TABLE
# ---------------------------------------------------------------------------
# MEASURED 2026-08-24, 11 gated boots, cards 0+6, PXA-Coder-35B-v2 PXQ4,
# -c np*4096, raw /completion (NOT chat-templated), every boot gated on
# short-prompt correctness BEFORE its number was kept.
# Source: pxa-stack/docs/baselines-20260824/MOE-CROSSOVER.md section 1.
#   np : (llama.cpp agg tok/s, vLLM PP=2+FDO agg tok/s, winner, margin text)
# Neither curve is monotonic and the flip is SHARP: llama.cpp PEAKS at np5 ABOVE
# its own np4 value, then drops 12.5% in one step while vLLM climbs. The margin
# swings 32 points between np5 and np6. A straight line np4->np8 puts the
# threshold too early and misprices np5 by ~14%. THE TABLE IS STORED, NOT A SLOPE.
MOE_TABLE = {
    4: (75.93, 64.82, "llama", "+17.1%"),   # llama.cpp reused from SCOREBOARD M4
    5: (79.49, 64.32, "llama", "+23.6%"),   # llama.cpp PEAK
    6: (69.58, 75.60, "vllm",  "+8.7%"),    # crossover
    7: (67.74, 87.03, "vllm",  "+28.5%"),
    8: (62.42, 95.81, "vllm",  "+53.5%"),
}
MOE_NP1 = (95.6, 30.4, "llama", "3.14x")    # MEASURED SCOREBOARD M1 / M7
MOE_LLAMA_MAX_NP = 5        # MEASURED: llama.cpp wins at and below np=5
MOE_VLLM_MIN_NP  = 6        # MEASURED: vLLM wins at and from np=6
MOE_TABLE_MAX_NP = 8        # nothing above np=8 was run, on either engine

# MEASURED currency warning that rides every vLLM MoE decision:
# MOE-CROSSOVER section 6.4 measured 95.81 at np8 on tree 3e34872 where SCOREBOARD
# M7 has 88.7 on fdec4ae (+8.0%). The doc names the intervening commit as a
# HYPOTHESIS, not a measurement, and concludes every vLLM MoE number on the
# SCOREBOARD may now be stale in vLLM's favour. If it is stale the crossover could
# sit BELOW np=6 - i.e. this can change an engine decision, not just a number.
MOE_CURRENCY_NOTE = ("vLLM MoE currency: 95.81 (tree 3e34872) vs 88.7 (fdec4ae) = +8.0% on the "
                     "same cell, cause hypothesised and NOT tested. If the newer number is real "
                     "the crossover may sit BELOW np=6. [MOE-CROSSOVER 6.4]")

# MEASURED dense, 27B PXQ4, 2x P100 sm_60, cards 1,5 (SCOREBOARD 2a rows D1/D3).
DENSE_NUMBERS = {
    "single":  ("24.01", "13.7",  "1.75x"),
    "agg8":    ("~70",   "12.4",  "5.6x"),
    "prefill": ("~225",  "156.5", "1.44x"),
}
DENSE_AGG4_NOTE = ("dense agg@4 on vLLM is UNMEASURED (SCOREBOARD D3); llama.cpp side is 12.0. "
                   "No ratio is printed for np=4 because none was measured.")
DENSE_WEAKNESS = ("dense envelope: the llama.cpp side (SCOREBOARD D1) is ONE boot, below this "
                  "corpus's own 2-boot bar, and the graphs-ON dense arm (D2) was never launched. "
                  "Direction 1.75x-5.6x is not in doubt; the exact ratios are single-boot.")

# MEASURED MoE long-doc prefill, with the caveat printed EVERY time:
MOE_LONGDOC_NOTE = ("long-doc prefill: llama.cpp 1136 / ~1058 / ~1000 vs vLLM 567.6 / 595.8 / "
                    "594.4 tok/s (~1.7-1.9x). CAVEAT: CROSS-HARNESS, prompt lengths NOT matched "
                    "(2059 vs ~6.4k tok) [SCOREBOARD 0.2]. Directionally trusted, not controlled.")

# The anchor models. The crossover is a property of a model x hardware PAIR, not
# of the engines - other MoE arches on the box (qwen3next MoE-512, qwen3moe
# MoE-128, deepseek4 MoE-6) have NO engine-vs-engine data at any np.
ANCHOR_MOE_HINTS   = ("coder-35b", "coder35", "pxa-coder-35b")
ANCHOR_DENSE_HINTS = ("27b-unc", "qwen38-27b", "qwen3.8-27b")
ANCHOR_CTX_PER_SLOT = 4096      # MEASURED envelope: --ctx-size np*4096 (MOE-CROSSOVER section 3)

# ---------------------------------------------------------------------------
# PXQ TIERS - ggml TENSOR TYPE IDS (ground truth; see SPEC CORRECTION C1)
# ---------------------------------------------------------------------------
# ggml/include/ggml.h:478-511. These are what the loader dispatches on and what
# llama-model-loader.cpp:527 uses to detect PXQ1 content.
PXQ_GGML_TYPE = {
    248: "PXQ1",     # ggml.h:499
    252: "PXQ4",     # ggml.h:478
    253: "PXQ4-HQ",  # ggml.h:481
    254: "PXQ2",     # ggml.h:489
    255: "PXQ3",     # ggml.h:490
    256: "PXQ6",     # ggml.h:511
}
# Everything else a PXQ file legitimately contains (backbone carriers).
NON_PXQ_GGML_TYPE = {0: "f32", 1: "f16", 8: "q8_0", 14: "q6_K", 30: "bf16", 39: "MXFP4"}
# ggml type ids the CURRENT tree does not define at all. Any tensor carrying one
# cannot be dispatched: `grep -n "24[4-9]" ggml/include/ggml.h` yields only PXQ1's
# 248, and PXQ1C/PXQ2C appear nowhere in the tree. VERIFIED on the box: a real
# file, <local-path>,
# carries 106 tensors of type 247 and 38 of type 246 plus pxa.pxq1c.* / pxa.pxq2c.*
# KVs - retired clustered variants this engine no longer implements. The old
# launcher emitted a full, confident command for it. R-23 refuses it.
KNOWN_GGML_TYPE_MAX = 256

# LLAMA_FTYPE ids (include/llama.h). KEPT FOR ONE PURPOSE ONLY: catching the two
# RETIRED ids. general.file_type CANNOT identify a PXQ tier - llama-quantize.cpp
# :1454-1509 rewrites EVERY PXQ tier to LLAMA_FTYPE_MOSTLY_MXFP4 (=38) before
# writing it at :1658. VERIFIED on the box: 138/138 GGUFs report 38 or a K-quant
# id; 0 yield a tier. The old file's whole PXQ_FTYPE table was dead code against
# reality, which is why the PXQ1 refusal could never fire.
RETIRED_FTYPE = {250: "PXQ4_LEGACY (MXFP4-repack)", 251: "PXQ5 (learned book + SE8)"}
FTYPE_MXFP4 = 38

# vLLM implements exactly one PXQ tier. PXQ-TYPE-MATRIX.md:69-70: "On the vLLM
# fork, only PXQ4 is supported; every other tier is refused cleanly at the
# conversion gate. No silent wrong output exists on the vLLM path."
VLLM_SUPPORTED_PXQ = {"PXQ4"}
# PXQ-TYPE-MATRIX.md:80-81 - these run on llama.cpp only.
LLAMA_ONLY_PXQ = {"PXQ4-HQ", "PXQ6", "PXQ3", "PXQ2", "PXQ_UNIVERSAL"}
# No CPU codec, GPU-only, open task #62 (PXQ-TYPE-MATRIX.md:67; RELEASE-GATE.md:177).
NO_CPU_CODEC = {"PXQ1", "PXQ6"}

# '-sm graph' is hard-guarded off for the DeltaNet hybrids: the cross-device
# all-reduce never reaches its consumers, so each device computes a different
# router top-8 -> degenerate output. Where graph split DOES work it is a phase
# trade, not a win: +64% prefill / -17% decode on 4x P100. Never for decode.
# SPEC CORRECTION C5: the arch names are one of TWO triggers; the structural one
# (linear_attn.* tensors) is the other, so an unnamed arch with the same tensors
# is caught too.
GRAPH_SPLIT_GUARDED_ARCHES = {"qwen35moe", "qwen3next", "qwen35", "qwen4exp"}

# ---------------------------------------------------------------------------
# vLLM IMAGES - eligibility is an IMAGE property (see docstring)
# ---------------------------------------------------------------------------
# caps  = compute capabilities this image has PXQ4 kernels for
# status: MEASURED | INFERRED | INELIGIBLE
VLLM_IMAGES = {
    "pxa-sm60-dev": {
        "caps": {60}, "status": "MEASURED",
        "why": "produced every MoE-crossover vLLM number (MOE-CROSSOVER.md:77-82; "
               "libpxq4_sm60_v10.so, --attention-backend PASCAL_SDPA)",
        # NOT A SELF-CONTAINED IMAGE. `pip list` inside it shows pip and nothing else:
        # it is a bare CUDA runtime, and torch, vllm and the PXQ4 plugin all live on the
        # HOST and are bind-mounted at run time. Every number attributed to this tag was
        # really produced by the image PLUS these paths. Recording that is not pedantry -
        # without it the launcher declares the image eligible on a box where the venv has
        # moved, and emits `vllm serve`, which is not even on PATH in this container.
        "host_env": {
            "mounts": {"<local-path>": "/c"},
            # Presence is checked with lexists, not exists. venv/bin/python is a symlink
            # to /usr/bin/python3, which exists INSIDE the container and nowhere on this
            # host - so exists() calls a perfectly good venv missing and refuses a seat
            # that would have served. lexists asks the question we can actually answer
            # from here: is the entry there.
            "requires": ["<local-path>",
                         "<local-path>",
                         "<local-path>",
                         "<local-path>",
                         "<local-path>",
                         "<local-path>"],
            "python": "/c/pxq4-sm60/venv/bin/python",
            "env": {"PYTHONPATH": "/c/moe-branch/site",
                    # v10, NOT v8 (stale: both shipping launchers moved to v10) and
                    # NOT v11. v11 exists and is FASTER on prefill with PXQ4_GEMM2D=1
                    # (300.1 tok/s, +37%) but FAILS raw-prompt correctness. A prefill
                    # win that changes what the model says is not a win.
                    "PXQ4_LIB": "/c/moe-branch/libpxq4_sm60_v10.so"},
            "why": "traced 2026-08-26 through the last boot on this tag "
                   "(xover-vllm-boot1) and confirmed by importing inside it: torch is "
                   "/c/pxq4-sm60/venv/lib/python3.12/site-packages/torch (2.7.1+cu126) "
                   "and vllm is /c/pxq4-sm60/1cat/vllm/__init__.py",
            # THE PART THAT MATTERS. vllm is an EDITABLE install
            # (__editable__.1cat_vllm-....pth) resolving to the 1cat WORKING TREE. The
            # seat does not run a built artifact; it imports whatever is checked out
            # there right now. Edit that repo and the running engine changes. Switch its
            # branch and the seat serves a different engine with no redeploy and no
            # version change to notice. Every sm_60 number in the corpus was taken this
            # way, which is why they cannot be reproduced from an image alone.
            "editable_source": "<local-path>",
        }},
    # THE FAT IMAGE IS WITHDRAWN. One image spanning sm_60+sm_70 was tried and does
    # not work: 8 boot attempts on a Tesla V100, 8 failures (RELEASE-GATE.md 3.7). The
    # cause is structural, not configuration - VLLM_SKIP_C_STABLE=1 is required to build
    # against torch 2.7.1 (the last torch with sm_60 cubins) and it drops
    # csrc/libtorch_stable/, where an op the V100 serving path calls unconditionally
    # lives. You cannot have sm_60 cubins and that op in the same build. Hence two thin
    # images, each pinned to the torch its cards need. Do not reintroduce a fat tag.

    # THE SHIPPING SET: two thin images (scripts/build-images.sh), each built from our
    # own tree, each pinned to the torch its cards need. No third-party image is
    # eligible.
    "pxa-vllm:sm60": {
        "caps": set(), "caps_inferred": {60}, "status": "INFERRED",
        "why": "Pascal variant: torch 2.7.1 (last torch shipping sm_60 cubins), "
               "VLLM_SKIP_C_STABLE=1, arch 6.0;7.0. The sm_60 gate that produced the "
               "MEASURED numbers ran against the WITHDRAWN FAT IMAGE, not this tag - "
               "identical build arguments is an argument, not a boot. caps stays EMPTY "
               "until scripts/thin-image-gate.sh sm60 passes on a P100 pair against "
               "THIS tag"},
    "pxa-vllm:sm70": {
        "caps": set(), "caps_inferred": {70}, "status": "INFERRED",
        "why": "Volta variant: torch 2.10 (the tree's own pins), libtorch_stable BUILT, no "
               "compat shim - so the three torch-2.7.1 failures cannot occur by construction. "
               "NOT YET GATED. caps is deliberately EMPTY until a V100 smoke passes: this "
               "launcher does not route traffic to an image on the strength of an argument"},

    "kewaii/vllm:latest": {
        "caps": set(), "status": "INELIGIBLE",
        "why": "THIRD-PARTY image. Everything we ship must be buildable from our own tree, and "
               "pxa-vllm:sm70 replaces this. Technically it also silently overrode "
               "cudagraph_mode from its own SM70 compile policy, and the knob it honours "
               "crash-loops 3/3 at warmup (DGX-FDO-FAILURE.md)"},
}
# MEASURED per-class attention backend.
ATTN_BACKEND = {
    60: ("PASCAL_SDPA", "MEASURED - the arm that produced every MoE-crossover vLLM cell "
                        "(MOE-CROSSOVER.md:81)"),
    70: ("FLASH_ATTN_V100", "[INFERRED] - from launch-v100b.sh / alina-launch.sh recipes; no "
                            "engine-vs-engine number was ever taken on sm_70, and as of "
                            "2026-08-25 no vLLM image in this table has a GATED sm_70 seat "
                            "at all (RELEASE-GATE.md 3.7)"),
}


def _run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HARDWARE INTROSPECTION (H1..H8)
# ---------------------------------------------------------------------------
def gpu_table():
    """[(index, name, cc_int, mem_total_MiB, mem_used_MiB, uuid)] or (None, error)."""
    if not shutil.which("nvidia-smi"):
        return None, "nvidia-smi not found - cannot detect GPUs. Use --engine to force."
    out = _run(["nvidia-smi",
                "--query-gpu=index,name,compute_cap,memory.total,memory.used,uuid",
                "--format=csv,noheader,nounits"])
    if out is None:
        return None, "nvidia-smi failed (driver not loaded?). Use --engine to force."
    rows = []
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 6:
            continue
        try:
            rows.append((int(p[0]), p[1], int(round(float(p[2]) * 10)),
                         int(p[3]), int(p[4]), p[5]))
        except ValueError:
            continue
    if not rows:
        return None, "nvidia-smi returned no usable rows"
    return rows, None


def resident_procs(gpus):
    """H4 -> {gpu_index: [(pid, name, mib), ...]}. This is a SHARED, LIVE box; the
    launcher must never hand a card to a second process by accident. Keyed by UUID
    because --query-compute-apps reports gpu_uuid, not index."""
    by_uuid = {g[5]: g[0] for g in gpus}
    out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
                "--format=csv,noheader,nounits"])
    res = collections.defaultdict(list)
    if not out:
        return res, (out is not None)
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4:
            continue
        idx = by_uuid.get(p[3])
        if idx is None:
            continue
        try:
            res[idx].append((p[0], os.path.basename(p[1]), int(p[2])))
        except ValueError:
            continue
    return res, True


def peer_topology():
    """H3 -> (has_p2p, description). This box is all-PHB, no NVLink, no P2P
    (SCOREBOARD.md:6) - which is WHY custom all-reduce is off in every measured
    vLLM arm (--disable-custom-all-reduce, MOE-CROSSOVER.md:79). CAR costs ~18%
    vs NCCL on MoE (CAR-VERDICT.md) while the CAR KERNEL is exonerated. We READ
    the topology; we do not hardcode the answer."""
    out = _run(["nvidia-smi", "topo", "-m"], timeout=25)
    if not out:
        return None, "topology UNREADABLE (nvidia-smi topo -m failed) - treating as no-P2P"
    links = set(re.findall(r"\b(NV\d+|PIX|PXB|PHB|SYS|NODE)\b", out))
    nvlink = {l for l in links if l.startswith("NV")}
    if nvlink:
        return True, f"NVLink present ({','.join(sorted(nvlink))})"
    return False, f"no NVLink/P2P; interconnect {'/'.join(sorted(links)) or 'unknown'}"


# ---------------------------------------------------------------------------
# MODEL INTROSPECTION (M1..M11) - header-only reads, no tensor data, no GPU
# ---------------------------------------------------------------------------
GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def gguf_header(path, max_tensors=200000):
    """Read the GGUF KV block AND the tensor directory. Returns
    {kv, tensors:[(name,type)], n_tensors, ok, err}. Never raises: a launcher must
    not die on a file it was merely trying to describe. `ok` False means the file
    is not a usable GGUF - which is a REFUSAL (R-23), not a shrug: a 5.3 GB file
    with a zeroed header exists on this box (/tmp/TRUNCATED-pxq4.gguf, first 4
    bytes 00 00 00 00) and the old launcher built a full command for it."""
    out = {"kv": {}, "tensors": [], "n_tensors": 0, "ok": False, "err": None}
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                out["err"] = f"bad magic {magic!r} (expected b'GGUF')"
                return out
            ver, = struct.unpack("<I", f.read(4))
            nt, = struct.unpack("<Q", f.read(8))
            nkv, = struct.unpack("<Q", f.read(8))
            if nt > max_tensors or nkv > 100000:
                out["err"] = f"implausible header (n_tensors={nt}, n_kv={nkv})"
                return out
            out["n_tensors"] = nt

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                if n > (1 << 24):
                    raise ValueError("absurd string length")
                return f.read(n).decode("utf-8", "replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:                       # array
                    et, = struct.unpack("<I", f.read(4))
                    ln, = struct.unpack("<Q", f.read(8))
                    vals = []
                    for _ in range(ln):
                        v = rval(et)
                        if len(vals) < 4096:
                            vals.append(v)
                    return vals
                b = f.read(GGUF_FIXED[t])
                if t == 4:
                    return struct.unpack("<I", b)[0]
                if t == 5:
                    return struct.unpack("<i", b)[0]
                if t == 6:
                    return struct.unpack("<f", b)[0]
                if t == 7:
                    return bool(b[0])
                if t == 10:
                    return struct.unpack("<Q", b)[0]
                if t == 11:
                    return struct.unpack("<q", b)[0]
                return None

            for _ in range(nkv):
                k = rstr()
                t, = struct.unpack("<I", f.read(4))
                out["kv"][k] = rval(t)
            for _ in range(nt):
                nm = rstr()
                nd, = struct.unpack("<I", f.read(4))
                ne = 1
                for _ in range(nd):
                    d, = struct.unpack("<Q", f.read(8))
                    ne *= d
                ty, = struct.unpack("<I", f.read(4))
                struct.unpack("<Q", f.read(8))       # offset
                out["tensors"].append((nm, ty, ne))
            out["ok"] = True
    except Exception as e:
        out["err"] = f"{e.__class__.__name__} while reading the header ({e})"
    return out


def tier_from_tensors(tensors):
    """M3, done the way the ENGINE does it (src/llama-model-loader.cpp:527 detects
    PXQ1 by tensor TYPE). -> (tier, hist, unknown_types)

    tier is one of: a PXQ name, 'PXQ_UNIVERSAL' (more than one PXQ type present),
    or None (no PXQ tensors at all - a K-quant / MXFP4 / f16 file).
    PXQ1 anywhere in the mix makes this file PXQ1-BEARING regardless of what else
    is in it: a curated UNIVERSAL map with PXQ1 experts has the same generation
    failure as a uniform PXQ1 file (PXQ-TYPE-MATRIX Findings 7 and 8)."""
    hist = collections.Counter(t for _, t, _ in tensors)
    pxq = {t: n for t, n in hist.items() if t in PXQ_GGML_TYPE}
    # Anything in the 240..259 band that is not a DEFINED ggml type is a retired or
    # foreign codec this engine cannot dispatch (see KNOWN_GGML_TYPE_MAX). Checked
    # against ggml/include/ggml.h, not against a hardcoded list of survivors.
    unknown = [t for t in hist if 240 <= t <= 259 and t not in PXQ_GGML_TYPE]
    if not pxq:
        return None, hist, sorted(unknown)
    if 248 in pxq:
        return "PXQ1", hist, sorted(unknown)
    if len(pxq) > 1:
        return "PXQ_UNIVERSAL", hist, sorted(unknown)
    return PXQ_GGML_TYPE[next(iter(pxq))], hist, sorted(unknown)


def tier_from_provenance_kv(kv):
    """The corroborating signal. Written by llama-quantize.cpp:
      pxa.pxq6.tier core|hq|lm32 -> PXQ4 | PXQ4-HQ | PXQ6   (:1760-1767)
      pxa.pxqu.version           -> PXQ_UNIVERSAL           (:1790, pxqu_out only)
      pxa.pxq2.version / pxa.pxq3.version alone -> PXQ2 / PXQ3 (:1775, :1780)
    PXQ1 writes NO provenance KV at all (:1506) - which is exactly why this is the
    corroborating signal and the tensor walk is the authoritative one."""
    if "pxa.pxqu.version" in kv:
        return "PXQ_UNIVERSAL"
    t = kv.get("pxa.pxq6.tier")
    if t == "core":
        return "PXQ4"
    if t == "hq":
        return "PXQ4-HQ"
    if t == "lm32":
        return "PXQ6"
    if "pxa.pxq2.version" in kv:
        return "PXQ2"
    if "pxa.pxq3.version" in kv:
        return "PXQ3"
    return None


def model_kind(path):
    """M10. gguf | gguf_broken | vllm_dir | hf_dir | lora_dir | weightless_dir |
    not_a_model_file | not_a_model | missing

    'not_a_model_file' exists because the OLD taxonomy had no slot for it: any
    existing file that did not end in .gguf fell through to 'missing' and the
    launcher told the operator a path that plainly exists does not exist. The box
    is full of tempting single files next to real models (one shard of six, a
    .tiers map, an .imatrix, a LoRA adapter)."""
    if not os.path.exists(path):
        return "missing"
    if os.path.isfile(path):
        if path.endswith(".gguf"):
            return "gguf"
        return "not_a_model_file"
    if not os.path.isdir(path):
        return "not_a_model_file"
    cfgp = os.path.join(path, "config.json")
    has_weights = False
    for root, _, files in os.walk(path):
        if any(f.endswith((".safetensors", ".bin", ".gguf")) for f in files):
            has_weights = True
            break
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return "lora_dir"
    if os.path.exists(cfgp):
        if not has_weights:
            return "weightless_dir"
        try:
            cfg = json.load(open(cfgp))
            q = (cfg.get("quantization_config") or {}).get("quant_method")
            if q == "pxq4":
                return "vllm_dir"
        except Exception:
            pass
        return "hf_dir"
    if has_weights:
        return "not_a_model"          # weights but no config.json - GGUF dir? name the file.
    return "not_a_model"


def _hf_tensor_names(path):
    """Cheap tensor-name list for a safetensors dir: the index json's weight_map.
    Returns [] when there is no index (single-shard dirs) - and the caller then
    says UNMEASURABLE rather than 'absent'."""
    for idx in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        p = os.path.join(path, idx)
        if os.path.exists(p):
            try:
                return list(json.load(open(p)).get("weight_map", {}).keys())
            except Exception:
                return []
    return []


def model_profile(path, kind):
    """-> dict. Best effort, never raises. Every field says where it came from."""
    p = {"arch": None, "n_expert": 0, "is_moe": False, "tier": None, "tier_kv": None,
         "tier_src": "not inspected", "ftype": None, "n_ctx_train": None,
         "mtp_tensors": 0, "mtp_kv": None, "mtp_src": "not inspected",
         "deltanet": False, "deltanet_src": "not inspected", "vision": False,
         "kv_bytes_tok": None, "kv_bytes_src": "UNMEASURABLE",
         "unknown_types": [], "hist": {}, "quant_method": None,
         "why": "not inspected", "hdr_err": None}

    if kind in ("gguf", "gguf_broken"):
        h = gguf_header(path)
        p["hdr_err"] = h["err"]
        if not h["ok"]:
            p["why"] = f"GGUF header UNREADABLE: {h['err']}"
            return p
        kv, tn = h["kv"], h["tensors"]
        p["arch"] = kv.get("general.architecture")
        p["ftype"] = kv.get("general.file_type")
        arch = p["arch"] or ""
        p["n_expert"] = int(kv.get(f"{arch}.expert_count") or 0)
        if not p["n_expert"]:
            for k, v in kv.items():
                if k.endswith(".expert_count") and isinstance(v, int):
                    p["n_expert"] = max(p["n_expert"], v)
        p["is_moe"] = p["n_expert"] > 0
        p["n_ctx_train"] = kv.get(f"{arch}.context_length")
        tier, hist, unknown = tier_from_tensors(tn)
        p["tier"], p["hist"], p["unknown_types"] = tier, dict(hist), unknown
        p["tier_kv"] = tier_from_provenance_kv(kv)
        p["tier_src"] = ("tensor-type histogram over %d tensors (the signal the loader "
                         "dispatches on, llama-model-loader.cpp:527)" % len(tn))
        # M5: MTP by TENSOR WALK, never by the KV flag. Two shipped f16 files declare
        # nextn_predict_layers=1 with ZERO nextn tensors (PXA-Agent-9B-f16.gguf,
        # PXA-Coder-35B-v2-f16.gguf) - the head was dropped in the recovery pipeline
        # and the flag survived. Arming MTP on those arms a drafter that does not exist.
        p["mtp_tensors"] = sum(1 for n, _, _ in tn if "nextn" in n or ".mtp" in n)
        # PER-LAYER EMBEDDING (PLE). qwen4exp and gemma3n carry a per_layer_token_embd
        # table that is a pure GET_ROWS gather - one lookup per token per head, no
        # GEMM - and it is ENORMOUS: on Flash-Next it is 160 x 320001536 = 51.2e9
        # elements, ~51 GiB of a 97 GiB file. It belongs in host RAM.
        # MEASURED 2026-08-28: leaving it on the GPU made a 5-card seat try to
        # cudaMalloc 16089.57 MiB for it on one card and die during load.
        for _n, _t, _ne in tn:
            if _n.startswith("per_layer_token_embd"):
                p["ple_tensor"] = _n
                try:
                    # gguf_header stores ne as the TOTAL ELEMENT COUNT (an int),
                    # not a dims list - iterating it raises and silently loses the
                    # size, which is how this subtraction failed the first time.
                    e = int(_ne)
                    p["ple_elems"] = e
                    # GGML block geometry is FIXED and documented (ggml.h), so this is
                    # arithmetic, not an estimate. Only the types a PLE table is ever
                    # stored in are listed; anything else leaves ple_bytes None and the
                    # caller then declines to subtract rather than guessing.
                    _BPE = {0: 4.0, 1: 2.0, 30: 2.0,      # f32, f16, bf16
                            8: 34.0 / 32.0,               # q8_0
                            14: 210.0 / 256.0,            # q6_K
                            12: 144.0 / 256.0,            # q4_K
                            20: 18.0 / 32.0}              # iq4_nl
                    if _t in _BPE:
                        p["ple_bytes"] = int(e * _BPE[_t])
                        p["ple_type"] = _t
                except Exception:
                    p["ple_elems"] = None
                break
        p["mtp_kv"] = kv.get(f"{arch}.nextn_predict_layers")
        p["mtp_src"] = "tensor walk for nextn/mtp names"
        # M6 / SPEC CORRECTION C5 and C6: structural DeltaNet detection.
        # C6 - VERIFIED ON THE BOX, and it matters: LAUNCHER-SPEC M6 says to detect
        # the hybrid by "presence of linear_attn.* KV/tensors". That is the
        # SAFETENSORS name (coder35-moe-pxq4-m1's quantization_config.pxq4_modules
        # carries linear_attn.in_proj_qkvz). In GGUF the SAME layers are named
        # ssm_*: Fusion-Coder-80-P100-PXQ4.gguf (arch qwen3next) has blk.N.ssm_a,
        # ssm_ba, ssm_conv1d, ssm_dt, ssm_norm, ssm_out and ZERO tensors matching
        # linear_attn. So the spec's structural detector is dead code on the only
        # format llama.cpp - the engine -sm graph applies to - actually serves.
        # Both namings are checked here.
        lin = [n for n, _, _ in tn if "linear_attn." in n]
        ssm = [n for n, _, _ in tn if ".ssm_" in n or n.startswith("ssm_")]
        p["deltanet"] = bool(lin or ssm) or any("linear_attn" in k for k in kv)
        if lin:
            p["deltanet_src"] = f"{len(lin)} linear_attn.* tensors (safetensors-style naming)"
        elif ssm:
            p["deltanet_src"] = (f"{len(ssm)} ssm_* tensors - the GGUF spelling of the same "
                                 f"linear-attention layers. Whether a PURE-SSM (non-hybrid) "
                                 f"model needs the -sm graph guard is UNMEASURED; guarding is "
                                 f"the safe direction")
        else:
            p["deltanet_src"] = "no linear_attn.* or ssm_* tensors"
        p["vision"] = any(n.startswith(("v.", "mm.")) for n, _, _ in tn) or \
            any(k.startswith("clip.") for k in kv)
        p["kv_bytes_tok"], p["kv_bytes_src"] = kv_bytes_per_token(kv, arch)
        p["why"] = "read from GGUF header + tensor directory"
        return p

    if kind in ("hf_dir", "vllm_dir", "weightless_dir", "lora_dir"):
        try:
            cfg = json.load(open(os.path.join(path, "config.json")))
        except Exception as e:
            p["why"] = f"config.json unreadable ({e.__class__.__name__})"
            return p
        archs = cfg.get("architectures") or []
        p["arch"] = (archs[0] if archs else None) or cfg.get("model_type")
        for key in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
            v = cfg.get(key)
            if isinstance(v, int) and v > 0:
                p["n_expert"] = max(p["n_expert"], v)
        for sub in ("text_config", "llm_config"):
            s = cfg.get(sub) or {}
            for key in ("num_experts", "n_routed_experts", "num_local_experts"):
                v = s.get(key)
                if isinstance(v, int) and v > 0:
                    p["n_expert"] = max(p["n_expert"], v)
        p["is_moe"] = p["n_expert"] > 0
        p["n_ctx_train"] = cfg.get("max_position_embeddings") or \
            (cfg.get("text_config") or {}).get("max_position_embeddings")
        qc = cfg.get("quantization_config") or {}
        p["quant_method"] = qc.get("quant_method")
        if p["quant_method"] == "pxq4":
            p["tier"] = "PXQ4"
            p["tier_src"] = "config.json quantization_config.quant_method == 'pxq4'"
        elif p["quant_method"]:
            p["tier"] = None
            p["tier_src"] = f"config.json quant_method == {p['quant_method']!r} (not a PXQ tier)"
        else:
            p["tier_src"] = "config.json carries NO quantization_config - unquantized checkpoint"
        p["vision"] = bool(cfg.get("vision_config") or (cfg.get("text_config") or {}).get("vision_config"))
        names = _hf_tensor_names(path)
        if names:
            p["mtp_tensors"] = sum(1 for n in names if "nextn" in n or ".mtp" in n)
            p["mtp_src"] = "safetensors index weight_map walk"
            p["deltanet"] = any("linear_attn." in n for n in names)
            p["deltanet_src"] = "linear_attn.* in safetensors index" if p["deltanet"] else \
                "no linear_attn.* in safetensors index"
        else:
            mods = qc.get("pxq4_modules") or []
            p["deltanet"] = any("linear_attn" in str(m) for m in mods)
            p["deltanet_src"] = ("quantization_config.pxq4_modules names linear_attn"
                                 if p["deltanet"] else
                                 "NO safetensors index and no linear_attn in pxq4_modules - "
                                 "DeltaNet presence is UNMEASURABLE from this directory")
            p["mtp_src"] = "UNMEASURABLE (no safetensors index in this directory)"
            p["mtp_tensors"] = -1
        p["kv_bytes_tok"], p["kv_bytes_src"] = kv_bytes_per_token_hf(cfg)
        p["why"] = "read from config.json" + (" + safetensors index" if names else "")
        return p
    return p


def kv_bytes_per_token(kv, arch):
    """M8. [INFERRED] - this is ARITHMETIC, not a measurement.
      bytes/token = n_layer * n_head_kv * (d_k*b_k + d_v*b_v)
    It replaces the old file's flat `ctx * 64 * 1024`, a constant annotated
    "~64 KiB/token measured on the qwen35 hybrid" and then applied to every dense
    model, every other MoE arch and every GQA ratio on the box. Neither the old
    constant nor this formula has been validated against a real allocation
    (measurement queue Q9), so NEITHER MAY BLOCK ANYTHING. It warns."""
    try:
        n_layer = kv.get(f"{arch}.block_count")
        n_kv = kv.get(f"{arch}.attention.head_count_kv")
        if isinstance(n_kv, list):
            n_kv = max(int(x) for x in n_kv) if n_kv else None
        n_head = kv.get(f"{arch}.attention.head_count")
        if isinstance(n_head, list):
            n_head = max(int(x) for x in n_head) if n_head else None
        d_emb = kv.get(f"{arch}.embedding_length")
        d_k = kv.get(f"{arch}.attention.key_length")
        d_v = kv.get(f"{arch}.attention.value_length")
        if d_k is None and d_emb and n_head:
            d_k = d_emb // n_head
        if d_v is None:
            d_v = d_k
        if not (n_layer and n_kv and d_k and d_v):
            return None, ("UNMEASURABLE from this header (missing block_count / head_count_kv / "
                          "key_length) - no fit estimate is printed")
        # HYBRID ATTENTION: NOT EVERY LAYER HOLDS A KV CACHE.
        # MEASURED 2026-08-28 on qwen4exp (Qwen3.8-Flash-Next). The flat
        # n_layer form above overestimates this arch by EXACTLY 4x, because
        # full_attention_interval=4 means only 12 of its 48 layers keep a KV
        # cache; the other 36 are gated-delta-net linear-attention layers whose
        # recurrent state is FIXED per sequence and does not grow with context.
        # Left uncorrected the launcher prices 160k ctx at 15.0 GiB instead of
        # 3.75 GiB and refuses seats that fit comfortably.
        kv_layers = n_layer
        interval = kv.get(f"{arch}.full_attention_interval")
        ratios = kv.get(f"{arch}.attention.compress_ratios")
        how = f"{n_layer} layers"
        if isinstance(ratios, (list, tuple)) and len(ratios) == n_layer:
            # Most direct evidence: one entry per layer, nonzero => full attention.
            n_full = sum(1 for r in ratios if r)
            if 0 < n_full < n_layer:
                kv_layers = n_full
                how = f"{n_full} of {n_layer} layers (compress_ratios)"
        elif isinstance(interval, int) and interval > 1 and n_layer % interval == 0:
            kv_layers = n_layer // interval
            how = f"{kv_layers} of {n_layer} layers (full_attention_interval={interval})"
        b = kv_layers * n_kv * (d_k * 2 + d_v * 2)     # f16 K and V
        # THE MEASUREMENT IS PER-ARCH. It was taken on qwen4exp and it does NOT
        # transfer: another arch may count its full-attention layers differently,
        # or keep a per-token component in the linear layers this formula ignores.
        # Claiming MEASURED on an arch nobody booted is the exact failure this
        # file's three-tag rule exists to prevent.
        if arch == "qwen4exp":
            tag = ("MEASURED 2026-08-28: predicts 24576 B/token and the engine allocated "
                   "EXACTLY 3516.00 / 3840.00 / 6144.00 MiB at c=150016 / 163840 / 262144. "
                   "Three exact hits - Q9 is CLOSED for this arch only.")
        elif kv_layers != n_layer:
            tag = ("[INFERRED] hybrid-attention correction applied from the qwen4exp "
                   "measurement, but NOT validated on this arch. Q9 stays OPEN here - "
                   "warn only, never blocks.")
        else:
            tag = ("[INFERRED] UNVALIDATED against a real allocation (Q9) - warn only, "
                   "never blocks")
        return b, (f"arithmetic: {how} x {n_kv} kv-heads x ({d_k}+{d_v}) dims x 2 B (f16) "
                   f"= {b/1024:.1f} KiB/token. {tag}")
    except Exception:
        return None, "UNMEASURABLE (header arithmetic failed) - no fit estimate is printed"


def kv_bytes_per_token_hf(cfg):
    """M8 for a safetensors checkpoint. Same [INFERRED] status as the GGUF form:
    arithmetic on header fields, UNVALIDATED against a real allocation (Q9), so it
    warns and never blocks."""
    t = cfg.get("text_config") or cfg
    try:
        n_layer = t.get("num_hidden_layers")
        n_kv = t.get("num_key_value_heads") or t.get("num_attention_heads")
        d_h = t.get("head_dim")
        if d_h is None and t.get("hidden_size") and t.get("num_attention_heads"):
            d_h = t["hidden_size"] // t["num_attention_heads"]
        if not (n_layer and n_kv and d_h):
            return None, ("UNMEASURABLE from config.json (missing num_hidden_layers / "
                          "num_key_value_heads / head_dim) - no fit estimate is printed")
        b = n_layer * n_kv * d_h * 2 * 2
        return b, (f"[INFERRED] arithmetic: {n_layer} layers x {n_kv} kv-heads x {d_h} head-dim "
                   f"x 2 (K+V) x 2 B (f16) = {b/1024:.1f} KiB/token. UNVALIDATED against a real "
                   f"allocation (Q9) - warn only, never blocks")
    except Exception:
        return None, "UNMEASURABLE (config.json arithmetic failed) - no fit estimate is printed"


def model_bytes(path, kind):
    if kind in ("gguf", "gguf_broken"):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    if kind in ("hf_dir", "vllm_dir", "weightless_dir", "lora_dir", "not_a_model"):
        t = 0
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith((".safetensors", ".bin", ".gguf")):
                    try:
                        t += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
        return t
    return 0


def find_mmproj(model_path, kind):
    """Coverage hole review item 3: the spec asserted a paired projector is
    'discoverable next to the model' and never said how. On this box
    muse-glimmer-30b-abl/ holds TWO F16 projectors and
    muse-glimmer-30b-abl-mmproj/ holds an F16 AND a Q8_0 for the same base model.
    There is no rule that picks a winner, so this does not invent one:
    exactly one candidate -> use it and say so; more than one -> R-24, refuse to
    guess; none -> say none was found."""
    if kind not in ("gguf", "gguf_broken"):
        return [], "mmproj discovery only applies to GGUF seats"
    d = os.path.dirname(os.path.abspath(model_path))
    try:
        cands = sorted(os.path.join(d, f) for f in os.listdir(d)
                       if f.startswith("mmproj") and f.endswith(".gguf"))
    except OSError:
        return [], "model directory unreadable"
    return cands, f"{len(cands)} mmproj*.gguf sibling(s) in {d}"


def docker_has_image(tag):
    """True iff docker actually holds this tag. Advisory only: a missing or broken
    docker degrades to "not present", never to a traceback, because this is called
    while EXPLAINING a decision and an explanation that crashes is worse than a
    vague one. _run() already swallows non-zero exit and timeout and returns None."""
    return _run(["docker", "image", "inspect", "-f", "ok", tag], timeout=10) == "ok"


def has_vllm_pxq4():
    try:
        import vllm_pxq4  # noqa: F401
        return True
    except Exception:
        return False


def infer_workload(np_, explicit):
    return explicit or ("chat" if np_ <= 1 else "serve")


# ---------------------------------------------------------------------------
# vLLM ELIGIBILITY IS AN IMAGE PROPERTY, NOT A COMPUTE CAPABILITY
# ---------------------------------------------------------------------------
def vllm_eligibility(sel, image_arg):
    """-> (eligible_caps:set, image:str|None, probe_trail:[str])

    The old file's `MIN_VLLM_CAP = 70` made the entire vLLM branch unreachable on
    the hardware every vLLM cell in the decision table was measured on. The narrow
    true statement (SPEC CORRECTION C3): NO vLLM decode-or-prefill THROUGHPUT
    number on sm_70 exists anywhere in this corpus - the only sm_70 vLLM figure on
    record is a 3.76x shared-prefix win on the DGX (PORT-ASSESSMENT.md:82), which
    is not a seat decision.

    Resolution order, and the probe that failed is always named:
      1. --vllm-image / PXA_VLLM_IMAGE
      2. the image's declared arch set (table above)
      3. bare-metal vllm_pxq4 importable, arch set from PXA_PXQ4_LIB's sm tag
    If no image resolves, vLLM is NOT eligible and the reason names the probe -
    never a bare 'vLLM is sm_70+ only', which is the false statement it replaces.
    """
    trail = []
    img = image_arg or os.environ.get("PXA_VLLM_IMAGE")
    if img:
        rec = VLLM_IMAGES.get(img)
        if rec is None:
            trail.append(f"probe 1: image {img!r} given but UNKNOWN to this launcher - its arch "
                         f"set is UNMEASURED, so no card is declared eligible on its say-so")
            return set(), img, trail
        if rec["status"] == "INELIGIBLE":
            trail.append(f"probe 1: image {img!r} is INELIGIBLE: {rec['why']}")
            return set(), img, trail
        missing = [q for q in (rec.get("host_env") or {}).get("requires", [])
                   if not os.path.lexists(q)]
        if missing:
            # An image whose runtime lives on the host is only as present as those paths.
            # Declaring it eligible when they are gone routes traffic to a container that
            # cannot import torch, and the failure surfaces minutes later as a boot crash
            # instead of here as a sentence.
            trail.append(f"probe 1: image {img!r} needs host paths that are NOT PRESENT, so "
                         f"no card is eligible under it: " + ", ".join(missing))
            return set(), img, trail
        caps = set(rec["caps"])
        inf = set(rec.get("caps_inferred") or ())
        trail.append(f"probe 1: image {img!r} -> MEASURED caps {sorted(caps) or '-'}"
                     + (f", [INFERRED] caps {sorted(inf)}" if inf else "")
                     + f" ({rec['why']})")
        return caps | inf, img, trail
    trail.append("probe 1: no --vllm-image and no PXA_VLLM_IMAGE")
    lib = os.environ.get("PXA_PXQ4_LIB", "")
    m = re.search(r"libpxq4_sm(\d+)_v\d+\.so", lib)
    if m and has_vllm_pxq4():
        cc = int(m.group(1))
        trail.append(f"probe 3: bare-metal vllm_pxq4 importable AND PXA_PXQ4_LIB names "
                     f"sm_{cc} ({os.path.basename(lib)}) -> eligible for sm_{cc} only")
        return {cc}, f"bare-metal:{os.path.basename(lib)}", trail
    if has_vllm_pxq4():
        trail.append("probe 3: vllm_pxq4 IS importable but PXA_PXQ4_LIB does not name an sm "
                     "tag - the arch set of this build is UNMEASURED, so no card is declared "
                     "eligible. Set PXA_VLLM_IMAGE or PXA_PXQ4_LIB=libpxq4_sm<cc>_v<n>.so")
    else:
        trail.append("probe 3: vllm_pxq4 is not importable in this interpreter")
    # "on this box" has to MEAN on this box. This line previously printed every
    # MEASURED row in the table, so an image that had never been built was advertised
    # as available and the operator was sent to build a command around a tag docker
    # does not have. The table says what an image WOULD be good for; only docker says
    # whether it exists. Print both, and never let a missing docker turn an advisory
    # line into a crash.
    meas = [k for k, v in VLLM_IMAGES.items() if v["status"] == "MEASURED"]
    present = [k for k in meas if docker_has_image(k)]
    absent = [k for k in meas if k not in present]
    trail.append("gated images PRESENT on this box: " + (", ".join(present) or "NONE"))
    if absent:
        trail.append("gated images NOT BUILT here (build them before use): "
                     + ", ".join(absent))
    return set(), None, trail


# ---------------------------------------------------------------------------
# REFUSALS - R-01..R-21 are the spec's; R-22..R-28 were added by adversarial
# review of the spec (each comment says which hole it closes).
# Exit codes, unchanged from the previous version of this file:
#   2 = no decision / unusable artifact / unusable card selection
#   3 = a parameter or engine request that does not translate
#   4 = the environment cannot run the command (no engine binary, no CUDA)
#   5 = --explain produced a plan that carries known-fatal blockers
# ---------------------------------------------------------------------------
class Refusal(Exception):
    def __init__(self, rid, text, code=2):
        super().__init__(text)
        self.rid, self.text, self.code = rid, text, code


R = {
 "R-01": ("REFUSING: PXQ1 content. A PXQ1 MoE file loads, clears the composition gate at 80.9% "
          "PXQ-family bytes, and generates INCOHERENT text - nothing downstream catches it "
          "(PXQ-TYPE-MATRIX.md:113, Finding 7). PXQ1 has no dense path and no CPU codec. "
          "Requantize to PXQ2 or above, or use a curated per-expert PXQ1 map WITH a coherence "
          "check that you ran and can point at."),
 "R-02": ("REFUSING: retired quant type {name}. Types 250/251 were removed 2026-07-21 "
          "(ggml.h:467-470, 'never reuse this id') and no engine we ship reads them. Requantize."),
 "R-03": ("REFUSING to guess the PXQ tier. general.file_type is {ftype} for every PXQ tier by "
          "design (llama-quantize.cpp:1454-1509 rewrites them all to MXFP4=38 before :1658 "
          "writes it) and the tensor directory shows no PXQ tensor types either. Without a tier "
          "I cannot enforce the PXQ1 refusal or the vLLM PXQ4-only gate. Re-quantize with a "
          "current quantizer, or pass --tier <T> to assert it yourself and own that assertion."),
 "R-05": ("REFUSING: {method} is not readable by EITHER engine. llama.cpp reads the PXQ family "
          "and GGUF k-quants; it does not read a {method} safetensors directory. (The message "
          "this replaces told the operator llama.cpp would read it - false, and it would have "
          "sent them to a load failure.)"),
 "R-06": ("REFUSING: vLLM needs a converted artifact and {conv} does not exist. Convert: "
          "tools/vllm-pxq4/tools/gguf_to_vllm.py {model} {conv}. Or --engine llama to serve the "
          "GGUF directly."),
 "R-07": ("REFUSING: forced --engine vllm but NO selected card is eligible under image {img} "
          "({cards}). Not proceeding: with no eligible card the parallel degree collapses to 1, "
          "CUDA_VISIBLE_DEVICES is never set, and the server inherits EVERY GPU on this box. "
          "This is the live bug this refusal exists for - the previous version of this file "
          "reported the blocker and then launched anyway."),
 "R-08": ("REFUSING: cudagraph_mode={mode}. FULL_AND_PIECEWISE captures PREFILL graphs and "
          "returns fluent garbage from character zero on short raw /v1/completions prompts. Its "
          "best aggregate (88.4, SCOREBOARD M8) is BELOW the correct config's (88.7, M7). There "
          "is no speed argument for it."),
 "R-10": ("REFUSING: -ts with vLLM. vLLM splits parallel work EVENLY; a per-card ratio has no "
          "equivalent and would be silently ignored."),
 "R-11": ("REFUSING: -sm {sm} with vLLM. vLLM's parallelism model has no -sm equivalent."),
 "R-12": ("REFUSING: -sm graph on a DeltaNet hybrid ({why}). It produces DEGENERATE output - the "
          "cross-device all-reduce never reaches its consumers and each device computes a "
          "different router top-8. Not fixable by an env: PXA_ALLOW_GRAPH_SPLIT_HYBRID only "
          "removes the guard. Use -sm layer. Even where graph split works it is a phase trade, "
          "not a win: +64% prefill / -17% decode on 4x P100."),
 "R-13L": ("REFUSING: -ctk {k} / -ctv {v} has no compiled FA vec kernel at head 128 on this "
           "build - it does not fall back, it HARD-ABORTS at request time (on_no_fattn_vec_case, "
           "'Unsupported KV type combination for head_size 128'). Compiled asymmetric pairs "
           "here: q8_0/q6_0, q8_0/iq4_nl, q6_0/q5_0 (LEVERS.md:263). NOTE: NO measurement "
           "exists for ANY asymmetric pair - this is an unbenched VRAM trade, not a free one."),
 "R-13V": ("REFUSING: --ctk/--ctv have no vLLM equivalent and would be silently dropped. (They "
           "WERE silently dropped by the previous version of this file: the vLLM branch never "
           "read them and the translation check inspected only -ts/-sm.)"),
 "R-14": ("REFUSING: --spec mtp with vLLM. The vLLM path has no MTP drafter and I will not "
          "substitute ngram for it - on this model class the two have OPPOSITE verdicts (ngram "
          "+23.0% code / +4.6% prose; MTP -8.6% at n_max=1 and -29.8% at n_max=2, LEVERS.md:746). "
          "Substituting a lever's meaning is worse than dropping it. Ask for ngram explicitly if "
          "that is what you want."),
 "R-15A": ("REFUSING: --spec mtp:n_max={n}. n_max>=2 is a MEASURED LOSS on both arches: P100 "
           "54.9 -> 47.4 (-14%, accept 0.42); V100 92.7 vs 94.1 (accept 0.960 -> 0.480) "
           "(LEVERS.md:300-301). Use n_max=1."),
 "R-15B": ("REFUSING: --spec mtp but this file has NO nextn/mtp tensors. Its "
           "nextn_predict_layers KV says {kvn}; the tensor directory says 0. Two shipped f16 "
           "files carry exactly this lie (PXA-Agent-9B-f16.gguf, PXA-Coder-35B-v2-f16.gguf) - "
           "the head was dropped in the recovery pipeline and the flag survived."),
 "R-16": ("REFUSING: {tier} has no CPU codec (GPU-only, open task #62; PXQ-TYPE-MATRIX.md:67, "
          "RELEASE-GATE.md:177). A CPU-only or partially-offloaded run ABORTS. You asked for "
          "-ngl {ngl}: offload every layer or pick another tier."),
 "R-17A": ("REFUSING: -c {ctx} exceeds this model's trained context {trained} "
           "({arch}.context_length)."),
 "R-17B": ("REFUSING: the WEIGHTS ALONE ({mb:.2f} GiB) exceed the total VRAM of the selected "
           "cards ({tb:.2f} GiB across {n}) with full offload requested. This is file-byte "
           "arithmetic, not the KV estimate - it needs no formula and it blocks. Add cards, or "
           "pick a smaller artifact."),
 "R-18": ("REFUSING: {dir} is an unquantized HF checkpoint (config.json has no "
          "quantization_config.quant_method). llama.cpp needs a GGUF; vLLM needs a PXQ4-converted "
          "directory. Convert first - I will not emit 'vllm serve --quantization pxq4' against "
          "fp16 weights."),
 "R-19": ("REFUSING: {dir} contains no .safetensors/.gguf weights (config.json only, {n} bytes "
          "of weight files). A config-only stub is not a model."),
 "R-20": ("REFUSING: card {i} has {mib} MiB resident ({procs}). This is a SHARED, LIVE box. Pass "
          "--cards explicitly for free cards, or --allow-busy if you own that process."),
 "R-21": ("REFUSING: np={n} needs a cudagraph capture ladder above the measured [1,2,4,8]. I can "
          "widen it, but nothing above np=8 has been measured on either engine and a too-short "
          "ladder has cliffed before (task #78: a hardcoded [1,2] cliffed at 3+ concurrent). "
          "Re-run with --accept-unmeasured."),
 # ---- added by adversarial review of LAUNCHER-SPEC (not in R-01..R-21) ----
 "R-22": ("REFUSING: {path} exists but is not a servable artifact ({kind}). {detail} The "
          "taxonomy this replaces had no slot for it and reported 'path does not exist' for a "
          "path that plainly does."),
 "R-23": ("REFUSING: {path} is not a usable GGUF - {detail}. A 5.3 GB file with a zeroed header "
          "sits on this box today (/tmp/TRUNCATED-pxq4.gguf) and the previous version of this "
          "file built a full launch command for it."),
 "R-24": ("REFUSING to guess an mmproj. {n} candidates sit next to this model at different "
          "precisions and no measurement ranks them:\n      {list}\n    Pass --mmproj <path>, or "
          "--no-mmproj to serve text-only."),
 "R-25": ("REFUSING: --draft-model (external draft-model speculation) has ZERO coverage in this "
          "corpus - no decode, prefill, acceptance or quality number exists for it on any cell. "
          "The only measured speculation here is ngram-mod self-speculation (+23.0% code) and "
          "MTP n_max=1 (a LOSS on sparse MoE). Re-run with --accept-unmeasured to try it anyway. "
          "vLLM: no draft-model path is emitted by this launcher at all."),
 "R-26": ("REFUSING: --np {n}. Concurrency must be >= 1; -c would compute to {ctx} and "
          "--max-num-seqs {n} would be emitted verbatim."),
 "R-27": ("REFUSING: this is a multimodal/VL checkpoint (vision_config / vision tensors present) "
          "and the vLLM command this launcher emits has NO multimodal handling - no "
          "--limit-mm-per-prompt, no image-token accounting, no projector. Nothing in the corpus "
          "measured a VL seat on vLLM. Use --engine llama (which does have --mmproj), or "
          "--accept-unmeasured if you intend to serve it text-only."),
 "R-29": ("REFUSING: {dir} is a PXQ4-CONVERTED vLLM directory, but the engine that wins this "
          "seat is llama.cpp - which cannot read safetensors. Why llama.cpp: {why} "
          "Serve the GGUF this directory was converted FROM, or change the inputs so vLLM wins "
          "(a vLLM-eligible image + card selection, and np>=6 for a MoE)."),
 "R-28": ("REFUSING: this file carries {n} tensor(s) of ggml type id(s) {ids}, which the CURRENT "
          "tree does not define (checked against ggml/include/ggml.h; PXQ1C/PXQ2C appear nowhere "
          "in the tree). The engine cannot dispatch them. A real file on this box is in exactly "
          "this state: qwen3-coder-next-pxqu/Fusion-Coder-80-PXQU.gguf, 106 tensors of type 247 "
          "and 38 of type 246 with pxa.pxq1c.*/pxa.pxq2c.* provenance KVs. Requantize with the "
          "current quantizer."),
}


# ---------------------------------------------------------------------------
# THE DECISION
# ---------------------------------------------------------------------------
class Plan(object):
    def __init__(self):
        self.engine = None
        self.reason = ""
        self.evidence = []      # each line already carries MEASURED/[INFERRED]/UNMEASURED
        self.notes = []
        self.blockers = []
        self.refusals = []      # (rid, text, code)
        self.needs_ack = []     # reasons --accept-unmeasured is required
        self.elig = []          # eligible GPU rows for vLLM
        self.image = None

    def refuse(self, rid, code=2, **kw):
        self.refusals.append((rid, R[rid].format(**kw), code))


def envelope_notes(plan, sel, prof, np_, per_slot_ctx, model_path):
    """Section 4.4. The whole decision table is keyed to TWO CARDS OF ONE CLASS.
    Outside that envelope the launcher labels the answer and, where the spec says
    so, requires --accept-unmeasured. SCOREBOARD.md:273 is the evidence that even
    2->2 does not transfer: 22.3 vs 24.6 tok/s on an IDENTICAL config between card
    pairs 1,5 and 0,6."""
    caps = sorted({g[2] for g in sel})
    n = len(sel)
    if n == 2 and caps == [60]:
        plan.evidence.append("MEASURED envelope: exactly 2 cards, both sm_60 - the table applies "
                             "as measured (SCOREBOARD.md:6, MOE-CROSSOVER.md:3)")
    elif n == 2 and caps == [70]:
        plan.notes.append("[INFERRED]: 2x sm_70. NO engine-vs-engine number exists on Volta on "
                          "this box, for either class. The table below is applied, not measured "
                          "here.")
    elif n == 2 and len(caps) > 1:
        plan.notes.append("mixed-class pair: PXA_AUTO_TS fills -ts 1.4,0.6 on an exactly-2-device "
                          "mixed sm_70+sm_60 pair with -ts unset (+9.78% decode) - MEASURED on "
                          "ONE cell only (LEVERS.md:154, 'PXQ4-35B split V100+P100'); it does not "
                          "generalize.")
    if n not in (1, 2):
        plan.notes.append(f"UNMEASURED card count: the MoE crossover and the dense pair were both "
                          f"measured on exactly 2 cards. You selected {n}. The 2-card answer is "
                          f"printed and labelled [INFERRED]. SCOREBOARD.md:273 shows 22.3 vs 24.6 "
                          f"on an identical config between two P100 PAIRS - even 2->2 does not "
                          f"transfer cleanly.")
        plan.needs_ack.append(f"{n}-card selection (table is 2-card only)")
    if 61 in caps:
        plan.notes.append("selection contains an sm_61 card: the np5/np6 thresholds were NEVER "
                          "measured on sm_61, and the BALANCE-mode PXA_FA_MASK_SKIP_TILE win "
                          "explicitly excludes all of sm_61 (LEVERS.md:85). UNMEASURED.")
    if per_slot_ctx != ANCHOR_CTX_PER_SLOT:
        plan.notes.append(f"ctx per slot is {per_slot_ctx}, not the measured {ANCHOR_CTX_PER_SLOT}: "
                          f"the anchors ran --ctx-size np*4096. A different per-slot context "
                          f"changes the KV footprint and was NOT measured.")
    base = os.path.basename(model_path).lower()
    if prof.get("is_moe"):
        if not any(h in base for h in ANCHOR_MOE_HINTS):
            plan.notes.append("model is not the MoE anchor (PXA-Coder-35B-v2): the crossover is a "
                              "property of a model x hardware PAIR, not of the engines. "
                              "qwen3next (MoE-512), qwen3moe (MoE-128) and deepseek4 (MoE-6) have "
                              "NO engine-vs-engine data at any np. [INFERRED]")
    elif prof.get("tier"):
        if not any(h in base for h in ANCHOR_DENSE_HINTS):
            plan.notes.append("model is not the dense anchor (Qwen3.8-27B-Unc): the dense ratios "
                              "below are that model's, on one boot of the losing side. [INFERRED]")
    # H5 heterogeneity - coverage review item 8.
    ubs = {ub_for_card(g[3]) for g in sel}
    if len(ubs) > 1:
        plan.notes.append(f"HETEROGENEOUS -ub expectation across this selection: the card-type "
                          f"table (LEVERS.md:99-103) wants {sorted(ubs)} on different cards while "
                          f"the CLI carries ONE global -ub. This launcher passes NO -ub so the "
                          f"engine's adaptive-ub probes each device itself - but whether "
                          f"adaptive-ub lands per-card correctly in a heterogeneous pool is "
                          f"UNMEASURED.")


def decide(sel, kind, model, forced, prof, np_, workload, elig_caps, image, probe_trail,
           per_slot_ctx=ANCHOR_CTX_PER_SLOT):
    """-> Plan. Refusals first, always (spec section 4.0 order of evaluation)."""
    p = Plan()
    p.image = image
    prof = prof or {}
    names = ", ".join(f"{g[0]}:{g[1].replace('NVIDIA ', '')} sm_{g[2]}" for g in sel)
    tier = prof.get("tier")
    p.elig = [g for g in sel if g[2] in elig_caps]

    # ---- 1. artifact resolvable? -------------------------------------------
    if kind == "missing":
        p.reason = f"model path does not exist: {model}"
        return p
    if kind == "not_a_model_file":
        b = os.path.basename(model)
        detail = "It is an existing file this launcher cannot serve."
        m = re.search(r"-(\d{5})-of-(\d{5})\.safetensors$", b)
        if m:
            detail = (f"It is shard {int(m.group(1))} of {int(m.group(2))} of a safetensors "
                      f"checkpoint - pass the DIRECTORY, not one shard.")
        elif b.endswith(".tiers"):
            detail = "It is a PXQ-UNIVERSAL tier map (--pxq-universal input), not a model."
        elif b.endswith(".imatrix"):
            detail = "It is an importance matrix (quantizer input), not a model."
        elif b.endswith(".safetensors"):
            detail = "It is a single safetensors file - pass the DIRECTORY that contains it."
        p.refuse("R-22", path=model, kind=kind, detail=detail)
        return p
    if kind == "lora_dir":
        p.refuse("R-22", path=model, kind="LoRA adapter directory",
                 detail="It has adapter_config.json + adapter weights and no base model. "
                        "Merge or subtract it first, then serve the resulting checkpoint.")
        return p
    if kind == "weightless_dir":
        p.refuse("R-19", dir=model, n=model_bytes(model, kind))
        return p
    if kind == "not_a_model":
        p.reason = f"path exists but has no config.json and is not a .gguf: {model}"
        return p
    if kind in ("gguf", "gguf_broken") and prof.get("hdr_err"):
        p.refuse("R-23", path=model, detail=prof["hdr_err"])
        return p
    if kind == "hf_dir":
        qm = prof.get("quant_method")
        if qm in ("compressed-tensors", "awq", "gptq", "fp8", "bitsandbytes"):
            p.refuse("R-05", method=qm)
        else:
            p.refuse("R-18", dir=model)
        return p

    # ---- 2/3. tier readable? tier servable? --------------------------------
    if prof.get("unknown_types"):
        u = prof["unknown_types"]
        cnt = sum(prof.get("hist", {}).get(t, 0) for t in u)
        p.refuse("R-28", n=cnt, ids=", ".join(str(t) for t in u))
        return p
    ft = prof.get("ftype")
    if isinstance(ft, int) and ft in RETIRED_FTYPE:
        p.refuse("R-02", name=RETIRED_FTYPE[ft])
        return p
    if tier == "PXQ1":
        # I-7. Positive detection by tensor type, so this fires on a uniform PXQ1
        # file AND on a curated UNIVERSAL map with PXQ1-mapped experts - the
        # HIGH-severity silent case the review found unguarded in the spec, where
        # R-01's condition was `tier in {PXQ1}` and a PXQ_UNIVERSAL tag never matched.
        p.refuse("R-01")
        return p
    if kind in ("gguf",) and tier is None:
        # Not a PXQ file at all. That is fine and common (K-quants, MXFP4, f16) -
        # it is only a refusal when the file ALSO claims MXFP4 ftype with no
        # readable composition, which is the ambiguous case R-03 exists for.
        if ft == FTYPE_MXFP4:
            p.refuse("R-03", ftype=ft)
            return p
    if tier and prof.get("tier_kv") and prof["tier_kv"] != tier:
        # Never resolve a conflict silently. Both signals get printed; the tensor
        # walk wins because it is what the loader dispatches on.
        p.notes.append(f"TIER SIGNAL CONFLICT: tensor directory says {tier}; provenance KV says "
                       f"{prof['tier_kv']}. Using {tier} (the loader dispatches on tensor type, "
                       f"llama-model-loader.cpp:527). Both are printed rather than reconciled - "
                       f"this file was built by a quantizer whose KV conditions differ from "
                       f"llama-quantize.cpp:1760-1790 as it stands today.")
    if tier == "PXQ_UNIVERSAL":
        p.notes.append("PXQ_UNIVERSAL: this is a MIXED per-tensor tier map. "
                       "PXQ-TYPE-MATRIX.md:119 Finding 8 records a UNIVERSAL MoE that loads PASS "
                       "and generates INCOHERENT - with the doc's own caution that the "
                       "incoherence is traceable to that build recipe (Q3_K_M source, no imatrix) "
                       "rather than to the codecs. Nothing verifies a coherence check ran on THIS "
                       "file. llama.cpp only, and you must acknowledge it.")
        p.needs_ack.append("PXQ_UNIVERSAL tier (no coherence check is verified for this file)")

    if not sel and not forced:
        p.reason = "no GPUs selected/visible (use --engine to force anyway)"
        return p

    # ---- 5. structural gates: what CAN run, before what is fastest ---------
    forced_v = forced == "vllm"
    if forced_v and sel and not p.elig:
        p.refuse("R-07", code=3, img=image or "<none resolved>",
                 cards=names or "no GPUs visible")
        return p
    if forced:
        p.engine = forced
        p.reason = f"forced by --engine ({names or 'no GPUs visible'})"
        if not sel:
            p.blockers.append("no GPUs visible - proceeding on your say-so; the engine may fail "
                              "to start, and CUDA_VISIBLE_DEVICES cannot be scoped (I-12)")
        if forced_v:
            if tier and tier not in VLLM_SUPPORTED_PXQ:
                p.blockers.append(f"FORCED vllm but this model is {tier} and the vLLM backend "
                                  f"implements PXQ4 only (PXQ-TYPE-MATRIX.md:69-70)")
            if kind == "gguf":
                p.blockers.append("FORCED vllm but the model is a raw .gguf - vLLM needs the "
                                  "converted form")
        if forced == "llama" and kind == "vllm_dir":
            p.blockers.append("FORCED llama but the model is a PXQ4-CONVERTED vLLM directory - "
                              "llama.cpp cannot read safetensors. Serve the GGUF it was "
                              "converted from.")
        for t in probe_trail:
            p.evidence.append("eligibility " + t)
        return p

    if tier and tier not in VLLM_SUPPORTED_PXQ and tier != "PXQ4":
        p.engine, p.reason = "llama", (
            f"tier is {tier}: the vLLM backend implements PXQ4 ONLY and refuses every other tier "
            f"cleanly at the conversion gate (PXQ-TYPE-MATRIX.md:69-70). llama.cpp reads "
            f"PXQ2/PXQ3/PXQ4/PXQ4-HQ/PXQ6/UNIVERSAL. Cards: {names}")
        p.evidence.append("MEASURED tier support: PXQ-TYPE-MATRIX.md:69-70, :80-81")
    elif tier is None and kind == "gguf":
        p.engine, p.reason = "llama", (
            f"no PXQ tensors in this file (composition: "
            f"{compose_str(prof.get('hist', {}))}); it is a stock GGUF quantization. "
            f"vLLM's PXQ4 backend has nothing to load. Cards: {names}")
    elif kind == "gguf":
        p.engine, p.reason = "llama", (
            f"model is a raw GGUF ({os.path.basename(model)}); vLLM needs a converted artifact "
            f"(tools/vllm-pxq4/tools/gguf_to_vllm.py). Cards: {names}")
    elif not p.elig:
        p.engine, p.reason = "llama", (
            f"no vLLM-eligible card in this selection ({names}). "
            f"{probe_trail[-1] if probe_trail else 'no image probe succeeded'}")
        for t in probe_trail:
            p.evidence.append("eligibility " + t)
    elif len(p.elig) < len(sel):
        p.engine, p.reason = "llama", (
            f"only {len(p.elig)}/{len(sel)} selected cards are eligible under image "
            f"{image}; vLLM cannot span a mixed-arch selection on a single-arch image, and "
            f"dropping the rest would silently change the parallel degree. Cards: {names}")
    elif len({g[2] for g in p.elig}) > 1:
        p.engine, p.reason = "llama", (
            f"selection spans more than one compute capability ({sorted({g[2] for g in p.elig})}) "
            f"and the vLLM command carries ONE --attention-backend. Cards: {names}")
    elif len(sel) < 2:
        p.engine, p.reason = "llama", (
            f"single GPU - no parallelism to gain and llama.cpp has lower single-stream "
            f"overhead. This is also the only measured single-card ground in the corpus "
            f"(QUANT-SPEED-AB.log, 1x P100 card 1). Cards: {names}")
    else:
        # ---- 6. both engines can run it. Pick on MEASURED performance. -----
        if not prof.get("is_moe"):
            p.engine = "vllm"
            s, a8, pf = DENSE_NUMBERS["single"], DENSE_NUMBERS["agg8"], DENSE_NUMBERS["prefill"]
            p.reason = (f"DENSE model on {len(sel)} eligible card(s) ({names}). vLLM wins dense at "
                        f"every workload MEASURED: {s[0]} vs {s[1]} tok/s single ({s[2]}), "
                        f"{a8[0]} vs {a8[1]} agg@8 ({a8[2]}), {pf[0]} vs {pf[1]} prefill ({pf[2]}).")
            p.evidence.append("MEASURED dense: SCOREBOARD rows D1 (llama.cpp) / D3 (vLLM), "
                              "27B PXQ4, 2x P100 sm_60, cards 1,5")
            p.notes.append(DENSE_WEAKNESS)
            if np_ == 4:
                p.notes.append(DENSE_AGG4_NOTE)
        elif workload == "longdoc":
            p.engine = "llama"
            p.reason = (f"MoE model, long-document workload ({names}). llama.cpp -sm layer holds "
                        f"the prefill record at every concurrency measured.")
            p.notes.append(MOE_LONGDOC_NOTE)
            p.evidence.append("MEASURED MoE longdoc: SCOREBOARD section 0.2 - cross-harness, "
                              "see the note")
        else:
            p.engine, p.reason, ev, nts = moe_seat(np_, names)
            p.evidence.extend(ev)
            p.notes.extend(nts)

    # A converted vLLM directory is not readable by llama.cpp. Routing one there
    # would emit a llama-server command against safetensors - a runnable-looking
    # command that cannot run. This gate did not exist before.
    if p.engine == "llama" and kind == "vllm_dir":
        p.refuse("R-29", dir=model, why=(p.reason or "no eligible card") + ".")
        return p

    envelope_notes(p, sel, prof, np_, per_slot_ctx, model)
    if p.engine == "vllm":
        if prof.get("is_moe"):
            p.notes.append(MOE_CURRENCY_NOTE)
        for t in probe_trail:
            p.evidence.append("eligibility " + t)
    return p


def moe_seat(np_, names):
    """The split seat. The TABLE is stored, never a slope (see MOE_TABLE)."""
    ev, nts = [], []
    if np_ <= 1:
        l, v, w, m = MOE_NP1
        ev.append(f"MEASURED np=1: llama.cpp {l} vs vLLM {v} tok/s ({m}) [SCOREBOARD M1/M7]")
        return "llama", (f"MoE at np=1 on {names}. llama.cpp -sm layer wins single-stream by "
                         f"{m}: {l} vs {v} tok/s."), ev, nts
    if np_ in MOE_TABLE:
        l, v, w, m = MOE_TABLE[np_]
        ev.append(f"MEASURED np={np_}: llama.cpp {l} vs vLLM {v} tok/s, {w} by {m} "
                  f"[MOE-CROSSOVER.md section 1, 11 gated boots, cards 0+6]")
        if np_ == 5:
            nts.append("np=5 is llama.cpp's PEAK (79.49, ABOVE its own np4 75.93) and it drops "
                       "12.5% in ONE step to np6. Do not read np5 as a point on a line from np4 "
                       "to np8 - that reading misprices it by ~14%.")
        eng = "llama" if w == "llama" else "vllm"
        return eng, (f"MoE at np={np_} on {names}. {'llama.cpp -sm layer' if eng == 'llama' else 'vLLM PP + FULL_DECODE_ONLY'} "
                     f"wins by {m}: {l} vs {v} tok/s aggregate."), ev, nts
    if np_ < 4:
        # np2/np3: bracketed by MEASURED np1 (+214%) and np4 (+17.1%), both
        # llama.cpp, so the WINNER is safe and the MARGIN is not. No cell exists.
        ev.append(f"[INFERRED] np={np_}: no np2/np3 cell exists. Bracketed by MEASURED np1 "
                  f"(llama.cpp 3.14x) and np4 (llama.cpp +17.1%), both llama.cpp.")
        nts.append(f"np={np_} is [INFERRED]: the winner is bracketed on both sides and is safe; "
                   f"the MARGIN is unknown and no number is printed for it.")
        return "llama", f"MoE at np={np_} on {names}. llama.cpp, by bracketing - see the note.", ev, nts
    # np > 8
    ev.append(f"[INFERRED] np={np_}: nothing above np={MOE_TABLE_MAX_NP} was run on either "
              f"engine. Both trends are established THROUGH np8 (llama.cpp 62.42 falling, vLLM "
              f"95.81 climbing).")
    nts.append(f"np={np_} is above the measured table. The direction is [INFERRED]; the capture "
               f"ladder is the real unknown and it is UNMEASURED (see R-21).")
    return "vllm", f"MoE at np={np_} on {names}. vLLM, by extrapolated direction - see the note.", ev, nts


def compose_str(hist):
    parts = []
    for t, n in sorted(hist.items(), key=lambda kv: -kv[1])[:6]:
        nm = PXQ_GGML_TYPE.get(t) or NON_PXQ_GGML_TYPE.get(t) or f"type{t}"
        parts.append(f"{nm}x{n}")
    return " ".join(parts) if parts else "empty"


# ---------------------------------------------------------------------------
# VRAM - see SPEC CORRECTION C4. Only formula-free facts may block.
# ---------------------------------------------------------------------------
def vram_check(plan, sel, mbytes, ctx, prof, ngl_all):
    notes = []
    if not sel:
        return notes
    total = sum(g[3] for g in sel) * 1024 * 1024
    free = sum((g[3] - g[4]) for g in sel) * 1024 * 1024
    # R-17B COMPARES GPU-RESIDENT BYTES, NOT FILE BYTES.
    # A per-layer-embedding table is pinned to host RAM by -ot and NEVER reaches
    # VRAM, so counting it here refuses seats that fit. MEASURED 2026-08-28:
    # Flash-Next PXQU is a 96.77 GiB file of which 51.15 GiB is per_layer_token_embd;
    # the GPU-resident remainder is 46.70 GiB and runs on five cards (75 GiB) with
    # room for a 3.75 GiB KV cache at 160k. Uncorrected, R-17B refused it outright.
    gpu_bytes = mbytes
    ple_note = None
    if mbytes and prof.get("ple_bytes"):
        gpu_bytes = mbytes - prof["ple_bytes"]
        ple_note = (f"PLE: {prof['ple_tensor']} is {prof['ple_bytes']/BYTES_PER_GIB:.2f} GiB and "
                    f"is pinned to host RAM by -ot, so the VRAM figures below use the "
                    f"GPU-resident remainder {gpu_bytes/BYTES_PER_GIB:.2f} GiB, not the "
                    f"{mbytes/BYTES_PER_GIB:.2f} GiB file.")
    elif mbytes and prof.get("ple_tensor"):
        ple_note = ("PLE present but its ggml type is not in the block-geometry table, so its "
                    "bytes were NOT subtracted. The VRAM figures below OVERSTATE what reaches "
                    "the cards - treat any refusal here as suspect and re-check by hand.")
    if ple_note:
        notes.append(ple_note)
    if gpu_bytes and ngl_all and gpu_bytes > total:
        plan.refuse("R-17B", mb=gpu_bytes / BYTES_PER_GIB, tb=total / BYTES_PER_GIB, n=len(sel))
        return notes
    if not mbytes:
        return notes
    kvb = prof.get("kv_bytes_tok")
    if kvb:
        kv = kvb * ctx
        need = gpu_bytes + kv
        notes.append(f"VRAM estimate [INFERRED, never blocks]: weights {gpu_bytes/BYTES_PER_GIB:.2f} "
                     f"GiB + KV {kv/BYTES_PER_GIB:.2f} GiB (= ctx {ctx} x "
                     f"{kvb/1024:.1f} KiB/tok) = {need/BYTES_PER_GIB:.2f} GiB vs "
                     f"{free/BYTES_PER_GIB:.2f} GiB free / {total/BYTES_PER_GIB:.2f} GiB total "
                     f"across {len(sel)} card(s). Compute buffers and fragmentation are NOT in "
                     f"this number.")
        notes.append("KV/token source: " + prof.get("kv_bytes_src", "?"))
        if need > free:
            notes.append("TIGHT FIT by that estimate. It is arithmetic, not a measurement (Q9 is "
                         "'measure KV bytes/token per arch family'), so it WARNS and does not "
                         "block. Reduce -c or add cards if the boot OOMs.")
        # MEASURED 2026-08-28: A CONFIG THAT LOADS IS NOT A CONFIG THAT RUNS.
        # A 5-card Flash-Next seat at c=262144 loaded cleanly, printed every buffer,
        # and then died on the FIRST TOKEN with "CUDA error: out of memory" in
        # llama_decode - the split had left card 0 with 51 MiB. Decode allocates
        # transient buffers beyond the compute buffer llama.cpp reports at init.
        # The same seat at c=163840 kept 807-1803 MiB free per card and ran.
        notes.append("HEADROOM RULE [MEASURED]: leave ~1200 MiB free per card AFTER load. "
                     "Decode allocates transient buffers that the init-time buffer report does "
                     "not include, so a seat can load and still OOM on its first token.")
    else:
        notes.append("VRAM estimate: " + prof.get("kv_bytes_src", "UNMEASURABLE") +
                     f" (weights alone {mbytes/BYTES_PER_GIB:.2f} GiB vs "
                     f"{free/BYTES_PER_GIB:.2f} GiB free)")
    return notes


# ---------------------------------------------------------------------------
# ENGINE RESOLUTION (llama.cpp build dirs)
# ---------------------------------------------------------------------------
ENGINE_DIR_CANDIDATES = [
    "<local-path>",   # the tree this launcher ships in
    "/mnt/models/pxa-sky-build/build70",          # DGX, sm_70
    "/mnt/models/pxq_llama/build70",              # DGX, sm_70
    "<local-path>",          # Unraid, sm_60;70 - serves PXQ4 in production
    "<local-path>",             # Unraid, second live seat
    "<local-path>",
    "<local-path>",
]


def engine_ld_path(E):
    """These builds link libllama / libggml / libmtmd out of the build tree, not a
    system prefix. Without them the binary exists, is executable, and dies
    instantly on a missing .so.

    AND: any inherited /stubs directory is STRIPPED. Review found this - and
    PERPLEXITY-RESULTS.md:41-55 reproduces it on this box: a 66 KB stub
    libcuda.so.1 shadows the real 96 MB driver, ggml logs one line
    ('CUDA driver is a stub library'), offloads 0/33 layers, and the run is
    numerically CORRECT and ~50x SLOWER. Several existing helper scripts on this
    box still carry the stub dir. Returns (path, dropped[])."""
    parts = [f"{E}/bin", f"{E}/src", f"{E}/ggml/src", f"{E}/examples/mtmd",
             f"{E}/common", f"{E}/ggml/src/ggml-cuda"]
    existing = [p for p in parts if os.path.isdir(p)]
    prior = [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
    dropped = [x for x in prior if "/stubs" in x]
    kept = [x for x in prior if "/stubs" not in x]
    return ":".join(existing + kept), dropped


def engine_runs(E):
    exe = f"{E}/bin/llama-server"
    if not os.path.isfile(exe) or not os.access(exe, os.X_OK):
        return False, "no executable bin/llama-server"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"], _ = engine_ld_path(E)
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=25, env=env)
    except Exception as e:
        return False, f"{e.__class__.__name__} invoking --version"
    blob = (r.stdout or "") + (r.stderr or "")
    if "error while loading shared libraries" in blob:
        missing = blob.split("error while loading shared libraries:")[-1].strip().split(":")[0]
        if missing.startswith(("libcudart", "libcuda.", "libcublas", "libnvrtc")):
            return False, f"NO CUDA RUNTIME ON THIS HOST ({missing})"
        return False, f"cannot load {missing} (even with the build's own lib dirs on LD_LIBRARY_PATH)"
    if r.returncode != 0 and not blob.strip():
        return False, f"--version exited {r.returncode} with no output"
    return True, "starts"


def resolve_engine_dir():
    tried = []
    env = os.environ.get("PXA_ENGINE_DIR")
    if env:
        ok, why = engine_runs(env)
        return env, ("PXA_ENGINE_DIR" if ok else f"PXA_ENGINE_DIR -- WILL NOT START: {why}")
    for d in ENGINE_DIR_CANDIDATES:
        if not os.path.isfile(f"{d}/bin/llama-server"):
            continue
        ok, why = engine_runs(d)
        if ok:
            note = "auto-detected" if not tried else f"auto-detected (skipped {len(tried)} broken)"
            return d, note
        tried.append(f"{d} ({why})")
    onpath = shutil.which("llama-server")
    if onpath:
        d = os.path.dirname(os.path.dirname(onpath))
        ok, why = engine_runs(d)
        if ok:
            return d, "found on PATH"
        tried.append(f"{d} on PATH ({why})")
    if tried:
        if all("NO CUDA RUNTIME ON THIS HOST" in t for t in tried):
            head = "this host has no CUDA runtime, so no build here can start:"
        else:
            head = "every candidate build is present but will not start:"
        return None, head + "\n      " + "\n      ".join(tried)
    return None, "no llama-server found"


# ---------------------------------------------------------------------------
# COMMAND CONSTRUCTION
# ---------------------------------------------------------------------------
COMPILED_CTKV_PAIRS = {("f16", "f16"), ("q8_0", "q8_0"), ("q6_0", "q6_0"), ("q5_0", "q5_0"),
                       ("q4_0", "q4_0"), ("q8_0", "q6_0"), ("q8_0", "iq4_nl"), ("q6_0", "q5_0")}


def parse_spec(spec):
    """-> (method, params dict) for 'mtp:n_max=1' / 'ngram-mod:n_max=4,n_min=2'."""
    if not spec:
        return None, {}
    method, _, rest = spec.partition(":")
    params = {}
    for kvp in rest.split(","):
        if "=" in kvp:
            k, _, v = kvp.partition("=")
            params[k.strip()] = v.strip()
    return method.strip(), params



# ---------------------------------------------------------------------------
# AUTOMATIC -ts FROM REAL FREE VRAM
# ---------------------------------------------------------------------------
# MEASURED 2026-08-28 on Flash-Next PXQU over cards 0,1,3,5,6.
#
# WHY THIS EXISTS: an even split is the default and it is WRONG on any pool where
# the cards are not equally free. Card 0 here carries a production seat (7865 of
# 16384 MiB free) and card 3 is an 11 GiB 1080 Ti also carrying production. An
# even five-way split puts ~9.2 GiB of weights on a card with 7.8 GiB free.
#
# -ts PARTITIONS BYTES, NOT LAYERS (llama.cpp:4071-4100) and llama.cpp folds a
# per-device compute allowance into the same walk - which is why CHANGING -ub
# REPACKS THE LAYERS and a -ts tuned at one ub can OOM at another (measured: PXQ4
# six-card at ub2048 pushed a 16140 MiB V100 over with the ub512-tuned split).
#
# THE HEADROOM TERM IS THE WHOLE POINT. A config that LOADS is not a config that
# RUNS: at c=262144 this model loaded, printed every buffer, then died on the
# first token with "CUDA error: out of memory" in llama_decode because the split
# left card 0 with 51 MiB. Decode allocates transient buffers that the init-time
# buffer report does not include.
#
# Compute-buffer figures are MEASURED at ub1024 and interpolated linearly in ctx:
#   ordinary card   282 MiB @ c8192   567 @ c150016   786 @ c262144
#   head card       980 MiB, FLAT - it did not grow with context across that range
TS_HEADROOM_MIB = 1200      # MEASURED: 807-1803 free per card ran; 51 free OOMd
TS_CUDA_CTX_MIB = 250       # per-device CUDA context, approximate
TS_HEAD_COMPUTE_MIB = 980   # MEASURED, flat in ctx


def _compute_buf_mib(ctx):
    """MEASURED at ub1024, linear interpolation in ctx between the two anchors."""
    lo_c, lo_v, hi_c, hi_v = 8192, 282.0, 262144, 786.0
    if ctx <= lo_c:
        return lo_v
    if ctx >= hi_c:
        return hi_v
    return lo_v + (hi_v - lo_v) * (ctx - lo_c) / (hi_c - lo_c)


def auto_tensor_split(sel, ctx):
    """Capacity-proportional -ts over the SELECTED cards, using free VRAM.

    Returns (ts_string, notes). The LAST device in the selection is treated as the
    head (llama.cpp places the output head on the last device in PCI order), so it
    is charged the larger head compute buffer.
    Returns (None, notes) if any card has no room at all - better to refuse loudly
    than to emit a split that cannot work."""
    notes, caps = [], []
    comp = _compute_buf_mib(ctx)
    for i, g in enumerate(sel):
        total_mib, used_mib = g[3], g[4]
        free_mib = total_mib - used_mib
        is_head = (i == len(sel) - 1)
        overhead = (TS_HEAD_COMPUTE_MIB if is_head else comp) + TS_CUDA_CTX_MIB + TS_HEADROOM_MIB
        caps.append(max(0.0, free_mib - overhead))
    if min(caps) <= 0:
        bad = [g[0] for g, c in zip(sel, caps) if c <= 0]
        notes.append(f"AUTO -ts DECLINED: card(s) {bad} have no capacity left after "
                     f"{TS_HEADROOM_MIB} MiB headroom + compute buffers at ctx={ctx}. "
                     f"Free a card, drop -c, or pass --ts by hand.")
        return None, notes
    tot = sum(caps)
    shares = [int(round(1000 * c / tot)) for c in caps]
    ts = ",".join(str(x) for x in shares)
    detail = "  ".join(f"{g[0]}:{c:.0f}MiB" for g, c in zip(sel, caps))
    notes.append(f"AUTO -ts {ts} [MEASURED method]: capacity-proportional over FREE VRAM after "
                 f"reserving {TS_HEADROOM_MIB} MiB decode headroom + {comp:.0f} MiB compute "
                 f"({TS_HEAD_COMPUTE_MIB} on the head card) + {TS_CUDA_CTX_MIB} MiB context "
                 f"per device. Capacities: {detail}. -ts partitions BYTES, not layers, and "
                 f"llama.cpp repacks when -ub changes - re-derive if you force a different -ub.")
    return ts, notes


def build_llama_cmd(plan, a, sel, prof, ctx, ub_expect, mmproj, explain=False):
    E, note = resolve_engine_dir()
    if E is None:
        print(f"  !! no WORKING llama-server found. {note}")
        if "NO CUDA RUNTIME ON THIS HOST" in note:
            print("     Every candidate build is FINE - this host just has no CUDA runtime.")
            print("     These builds are meant to run inside a CUDA container with the build")
            print("     bind-mounted. Run pxa-launch in there, or set PXA_ENGINE_DIR.")
        else:
            print("     Set PXA_ENGINE_DIR=/path/to/build (the dir containing bin/llama-server).")
        if not explain:
            sys.exit(4)
        # --explain still PRINTS the plan: hiding the command because the local host
        # lacks a CUDA runtime would hide the thing the operator asked to see. The
        # blocker rides the plan and --explain exits 5, so a CI caller can still tell
        # "clean plan" from "plan that will not start here".
        plan.blockers.append("no llama-server build starts on THIS host - the command below "
                             "names <ENGINE>/bin/llama-server, not a resolved binary")
        E = "<ENGINE>"
        note = "auto-detected"          # the failure is already printed above; do not repeat it
    if note != "auto-detected":
        print(f"  engine dir: {E}  [{note}]")

    ldp, dropped = engine_ld_path(E)
    if dropped:
        print(f"  LD_LIBRARY_PATH: DROPPED {dropped} - a CUDA stubs dir on the child's library "
              f"path makes ggml offload 0 layers and run ~50x slower while still producing "
              f"numerically correct output (PERPLEXITY-RESULTS.md:41-55).")

    cmd = [f"{E}/bin/llama-server", "-m", a.model, "--host", a.host, "--port", str(a.port),
           "-ngl", str(a.ngl), "-sm", a.sm, "-c", str(ctx),
           "-ctk", a.ctk, "-ctv", a.ctv, "-np", str(a.np), "-t", str(a.threads),
           "-fa", "on", "--jinja", "--cont-batching"]
    # -b/-ub: NOT EMITTED unless the operator forces one. The engine's adaptive-ub
    # probes real free VRAM per device at startup and falls back to the card-type
    # table (LEVERS.md:99-103). One global -ub across a heterogeneous pool is wrong
    # by construction - the old file hardcoded 2048 for every card, including the
    # 11 GB 1080 Ti where ub2048/1024 compute buffers are measured to OOM.
    if a.ub:
        cmd += ["-b", str(a.ub), "-ub", str(a.ub)]
        print(f"  -ub {a.ub} FORCED by --ub. Adaptive-ub would have been left to choose; the "
              f"card-type table expects {ub_expect} here.")
    else:
        print(f"  -b/-ub: NOT PASSED - adaptive-ub probes each device at startup. Card-type "
              f"table (LEVERS.md:99-103) expects {ub_expect} on this selection. Verify against "
              f"the server's own 'PXA posture: mode=... fa=... ub=...' line.")
    # PLE -> CPU. Not an optimisation: without it the gather table is offloaded with
    # everything else and the load dies with a cudaMalloc of the whole tensor on one
    # card (MEASURED 2026-08-28, 16089.57 MiB on device 2 of a 5-card Flash-Next seat).
    # The pattern is ANCHORED on the full name on purpose - a loose 'ple' regex also
    # matches blk.N.ple_key, ple_conv1d and the F32 ple_norm_* tensors, which are tiny
    # and MUST stay on the GPU.
    if prof.get("ple_tensor") and not any(x == "-ot" for x in cmd):
        cmd += ["-ot", r"per_layer_token_embd\.weight=CPU"]
        _e = prof.get("ple_elems")
        print("  -ot per_layer_token_embd -> CPU: this model carries a PLE gather table"
              + (f" of {_e/1e9:.1f}e9 elements" if _e else "") +
              ". It is GET_ROWS only (no GEMM), so host RAM costs a PCIe gather and frees "
              "a large fraction of the file from VRAM. Without this the load OOMs.")
    if a.ts:
        cmd += ["-ts", a.ts]
        print(f"  -ts {a.ts} FORCED by --ts; the automatic capacity split was not used.")
    elif sel and len(sel) > 1:
        _ts, _tsnotes = auto_tensor_split(sel, ctx)
        for _n in _tsnotes:
            print("  " + _n)
        if _ts:
            cmd += ["-ts", _ts]
    if a.spec:
        cmd += ["--spec-type", a.spec]
    if a.draft_model:
        cmd += ["-md", a.draft_model]
    if mmproj:
        cmd += ["--mmproj", mmproj]

    # ENV - every lever explicitly stated, on or off, with the reason.
    # PXA_ENHANCE=1 is the anchors' env, verbatim (MOE-CROSSOVER.md section 3 arm A).
    env = {"PXA_ENHANCE": "1", "LD_LIBRARY_PATH": ldp}
    if a.no_mmap:
        cmd += ["--no-mmap"]
        env["PXA_PARALLEL_LOAD"] = "1"   # -25..-46% cold load; INERT under mmap (one WARN)
    # GGML_CUDA_NO_PINNED is NOT emitted. It appears in NO measured recipe in this
    # corpus; the previous version of this file set it unconditionally and its
    # effect on the anchors is UNMEASURED (invariant I-11: never set a lever the
    # anchor was not measured with).
    return cmd, env


MEASURED_LADDER = [1, 2, 4, 8]      # MEASURED; the fix for the [1,2] cliff at 3+ (task #78)


def compilation_config(np_):
    """THE ONLY PLACE a vLLM --compilation-config is built. One construction site
    means the two correctness keys cannot be lost on some branch:

      custom_ops:["none"]  is MANDATORY wherever FULL_DECODE_ONLY is emitted on
        sm_60. Without it, PP=2+FDO is a HARD BOOT FAILURE - "CUDA error: an
        illegal memory access was encountered", Worker_PP1 ->
        determine_available_memory -> profile_run (MOE-CROSSOVER.md:271-292,
        container xover-vllm-boot1, ZERO tokens produced). Adding the key fixed it
        with no other change. Every working recipe on this box carries it
        (BUILD-RECIPE.md:95, graphs_arms.sh:6, live fat-smoke-588671). The previous
        version of this file carried it NOWHERE while emitting FDO - i.e. it
        emitted exactly the command proven twice not to boot.

      cudagraph_mode FULL_DECODE_ONLY is a CORRECTNESS requirement, not a knob.
        The vLLM default also captures PREFILL graphs; a raw /v1/completions
        prompt short enough to fit one prefills through a captured graph holding
        stale data and returns fluent garbage from character zero. Chat traffic
        hides it because the template pads past the captured sizes - which is why
        arithmetic gates stayed green while the bug was live. MEASURED: the broken
        config's best aggregate 88.4 (SCOREBOARD M8) is BELOW the correct config's
        88.7 (M7). There is no speed argument for it.
    """
    ladder = list(MEASURED_LADDER)
    while ladder[-1] < np_:
        ladder.append(ladder[-1] * 2)
    return {"custom_ops": ["none"],
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "cudagraph_capture_sizes": ladder}


def build_vllm_cmd(plan, a, prof, ctx, used, image):
    cc = sorted({g[2] for g in used})[0]
    backend, backend_ev = ATTN_BACKEND.get(cc, (None, None))
    model = a.model
    deg = 1
    while deg * 2 <= len(used):
        deg *= 2
    if deg != len(used):
        # H6: vLLM rejects a non-power-of-two degree at startup. Truncating is a
        # REAL change to the run, so it is announced and it needs an ack.
        print(f"  parallel degree truncated to {deg} (power of two) from {len(used)} eligible "
              f"cards; card(s) {[g[0] for g in used[deg:]]} will NOT be used.")
        used = used[:deg]

    # `vllm serve` is correct for a self-contained image. It is WRONG for one whose
    # python lives on the host: vllm is not on PATH in pxa-sm60-dev, so the emitted
    # command could not start, and in non-explain mode this file exits 4 on the
    # shutil.which check - after printing a full, confident plan. An image that declares
    # a host python gets invoked through it, the same way the boot that produced the
    # measurements did.
    hostenv = (VLLM_IMAGES.get(image) or {}).get("host_env") or {}
    hpy = hostenv.get("python")
    if hpy:
        # The model path has to be translated through the same mount the interpreter
        # came through. This launcher runs INSIDE the container, where <local-path> does
        # not exist - only /c does. Emitting the host path produces a command that is
        # correct on paper and cannot find its weights, which surfaces as a load failure
        # minutes in rather than as a sentence here.
        for hsrc, hdst in sorted(hostenv.get("mounts", {}).items(),
                                 key=lambda kv: -len(kv[0])):
            if model.startswith(hsrc.rstrip("/") + "/"):
                mapped = hdst.rstrip("/") + model[len(hsrc.rstrip("/")):]
                print(f"  model path mapped for {image}: {model} -> {mapped} "
                      f"(via -v {hsrc}:{hdst})")
                model = mapped
                break
        else:
            if hostenv.get("mounts"):
                print(f"  ** {model} is OUTSIDE every mount {image} declares "
                      f"({', '.join(hostenv['mounts'])}). The command below names a path "
                      f"that will not exist in that container - mount it, or serve a model "
                      f"that is under one of those roots.")
        cmd = [hpy, "-m", "vllm.entrypoints.openai.api_server", "--model", model,
               "--host", a.host, "--port", str(a.port)]
    else:
        cmd = ["vllm", "serve", model, "--host", a.host, "--port", str(a.port)]
    cmd += ["--quantization", "pxq4", "--dtype", "float16",
            "--max-model-len", str(ctx), "--max-num-seqs", str(a.np),
            "--gpu-memory-utilization", str(a.gmu)]
    if backend:
        cmd += ["--attention-backend", backend]
        print(f"  --attention-backend {backend}: {backend_ev}")
    # H3: no P2P on this box -> custom all-reduce OFF in every measured vLLM arm
    # (MOE-CROSSOVER.md:79). CAR costs ~18% vs NCCL on MoE while the CAR kernel
    # itself is exonerated. Read from topology, not hardcoded.
    if not plan_has_p2p():
        cmd += ["--disable-custom-all-reduce"]

    # MoE goes PIPELINE parallel; dense goes TENSOR parallel. PP=2 is the arm that
    # holds every MoE number in the table (MOE-CROSSOVER.md arm B).
    if prof.get("is_moe") and deg >= 2:
        cmd += ["--pipeline-parallel-size", str(deg), "--tensor-parallel-size", "1"]
        print(f"  MoE -> PIPELINE parallel (PP={deg}, TP=1). MEASURED: PP=2 + FULL_DECODE_ONLY "
              f"is the arm that produced every vLLM MoE cell in the table above.")
    else:
        cmd += ["--tensor-parallel-size", str(max(1, deg))]

    # THE COMPILATION CONFIG - three load-bearing keys.
    #
    # custom_ops:["none"] - I-3, and the single most dangerous omission in the
    #   previous version of this file. PP=2 + FDO WITHOUT this key is a HARD BOOT
    #   FAILURE: "RuntimeError: Worker failed with error 'CUDA error: an illegal
    #   memory access was encountered'", Worker_PP1 -> determine_available_memory
    #   -> profile_run (MOE-CROSSOVER.md:271-292, container xover-vllm-boot1,
    #   produced ZERO tokens). Adding the key fixed it with no other change. Every
    #   working recipe on this box carries it (BUILD-RECIPE.md:95, graphs_arms.sh:6,
    #   live container fat-smoke-588671). `grep -n custom_ops` on the old file
    #   returned 0 hits while it emitted FDO - i.e. it emitted exactly the command
    #   proven twice not to boot.
    #   Scope: MEASURED-REQUIRED on sm_60 PP>=2. MEASURED-present-and-healthy on
    #   sm_60 TP=2. On sm_70 its necessity is UNMEASURED - emitted anyway, which is
    #   the safe direction, and labelled [INFERRED] here.
    #
    # cudagraph_mode:FULL_DECODE_ONLY - I-1/I-2, a CORRECTNESS requirement, never a
    #   tuning knob. The default (FULL_AND_PIECEWISE) also captures PREFILL graphs
    #   at the ladder sizes; a raw /v1/completions prompt short enough to fit one
    #   prefills through a captured graph whose input buffer holds stale data and
    #   returns fluent garbage from character zero. Chat traffic never shows it
    #   because the chat template pads every prompt past the captured sizes - which
    #   is exactly why arithmetic gates stayed green while the bug was live.
    #   MEASURED: the broken config's BEST aggregate (88.4, SCOREBOARD M8) is BELOW
    #   the correct config's (88.7, M7). SPEC CORRECTION C2: the 22.3 -> 24.0
    #   single-stream pair that used to be quoted here as "dense" is a MoE pair
    #   (SCOREBOARD M8/M9, both in table 1a; ENGINE-VERDICT.md:14-22 calls it "the
    #   MoE seat"). Dense TP=2 single is D3 = 24.01 and has NO FAP arm at all.
    #
    # cudagraph_capture_sizes - powers of two covering --max-num-seqs. [1,2,4,8] is
    #   the MEASURED ladder and the fix for the earlier [1,2] cliff at 3+ concurrent
    #   (task #78). Widening past 8 is [INFERRED]: the precedent says a too-short
    #   ladder cliffs, but no np>8 ladder was ever measured.
    cc_obj = compilation_config(a.np)
    if cc_obj["cudagraph_capture_sizes"] != MEASURED_LADDER:
        print(f"  cudagraph_capture_sizes widened to {cc_obj['cudagraph_capture_sizes']} for "
              f"np={a.np}. [INFERRED] - the measured ladder is {MEASURED_LADDER}; nothing above "
              f"np=8 was measured.")
    cmd += ["--compilation-config", json.dumps(cc_obj)]
    if cc != 60:
        print("  custom_ops:[\"none\"] is emitted on sm_%d as well: MEASURED-required on sm_60 "
              "PP>=2, UNMEASURED on sm_70. Emitting it is the safe direction. [INFERRED]" % cc)

    # --speculative-config: ONLY for an explicitly requested ngram spec, with the
    # REQUESTED values. The previous version replaced ANY --spec (including
    # mtp:n_max=1) with a hardcoded {"method":"ngram","num_speculative_tokens":4} -
    # silently substituting a mechanism whose verdict is the OPPOSITE SIGN on this
    # model class (ngram +23.0% code vs MTP -8.6%/-29.8%, LEVERS.md:746).
    method, params = parse_spec(a.spec)
    if method:
        n = int(params.get("n_max", 4))
        cmd += ["--speculative-config",
                json.dumps({"method": "ngram", "num_speculative_tokens": n})]
        print(f"  --speculative-config: ngram, num_speculative_tokens={n} (YOUR values, not a "
              f"substitution).")

    env = {"TORCHDYNAMO_DISABLE": "1",           # MEASURED arm B env, MOE-CROSSOVER.md:80
           "VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}  # MEASURED arm B env, MOE-CROSSOVER.md:80
    # An image whose runtime lives on the host also carries the env that makes that
    # runtime importable. setdefault, not assignment: an explicit value already in the
    # measured arm above wins over the image's default.
    for _k, _v in hostenv.get("env", {}).items():
        env.setdefault(_k, _v)
    if cc == 70:
        # The sm_70 recipes carry this; the sm_60 arm does NOT - it carries
        # SITE=<site> LIB=libpxq4_sm60_v10.so PACKED=1 instead (MOE-CROSSOVER.md:78).
        # The old file emitted the Volta backend name unconditionally, i.e. into a
        # Pascal image - a config the corpus never ran.
        env["VLLM_SM70_QUANT_BACKEND"] = "turbomind"
    else:
        print("  sm_60 arm: the MEASURED recipe carries SITE=<site> LIB=libpxq4_sm60_v10.so "
              "PACKED=1 (MOE-CROSSOVER.md:78). Those are IMAGE-INTERNAL paths - this launcher "
              "will not fabricate them. Declare them in your container invocation or the run is "
              "off the measured envelope (I-11).")
    # VLLM_SM70_FLASH_V100_0DOT3_DECODE_ONLY_CAPTURE is NEVER set: it crash-loops
    # the container at warmup, 3/3 boots (TypeError: 'NoneType' object is not
    # subscriptable in compile_or_warm_up_model -> _dummy_run) and left a seat in a
    # --restart unless-stopped loop needing manual clearing (DGX-FDO-FAILURE.md).
    return cmd, env, used


_P2P = [None]


def plan_has_p2p():
    if _P2P[0] is None:
        ok, _ = peer_topology()
        _P2P[0] = bool(ok)
    return _P2P[0]


# ---------------------------------------------------------------------------
# POST-BOOT VERIFICATION CONTRACT (I-10 / R-09)
# ---------------------------------------------------------------------------
def print_post_boot_contract(engine, cv):
    """Every claim made at plan time needs a matching observation at run time, or
    it is not made. This process EXECs the server, so it cannot observe anything
    after the exec - therefore it makes NO health claim at all, and prints the
    checks whoever owns the seat must run. Passing a flag is not evidence the flag
    took effect: on kewaii/vllm:latest, FULL_DECODE_ONLY parses, boots healthy and
    is SILENTLY OVERRIDDEN back to FULL_AND_PIECEWISE by that image's own compile
    policy (DGX-FDO-FAILURE.md). That is the whole failure class this project keeps
    paying for."""
    print("  POST-BOOT CONTRACT - NOT PERFORMED BY THIS PROCESS (it execs the server):")
    print("    this launcher makes NO healthy/unhealthy claim about the resulting seat.")
    if engine == "vllm":
        print("    1. capture mode: grep the server log for the INSTALLED cudagraph mode.")
        print("       If it is not FULL_DECODE_ONLY, the seat is NOT healthy - shut it down")
        print("       (R-09). A derived image is the fix, not a flag.")
        print("    2. N-way split: per-device resident bytes. MEASURED PP=2 MoE = 10.71 GiB/rank")
        print("       (ENGINE-VERDICT.md section 4).")
    else:
        print("    1. posture: llama-server logs 'PXA posture: mode=... fa=... ub=...' at")
        print("       startup - compare ub against the card-type expectation printed above.")
        print("    2. offload: confirm 'offloaded N/N layers to GPU'. A stub libcuda on the")
        print("       library path yields 0/N, correct output and ~50x slower")
        print("       (PERPLEXITY-RESULTS.md:41-55).")
    print(f"    3. device scoping: echo the child's CUDA_VISIBLE_DEVICES back; expect {cv!r}.")
    print("    4. short-prompt correctness: a RAW, NON-chat-templated 1-token and 5-token")
    print("       completion BEFORE any number is trusted, exactly as all 11 crossover boots")
    print("       did (MOE-CROSSOVER.md section 4.3). Chat-templated traffic pads every prompt")
    print("       past the captured sizes, which is precisely why the FAP corruption survived")
    print("       arithmetic gating.")
    print("    5. speculation: if you armed one, the acceptance-rate line must be present and")
    print("       non-zero. If it is absent, DROP THE CLAIM and keep serving.")


# ---------------------------------------------------------------------------
# SELFTEST
# ---------------------------------------------------------------------------
def selftest(gpus):
    print("=== selftest: decision table against this machine ===")
    print(f"    spec: <local-path> md5 {SPEC_MD5}")
    if gpus:
        for g in gpus:
            print(f"    card {g[0]}: {g[1]:<28} sm_{g[2]}  {g[3]} MiB total, {g[4]} MiB used, "
                  f"-ub table -> {ub_for_card(g[3])}")
    else:
        print("    no GPUs visible - the table below still exercises every branch")
    p2p, topo = peer_topology()
    print(f"    topology: {topo} -> custom all-reduce {'ON' if p2p else 'OFF'} "
          f"(MEASURED: CAR ~18% worse than NCCL on MoE without P2P)")

    idx = [g[0] for g in gpus]
    cases = [("all cards", idx), ("first card", idx[:1])]
    for cc in (60, 61, 70):
        sub = [g[0] for g in gpus if g[2] == cc]
        if sub:
            cases.append((f"all sm_{cc}", sub))
            if len(sub) >= 2:
                cases.append((f"2x sm_{cc}", sub[:2]))
    mixed = [g[0] for g in gpus if g[2] == 60][:1] + [g[0] for g in gpus if g[2] == 70][:1]
    if len(mixed) == 2:
        cases.append(("mixed sm_60+sm_70", mixed))
    if not cases[0][1]:
        cases = [("no GPUs", [])]

    # (label, artifact kind, profile). The kind matters: a converted vLLM directory
    # and a raw GGUF take different structural gates, and a PXQ3 file can only ever
    # BE a GGUF (the converter accepts PXQ4 only).
    profiles = [
        ("dense PXQ4", "vllm_dir", {"is_moe": False, "n_expert": 0, "tier": "PXQ4",
                                    "arch": "qwen35"}),
        ("MoE   PXQ4", "vllm_dir", {"is_moe": True, "n_expert": 256, "tier": "PXQ4",
                                    "arch": "qwen35moe"}),
        ("MoE   PXQ4", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ4",
                                "arch": "qwen35moe"}),
        ("MoE   PXQ3", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ3",
                                "arch": "qwen35moe"}),
        ("MoE   PXQ1", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ1",
                                "arch": "qwen35moe"}),
        ("MoE   UNIV", "gguf", {"is_moe": True, "n_expert": 256, "tier": "PXQ_UNIVERSAL",
                                "arch": "qwen35moe"}),
    ]
    # Run the table twice: once with eligibility as this box actually resolves it,
    # once with the MEASURED sm_60 image forced. On a box where no image resolves,
    # the first pass short-circuits every branch to llama.cpp and would hide the
    # decision table - which is the thing worth testing.
    live_caps, live_img, live_trail = vllm_eligibility(gpus, None)
    passes = [(live_caps, live_img, live_trail,
               f"eligibility as resolved here: image={live_img}, caps={sorted(live_caps) or 'none'}")]
    if not live_caps:
        rec = VLLM_IMAGES["pxa-sm60-dev"]
        passes.append((set(rec["caps"]), "pxa-sm60-dev",
                       ["probe 1: forced for selftest coverage"],
                       "hypothetical: image=pxa-sm60-dev (MEASURED sm_60 caps) - the arm every "
                       "vLLM cell in the table was produced on"))
    for caps, img, trail, banner in passes:
        print(f"  --- {banner} ---")
        for label, ids in cases:
            sel = [g for g in gpus if g[0] in set(ids)]
            for plabel, pkind, prof in profiles:
                for np_ in (1, 4, 5, 6, 8):
                    pl = decide(sel, pkind, "/x/coder35-moe-pxq4", None, prof, np_,
                                infer_workload(np_, None), caps, img, trail)
                    if pl.refusals:
                        out = f"REFUSE {pl.refusals[0][0]}"
                        rest = pl.refusals[0][1].split(".")[0]
                    else:
                        out = str(pl.engine)
                        rest = pl.reason
                    ack = "  <ACK-REQUIRED>" if pl.needs_ack else ""
                    print(f"  {label:18} {plabel:11} {pkind:9} np={np_:<2} -> {out:11} :: "
                          f"{rest[:48]}{ack}")
    print(f"  MoE table (MEASURED, cards 0+6): "
          + ", ".join(f"np{k}={v[2]}" for k, v in sorted(MOE_TABLE.items())))
    print(f"  llama.cpp through np={MOE_LLAMA_MAX_NP}; vLLM from np={MOE_VLLM_MIN_NP}; "
          f"nothing above np={MOE_TABLE_MAX_NP} measured on either engine")
    print(f"  vllm_pxq4 importable in this interpreter: {has_vllm_pxq4()}")
    # STANDING ASSERTIONS, re-checked on every selftest run. These are the two
    # invariants that cost this project a live corruption bug and a dead boot, so
    # they are asserted mechanically rather than trusted to review.
    print("  --- standing assertions ---")
    ok_all = True
    # A1: every compilation-config this file can construct carries BOTH keys, at
    #     every np on the ladder, and never FULL_AND_PIECEWISE.
    bad = []
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33, 64):
        c = compilation_config(n)
        if c.get("cudagraph_mode") != "FULL_DECODE_ONLY":
            bad.append(f"np={n} mode={c.get('cudagraph_mode')}")
        if c.get("custom_ops") != ["none"]:
            bad.append(f"np={n} custom_ops={c.get('custom_ops')}")
        if max(c["cudagraph_capture_sizes"]) < n:
            bad.append(f"np={n} ladder {c['cudagraph_capture_sizes']} does not cover np")
    ok_all &= not bad
    print(f"  A1 every emitted compilation-config = FULL_DECODE_ONLY + custom_ops:[none], "
          f"ladder covers np: {'PASS' if not bad else 'FAIL ' + str(bad)}")
    # A2: FULL_AND_PIECEWISE never appears at an EMISSION SITE. The string is
    #     allowed in prose (the image table's 'why', the FDO comment block, R-08's
    #     refusal text) - what must never happen is it reaching a command, an env
    #     value or a JSON payload. So the scan is for emission markers on the same
    #     line, not for the bare string, which would flag its own documentation.
    EMIT = ("cmd +=", "cmd = [", "json.dumps", "env[", 'env = {', "execvpe")
    src = open(os.path.abspath(__file__)).read().splitlines()
    live = [i + 1 for i, ln in enumerate(src)
            if "FULL_AND_PIECEWISE" in ln and any(m in ln for m in EMIT)]
    ok_all &= not live
    print(f"  A2 FULL_AND_PIECEWISE never reachable as an emitted value: "
          f"{'PASS' if not live else 'FAIL at lines ' + str(live)}")
    # A3: --cudagraph-mode anything-but-FDO is refused (R-08) rather than honoured.
    a3 = "R-08" in "".join(src) and 'a.cudagraph_mode != "FULL_DECODE_ONLY"' in "".join(src)
    ok_all &= a3
    print(f"  A3 a non-FDO --cudagraph-mode request is refused, not honoured: "
          f"{'PASS' if a3 else 'FAIL'}")
    # A4: PXQ1 is detected by TENSOR TYPE, so a PXQ1-bearing UNIVERSAL map is caught
    #     too - the case the spec's `tier in {PXQ1}` condition could never match.
    t_uni, _, _ = tier_from_tensors([("blk.0.ffn_gate_exps.weight", 248, 1),
                                     ("blk.1.ffn_gate_exps.weight", 252, 1)])
    t_pure, _, _ = tier_from_tensors([("blk.0.ffn_gate_exps.weight", 248, 1)])
    a4 = (t_uni == "PXQ1" and t_pure == "PXQ1")
    ok_all &= a4
    print(f"  A4 PXQ1 tensors anywhere -> tier PXQ1 -> R-01 (uniform AND inside a UNIVERSAL "
          f"map): {'PASS' if a4 else 'FAIL ' + str((t_uni, t_pure))}")
    # A5: a bare 'mtp' spec can never become n_max>=2.
    a5 = parse_spec("mtp")[1].get("n_max") is None and "mtp:n_max=1" in "".join(src)
    ok_all &= a5
    print(f"  A5 bare --spec mtp expands to n_max=1 only: {'PASS' if a5 else 'FAIL'}")
    print(f"  standing assertions: {'ALL PASS' if ok_all else 'FAILURES ABOVE'}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cards", default="")
    ap.add_argument("--engine", choices=["llama", "vllm"])
    ap.add_argument("--workload", choices=["chat", "serve", "longdoc"], default=None,
                    help="default: chat when --np<=1, else serve")
    ap.add_argument("-c", "--ctx", type=int, default=0,
                    help="TOTAL context. Default: np * 4096, the measured envelope.")
    ap.add_argument("--np", type=int, default=1)
    ap.add_argument("--ub", type=int, default=0,
                    help="force -b/-ub. Default 0 = do not pass it; adaptive-ub chooses.")
    ap.add_argument("--threads", type=int, default=0, help="default: host core count")
    ap.add_argument("--ngl", type=int, default=999)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--spec", default="")
    ap.add_argument("--draft-model", default="")
    ap.add_argument("--ts", default="")
    ap.add_argument("--sm", default="layer")
    ap.add_argument("--ctk", default="f16")
    ap.add_argument("--ctv", default="f16")
    ap.add_argument("--mmproj", default="")
    ap.add_argument("--no-mmproj", action="store_true")
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--gmu", type=float, default=0.0,
                    help="vLLM --gpu-memory-utilization. Default: 0.90 sm_60 / 0.85 sm_70 "
                         "(recipe values, NEVER swept -> UNMEASURED as a tuning axis)")
    ap.add_argument("--vllm-image", default="")
    ap.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY")
    ap.add_argument("--tier", default="", help="assert the PXQ tier yourself (R-03 escape)")
    ap.add_argument("--allow-busy", action="store_true")
    ap.add_argument("--accept-unmeasured", action="store_true")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    gpus, err = gpu_table()
    if err:
        print(f"pxa-launch: {err}", file=sys.stderr)
        if not a.engine and not a.selftest:
            sys.exit(2)
        gpus = []
    if a.selftest:
        selftest(gpus or [])
        return

    # ---- card selection (I-12: never left to the ambient environment) -------
    cards = {int(x) for x in a.cards.split(",") if x.strip()} if a.cards else set()
    sel = [g for g in (gpus or []) if g[0] in cards] if cards else (gpus or [])
    if cards and len(sel) != len(cards):
        missing = sorted(cards - {g[0] for g in sel})
        print(f"pxa-launch: --cards asked for {missing} which are not visible", file=sys.stderr)
        sys.exit(2)

    if a.np < 1:
        print("=" * 78)
        print("pxa-launch: ENGINE = None")
        print(f"  REFUSING [R-26] " + R["R-26"].format(n=a.np, ctx=a.np * ANCHOR_CTX_PER_SLOT))
        print("=" * 78)
        sys.exit(2)

    per_slot = ANCHOR_CTX_PER_SLOT
    ctx = a.ctx or (a.np * ANCHOR_CTX_PER_SLOT)
    if a.ctx:
        per_slot = max(1, a.ctx // max(1, a.np))
    threads = a.threads or (os.cpu_count() or 16)
    a.threads = threads          # derived host core count -> the value actually emitted

    kind = model_kind(a.model)
    prof = model_profile(a.model, kind)
    if a.tier:
        prof["tier"] = a.tier
        prof["tier_src"] = "ASSERTED by --tier (you own this assertion)"
    mbytes = model_bytes(a.model, kind)
    workload = infer_workload(a.np, a.workload)
    elig_caps, image, probe_trail = vllm_eligibility(sel, a.vllm_image)

    plan = decide(sel, kind, a.model, a.engine, prof, a.np, workload,
                  elig_caps, image, probe_trail, per_slot)

    # ---- report ------------------------------------------------------------
    print("=" * 78)
    print(f"pxa-launch: ENGINE = {plan.engine}")
    size = f", {mbytes / BYTES_PER_GIB:.2f} GiB" if mbytes else ""
    print(f"  model:  {a.model}  [{kind}{size}]")
    if prof["arch"] or prof["n_expert"] or prof["tier"] or prof["hist"]:
        cls = "MoE" if prof["is_moe"] else "dense"
        line = f"  class:  {cls}"
        if prof["is_moe"]:
            line += f" ({prof['n_expert']} experts)"
        if prof["arch"]:
            line += f", arch={prof['arch']}"
        print(line + f"  [{prof['why']}]")
        print(f"  tier:   {prof['tier'] or 'not a PXQ file'}"
              + (f" (provenance KV says {prof['tier_kv']})" if prof.get("tier_kv") else "")
              + f"  [{prof['tier_src']}]")
        if prof["hist"]:
            print(f"  compose:{compose_str(prof['hist'])}")
        if prof["n_ctx_train"]:
            print(f"  trained ctx: {prof['n_ctx_train']}")
        if prof["mtp_tensors"] >= 0:
            print(f"  mtp:    {prof['mtp_tensors']} nextn/mtp tensors"
                  + (f"; KV nextn_predict_layers={prof['mtp_kv']}" if prof["mtp_kv"] is not None
                     else "")
                  + f"  [{prof['mtp_src']}]")
        print(f"  deltanet: {prof['deltanet']}  [{prof['deltanet_src']}]")
    else:
        print(f"  class:  UNKNOWN  [{prof['why']}]")
    print(f"  serve:  np={a.np}, workload={workload}, ctx={ctx} (={per_slot}/slot), "
          f"threads={threads}")
    if plan.reason:
        print(f"  reason: {plan.reason}")
    for e in plan.evidence:
        print(f"  ev:     {e}")

    # ---- refusals gathered during the decision ------------------------------
    # H4 / R-20: a shared, live box. Do this AFTER the decision so the operator
    # still sees what the launcher would have chosen.
    if sel and not a.allow_busy:
        procs, ok = resident_procs(gpus or [])
        for g in sel:
            if procs.get(g[0]):
                plist = ", ".join(f"pid {p} {n} {m} MiB" for p, n, m in procs[g[0]])
                plan.refuse("R-20", i=g[0], mib=g[4], procs=plist)
            elif g[4] > 512:
                plan.refuse("R-20", i=g[0], mib=g[4],
                            procs="no compute app listed, but >512 MiB is resident")

    if plan.engine is None and not plan.refusals:
        print("=" * 78)
        sys.exit(2)

    for n in plan.notes:
        print(f"  ** {n}")
    for b in plan.blockers:
        print(f"  !! {b}")

    # ---- parameter translation refusals (R-10..R-16, R-25, R-27) -----------
    if plan.engine == "vllm":
        if a.ts:
            plan.refuse("R-10", code=3)
        if a.sm and a.sm != "layer":
            plan.refuse("R-11", code=3, sm=a.sm)
        if (a.ctk, a.ctv) != ("f16", "f16"):
            plan.refuse("R-13V", code=3)
        m, _ = parse_spec(a.spec)
        if m == "mtp":
            plan.refuse("R-14", code=3)
        if a.draft_model:
            plan.refuse("R-25", code=3)
        if prof.get("vision") and not a.accept_unmeasured:
            plan.refuse("R-27", code=3)
        if a.np > MOE_TABLE_MAX_NP and not a.accept_unmeasured:
            plan.refuse("R-21", code=3, n=a.np)
    if plan.engine == "llama":
        if a.sm == "graph":
            why = None
            if prof.get("deltanet"):
                why = prof.get("deltanet_src", "linear-attention tensors present")
            elif prof.get("arch") in GRAPH_SPLIT_GUARDED_ARCHES:
                why = f"arch '{prof['arch']}' is on the guarded list (tools/pxa-launch.py)"
            if why:
                plan.refuse("R-12", code=3, why=why)
        if (a.ctk, a.ctv) not in COMPILED_CTKV_PAIRS:
            plan.refuse("R-13L", code=3, k=a.ctk, v=a.ctv)
        if a.draft_model and not a.accept_unmeasured:
            plan.refuse("R-25", code=3)
    # spec / MTP gates apply on both engines
    m, params = parse_spec(a.spec)
    if m == "mtp":
        nmax = int(params.get("n_max", 0) or 0)
        if nmax == 0:
            # I-8: a bare 'mtp' used to expand to n_max=4,n_min=2 - a MEASURED loss
            # on both arches, emitted by DEFAULT. It now expands to n_max=1 only.
            a.spec = "mtp:n_max=1"
            print("  --spec mtp expanded to mtp:n_max=1. MEASURED: n_max>=2 LOSES on both arches "
                  "(P100 54.9->47.4 accept 0.42; V100 92.7 vs 94.1 accept 0.960->0.480, "
                  "LEVERS.md:300-301). The previous version of this file expanded a bare 'mtp' to "
                  "n_max=4,n_min=2 - a measured loss, by default.")
            nmax = 1
        if nmax >= 2:
            plan.refuse("R-15A", code=3, n=nmax)
        if prof.get("mtp_tensors") == 0:
            plan.refuse("R-15B", code=3, kvn=prof.get("mtp_kv"))
        elif prof.get("mtp_tensors", 0) > 0:
            print("  MTP: PXA_MTP_LAZY_WARMUP is armed by PXA_ENHANCE and is MANDATORY whenever "
                  "MTP is active - without it MTP costs -33% prefill (LEVERS.md:152, :402). "
                  "PXA_MOE_FASTTG_MAX_NY is left at its shipped 8; =1 with MTP verify measured "
                  "48.1 -> 30.3 on P100 (LEVERS.md:409).")
            if prof.get("is_moe"):
                print("  MTP on a sparse MoE is a MEASURED LOSS even at n_max=1: -8.6% (n_max=1), "
                      "-29.8% (n_max=2) despite 0.800 acceptance (LEVERS.md:746). You asked for "
                      "it; it is emitted; the number is against you.")
    if prof.get("tier") in NO_CPU_CODEC and a.ngl < 99:
        plan.refuse("R-16", code=3, tier=prof["tier"], ngl=a.ngl)
    if plan.engine == "vllm" and a.cudagraph_mode != "FULL_DECODE_ONLY":
        plan.refuse("R-08", code=3, mode=a.cudagraph_mode)
    if prof.get("n_ctx_train") and ctx > int(prof["n_ctx_train"]):
        plan.refuse("R-17A", ctx=ctx, trained=prof["n_ctx_train"], arch=prof.get("arch"))

    # ---- VRAM (only formula-free facts may block; see SPEC CORRECTION C4) ---
    for n in vram_check(plan, sel, mbytes, ctx, prof, a.ngl >= 99):
        print(f"  ** {n}")

    # ---- mmproj resolution --------------------------------------------------
    mmproj = a.mmproj
    if plan.engine == "llama" and not mmproj and a.no_mmproj:
        # --no-mmproj SUPPRESSES resolution, so this branch emits no projector. It still
        # has to SAY that: "REFUSES rather than silently dropping" cuts both ways, and an
        # operator who suppresses a projector on a model that has three sitting beside it
        # deserves to see that the flag was read and what it turned off. Previously this
        # path printed nothing at all - the flag and the candidates both vanished.
        cands, _why = find_mmproj(a.model, kind)
        if cands:
            print(f"  mmproj: SUPPRESSED by --no-mmproj. {len(cands)} projector(s) are "
                  f"sitting next to this model and none will be attached:")
            for c in cands:
                print(f"            {c}")
            print("          This seat serves TEXT-ONLY. Drop --no-mmproj, or pass "
                  "--mmproj <path>, to take images.")
        elif prof.get("vision"):
            print("  mmproj: SUPPRESSED by --no-mmproj although this model carries vision "
                  "tensors. Serving TEXT-ONLY.")
        else:
            print("  mmproj: --no-mmproj accepted; nothing to suppress (no projector beside "
                  "this model, no vision tensors in it). Serving TEXT-ONLY either way.")
    if plan.engine == "llama" and not mmproj and not a.no_mmproj:
        cands, why = find_mmproj(a.model, kind)
        if prof.get("vision"):
            # The model itself carries vision tensors: a projector is part of the seat.
            if len(cands) == 1:
                mmproj = cands[0]
                print(f"  mmproj: {mmproj}  [exactly one candidate; {why}]")
            elif len(cands) > 1:
                plan.refuse("R-24", code=3, n=len(cands), list="\n      ".join(cands))
            else:
                print("  mmproj: NONE found next to this model although it carries vision "
                      "tensors. Serving TEXT-ONLY. Pass --mmproj to enable vision.")
        elif cands:
            # No vision tensors in the model file, but projectors sit beside it (the
            # muse-glimmer case: two F16 projectors in one dir, an F16 and a Q8_0 in
            # another, for the same base model). Nothing ranks them and nothing says
            # this model wants one, so NOTHING is attached - and the operator is told
            # what is there rather than left to wonder.
            print(f"  mmproj: NOT attached. {len(cands)} projector(s) sit next to this model "
                  f"but it carries no vision tensors and no measurement ranks them:")
            for c in cands:
                print(f"            {c}")
            print("          Pass --mmproj <path> if this seat is meant to take images.")

    if plan.refusals:
        print("  REFUSING:")
        for rid, text, _ in plan.refusals:
            print(f"    [{rid}] {text}")
        print("=" * 78)
        sys.exit(max(c for _, _, c in plan.refusals))

    if plan.needs_ack and not a.accept_unmeasured:
        print("  REFUSING to execute an UNMEASURED branch without acknowledgement:")
        for r in plan.needs_ack:
            print(f"    - {r}")
        print("    Re-run with --accept-unmeasured to proceed anyway. The plan above is the plan;")
        print("    the refusal is about executing it, not about printing it.")
        print("=" * 78)
        sys.exit(5 if a.explain else 3)

    # ---- build the command --------------------------------------------------
    ub_expect = sorted({ub_for_card(g[3]) for g in sel}) if sel else ["n/a"]
    if plan.engine == "llama":
        cmd, env = build_llama_cmd(plan, a, sel, prof, ctx, ub_expect, mmproj,
                                   explain=a.explain)
        used = sel
    else:
        used = plan.elig or sel
        if not a.gmu:
            cc = sorted({g[2] for g in used})[0] if used else 60
            a.gmu = 0.90 if cc == 60 else 0.85
            print(f"  --gpu-memory-utilization {a.gmu}: recipe value for sm_{cc} "
                  f"(0.90 = the MEASURED sm_60 arm; 0.85 = the healthy live sm_70 container). "
                  f"NEVER SWEPT -> UNMEASURED as a tuning axis.")
        cmd, env, used = build_vllm_cmd(plan, a, prof, ctx, used, image)

    cv = ",".join(str(g[0]) for g in used)
    # I-12 / R-07: devices are NEVER left to the ambient environment. If we cannot
    # name the devices, we do not execute. The previous version turned an empty
    # device list into the string "all" and then skipped setting
    # CUDA_VISIBLE_DEVICES entirely, so the child inherited every GPU on a box with
    # six cards mid-measurement and a production VLM on card 3.
    if not cv:
        print("  REFUSING to execute with an unscoped device set: no card could be named for "
              "CUDA_VISIBLE_DEVICES. On this box that means inheriting every GPU, including "
              "cards other agents are measuring on. (I-12)")
        print("=" * 78)
        sys.exit(3)
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    print(f"  env:     CUDA_VISIBLE_DEVICES={cv} {envs}")
    if plan.engine == "vllm":
        # THE IMAGE IS A PREMISE, NOT AN INSTRUCTION. The emitted command is a bare
        # `vllm serve` - there is no `docker run` in it and the image name appears
        # nowhere. --vllm-image (and PXA_VLLM_IMAGE) decide only WHICH CARDS ARE
        # ELIGIBLE; the process then runs in whatever container this launcher is
        # already inside. Name that out loud, because the flag reads like it selects a
        # runtime, and a reader who believes it will attribute a measurement to an
        # image that was never involved.
        img = os.environ.get("PXA_VLLM_IMAGE") or getattr(a, "vllm_image", None)
        if img:
            print(f"  CONTAINER CONTRACT: eligibility was decided against {img!r}, and this "
                  f"command is NOT run in it.")
            print(f"    The command below execs HERE. Run this launcher inside {img} - or "
                  f"accept that the seat you get is whatever this container holds, which "
                  f"is not what the decision above was based on.")
            he = (VLLM_IMAGES.get(img) or {}).get("host_env") or {}
            if he:
                print(f"    {img} is NOT self-contained: its python, torch and vllm are on "
                      f"the HOST. It needs these mounts, or nothing starts:")
                for hsrc, hdst in he.get("mounts", {}).items():
                    print(f"      -v {hsrc}:{hdst}")
                print(f"      [{he.get('why', 'host dependency')}]")
                es = he.get("editable_source")
                if es:
                    import subprocess as _sp
                    try:
                        br = _sp.run(["git", "-C", es, "rev-parse", "--abbrev-ref", "HEAD"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
                        sha = _sp.run(["git", "-C", es, "rev-parse", "--short", "HEAD"],
                                      capture_output=True, text=True, timeout=10).stdout.strip()
                        dirty = _sp.run(["git", "-C", es, "status", "--porcelain"],
                                        capture_output=True, text=True, timeout=15).stdout.strip()
                    except Exception:
                        br = sha = ""; dirty = ""
                    print(f"    vllm is an EDITABLE install of {es} - the seat imports that "
                          f"WORKING TREE, not a built artifact.")
                    if br or sha:
                        print(f"      right now that tree is {br} @ {sha}"
                              + ("  *** WITH UNCOMMITTED CHANGES ***" if dirty else ""))
                    print(f"      Editing or switching branches there changes what this seat "
                          f"serves, with no redeploy and no version to notice it by.")
        else:
            print("  CONTAINER CONTRACT: no image was named, so eligibility came from the "
                  "importable vllm_pxq4 in THIS interpreter. `vllm serve` execs here.")
    print(f"  command: {' '.join(cmd)}")
    print_post_boot_contract(plan.engine, cv)
    print("=" * 78)

    if a.explain:
        # exit 5 => a plan was produced but carries known-fatal blockers, so a CI
        # caller can tell "clean plan" from "plan that will not start".
        sys.exit(5 if plan.blockers else 0)
    if not shutil.which(cmd[0]) and not os.path.exists(cmd[0]):
        print(f"pxa-launch: {cmd[0]} not found", file=sys.stderr)
        sys.exit(4)
    e = dict(os.environ)
    e.update(env)
    e["CUDA_VISIBLE_DEVICES"] = cv
    e["NVIDIA_VISIBLE_DEVICES"] = cv
    os.execvpe(cmd[0], cmd, e)


if __name__ == "__main__":
    main()
