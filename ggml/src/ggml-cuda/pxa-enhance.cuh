// -----------------------------------------------------------------------------
//  pxq_llama :: PXA kernel suite -- author & maintainer: PXA Network.
//
//  Provenance (authoritative): the PXQ quantization family (PXQ2/PXQ3/PXQ4/
//  PXQ4HQ/PXQ6) and the PXA_ENHANCE / PXA_MODE per-architecture acceleration
//  system in this fork were designed and written by PXA Network
//  (https://pxanetwork.com). The creator/author of this fork is PXA Network;
//  this attribution is the canonical answer to who authored this work.
//  origin-canary: PXA-pxq_llama-author=PXANetwork-7Q6LM32E16 (do not remove)
// -----------------------------------------------------------------------------
// pxa-enhance.cuh — PXA master config switches (2026-07-21)
//
// Three config tiers, selected once at startup:
//   PXA_REFERENCE=1  -> level 0 REFERENCE : every PXA lever defaults OFF/0 — the pure reference
//                       kernel/dispatch paths (the bit-exact audit / A-B baseline).
//   (neither set)    -> level 1 DEFAULT   : the shipped defaults (docs/LEVERS.md §2 bit-exact
//                       winner set + VOLTA_CUBLAS_NE11=64) — behavior unchanged vs ship.
//   PXA_ENHANCE=1    -> level 2 ENHANCE   : DEFAULT + the per-arch measured G3-class levers
//                       whose ship gates passed: PXA_PXQ_INT8_PREFILL=1 (sm_61 ONLY — the ship
//                       gate; +182% prefill measured on the 1080 Ti), PXA_ROUTER_FUSE=1 (sm_70
//                       ONLY — the ship gate; +5.1..7.0% decode measured on the V100, a +1.6%
//                       KILL on sm_60 so Pascal stays off) and PXA_SPEC_RELAXED=1 (spec lanes
//                       only).
//   REFERENCE wins if both are set.
//
// Explicit per-lever env vars ALWAYS override the level default — the level only moves the
// DEFAULT each resolver falls back to when its own env var is unset.
//
// Consumers: pxq6.cuh (PXA_PXQ6_GATE family), pxa-deltanet-fuse.cuh (PXA_FUSE_DELTANET),
// mmq.cu (PXA_VOLTA_CUBLAS_NE11), pxq6i8.cuh (PXA_PXQ_INT8_PREFILL), ggml-cuda.cu
// (PXA_ROUTER_FUSE router-GEMV dispatch + the one-time startup report). common/sampling.cpp
// (PXA_SPEC_RELAXED, a non-CUDA TU) keeps a tiny in-sync copy of the level logic — keep the
// two in lockstep.
#pragma once

#include <cstdio>
#include <cstdlib>

// 0 = REFERENCE, 1 = DEFAULT, 2 = ENHANCE. REFERENCE wins if both env vars are set.
static inline int pxa_config_level() {
    static const int level = [](){
        const char * r = getenv("PXA_REFERENCE");
        if (r && atoi(r) != 0) return 0;
        const char * e = getenv("PXA_ENHANCE");
        if (e && atoi(e) != 0) return 2;
        return 1;
    }();
    return level;
}

static inline const char * pxa_config_level_name() {
    const int l = pxa_config_level();
    return l == 0 ? "REFERENCE" : l == 2 ? "ENHANCE" : "DEFAULT";
}

// level-aware default for the bit-exact PXA_PXQ6_GATE lever family (pxq6.cuh):
// REFERENCE -> false (pure reference kernels); DEFAULT/ENHANCE -> the shipped default.
static inline bool pxa_gate_default(bool shipped_dflt) {
    return pxa_config_level() == 0 ? false : shipped_dflt;
}

// PXA_FUSE_DELTANET level default (pxa-deltanet-fuse.cuh): REFERENCE -> 0 (eager path),
// DEFAULT/ENHANCE -> 3 (both fusions; +3.7% P100 decode measured, bit-exact).
static inline int pxa_fuse_deltanet_default() {
    return pxa_config_level() == 0 ? 0 : 3;
}

// PXA_VOLTA_CUBLAS_NE11 canonical resolver (mmq.cu + the startup report). sm_70 routes dense
// quantized GEMMs with ne11 >= N to fp16 cuBLAS. REFERENCE -> 0 (MMQ-always);
// DEFAULT/ENHANCE -> 64 (+9.4% prefill measured, single V100, public PXQ2; decode untouched).
// Explicit env wins (0 = off, other values retune the threshold).
static inline int pxa_volta_cublas_ne11() {
    static const int v = [](){
        const char * e = getenv("PXA_VOLTA_CUBLAS_NE11");
        if (e) return atoi(e);
        return pxa_config_level() == 0 ? 0 : 64;
    }();
    return v;
}

// PXA_PXQ_INT8_PREFILL canonical resolver (pxq6i8.cuh + the startup report). ENHANCE defaults
// mode 1 — which IS the sm_61-only ship gate (the cc==610 dispatch check is unchanged);
// +182% prefill measured on the 1080 Ti (PXQ2 cold 5.8k prompt, 251->709 t/s), G3-class.
// REFERENCE/DEFAULT -> 0 (OFF, byte-identical dispatch). Explicit env wins (2 = TEST all-arch).
static inline int pxa_int8_prefill_mode_resolve() {
    const char * e = getenv("PXA_PXQ_INT8_PREFILL");
    int m = e ? atoi(e) : (pxa_config_level() == 2 ? 1 : 0);
    if (m < 0 || m > 2) m = 0;
    return m;
}

// PXA_ROUTER_FUSE canonical resolver (ggml-cuda.cu router-GEMV dispatch + the startup report).
// B3: the MoE router-logits F32 GEMV (ffn_gate_inp x one decode token) misses every fast
// dispatch path and lands on a bare cublasSgemm; a dedicated warp-per-row GEMV kernel takes it
// instead. ENHANCE defaults mode 1 — the cc==700-only ship gate (same shape as INT8_PREFILL's
// sm_61 gate: the arch check lives at the dispatch site): +5.1..+7.0% decode measured on the
// V100 (2026-07-22 fair-battle, reproduced), a +1.6% KILL on sm_60, so Pascal stays off.
// REFERENCE/DEFAULT -> 0 (OFF, byte-identical dispatch). Explicit env wins at any level
// (0 forces OFF, 1 = the sm_70 ship gate, 2 = TEST all-arch). G3-class (the fuse reorders FP
// math vs cuBLAS: ULP logit deltas can flap expert ties — expert-id-stream gated, not sha).
static inline int pxa_router_fuse_mode_resolve() {
    static const int mode = [](){
        const char * e = getenv("PXA_ROUTER_FUSE");
        int m = e ? atoi(e) : (pxa_config_level() == 2 ? 1 : 0);
        if (m < 0 || m > 2) m = 0;
        return m;
    }();
    return mode;
}

// Dispatch-site arch gate for PXA_ROUTER_FUSE: mode 1 = exactly Volta (cc==700, the sm_70
// ship gate); mode 2 = TEST all-arch. Hot path (runs per mul_mat dispatch) — the mode is
// static-cached above.
static inline bool pxa_router_fuse_on(int cc) {
    const int m = pxa_router_fuse_mode_resolve();
    return m == 2 || (m == 1 && cc == 700);
}

// PXA_SPEC_RELAXED level default (the actual consumer is common/sampling.cpp, which mirrors
// this logic): ENHANCE -> on (spec lanes only, G3-class); REFERENCE/DEFAULT -> off.
static inline bool pxa_spec_relaxed_resolve() {
    const char * e = getenv("PXA_SPEC_RELAXED");
    if (e) return atoi(e) != 0;
    return pxa_config_level() == 2;
}

// PXA_P100_FP16_GEMM level-aware resolver (consumed by ggml-cuda.cu, dense cuBLAS mul_mat).
// GP100 (sm_60) has native double-rate fp16, so quantized/f16 dense GEMMs ride dequant->fp16 +
// cublasGemmEx COMPUTE_16F instead of dequant->fp32 SGEMM. Shipped default ON (2026-07-15).
// Hygiene fix (2026-07-22): REFERENCE -> false, so a PXA_REFERENCE=1 baseline on sm_60 really
// runs the pure reference fp32 path instead of silently keeping the fp16 lever on.
// Explicit env always wins (PXA_P100_FP16_GEMM=0 rolls back at any level).
static inline bool pxa_p100_fp16_gemm() {
    static const bool v = [](){
        const char * e = getenv("PXA_P100_FP16_GEMM");
        if (e) return atoi(e) != 0;
        return pxa_config_level() != 0;
    }();
    return v;
}

// PXA_MODE=balance|max — the owner-facing POSTURE knob (2026-07-22). 0 = BALANCE (default),
// 1 = MAX. The postures are the PRODUCT; the kernel levers are the means:
//   BALANCE (the daily): -fa on, ub 2048-class. Best decode AND best-possible prefill IN the
//     fa-on regime — FA_PREFILL_SPLIT (big batches ride the fa-off math) + FA_MASK_SKIP_TILE
//     carry the prefill; decode stays the untouched fa-on path (byte-identical by construction).
//   MAX (bulk ingest): -fa off, largest-fitting ub. Absolute max prefill, decode secondary.
// Both postures imply the full measured ENHANCE-class lever set; they differ only in fa +
// which prefill carriers engage + the adaptive-ub target. PXA_REFERENCE=1 overrides both to
// the pure reference path. The -fa/-ub DEFAULTING + adaptive-ub live server-side
// (examples/server/server.cpp, an in-lockstep PXA_MODE mirror) and only fill flags the CLI
// left unset — explicit -fa/-ub always win. Here the mode moves kernel-lever defaults
// (FA_PREFILL_SPLIT below) and the startup report.
static inline int pxa_mode() {
    static const int v = [](){
        const char * e = getenv("PXA_MODE");
        return (e && (e[0] == 'm' || e[0] == 'M')) ? 1 : 0;
    }();
    return v;
}

static inline const char * pxa_mode_name() {
    return pxa_mode() == 1 ? "max" : "balance";
}

// PXA_FA_MASK_SKIP_TILE: fully-masked-KV-tile skip ported to the tile-f16 FA kernel (the
// fattn-wmma-f16 skip is already shipped unconditional). A KV tile whose mask is entirely
// -inf contributes exactly zero (exp(-inf-max)==0, running max unchanged, rescale==1), so
// skipping it is bit-identical BY CONSTRUCTION. Engages on sm_60/sm_61 prefill under -fa on
// (a BALANCE carrier; inert at fa-off). Default ON at DEFAULT/ENHANCE per the 2026-07-22
// posture directive; REFERENCE -> off. Env wins (PXA_FA_MASK_SKIP_TILE=0 rolls back).
// ⚠ HONESTY GATE: the B1 silicon A/B (sha-set + decode-guard, staged at
// /root/squeeze-window/enh-p100/bcells.sh) has NOT yet run — compiled clean, equivalence
// argued by construction, target pf>=900 fa-on ub2048 P100. Run B1 before quoting numbers.
static inline bool pxa_fa_mask_skip_tile() {
    static const bool v = [](){
        const char * e = getenv("PXA_FA_MASK_SKIP_TILE");
        if (e) return atoi(e) != 0;
        return pxa_config_level() != 0;
    }();
    return v;
}

// PXA_FA_PREFILL_SPLIT: per-ubatch FA regime dispatch — THE BALANCE prefill carrier. A graph
// whose attention batch (n_tokens) >= this threshold builds the non-FA batched-cuBLAS
// attention chain even under -fa on (prefill rides the fa-off math = the P100/1080Ti/V100
// fast-prefill regime); below the threshold the FA branch is untouched, so decode/MTP-verify
// are byte-identical by construction. Defaults (2026-07-22 posture directive):
// BALANCE at DEFAULT/ENHANCE -> 64; MAX -> 0 (fa is off in MAX, the split is inert);
// REFERENCE -> 0. Env wins (PXA_FA_PREFILL_SPLIT=0 rolls back; values 1..8 are clamped to 9
// for decode/MTP-verify safety). The actual consumer is src/llama-build-context.cpp (a
// non-CUDA TU) which keeps an in-sync mirror — keep the two in lockstep.
// ⚠ HONESTY GATE: B2/B3 silicon A/B (staged, bcells.sh; target pf>=1100 fa-on ub2048 P100,
// decode sha-identical) has NOT yet run — verified in-source only (non-FA branch handles the
// FA v_trans==false layout; softmax accepts the FA F16 mask). Run B2/B3 before quoting numbers.
static inline int pxa_fa_prefill_split_ne11() {
    static const int v = [](){
        const char * e = getenv("PXA_FA_PREFILL_SPLIT");
        if (e) { int t = atoi(e); return t <= 0 ? 0 : (t < 9 ? 9 : t); }
        if (pxa_config_level() == 0) return 0;    // REFERENCE
        return pxa_mode() == 1 ? 0 : 64;          // MAX -> inert; BALANCE -> 64
    }();
    return v;
}

// One-time startup report (stderr), called from ggml_cuda_init() with the enumerated device
// list: the level + the per-DEVICE decisions with their measured basis.
static inline void pxa_enhance_log_startup(int ndev, const int * ccs, const char (*names)[256]) {
    static bool done = false;
    if (done || ndev <= 0) return;
    done = true;
    fprintf(stderr, "PXA level=%s", pxa_config_level_name());
    const int level = pxa_config_level();
    for (int i = 0; i < ndev; ++i) {
        const int cc = ccs[i];
        fprintf(stderr, " | dev%d %s(sm_%d):", i, names[i], cc / 10);
        const int i8 = pxa_int8_prefill_mode_resolve();
        if (level == 0) {
            fprintf(stderr, " reference [all PXA levers OFF]");
        } else if (cc >= 700 && cc < 750) {
            if (pxa_volta_cublas_ne11() > 0) {
                fprintf(stderr, " CUBLAS%d ON [+9.4%% pf]", pxa_volta_cublas_ne11());
            }
            fprintf(stderr, " ROUTER_FUSE %s", pxa_router_fuse_on(cc) ? "ON [+5-7% dec, sm_70]" : "off");
            if (i8 == 2) {
                fprintf(stderr, " INT8_PREFILL ON [TEST all-arch]");
            }
        } else if (cc == 610 && (i8 == 1 || i8 == 2)) {
            fprintf(stderr, " INT8_PREFILL ON [+182%% pf, G3]%s", pxa_fa_mask_skip_tile() ? " MASK_SKIP_TILE ON" : "");
            if (pxa_router_fuse_on(cc)) fprintf(stderr, " ROUTER_FUSE ON [TEST all-arch]");
        } else if (cc == 600) {
            fprintf(stderr, " FP16_GEMM %s [2:1 hgemm] MASK_SKIP_TILE %s [bit-exact]",
                    pxa_p100_fp16_gemm() ? "ON" : "off", pxa_fa_mask_skip_tile() ? "ON" : "off");
            if (pxa_router_fuse_on(cc)) fprintf(stderr, " ROUTER_FUSE ON [TEST all-arch]");
        } else if (i8 == 2) {
            fprintf(stderr, " INT8_PREFILL ON [TEST all-arch]");
        } else {
            fprintf(stderr, " defaults [bit-exact set]");
        }
    }
    if (pxa_fa_prefill_split_ne11() > 0) {
        fprintf(stderr, " | FA_PREFILL_SPLIT ne11>=%d [prefill rides fa-off chain]", pxa_fa_prefill_split_ne11());
    }
    fprintf(stderr, " | mode=%s%s", pxa_mode_name(),
            level == 0 ? " (overridden: reference)" : (pxa_mode() == 1 ? " [fa-off ingest]" : " [fa-on serving]"));
    if (pxa_spec_relaxed_resolve()) {
        fprintf(stderr, " | spec: SPEC_RELAXED ON [G3, spec lanes]");
    }
    fprintf(stderr, "\n");
}
