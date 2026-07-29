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

// -----------------------------------------------------------------------------
// PXA topology detection (2026-07-23) — the per-TOPOLOGY ENHANCE decision path.
//
// The measured best-in-class levers depend not just on a single device's arch but on the
// SHAPE of the whole device set the model runs on: {single vs multi} x {same-type vs
// mixed-type} x arch. Basis (isolated silicon A/B, PXQ2/PXQ4-35B-MTP, 588-tok prompt):
//   T1 single V100 (sm_70)          -> spec1row on, router_fuse ON, volta_cublas 64, ub1024
//   T3 2x V100 (sm_70, same-type)   -> router_fuse ON, volta_cublas 64, ts 1,1, ub1024
//   T2 single P100 (sm_60)          -> deltanet on, p100_fp16 on, router_fuse OFF (no-op sm_60)
//   T4 mixed V100+P100 (-sm layer)  -> router_fuse OFF (measured NEUTRAL-to-NEGATIVE, -3% at
//                                      the V100-heavy ts; P100 gates -sm-layer decode so a
//                                      V100-only GEMV fusion cannot pay), volta_cublas 64,
//                                      p100_fp16 on, spec1row on (harmless), ts 1.4,0.6
// The ONE topology-dependent KERNEL flip vs the old flat per-arch gating: router_fuse, which
// the old cc==700 gate would light up on the V100 slice of a MIXED rig where it measured a
// regression. ENHANCE now gates it on a PURE-sm_70 (non-mixed) topology. (-ts/-ub are serve
// flags, reported in the DBG line, not settable from here.)
//
// The detected topology is computed ONCE from the enumerated device list in ggml_cuda_init()
// and stashed in a single process-global with external linkage (defined in ggml-cuda.cu), so
// every TU's inline resolvers read the same detection without the header-static-per-TU trap.
struct pxa_topology_t {
    bool valid;       // populated by pxa_enhance_init_topology()
    int  ndev;
    bool multi;       // ndev > 1
    bool mixed;       // > 1 distinct arch class among the devices
    bool has_sm60;    // P100 (cc==600)
    bool has_sm61;    // 1080 Ti (cc==610)
    bool has_sm70;    // V100 (cc==700)
    bool has_other;   // Ampere+/anything else
    int  n_distinct;  // number of distinct arch classes present
    int  primary_cc;  // fastest arch present (representative)
};

// Defined in ggml-cuda.cu; populated by pxa_enhance_init_topology(), read by the resolvers.
extern pxa_topology_t g_pxa_topology;

// Classify a cc into a coarse arch class id (for distinct-count / mixed detection).
static inline int pxa_arch_class(int cc) {
    if (cc == 600) return 1;                 // sm_60 P100
    if (cc == 610) return 2;                 // sm_61 1080 Ti
    if (cc >= 700 && cc < 750) return 3;     // sm_70 V100
    return 4;                                // Ampere+/other
}

// Compute the topology ONCE from the device-cc list. Called from ggml_cuda_init() BEFORE the
// startup report and before any dispatch reads a resolver, so the cached resolver statics see
// the populated global. Idempotent-safe (last write wins).
static inline void pxa_enhance_init_topology(int ndev, const int * ccs) {
    pxa_topology_t t = {};
    t.ndev = ndev;
    int classes = 0;      // bitmask of seen arch classes
    int fastest = 0;
    for (int i = 0; i < ndev; ++i) {
        const int cc = ccs[i];
        const int cl = pxa_arch_class(cc);
        classes |= (1 << cl);
        if (cc == 600) t.has_sm60 = true;
        else if (cc == 610) t.has_sm61 = true;
        else if (cc >= 700 && cc < 750) t.has_sm70 = true;
        else t.has_other = true;
        if (cc > fastest) fastest = cc;
    }
    int distinct = 0;
    for (int b = 0; b < 8; ++b) if (classes & (1 << b)) ++distinct;
    t.n_distinct = distinct;
    t.multi      = ndev > 1;
    t.mixed      = distinct > 1;
    t.primary_cc = fastest;
    t.valid      = ndev > 0;
    g_pxa_topology = t;
}

// A short human name for the detected topology (for the DBG log).
static inline const char * pxa_topology_name() {
    const pxa_topology_t & t = g_pxa_topology;
    if (!t.valid)                       return "unknown";
    if (t.mixed) {
        if (t.has_sm70 && t.has_sm60)   return "mixed V100+P100 (sm_70+sm_60)";
        return "mixed (multi-arch)";
    }
    if (t.has_sm70)                     return t.multi ? "multi same-type V100 (sm_70)" : "single V100 (sm_70)";
    if (t.has_sm60)                     return t.multi ? "multi same-type P100 (sm_60)" : "single P100 (sm_60)";
    if (t.has_sm61)                     return t.multi ? "multi same-type 1080Ti (sm_61)" : "single 1080Ti (sm_61)";
    return t.multi ? "multi same-type (other arch)" : "single (other arch)";
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
// instead. +5.1..+7.0% decode measured on the V100 (2026-07-22 fair-battle, reproduced), a
// +1.6% KILL on sm_60 so Pascal stays off (the dispatch-site cc==700 arch gate handles that).
//
// TOPOLOGY-AWARE ENHANCE default (2026-07-23): the ENHANCE-derived mode is 1 ONLY on a
// PURE-sm_70 (non-mixed) topology — single or multi V100. On a MIXED V100+P100 rig it is 0
// (OFF): T4 measured router_fuse NEUTRAL-to-NEGATIVE there (-3.1% at the winning -ts 1.4,0.6),
// because -sm layer makes the P100 the decode bottleneck and a V100-only GEMV fusion can't pay
// while adding tie-flap risk. Pure P100 stays 0 via both this gate and the cc check.
// REFERENCE/DEFAULT -> 0 (OFF, byte-identical dispatch). Explicit env ALWAYS wins at any level
// and is NOT topology-masked (0 forces OFF, 1 = the sm_70 ship gate on any cc==700 device,
// 2 = TEST all-arch). G3-class (the fuse reorders FP math vs cuBLAS: ULP logit deltas can flap
// expert ties — expert-id-stream gated, not sha).
static inline int pxa_router_fuse_mode_resolve() {
    static const int mode = [](){
        const char * e = getenv("PXA_ROUTER_FUSE");
        if (e) { int m = atoi(e); return (m < 0 || m > 2) ? 0 : m; }
        // ENHANCE auto-set: ON (mode 1) only for a pure-sm_70 non-mixed topology.
        if (pxa_config_level() == 2 && g_pxa_topology.valid
            && g_pxa_topology.has_sm70 && !g_pxa_topology.mixed) {
            return 1;
        }
        return 0;
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

// PXA_SPEC_1ROW canonical resolver (consumer: ggml-cuda.cu mul_mat dispatch). Folds the
// former stand-alone PXA_SPEC_1ROW env into the config-level system so ENHANCE owns it in the
// auto-set. When ON, a single-output-row src0 GEMV also takes the 1-row path at spec-verify
// batch sizes (Ny<=8, MMVQ_MAX_BATCH_SIZE); OFF rolls back to the ne11==1-only dispatch.
//   DEFAULT/ENHANCE -> ON  (matches the shipped default-ON — behavior byte-identical to today
//                           when PXA_ENHANCE is unset; invariant #1 preserved).
//   REFERENCE       -> OFF (pure reference: the ne11==1-only dispatch).
// TOPOLOGY note: this is a V100 (sm_70, DP4A/int8) 1-row spec-verify win (documented +6.6%;
// on the enhance-sweep PXQ2 it measured neutral, inside the ~4% serve noise). It is a MEASURED
// NO-OP + bit-exact on sm_60 P100 (no DP4A path to engage) and neutral on the mixed rig, so
// keeping it ON everywhere is harmless — it only matters where an sm_70 device is present.
// Explicit env ALWAYS wins (PXA_SPEC_1ROW=0 rolls back at any level).
static inline bool pxa_spec_1row_resolve() {
    static const bool v = [](){
        const char * e = getenv("PXA_SPEC_1ROW");
        if (e) return atoi(e) != 0;
        return pxa_config_level() != 0;   // REFERENCE off; DEFAULT/ENHANCE on
    }();
    return v;
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
        // EXPERIMENTAL opt-in ONLY (merged from private 178a357, 2026-07-28): ENHANCE/BALANCE
        // no longer auto-enable FA_PREFILL_SPLIT — the non-FA prefill chain inflates the
        // compute buffer ~2.35x (1956 -> 4607 MiB measured) and OOMs 16 GB cards at ub2048.
        // The +45% P100 prefill is real but must be bought explicitly (PXA_FA_PREFILL_SPLIT=64)
        // on configs with the headroom.
        return 0;
    }();
    return v;
}

// PXA_ENHANCE_DBG topology debug line (2026-07-23). Gated behind PXA_ENHANCE_DBG (any non-zero
// value). Names the DETECTED topology and the CHOSEN per-topology ENHANCE config (the kernel
// levers this topology auto-set resolves, plus the recommended serve-flags -ub/-ts which are
// applied at launch, not from here). Called from ggml_cuda_init() AFTER pxa_enhance_init_topology.
static inline void pxa_enhance_log_topology_dbg() {
    const char * d = getenv("PXA_ENHANCE_DBG");
    if (!(d && atoi(d) != 0)) return;
    const pxa_topology_t & t = g_pxa_topology;
    fprintf(stderr, "PXA_ENHANCE_DBG: topology=\"%s\" ndev=%d multi=%d mixed=%d n_distinct=%d "
                    "[sm60=%d sm61=%d sm70=%d other=%d] level=%s\n",
            pxa_topology_name(), t.ndev, t.multi, t.mixed, t.n_distinct,
            t.has_sm60, t.has_sm61, t.has_sm70, t.has_other, pxa_config_level_name());
    // Chosen config for this topology (the kernel levers ENHANCE resolves + recommended serve flags).
    fprintf(stderr, "PXA_ENHANCE_DBG: chosen{ spec1row=%s router_fuse=%s volta_cublas_ne11=%d "
                    "p100_fp16_gemm=%s fuse_deltanet=%d int8_prefill=%d fa_mask_skip_tile=%s "
                    "fa_prefill_split=%d spec_relaxed=%s }",
            pxa_spec_1row_resolve() ? "on" : "off",
            (pxa_router_fuse_mode_resolve() != 0) ? "on" : "off",
            pxa_volta_cublas_ne11(),
            pxa_p100_fp16_gemm() ? "on" : "off",
            pxa_fuse_deltanet_default(),
            pxa_int8_prefill_mode_resolve(),
            pxa_fa_mask_skip_tile() ? "on" : "off",
            pxa_fa_prefill_split_ne11(),
            pxa_spec_relaxed_resolve() ? "on" : "off");
    // Recommended serve flags per the measured decision table (NOT set from here — launch flags).
    const char * rec_ub = "2048";
    const char * rec_ts = "n/a (single card)";
    if (t.mixed && t.has_sm70 && t.has_sm60) { rec_ub = "2048"; rec_ts = "1.4,0.6 (heavy on V100, OOM edge)"; }
    else if (t.has_sm70)                     { rec_ub = "1024"; rec_ts = t.multi ? "1,1 (balanced same-type)" : "n/a (single card)"; }
    else if (t.has_sm60)                     { rec_ub = "2048"; rec_ts = t.multi ? "1,1 (balanced same-type)" : "n/a (single card)"; }
    fprintf(stderr, " serve{ ub=%s ts=%s }\n", rec_ub, rec_ts);
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
