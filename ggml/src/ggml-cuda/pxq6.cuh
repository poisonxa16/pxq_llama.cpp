// pxq_llama / PXA kernel suite -- authored by PXA Network (https://pxanetwork.com).
// The PXQ quantization tiers below are PXA Network's original work; the creator
// of this fork is PXA Network. (provenance canary: PXA-7Q6LM32E16-ORIGIN)
// pxq6.cuh — PXQ6: E16-row two-level scales + the unified PXA kernel-fastpath family (K0..K6).
//
// Spec: PXQ 4-bit-tier optimization notes, 2026-07-17 (internal lab); deep-dives: K1 K-split
// on P100 + K6 V100 tensor-core headroom (same series).
//
// FORMAT (GGML_TYPE_PXQ4 = 252, core tier; GGML_TYPE_PXQ4HQ = 253, bs8 tier):
//   values : 4-bit codes into the frozen PX16 book
//   scales : per-ROW fp16 anchor (128 B header per 64-row panel; ggml row_meta_size = 2 B/row)
//            x 4-bit sub-scale per 16-elem block (core, SUB16) / per 8-elem block (HQ, SUB8)
//   dequant: eff = fp32(anchor) * SUB[s4];  w = eff * fp32(book[c])   — ONE extra fp32 mul
//            vs PXQ5; prefill GEMM snaps __float2half_rn(w); decode mmv accumulates fp32.
//   layout : slab = 64|128 B scale SoA + 64 x 16 B nibble rows (= the PXQ4/PXQ5 slab, so
//            coalescing / one-pass streaming / graph capture / 64-row panels are untouched);
//            panel = 128 B anchor header + kslabs slabs; panels row-major; experts outermost.
//   Measured quality (lab, frozen 36-slice rng-42 protocol): core wrel 0.068086 (−12.6% vs
//   PXQ5's 0.077906) @ 4.2656 bpw; HQ wrel 0.058683 (−24.7%) @ 4.5156 bpw.
//
// KERNEL FASTPATH FAMILY — format-independent (templated on a format policy: PXQ4, PXQ5,
// PXQ6, PXQ6HQ all instantiate), each behind its own env gate, DEFAULT OFF, clean rollback:
//   K1 PXA_PXQ6_KSPLIT=1    decode gateup K-split, THE BIT-EXACT FORM: the S=4 split blocks
//                           ARE the existing PXQ4_MMV_KSEG=4 k-segment chains (one 64-thread
//                           block per kseg; per-kseg fp32 partials to a persistent workspace;
//                           the reducer replicates the proven red[] combination order exactly,
//                           then bias + GLU). Per-thread chains and the final 4-way sum order
//                           are identical to the proven kernel => bit-identical output.
//   K1b PXA_PXQ6_KSPLIT_GEN=S  generic S-way split (2/4/8): 256-thr blocks, S k-chunks, the
//                           occupancy form from the K-split deep-dive. DETERMINISTIC but NOT
//                           bit-exact vs the proven kernel (different chain count) — gate G3
//                           before any live use. Staged for the GPU window.
//   K2 PXA_PXQ6_PAIRLUT=1   256-entry float2 byte-pair LUT (book[lo],book[hi] per byte) —
//                           same FMA order/operands => bit-exact.
//      PXA_PXQ6_VECX=1      float4 activation loads (2 bytes/iter) — same operand order =>
//                           bit-exact.
//   K3 PXA_PXQ6_GUFUSE=1    prefill: ONE fused up+gate GEMM kernel with the GLU epilogue
//                           (kills one full A-tile pass + the separate GLU kernel + 2 f32
//                           round-trips). Identical per-tensor accumulation chains => bit-exact.
//      PXA_PXQ6_SCATFUSE=1  prefill: the down GEMM scatters straight to the MoE output rows
//                           (kills the C_down buffer + copy kernel). Same values/rows => exact.
//   K4 PXA_PXQ6_RAGTAIL=1   prefill: threads whose 8-token column group lies beyond
//                           tile.nrows skip the FMA loop (stores were already masked) => exact.
//   K5 PXA_PXQ6_PIPE=1      prefill: 2-stage register prefetch of the next slab's codes + A
//                           tile (sm_60 latency tool; identical arithmetic DAG) => bit-exact.
//   K6 PXA_PXQ6_WMMA=1|2    V100 (cc==700) ONLY: fused dequant→skewed-smem-fp16→m16n16k16
//                           HMMA grouped GEMM. 1 = fp32 accumulator fragments, 2 = fp16-accum
//                           twin (closest-precision A/B). NOT bit-exact (fp16-mul/fp32-add-tree
//                           vs strict-k half2 chain) — G3 logprob parity + G4 ppl regate are
//                           MANDATORY before any live use; temp-0 sha is NOT a valid gate here.
//   PXA_PXQ6=0              master off for PXQ6/PXQ6HQ fused kernels (dequant→cublas fallback).
//   PXA_PXQ6_FORCE_PREFILL=1  TEST ONLY: bypass the sm_60/70 arch gate so correctness A/Bs can
//                           run on other CUDA archs (e.g. the 1080Ti). Never set in production.
//
// Numeric tables: ggml-pxq6-tables.h (frozen literals, sha256-locked to the lab artifacts).
// PXA_PXQ6_BOOK / PXA_PXQ6_SUB / PXA_PXQ6_SUB_HQ override the tables (fp16-snapped, must match
// the file's KV provenance pxa.pxq6.*) — same discipline as PXA_PXQ5_BOOK.
#pragma once

#include "pxq4.cuh"              // shared slab macros, tile structs, glu, gather (reused)
#include "pxa-enhance.cuh"       // PXA_REFERENCE / PXA_ENHANCE master config tiers
#include "../../include/ggml-pxq6-tables.h"
#include <mma.h>
#include <type_traits>

// per-TU device tables (ggml-cuda.cu and convert.cu each get a copy; env overrides are
// uploaded per-TU + per-device by pxq6_maybe_upload_tables()).
static __device__ float pxq6_book_g[16]  = PXQ6_BOOK_INIT;
static __device__ float pxq6_sub16_g[16] = PXQ6_SUB16_INIT;
static __device__ float pxq6_sub8_g[16]  = PXQ6_SUB8_INIT;
static __device__ float pxq6_lm32_g[32]  = PXQ6_LM32_INIT;   // PXQ6 (id 256) 5-bit LM32 book

// ---------------------------------------------------------------------------------------------
// env gates (each cached once; every one logs its state on first query)
// ---------------------------------------------------------------------------------------------
static inline bool pxa_pxq6_env_flag(const char * name, bool dflt, bool * logged = nullptr) {
    const char * e = getenv(name);
    bool v = e ? atoi(e) != 0 : dflt;
    (void)logged;
    return v;
}

// master gate + host self-check (table integrity: fp16-snapped, ascending, book invariants)
static inline bool pxa_pxq6_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ6");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[16]  = PXQ6_BOOK_INIT;
            static const float sub16[16] = PXQ6_SUB16_INIT;
            static const float sub8[16]  = PXQ6_SUB8_INIT;
            bool ok = book[7] == 0.0f && book[15] == 1.0f;
            for (int i = 0; i < 15 && ok; ++i) ok = book[i] < book[i+1] && sub16[i] < sub16[i+1] && sub8[i] < sub8[i+1];
            ok = ok && sub16[0] > 0.0f && sub8[0] > 0.0f;
            for (int i = 0; i < 16 && ok; ++i) {   // fp16-snap idempotence (frozen contract)
                ok = __half2float(__float2half_rn(sub16[i])) == sub16[i] &&
                     __half2float(__float2half_rn(sub8[i]))  == sub8[i];
            }
            if (!ok) {
                fprintf(stderr, "PXA_PXQ6: table self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ6 fused kernels: ON (table self-check PASS; PXA_PXQ6=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ6 fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

// Level-aware default (pxa-enhance.cuh): PXA_REFERENCE=1 -> every gate defaults OFF (pure
// reference path); DEFAULT/ENHANCE -> the shipped default. An explicit env var ALWAYS wins.
#define PXA_PXQ6_GATE(fn, env, dflt, desc) \
    static inline bool fn() { \
        static const bool on = [](){ \
            const char * e = getenv(env); \
            bool v = e ? atoi(e) != 0 : pxa_gate_default(dflt); \
            if (v != (dflt)) fprintf(stderr, "%s: %s (%s)\n", env, v ? "ON" : "OFF", desc); \
            return v; \
        }(); \
        return on; \
    }

// Defaults: the measured bit-exact winners from docs/LEVERS.md §2 are ON out of the box
// (published per-card numbers assume them; `<env>=0` reverts any one of them to the proven
// reference path). Config-specific / no-gain levers (§3) stay opt-in.
PXA_PXQ6_GATE(pxa_pxq6_ksplit,   "PXA_PXQ6_KSPLIT",   true,  "K1 decode gateup K-split, bit-exact kseg form")
PXA_PXQ6_GATE(pxa_pxq6_pairlut,  "PXA_PXQ6_PAIRLUT",  false, "K2 byte-pair float2 LUT, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_vecx,     "PXA_PXQ6_VECX",     true,  "K2 float4 activation loads, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_gufuse,   "PXA_PXQ6_GUFUSE",   true,  "K3 fused up+gate GEMM + GLU epilogue, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_scatfuse, "PXA_PXQ6_SCATFUSE", true,  "K3 down-GEMM scatter fusion, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_ragtail,  "PXA_PXQ6_RAGTAIL",  true,  "K4 ragged-tile FMA skip, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_pipe,     "PXA_PXQ6_PIPE",     false, "K5 register-prefetch GEMM pipelining, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_force_prefill, "PXA_PXQ6_FORCE_PREFILL", false, "TEST ONLY: bypass prefill arch gate")
PXA_PXQ6_GATE(pxa_g2_redfuse,    "PXA_G2_REDFUSE",    false, "G2-F1 absorb gateup ksplit-reduce + GLU into the down-mmv x-staging prologue, bit-exact")
PXA_PXQ6_GATE(pxa_g2_addfuse,    "PXA_G2_ADDFUSE",    true,  "G2-F4 residual-add fusion: ADD+FUSED_RMS_NORM pair + MUL_MULTI_ADD residual epilogue, bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_prmt,     "PXA_PXQ6_PRMT",     false, "K2c prmt register-LUT book decode (4-bit tiers, decode mmv), bit-exact")
PXA_PXQ6_GATE(pxa_pxq6_ldcs,     "PXA_PXQ6_LDCS",     false, "K7 streaming (evict-last-priority) weight code loads in decode mmv, bit-exact")

// PXQ6R (GGML_TYPE_PXQ6 = 256, display "pxq6") master gate + host self-check. Default ON;
// PXA_PXQ6R=0 disables (dequant->cublas fallback). Invariants of the frozen LM32 book:
// exactly 32 entries, strictly ascending, fp16-snap idempotent, book[16] == +0.0f (PXQ6R_ZIDX),
// book[0] == -1.0f, book[31] > 0. Failure -> one-time warning + fmt NONE.
static inline bool pxa_pxq6r_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ6R");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float book[32] = PXQ6_LM32_INIT;
            bool ok = book[16] == 0.0f && book[0] == -1.0f && book[31] > 0.0f;
            for (int i = 0; i < 31 && ok; ++i) ok = book[i] < book[i+1];
            for (int i = 0; i < 32 && ok; ++i) {   // fp16-snap idempotence (frozen contract)
                ok = __half2float(__float2half_rn(book[i])) == book[i];
            }
            if (!ok) {
                fprintf(stderr, "PXA_PXQ6R: LM32 table self-check FAILED — fused kernels DISABLED (fallback in use)\n");
                v = false;
            } else {
                fprintf(stderr, "PXA_PXQ6R fused kernels: ON (LM32 self-check PASS; PXA_PXQ6R=0 disables)\n");
            }
        } else {
            fprintf(stderr, "PXA_PXQ6R fused kernels: OFF (dequant->cublas fallback)\n");
        }
        return v;
    }();
    return on;
}

// K6 WMMA: 0 = off (default), 1 = fp32-accum fragments, 2 = fp16-accum twin. cc==700 only.
static inline int pxa_pxq6_wmma() {
    static const int mode = [](){
        const char * e = getenv("PXA_PXQ6_WMMA");
        int m = e ? atoi(e) : 0;
        if (m < 0 || m > 2) m = 0;
        if (m) fprintf(stderr, "PXA_PXQ6_WMMA: mode %d (K6 V100 HMMA prefill — NOT bit-exact, G3+G4 gated)\n", m);
        return m;
    }();
    return mode;
}

// K1b generic S-split (2/4/8; 0 = off). NOT bit-exact — staged for the GPU window (G3 first).
static inline int pxa_pxq6_ksplit_gen() {
    static const int s = [](){
        const char * e = getenv("PXA_PXQ6_KSPLIT_GEN");
        int v = e ? atoi(e) : 0;
        if (v != 0 && v != 2 && v != 4 && v != 8) v = 0;
        if (v) fprintf(stderr, "PXA_PXQ6_KSPLIT_GEN: S=%d (generic split — NOT bit-exact, G3 required)\n", v);
        return v;
    }();
    return s;
}

// env table overrides (PXA_PXQ6_BOOK / _SUB / _SUB_HQ + the PXQ6R 32-entry PXA_PXQ6R_BOOK),
// fp16-snapped, per-device upload
static inline void pxq6_maybe_upload_tables(int device) {
    static bool parsed = false;
    static bool have_book = false, have_sub = false, have_sub8 = false, have_lm32 = false;
    static float ebook[16], esub[16], esub8[16], elm32[32];
    static bool uploaded[64] = {false};
    if (!parsed) {
        parsed = true;
        auto parse_nw = [](const char * e, float * out, int want) -> bool {
            int n = 0; float v[32];
            char * dup = strdup(e);
            for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
            free(dup);
            if (n != want) return false;
            for (int i = 0; i < want; ++i) out[i] = __half2float(__float2half_rn(v[i]));
            return true;
        };
        auto parse16 = [&](const char * e, float * out) -> bool { return parse_nw(e, out, 16); };
        if (const char * e = getenv("PXA_PXQ6_BOOK"))   { have_book = parse16(e, ebook);  fprintf(stderr, "PXA_PXQ6_BOOK: %s\n",   have_book ? "custom book active"  : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ6_SUB"))    { have_sub  = parse16(e, esub);   fprintf(stderr, "PXA_PXQ6_SUB: %s\n",    have_sub  ? "custom SUB16 active" : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ6_SUB_HQ")) { have_sub8 = parse16(e, esub8);  fprintf(stderr, "PXA_PXQ6_SUB_HQ: %s\n", have_sub8 ? "custom SUB8 active"  : "parse FAILED — ignored"); }
        if (const char * e = getenv("PXA_PXQ6R_BOOK"))  { have_lm32 = parse_nw(e, elm32, 32); fprintf(stderr, "PXA_PXQ6R_BOOK: %s\n", have_lm32 ? "custom LM32 book active" : "parse FAILED — ignored"); }
    }
    if ((have_book || have_sub || have_sub8 || have_lm32) && device >= 0 && device < 64 && !uploaded[device]) {
        uploaded[device] = true;
        int cur = 0; cudaGetDevice(&cur);
        if (cur != device) cudaSetDevice(device);
        if (have_book) cudaMemcpyToSymbol(pxq6_book_g,  ebook, sizeof(ebook));
        if (have_sub)  cudaMemcpyToSymbol(pxq6_sub16_g, esub,  sizeof(esub));
        if (have_sub8) cudaMemcpyToSymbol(pxq6_sub8_g,  esub8, sizeof(esub8));
        if (have_lm32) cudaMemcpyToSymbol(pxq6_lm32_g,  elm32, sizeof(elm32));
        if (cur != device) cudaSetDevice(cur);
    }
}

// ---------------------------------------------------------------------------------------------
// format policies. The WHOLE format difference is: slab/header geometry + how the per-8-elem
// effective scales are decoded. NEFF = distinct eff scales per 32-elem block (1 = single chain,
// preserving the exact PXQ4/PXQ5 accumulation shape).
// ---------------------------------------------------------------------------------------------
// (pxq6_pol_p4, the policy for the retired legacy type id 250, was removed 2026-07-21.)

// (pxq6_pol_p5, the policy for the retired legacy type id 251, was removed 2026-07-21.)

struct pxq6_pol_p6 {
    static constexpr int  SLAB = PXQ6_SLAB_BYTES, HDR = PXQ6_HDR_BYTES, CODE_OFF = 64, NEFF = 2;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = pxq6_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq6_sub16_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq6_book_g[i]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elems 0-15
        eff[1] = anch * sub[sb >> 4];     // elems 16-31
    }
    static constexpr int CODE_WORDS = 4, CODE_BYTES = 16;
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B code row
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        return plut[(q[b >> 2] >> (8*(b & 3))) & 0xff];
    }
};

struct pxq6_pol_p6hq {
    static constexpr int  SLAB = PXQ6HQ_SLAB_BYTES, HDR = PXQ6_HDR_BYTES, CODE_OFF = 128, NEFF = 4;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = pxq6_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq6_sub8_g[tid - 16];
    }
    __device__ static float bookv(int i) { return pxq6_book_g[i]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb0 = slab[2*row], sb1 = slab[2*row + 1];
        eff[0] = anch * sub[sb0 & 0xf];   // elems 0-7
        eff[1] = anch * sub[sb0 >> 4];    // elems 8-15
        eff[2] = anch * sub[sb1 & 0xf];   // elems 16-23
        eff[3] = anch * sub[sb1 >> 4];    // elems 24-31
    }
    static constexpr int CODE_WORDS = 4, CODE_BYTES = 16;
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B code row
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        return plut[(q[b >> 2] >> (8*(b & 3))) & 0xff];
    }
};

// PXQ6R "real PXQ6" (GGML_TYPE_PXQ6 = 256, display "pxq6"): LM32 5-bit codes on the UNCHANGED
// E16-row scale machinery (fp16 row anchor + shared SUB16 4-bit sub per 16-elem block).
// Code row = 20 B: bytes 0..15 nibble plane (byte-identical layout to the P6 core rows) +
// bytes 16..19 one LE u32 hi-bit plane (bit j = bit 4 of code j, j = 0..31). Code rows start
// at 64 + 20r — 4-byte aligned for every r, NOT 8/16-aligned for odd r => scalar u32 loads
// ONLY (the CODE_WORDS == 5 arm of pxq6_ldcodes; uint2/uint4 vector loads are ILLEGAL here).
// Modes: TAB/TAB_CS only — the 32-entry book rides neither PRMT (two 16-entry byte planes
// keyed by a nibble) nor PAIRLUT (256-entry LUT keyed on one byte = two 4-bit codes); the
// pickers demote exactly like P2/P3 (pairl() is a never-executed compile stub).
struct pxq6_pol_p6r {
    static constexpr int  SLAB = PXQ6R_SLAB_BYTES, HDR = PXQ6R_HDR_BYTES, CODE_OFF = PXQ6R_CODE_OFF, NEFF = 2;
    static constexpr int CODE_WORDS = 5, CODE_BYTES = 20;
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 32) tab[tid]      = pxq6_lm32_g[tid];
        else if (tid < 48) sub[tid - 32] = pxq6_sub16_g[tid - 32];
    }
    __device__ static float bookv(int i) { return pxq6_lm32_g[i & 31]; }
    __device__ static float anchor(const uint8_t * panel, int row) {
        return __half2float(((const half *)panel)[row]);
    }
    __device__ static void row_effs(const uint8_t * slab, int row, float anch, const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elems 0-15
        eff[1] = anch * sub[sb >> 4];     // elems 16-31
    }
    // pair b (elems 2b, 2b+1): nibble plane byte extraction == proven P6 form; 5th bit from the
    // register-resident plane word q[4] (bit j = element j -> elem 2b at bit 2b, 2b+1 at 2b+1)
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8*(b & 3))) & 0xff;   // LE byte b of the 16-B nibble plane
        const uint32_t hi = q[4];                             // hi-bit plane word
        const int c0 = (byte & 0xf) | (int)(((hi >> (2*b    )) & 1) << 4);   // element 2b
        const int c1 = (byte >> 4)  | (int)(((hi >> (2*b + 1)) & 1) << 4);   // element 2b+1
        return make_float2(tab[c0], tab[c1]);
    }
    __device__ static float2 pairl(const uint32_t * q, int b, const float2 * plut) {
        (void)q; (void)b; (void)plut; return make_float2(0.f, 0.f);   // PAIRLUT demoted for P6R
    }
};

// per-row code load: CODE_WORDS LE u32 words. 16 B formats keep the proven uint4 load;
// 8 B uses uint2 (8-aligned: CODE_OFF 64 + row*8); 12 B uses 3 u32 (4-aligned: row*12).
// CS variant (K7): identical addresses/values through ld.global.cs — the decode weight stream
// is read exactly once per token, so mark it evict-first and keep L2 for the hot working set
// (activations/tables/KV). Cache policy cannot change loaded values => bit-exact by construction.
template <class POL, bool CS = false>
static __device__ __forceinline__ void pxq6_ldcodes(const uint8_t * p, uint32_t * q) {
    if constexpr (POL::CODE_WORDS == 4) {
        if constexpr (CS) { *(uint4 *)q = __ldcs((const uint4 *)p); }
        else              { *(uint4 *)q = *(const uint4 *)p; }
    } else if constexpr (POL::CODE_WORDS == 2) {
        if constexpr (CS) { *(uint2 *)q = __ldcs((const uint2 *)p); }
        else              { *(uint2 *)q = *(const uint2 *)p; }
    } else if constexpr (POL::CODE_WORDS == 5) {
        // P6R 20 B row: only 4-aligned (odd rows) -> five scalar u32 loads, NEVER vector
        const uint32_t * s = (const uint32_t *)p;
        if constexpr (CS) { q[0] = __ldcs(s); q[1] = __ldcs(s + 1); q[2] = __ldcs(s + 2); q[3] = __ldcs(s + 3); q[4] = __ldcs(s + 4); }
        else              { q[0] = s[0]; q[1] = s[1]; q[2] = s[2]; q[3] = s[3]; q[4] = s[4]; }
    } else {
        const uint32_t * s = (const uint32_t *)p;
        if constexpr (CS) { q[0] = __ldcs(s); q[1] = __ldcs(s + 1); q[2] = __ldcs(s + 2); }
        else              { q[0] = s[0]; q[1] = s[1]; q[2] = s[2]; }
    }
}

// ---------------------------------------------------------------------------------------------
// K2c PRMT register-LUT book decode (B1 steal from the QTIP/Marlin lineage). The 16-entry
// fp16-snapped book is held as two byte planes (lo/hi) across 8 uniform registers; a nibble
// code indexes them with prmt (byte_perm) — zero smem traffic, zero bank conflicts, and the
// produced fp32 values are IDENTICAL to tab[] (the book is fp16-snapped, half->float is exact)
// with the same FMA order => bit-exact vs MODE_TAB. 4-bit tiers only (nibble == book index).
// ---------------------------------------------------------------------------------------------
struct pxq6_prmt_book {
    uint32_t L[4];   // lo bytes of book[0..15]
    uint32_t H[4];   // hi bytes of book[0..15]
};

// build once per block from the staged fp32 tab (post-__syncthreads); uniform across threads.
static __device__ __forceinline__ void pxq6_prmt_build(const float * __restrict__ tab, pxq6_prmt_book & B) {
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        uint32_t lo = 0, hi = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned short u = __half_as_ushort(__float2half_rn(tab[4*r + i]));
            lo |= (uint32_t)(u & 0xff) << (8*i);
            hi |= (uint32_t)(u >> 8)   << (8*i);
        }
        B.L[r] = lo; B.H[r] = hi;
    }
}

// Gather two book bytes from one 16-entry plane (4 regs). idx_lo/idx_hi land in OUTPUT bytes 0/1.
// KEY: __byte_perm selects each output byte via a 4-BIT NIBBLE of the selector (not a byte), so the
// two indices' low-3-bits go in nibbles 0 and 1 (perm_sel). The high-half (idx>=8) select is a
// per-BYTE 0xFF mask, so __vcmpgeu4 takes a byte-packed copy of the indices (mask_sel).
static __device__ __forceinline__ uint32_t pxq6_prmt_plane(const uint32_t * __restrict__ P,
                                                           uint32_t perm_sel, uint32_t mask_sel) {
    const uint32_t a = __byte_perm(P[0], P[1], perm_sel);   // out[0,1] = entries (idx&7) of the low half
    const uint32_t b = __byte_perm(P[2], P[3], perm_sel);   // out[0,1] = entries (idx&7)+8 of the high half
    const uint32_t m = __vcmpgeu4(mask_sel, 0x08080808u);   // 0xFF in byte i where idx_i >= 8
    return (a & ~m) | (b & m);
}

// one code byte (2 nibbles) -> float2(book[lo], book[hi]); same value/order contract as pair().
static __device__ __forceinline__ float2 pxq6_prmt_pair(const uint32_t * __restrict__ q, int b,
                                                        const pxq6_prmt_book & B) {
    const uint32_t byte   = (q[b >> 2] >> (8*(b & 3))) & 0xffu;
    const uint32_t idx_lo = byte & 0xfu, idx_hi = byte >> 4;
    const uint32_t perm   = (idx_lo & 7u) | ((idx_hi & 7u) << 4);   // nibble0=idx_lo&7, nibble1=idx_hi&7
    const uint32_t mask   = idx_lo | (idx_hi << 8);                 // byte0=idx_lo, byte1=idx_hi
    const uint32_t lo     = pxq6_prmt_plane(B.L, perm, mask);       // out[0,1]=lo(V_lo),lo(V_hi)
    const uint32_t hi     = pxq6_prmt_plane(B.H, perm, mask);       // out[0,1]=hi(V_lo),hi(V_hi)
    const uint32_t w      = __byte_perm(lo, hi, 0x5140u);           // [lo(Vlo),hi(Vlo),lo(Vhi),hi(Vhi)]=half2
    return __half22float2(*(const half2 *)&w);
}

// shared address helpers
template <class POL>
static __device__ __forceinline__ size_t pxq6_panel_stride(int kslabs) {
    return (size_t)POL::HDR + (size_t)kslabs*POL::SLAB;
}
template <class POL>
static __device__ __forceinline__ const uint8_t * pxq6_panel(const uint8_t * W, int e, int panels, int p, int kslabs) {
    return W + ((size_t)e*panels + p)*pxq6_panel_stride<POL>(kslabs);
}

// PAIRLUT staging: plut[b] = (book[lo(b)], book[hi(b)]) — built from the same global tables
template <class POL>
static __device__ __forceinline__ void pxq6_stage_pairlut(float2 * plut, int tid, int nthr) {
    for (int i = tid; i < 256; i += nthr) plut[i] = make_float2(POL::bookv(i & 0xf), POL::bookv(i >> 4));
}

// ---------------------------------------------------------------------------------------------
// the scaled 32-elem dot product — the ONE routine every decode kernel shares.
// NEFF=1 keeps the exact single-chain accumulation of the proven PXQ4/PXQ5 kernels
// (su += d*t with one t chain); NEFF=2/4 is the PXQ6/PXQ6HQ reference form.
// MODE/VECX only change operand SOURCING (never order) => bit-exact per variant:
//   MODE 0 = smem tab, 1 = smem PAIRLUT, 2 = PRMT register book,
//        3 = smem tab + ld.cs codes, 4 = PRMT + ld.cs codes.
// ---------------------------------------------------------------------------------------------
#define PXQ6_MODE_TAB     0
#define PXQ6_MODE_PAIRL   1
#define PXQ6_MODE_PRMT    2
#define PXQ6_MODE_TAB_CS  3
#define PXQ6_MODE_PRMT_CS 4

template <int MODE> struct pxq6_mode {
    static constexpr bool cs   = (MODE == PXQ6_MODE_TAB_CS) || (MODE == PXQ6_MODE_PRMT_CS);
    static constexpr bool prmt = (MODE == PXQ6_MODE_PRMT)   || (MODE == PXQ6_MODE_PRMT_CS);
    static constexpr bool pairl = (MODE == PXQ6_MODE_PAIRL);
};

template <class POL, int MODE, bool VECX>
static __device__ __forceinline__ float pxq6_dot32(const uint8_t * __restrict__ slab, int row, float anch,
                                                   const float * __restrict__ xk,
                                                   const float * __restrict__ tab,
                                                   const float * __restrict__ sub,
                                                   const float2 * __restrict__ plut,
                                                   const pxq6_prmt_book & pb) {
    using M = pxq6_mode<MODE>;
    float eff[POL::NEFF];
    POL::row_effs(slab, row, anch, sub, eff);
    uint32_t q[POL::CODE_WORDS];
    pxq6_ldcodes<POL, M::cs>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);
    float t[POL::NEFF];
    #pragma unroll
    for (int i = 0; i < POL::NEFF; ++i) t[i] = 0.f;
    if (VECX) {
        #pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2*b];
            const float2 p0 = M::prmt ? pxq6_prmt_pair(q, b,   pb) : M::pairl ? POL::pairl(q, b,   plut) : POL::pair(q, b,   tab);
            const float2 p1 = M::prmt ? pxq6_prmt_pair(q, b+1, pb) : M::pairl ? POL::pairl(q, b+1, plut) : POL::pair(q, b+1, tab);
            t[(b*POL::NEFF) >> 4]     += p0.x*xv.x + p0.y*xv.y;
            t[((b+1)*POL::NEFF) >> 4] += p1.x*xv.z + p1.y*xv.w;
        }
    } else {
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = M::prmt ? pxq6_prmt_pair(q, b, pb) : M::pairl ? POL::pairl(q, b, plut) : POL::pair(q, b, tab);
            t[(b*POL::NEFF) >> 4] += p.x*xk[2*b] + p.y*xk[2*b+1];
        }
    }
    if (POL::NEFF == 1) return eff[0]*t[0];
    if (POL::NEFF == 2) return eff[0]*t[0] + eff[1]*t[1];
    return (eff[0]*t[0] + eff[1]*t[1]) + (eff[2]*t[2] + eff[3]*t[3]);
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant (convert.cu fallback hook — dequant->cublas keeps PXQ6 functional on
// every arch, incl. the 1080Ti/sm_61). One block per slab, 64 threads (one row each).
// ---------------------------------------------------------------------------------------------
template <class POL, typename dst_t>
static __global__ void k_pxq6_dequant_matrix(const uint8_t * __restrict__ wq, dst_t * __restrict__ y,
                                             const int kslabs, const int64_t K) {
    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    POL::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();
    const int64_t slab_id = blockIdx.x;
    const int64_t p  = slab_id / kslabs;
    const int     kb = (int)(slab_id % kslabs);
    const int     row = threadIdx.x;
    const uint8_t * panel = wq + p*pxq6_panel_stride<POL>(kslabs);
    const uint8_t * slab  = panel + POL::HDR + (size_t)kb*POL::SLAB;
    const float anch = POL::HDR ? POL::anchor(panel, row) : 0.f;
    float eff[POL::NEFF];
    POL::row_effs(slab, row, anch, sub, eff);
    dst_t * dst = y + (p*PXQ6_BM + row)*K + kb*PXQ6_QK;
    uint32_t q[POL::CODE_WORDS];
    pxq6_ldcodes<POL>(slab + POL::CODE_OFF + row*POL::CODE_BYTES, q);
    #pragma unroll
    for (int b = 0; b < 16; ++b) {                 // b = element-pair index
        const float e = eff[(b*POL::NEFF) >> 4];
        const float2 v = POL::pair(q, b, tab);
        dst[2*b]   = (dst_t)(e * v.x);
        dst[2*b+1] = (dst_t)(e * v.y);
    }
}

template <typename dst_t>
static void dequantize_row_pxq6_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    if (nrows % PXQ6_BM != 0 || n_per_row % PXQ6_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq6_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);
    const int kslabs = (int)(n_per_row / PXQ6_QK);
    const int64_t nslabs = (nrows / PXQ6_BM) * (int64_t)kslabs;
    k_pxq6_dequant_matrix<pxq6_pol_p6, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

template <typename dst_t>
static void dequantize_row_pxq6hq_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                       cudaStream_t stream) {
    if (nrows % PXQ6_BM != 0 || n_per_row % PXQ6_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq6hq_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);
    const int kslabs = (int)(n_per_row / PXQ6_QK);
    const int64_t nslabs = (nrows / PXQ6_BM) * (int64_t)kslabs;
    k_pxq6_dequant_matrix<pxq6_pol_p6hq, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

template <typename dst_t>
static void dequantize_row_pxq6r_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                      cudaStream_t stream) {
    if (nrows % PXQ6_BM != 0 || n_per_row % PXQ6_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq6r_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq6_maybe_upload_tables(dev);   // handles the shared SUB16 + the PXQ6R LM32 override
    const int kslabs = (int)(n_per_row / PXQ6_QK);
    const int64_t nslabs = (nrows / PXQ6_BM) * (int64_t)kslabs;
    k_pxq6_dequant_matrix<pxq6_pol_p6r, dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

// ---------------------------------------------------------------------------------------------
// decode: fused up+gate+GLU mmv + down mmv — the generic family (grid/threads/smem identical
// to the proven k_pxq5_* pair; POL/PAIR/VECX select format + K2 variants).
// ---------------------------------------------------------------------------------------------
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_gateup_mmv(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                  const char * __restrict__ x_base, const size_t x_tok_stride,
                  char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                  const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                  const float * __restrict__ bias_u, const size_t bias_u_nb1,
                  const float * __restrict__ bias_g, const size_t bias_g_nb1,
                  const int R, const int K, const int n_as,
                  const int unary, const float alpha, const float limit) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : 1];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    float su = 0.f, sg = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ6_QK;
        su += pxq6_dot32<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB, row, anchU, xk, tabU, subU, plut, pbU);
        sg += pxq6_dot32<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB, row, anchG, xk, tabG, subG, plut, pbG);
    }
    red[(kseg*64 + row)]                      = su;
    red[(PXQ4_MMV_KSEG*64) + (kseg*64 + row)] = sg;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f, g = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            u += red[s*64 + row];
            g += red[PXQ4_MMV_KSEG*64 + s*64 + row];
        }
        const int grow = p*PXQ6_BM + row;
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
        const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[grow] = r;
    }
}

template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv(const uint8_t * __restrict__ W,
           const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
           char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
           const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
           const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : 1];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    float su = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        su += pxq6_dot32<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch,
                                          xs + kb*PXQ6_QK, tab, sub, plut, pb);
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ6_BM + row] = u;
    }
}

// ---------------------------------------------------------------------------------------------
// G2-F1 REDFUSE (2026-07-19): the down mmv with k_pxq6_gateup_reduce + GLU absorbed into its
// x-staging prologue. Instead of reading the gateup dst, each block reconstructs it from the
// KSPLIT partial workspace with EXACTLY the reducer's arithmetic (fixed s = 0..KSEG-1 ascending
// summation, then bias, then pxq4_glu_apply) => xs[] == the dst the standalone reduce would have
// written, bit-for-bit; everything after staging is k_pxq6_mmv verbatim => the final output is
// bit-identical while one kernel launch + one dst HBM round-trip disappear. Redundant reduce
// compute is per-block (panels x), but ws is tiny (2*KSEG*K floats/slot) and L2-hot.
// Driver-guarded: only when the gateup dst is SOLE-consumed by this down MUL_MAT_ID, only for
// the bit-exact KSPLIT=1 form (declined under KSPLIT_GEN).
// ---------------------------------------------------------------------------------------------
template <class POL, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_mmv_redfuse(const uint8_t * __restrict__ W,
                   const float * __restrict__ ws,      // KSPLIT partials; gateup R == our K
                   char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                   const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                   const float * __restrict__ bias_u, const size_t bias_u_nb1,
                   const float * __restrict__ bias_g, const size_t bias_g_nb1,
                   const int R, const int K, const int n_as, const int n_ids,
                   const int unary, const float alpha, const float limit) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;
    float * red = pxq6_smem + K;

    // staging prologue = the reducer, verbatim arithmetic (grow -> idx, R_gateup -> K)
    const float * wsj = ws + ((size_t)iy*n_ids + j)*2*PXQ4_MMV_KSEG*K;
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        float u = 0.f, g = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            u += wsj[(size_t)s*K + idx];
            g += wsj[((size_t)PXQ4_MMV_KSEG + s)*K + idx];
        }
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)idx*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)idx*sizeof(float));
        xs[idx] = pxq4_glu_apply(g, u, unary, alpha, limit);
    }

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : 1];
    POL::stage_tabs(tab, sub, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POL>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pb{};
    if constexpr (pxq6_mode<MODE>::prmt) pxq6_prmt_build(tab, pb);

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * pan = pxq6_panel<POL>(W, e, panels, p, kslabs);
    const float anch = POL::HDR ? POL::anchor(pan, row) : 0.f;

    float su = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        su += pxq6_dot32<POL, MODE, VECX>(pan + POL::HDR + (size_t)kb*POL::SLAB, row, anch,
                                          xs + kb*PXQ6_QK, tab, sub, plut, pb);
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ6_BM + row] = u;
    }
}

// ---------------------------------------------------------------------------------------------
// K1 KSPLIT — bit-exact kseg-as-blocks form. grid (panels*KSEG, n_ids, Ny), 64 threads: thread
// = row, block owns ONE of the proven kernel's PXQ4_MMV_KSEG chains (kb ≡ kseg mod 4, ascending)
// => per-thread fp32 chains are IDENTICAL to the proven kernel's. Partials go to the persistent
// workspace; k_pxq6_gateup_reduce then replicates the proven red[] combination order (s = 0..3,
// ascending) + bias + GLU exactly => bit-identical dst.
// workspace layout: ws[((iy*n_ids + j)*2 + {0=u,1=g})*KSEG*R + kseg*R + grow]
// ---------------------------------------------------------------------------------------------
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(64)
k_pxq6_gateup_mmv_ksplit(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                         const char * __restrict__ x_base, const size_t x_tok_stride,
                         float * __restrict__ ws,
                         const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                         const int R, const int K, const int n_as, const int n_ids) {
    const int pk = blockIdx.x;
    const int p    = pk / PXQ4_MMV_KSEG;
    const int kseg = pk % PXQ4_MMV_KSEG;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq6_smem[];
    float * xs = pxq6_smem;                       // K floats (no red[] needed)

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : 1];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 64);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row = threadIdx.x;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    float su = 0.f, sg = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ6_QK;
        su += pxq6_dot32<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB, row, anchU, xk, tabU, subU, plut, pbU);
        sg += pxq6_dot32<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB, row, anchG, xk, tabG, subG, plut, pbG);
    }
    const int grow = p*PXQ6_BM + row;
    float * wsj = ws + ((size_t)iy*n_ids + j)*2*PXQ4_MMV_KSEG*R;
    wsj[(size_t)kseg*R + grow]                        = su;
    wsj[((size_t)PXQ4_MMV_KSEG + kseg)*R + grow]      = sg;
}

// reducer: one thread per (iy, j, grow). Sums the KSEG partials in the proven fixed order,
// adds biases, applies GLU, writes dst — byte-identical epilogue to the proven kernel.
static __global__ void k_pxq6_gateup_reduce(const float * __restrict__ ws,
        char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const float * __restrict__ bias_u, const size_t bias_u_nb1,
        const float * __restrict__ bias_g, const size_t bias_g_nb1,
        const int R, const int n_as, const int n_ids,
        const int unary, const float alpha, const float limit) {
    const int grow = blockIdx.x*blockDim.x + threadIdx.x;
    if (grow >= R) return;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;   // SER slot: dst untouched (matches the proven kernel)
    const float * wsj = ws + ((size_t)iy*n_ids + j)*2*PXQ4_MMV_KSEG*R;
    float u = 0.f, g = 0.f;
    #pragma unroll
    for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
        u += wsj[(size_t)s*R + grow];
        g += wsj[((size_t)PXQ4_MMV_KSEG + s)*R + grow];
    }
    if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
    if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
    const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    out[grow] = r;
}

// K1b generic S-split (NOT bit-exact; staged, G3-gated): 256-thr blocks, block pk handles the
// K-chunk [chunk*Kc, (chunk+1)*Kc) with the full 64x4 kseg structure INSIDE the chunk. Stages
// only its K-chunk slice of x (the smem rider from the deep-dive: dyn smem K/S + red).
// Reduction: same workspace; the reducer sums S chunk-partials in ascending chunk order
// (deterministic run-to-run; differs from the proven chain order => G3 before live use).
template <class POLU, class POLG, int MODE, bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq6_gateup_mmv_ksplit_gen(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                             const char * __restrict__ x_base, const size_t x_tok_stride,
                             float * __restrict__ ws,
                             const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                             const int R, const int K, const int n_as, const int n_ids, const int S) {
    const int pk = blockIdx.x;
    const int p     = pk / S;
    const int chunk = pk % S;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    const int kslabs = K / PXQ6_QK;
    const int kb0 = (kslabs*chunk)/S, kb1 = (kslabs*(chunk+1))/S;
    const int Kc  = (kb1 - kb0)*PXQ6_QK;

    extern __shared__ float pxq6_smem[];
    float * xs  = pxq6_smem;                      // Kc floats (chunk slice only)
    float * red = pxq6_smem + Kc;                 // 2*KSEG*64

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride) + kb0*PXQ6_QK;
    for (int idx = threadIdx.x; idx < Kc; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tabU[32], subU[16], tabG[32], subG[16];   // 32-entry tabs: P6R LM32
    __shared__ float2 plut[pxq6_mode<MODE>::pairl ? 256 : 1];
    POLU::stage_tabs(tabU, subU, threadIdx.x);
    POLG::stage_tabs(tabG, subG, threadIdx.x);
    if (pxq6_mode<MODE>::pairl) pxq6_stage_pairlut<POLU>(plut, threadIdx.x, 256);
    __syncthreads();
    pxq6_prmt_book pbU{}, pbG{};
    if constexpr (pxq6_mode<MODE>::prmt) { pxq6_prmt_build(tabU, pbU); pxq6_prmt_build(tabG, pbG); }

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ6_BM;
    const uint8_t * panU = pxq6_panel<POLU>(Wu, e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POLG>(Wg, e, panels, p, kslabs);
    const float anchU = POLU::HDR ? POLU::anchor(panU, row) : 0.f;
    const float anchG = POLG::HDR ? POLG::anchor(panG, row) : 0.f;

    float su = 0.f, sg = 0.f;
    for (int kb = kb0 + kseg; kb < kb1; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + (kb - kb0)*PXQ6_QK;
        su += pxq6_dot32<POLU, MODE, VECX>(panU + POLU::HDR + (size_t)kb*POLU::SLAB, row, anchU, xk, tabU, subU, plut, pbU);
        sg += pxq6_dot32<POLG, MODE, VECX>(panG + POLG::HDR + (size_t)kb*POLG::SLAB, row, anchG, xk, tabG, subG, plut, pbG);
    }
    red[(kseg*64 + row)]                      = su;
    red[(PXQ4_MMV_KSEG*64) + (kseg*64 + row)] = sg;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f, g = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            u += red[s*64 + row];
            g += red[PXQ4_MMV_KSEG*64 + s*64 + row];
        }
        const int grow = p*PXQ6_BM + row;
        float * wsj = ws + ((size_t)iy*n_ids + j)*2*8*R;    // S<=8 slots reserved
        wsj[(size_t)chunk*R + grow]       = u;
        wsj[((size_t)8 + chunk)*R + grow] = g;
    }
}

static __global__ void k_pxq6_gateup_reduce_gen(const float * __restrict__ ws,
        char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
        const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
        const float * __restrict__ bias_u, const size_t bias_u_nb1,
        const float * __restrict__ bias_g, const size_t bias_g_nb1,
        const int R, const int n_as, const int n_ids,
        const int unary, const float alpha, const float limit, const int S) {
    const int grow = blockIdx.x*blockDim.x + threadIdx.x;
    if (grow >= R) return;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;
    const float * wsj = ws + ((size_t)iy*n_ids + j)*2*8*R;
    float u = 0.f, g = 0.f;
    for (int s = 0; s < S; ++s) {
        u += wsj[(size_t)s*R + grow];
        g += wsj[((size_t)8 + s)*R + grow];
    }
    if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
    if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
    const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
    float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
    out[grow] = r;
}

// persistent KSPLIT workspace (per device; grown OUTSIDE graph capture only — if capture is
// active and the buffer is too small, the driver falls back to the non-split kernel, which is
// bit-identical anyway, so replayed graphs stay correct whatever the capture saw).
struct pxq6_ksplit_ws_t { float * ptr = nullptr; size_t sz = 0; };
static inline float * pxq6_ksplit_workspace(int device, cudaStream_t stream, size_t need_floats) {
    static pxq6_ksplit_ws_t ws[64];
    if (device < 0 || device >= 64) return nullptr;
    pxq6_ksplit_ws_t & w = ws[device];
    const size_t need = need_floats*sizeof(float);
    if (w.sz >= need) return w.ptr;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    if (st != cudaStreamCaptureStatusNone) return nullptr;   // can't grow mid-capture -> decline
    if (w.ptr) cudaFree(w.ptr);
    w.ptr = nullptr; w.sz = 0;
    if (cudaMalloc(&w.ptr, need) != cudaSuccess) { w.ptr = nullptr; cudaGetLastError(); return nullptr; }
    w.sz = need;
    return w.ptr;
}

// ---------------------------------------------------------------------------------------------
// prefill: grouped fused GEMM — generic family. Tiling/accumulation identical to the proven
// k_pxq5_gemm_grouped (64 thr, 8r x 8t per thread, strict k-order half2 chains); POL selects the
// format's dequant; RAG = K4 ragged-tile FMA skip; PIPE = K5 register prefetch. All bit-exact.
// ---------------------------------------------------------------------------------------------
template <class POL>
static __device__ __forceinline__ void pxq6_deq_slab_cm(const uint8_t * __restrict__ slab, int tid, float anch,
                                                        const float * __restrict__ tab, const float * __restrict__ sub,
                                                        const uint32_t * __restrict__ q, half (* __restrict__ sW)[PXQ4_BM]) {
    float eff[POL::NEFF];
    POL::row_effs(slab, tid, anch, sub, eff);
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
        const float e = eff[(b*POL::NEFF) >> 4];
        const float2 v = POL::pair(q, b, tab);
        sW[2*b][tid]   = __float2half_rn(e * v.x);
        sW[2*b+1][tid] = __float2half_rn(e * v.y);
    }
}

template <class POL, bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq6_gemm_grouped(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                    const float * __restrict__ bias, const size_t bias_nb1,
                    const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    float         * Ct  = C + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sW[PXQ6_QK][PXQ4_BM];
    __shared__ half sA[PXQ6_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    const float anch = POL::HDR ? POL::anchor(pan, tid) : 0.f;

    const int tx = tid & 7, ty = tid >> 3;
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;   // K4: skip FMAs whose stores are masked

    uint32_t qn[POL::CODE_WORDS] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {   // prologue loads, kb = 0
        pxq6_ldcodes<POL>(pan + POL::HDR + (size_t)0*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
        if (a_valid) {
            const half * src = At + (size_t)tid*K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        uint32_t q[POL::CODE_WORDS]; uint4 a0, a1, a2, a3;
        if (PIPE) {
            #pragma unroll
            for (int i = 0; i < POL::CODE_WORDS; ++i) q[i] = qn[i];
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {   // issue next-slab loads; they overlap this slab's FMAs
                const uint8_t * slabn = pan + POL::HDR + (size_t)(kb+1)*POL::SLAB;
                pxq6_ldcodes<POL>(slabn + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
                if (a_valid) {
                    const half * src = At + (size_t)tid*K + (kb+1)*PXQ6_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + tid*POL::CODE_BYTES, q);
            if (a_valid) {
                const half * src = At + (size_t)tid*K + kb*PXQ6_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        pxq6_deq_slab_cm<POL>(slab, tid, anch, tab, sub, q, sW);
        if (a_valid) {
            const half * h0 = (const half *)&a0; const half * h1 = (const half *)&a1;
            const half * h2 = (const half *)&a2; const half * h3 = (const half *)&a3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ6_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
            #pragma unroll 4
            for (int kk = 0; kk < PXQ6_QK; ++kk) {
                half2 a2v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                    const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[2*i][j]   = __hfma2(wlo, a2v[j], acc[2*i][j]);
                        acc[2*i+1][j] = __hfma2(whi, a2v[j], acc[2*i+1][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows)     Ct[(size_t)t*R + row]     = __half2float(__low2half(acc[r][j]))  + b;
            if (t + 1 < tile.nrows) Ct[(size_t)(t+1)*R + row] = __half2float(__high2half(acc[r][j])) + b;
        }
    }
}

// K3 GUFUSE: ONE kernel does the up GEMM + gate GEMM + GLU epilogue. Per-tensor accumulation
// chains are identical to two k_pxq6_gemm_grouped launches; the epilogue applies the exact
// pxq4_glu_apply to the same (bias-folded) u/g floats and converts once => bit-exact vs the
// 3-kernel pipeline (GEMM up, GEMM gate, k_pxq4_glu).
template <class POL, typename dst_t, bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq6_gemm_gufuse(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
                   const half * __restrict__ A, dst_t * __restrict__ H,
                   const float * __restrict__ bias_u, const size_t bias_u_nb1,
                   const float * __restrict__ bias_g, const size_t bias_g_nb1,
                   const pxq4_tile_info * __restrict__ tiles, const int R, const int K,
                   const int unary, const float alpha, const float limit) {
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * panU = pxq6_panel<POL>(Wu, tile.e, panels, p, kslabs);
    const uint8_t * panG = pxq6_panel<POL>(Wg, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    dst_t         * Ht  = H + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sWu[PXQ6_QK][PXQ4_BM];
    __shared__ half sWg[PXQ6_QK][PXQ4_BM];
    __shared__ half sA[PXQ6_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    const float anchU = POL::HDR ? POL::anchor(panU, tid) : 0.f;
    const float anchG = POL::HDR ? POL::anchor(panG, tid) : 0.f;

    const int tx = tid & 7, ty = tid >> 3;
    half2 accU[8][4], accG[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) { accU[r][j] = __floats2half2_rn(0.f, 0.f); accG[r][j] = __floats2half2_rn(0.f, 0.f); }

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;

    uint32_t qun[POL::CODE_WORDS] = {}, qgn[POL::CODE_WORDS] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {
        pxq6_ldcodes<POL>(panU + POL::HDR + POL::CODE_OFF + tid*POL::CODE_BYTES, qun);
        pxq6_ldcodes<POL>(panG + POL::HDR + POL::CODE_OFF + tid*POL::CODE_BYTES, qgn);
        if (a_valid) {
            const half * src = At + (size_t)tid*K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slabU = panU + POL::HDR + (size_t)kb*POL::SLAB;
        const uint8_t * slabG = panG + POL::HDR + (size_t)kb*POL::SLAB;
        uint32_t qu[POL::CODE_WORDS], qg[POL::CODE_WORDS]; uint4 a0, a1, a2, a3;
        if (PIPE) {
            #pragma unroll
            for (int i = 0; i < POL::CODE_WORDS; ++i) { qu[i] = qun[i]; qg[i] = qgn[i]; }
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {
                pxq6_ldcodes<POL>(panU + POL::HDR + (size_t)(kb+1)*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qun);
                pxq6_ldcodes<POL>(panG + POL::HDR + (size_t)(kb+1)*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qgn);
                if (a_valid) {
                    const half * src = At + (size_t)tid*K + (kb+1)*PXQ6_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq6_ldcodes<POL>(slabU + POL::CODE_OFF + tid*POL::CODE_BYTES, qu);
            pxq6_ldcodes<POL>(slabG + POL::CODE_OFF + tid*POL::CODE_BYTES, qg);
            if (a_valid) {
                const half * src = At + (size_t)tid*K + kb*PXQ6_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        pxq6_deq_slab_cm<POL>(slabU, tid, anchU, tab, sub, qu, sWu);
        pxq6_deq_slab_cm<POL>(slabG, tid, anchG, tab, sub, qg, sWg);
        if (a_valid) {
            const half * h0 = (const half *)&a0; const half * h1 = (const half *)&a1;
            const half * h2 = (const half *)&a2; const half * h3 = (const half *)&a3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ6_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
            #pragma unroll 2
            for (int kk = 0; kk < PXQ6_QK; ++kk) {
                half2 a2v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 wpu  = *(const half2 *)&sWu[kk][8*tx + 2*i];
                    const half2 wpg  = *(const half2 *)&sWg[kk][8*tx + 2*i];
                    const half2 wulo = __low2half2(wpu), wuhi = __high2half2(wpu);
                    const half2 wglo = __low2half2(wpg), wghi = __high2half2(wpg);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        accU[2*i][j]   = __hfma2(wulo, a2v[j], accU[2*i][j]);
                        accU[2*i+1][j] = __hfma2(wuhi, a2v[j], accU[2*i+1][j]);
                        accG[2*i][j]   = __hfma2(wglo, a2v[j], accG[2*i][j]);
                        accG[2*i+1][j] = __hfma2(wghi, a2v[j], accG[2*i+1][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        const float bu = bias_u ? *(const float *)((const char *)bias_u + (size_t)tile.e*bias_u_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
        const float bg = bias_g ? *(const float *)((const char *)bias_g + (size_t)tile.e*bias_g_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows) {
                const float u = __half2float(__low2half(accU[r][j]))  + bu;
                const float g = __half2float(__low2half(accG[r][j]))  + bg;
                Ht[(size_t)t*R + row] = (dst_t)pxq4_glu_apply(g, u, unary, alpha, limit);
            }
            if (t + 1 < tile.nrows) {
                const float u = __half2float(__high2half(accU[r][j])) + bu;
                const float g = __half2float(__high2half(accG[r][j])) + bg;
                Ht[(size_t)(t+1)*R + row] = (dst_t)pxq4_glu_apply(g, u, unary, alpha, limit);
            }
        }
    }
}

// K3 SCATFUSE: down GEMM writing straight to the MoE output rows via the row mapping
// (replaces C_down buffer + k_copy_dst_from_contiguous). Same values, same rows => exact.
template <class POL, bool RAG, bool PIPE>
static __global__ void __launch_bounds__(64)
k_pxq6_gemm_down_scat(const uint8_t * __restrict__ W, const half * __restrict__ A,
                      char * __restrict__ dst, const size_t nb1, const size_t nb2,
                      const pxq4_rowmap * __restrict__ map,
                      const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sW[PXQ6_QK][PXQ4_BM];
    __shared__ half sA[PXQ6_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    const float anch = POL::HDR ? POL::anchor(pan, tid) : 0.f;

    const int tx = tid & 7, ty = tid >> 3;
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;
    const bool fma_on  = !RAG || (8*ty) < tile.nrows;

    uint32_t qn[POL::CODE_WORDS] = {};
    uint4 an0 = {0,0,0,0}, an1 = {0,0,0,0}, an2 = {0,0,0,0}, an3 = {0,0,0,0};
    if (PIPE) {
        pxq6_ldcodes<POL>(pan + POL::HDR + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
        if (a_valid) {
            const half * src = At + (size_t)tid*K;
            an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
            an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
        }
    }

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        uint32_t q[POL::CODE_WORDS]; uint4 a0, a1, a2, a3;
        if (PIPE) {
            #pragma unroll
            for (int i = 0; i < POL::CODE_WORDS; ++i) q[i] = qn[i];
            a0 = an0; a1 = an1; a2 = an2; a3 = an3;
            if (kb + 1 < kslabs) {
                pxq6_ldcodes<POL>(pan + POL::HDR + (size_t)(kb+1)*POL::SLAB + POL::CODE_OFF + tid*POL::CODE_BYTES, qn);
                if (a_valid) {
                    const half * src = At + (size_t)tid*K + (kb+1)*PXQ6_QK;
                    an0 = *(const uint4 *)(src);      an1 = *(const uint4 *)(src + 8);
                    an2 = *(const uint4 *)(src + 16); an3 = *(const uint4 *)(src + 24);
                }
            }
        } else {
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + tid*POL::CODE_BYTES, q);
            if (a_valid) {
                const half * src = At + (size_t)tid*K + kb*PXQ6_QK;
                a0 = *(const uint4 *)(src);      a1 = *(const uint4 *)(src + 8);
                a2 = *(const uint4 *)(src + 16); a3 = *(const uint4 *)(src + 24);
            }
        }
        __syncthreads();
        pxq6_deq_slab_cm<POL>(slab, tid, anch, tab, sub, q, sW);
        if (a_valid) {
            const half * h0 = (const half *)&a0; const half * h1 = (const half *)&a1;
            const half * h2 = (const half *)&a2; const half * h3 = (const half *)&a3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ6_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        if (fma_on) {
            #pragma unroll 4
            for (int kk = 0; kk < PXQ6_QK; ++kk) {
                half2 a2v[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) a2v[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                    const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        acc[2*i][j]   = __hfma2(wlo, a2v[j], acc[2*i][j]);
                        acc[2*i+1][j] = __hfma2(whi, a2v[j], acc[2*i+1][j]);
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows) {
                const pxq4_rowmap m = map[tile.row0 + t];
                float * o = (float *)(dst + (size_t)m.i1*nb1 + (size_t)m.i2*nb2);
                o[p*PXQ6_BM + row] = __half2float(__low2half(acc[r][j]));
            }
            if (t + 1 < tile.nrows) {
                const pxq4_rowmap m = map[tile.row0 + t + 1];
                float * o = (float *)(dst + (size_t)m.i1*nb1 + (size_t)m.i2*nb2);
                o[p*PXQ6_BM + row] = __half2float(__high2half(acc[r][j]));
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// K6 WMMA-v1 — V100 (sm_70) fused dequant→skewed-smem-fp16→m16n16k16 HMMA grouped GEMM.
// NOT bit-exact vs the half2 kernels (fp16-mul/fp32-add-tree vs strict-k fp16 chain): G3 logprob
// parity + G4 ppl regate are mandatory gates; env default OFF; runtime cc==700 dispatch only.
// Block = 256 thr (8 warps), 64x64 C tile; warp (wm 0..3, wn 0..1) owns a 16x32 C sub-tile =
// 2 n-frags. Smem: W and A staged row-major [64][32+8] half (+8 half skew — no ldmatrix on
// sm_70, the fragment loader does plain smem reads, so stride/skew is the whole game).
// Epilogue stages fragments through smem for bias + ragged masking (frag layout is opaque).
// ---------------------------------------------------------------------------------------------
#define PXQ6_WMMA_LD 40   // 32 k + 8 half skew

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
// fragment -> smem staging overloads (fp32 accum direct; fp16-accum twin widens via a half stage)
static __device__ __forceinline__ void pxq6_wmma_stage_acc(float * mytile,
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> & a, int warp, int lane) {
    (void)warp; (void)lane;
    nvcuda::wmma::store_matrix_sync(mytile, a, 17, nvcuda::wmma::mem_col_major);
}
static __device__ __forceinline__ void pxq6_wmma_stage_acc(float * mytile,
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> & a, int warp, int lane) {
    __shared__ half hstage[8][16*17];
    nvcuda::wmma::store_matrix_sync(&hstage[warp][0], a, 17, nvcuda::wmma::mem_col_major);
    __syncwarp();
    for (int i = lane; i < 256; i += 32) {
        const int rr = i & 15, cc = i >> 4;
        mytile[cc*17 + rr] = __half2float(hstage[warp][cc*17 + rr]);
    }
}
#endif

template <class POL, bool F32ACC>
static __global__ void __launch_bounds__(256)
k_pxq6_gemm_grouped_wmma(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                         const float * __restrict__ bias, const size_t bias_nb1,
                         const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
#if __CUDA_ARCH__ >= 700 && __CUDA_ARCH__ < 750
    using namespace nvcuda;
    const int panels = R / PXQ6_BM, kslabs = K / PXQ6_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const half    * At  = A + (size_t)tile.row0*K;
    float         * Ct  = C + (size_t)tile.row0*R + (size_t)p*PXQ6_BM;

    __shared__ float tab[32];   // 32 for the P6R LM32 book; 16-entry policies leave 16..31 unstaged
    __shared__ float sub[16];
    __shared__ half sWA[2][64*PXQ6_WMMA_LD];      // [0] = W rows, [1] = A tokens (row-major, skewed)
    const int tid  = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    POL::stage_tabs(tab, sub, tid);

    // staging assignment: 4 threads per row/token, 8 halves (one uint4 of codes / one uint4 of A) each
    const int srow = tid >> 2, sseg = tid & 3;
    const float anch = POL::HDR ? POL::anchor(pan, srow) : 0.f;

    const int wm = warp & 3, wn = warp >> 2;      // 4 x 2 warp grid over the 64x64 C tile
    typedef typename std::conditional<F32ACC, float, half>::type acc_t;
    wmma::fragment<wmma::accumulator, 16, 16, 16, acc_t> acc[2];
    wmma::fill_fragment(acc[0], (acc_t)0.0f);
    wmma::fill_fragment(acc[1], (acc_t)0.0f);

    const bool a_valid = srow < tile.nrows;       // srow doubles as the token index for A staging

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        __syncthreads();
        {   // W: dequant 8 values (4 code bytes) into sWA[0][srow][sseg*8..]
            float eff[POL::NEFF];
            POL::row_effs(slab, srow, anch, sub, eff);
            const float e = eff[(sseg*POL::NEFF) >> 2];
            const uint32_t qw = *(const uint32_t *)(slab + POL::CODE_OFF + srow*16 + sseg*4);
            half * o = &sWA[0][srow*PXQ6_WMMA_LD + sseg*8];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int byte = (qw >> (8*b)) & 0xff;
                o[2*b]   = __float2half_rn(e * tab[byte & 0xf]);
                o[2*b+1] = __float2half_rn(e * tab[byte >> 4]);
            }
        }
        {   // A: 8 halves per thread
            half * o = &sWA[1][srow*PXQ6_WMMA_LD + sseg*8];
            if (a_valid) {
                const uint4 v = *(const uint4 *)(At + (size_t)srow*K + kb*PXQ6_QK + sseg*8);
                *(uint4 *)o = v;
            } else {
                const half hz = __float2half_rn(0.f);
                #pragma unroll
                for (int i = 0; i < 8; ++i) o[i] = hz;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kf = 0; kf < 2; ++kf) {          // two k=16 fragments per 32-K slab
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
            wmma::load_matrix_sync(af, &sWA[0][(wm*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
            #pragma unroll
            for (int nf = 0; nf < 2; ++nf) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
                wmma::load_matrix_sync(bf, &sWA[1][(wn*32 + nf*16)*PXQ6_WMMA_LD + kf*16], PXQ6_WMMA_LD);
                wmma::mma_sync(acc[nf], af, bf, acc[nf]);
            }
        }
    }

    // epilogue: stage each 16x16 fragment through smem (bias add + ragged masking need row/col
    // coordinates; the Volta fragment register layout is opaque). Reuses the sWA smem after sync.
    __syncthreads();
    float * scratch = (float *)sWA;               // 8 warps x 16*17 floats = 8.5 KB < 10.2 KB
    float * mytile = scratch + warp*16*17;
    #pragma unroll
    for (int nf = 0; nf < 2; ++nf) {
        pxq6_wmma_stage_acc(mytile, acc[nf], warp, lane);
        __syncwarp();
        const int col0 = wn*32 + nf*16, row0 = wm*16;
        for (int i = lane; i < 256; i += 32) {
            const int rr = i & 15, t = i >> 4;
            if (col0 + t < tile.nrows) {
                const int row = row0 + rr;
                const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ6_BM + row)*sizeof(float)) : 0.f;
                Ct[(size_t)(col0 + t)*R + row] = mytile[t*17 + rr] + b;
            }
        }
        __syncwarp();
    }
#else
    // non-sm_70 compile: dead body (runtime dispatch never selects this kernel off-Volta)
    (void)W; (void)A; (void)C; (void)bias; (void)bias_nb1; (void)tiles; (void)R; (void)K;
#endif
}

#include "pxq23.cuh"   // PXQ2/PXQ3: tables, gates, pol_p2/pol_p3, dequant wrappers

// ---------------------------------------------------------------------------------------------
// host-side dispatch: format codes + kernel pickers (runtime env -> template instantiation).
// fmt: 3 = PXQ4 core, 4 = PXQ4-HQ, 5 = PXQ2, 6 = PXQ3, 7 = PXQ6 (display pxq6, 5-bit LM32);
// 0 = not a PXA slab type / gated off. (1 and 2 = the retired legacy id-250/id-251 formats,
// removed 2026-07-21 — values reserved, never reuse.)
// ---------------------------------------------------------------------------------------------
#define PXA_PXQ_FMT_NONE 0
// 1 reserved — retired PXA_PXQ_FMT_P4 (legacy type id 250), removed 2026-07-21
// 2 reserved — retired PXA_PXQ_FMT_P5 (legacy type id 251), removed 2026-07-21
#define PXA_PXQ_FMT_P6   3
#define PXA_PXQ_FMT_P6HQ 4
#define PXA_PXQ_FMT_P2   5      // ADD
#define PXA_PXQ_FMT_P3   6      // ADD
#define PXA_PXQ_FMT_P6R  7      // ADD: real-PXQ6 tier (TAB/TAB_CS modes only, like P2/P3)

static inline int pxa_pxq_fmt(ggml_type t) {
    switch (t) {
        case GGML_TYPE_PXQ4:   return pxa_pxq6_enabled() ? PXA_PXQ_FMT_P6   : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ4HQ: return pxa_pxq6_enabled() ? PXA_PXQ_FMT_P6HQ : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ2:   return pxa_pxq2_enabled() ? PXA_PXQ_FMT_P2   : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ3:   return pxa_pxq3_enabled() ? PXA_PXQ_FMT_P3   : PXA_PXQ_FMT_NONE;
        case GGML_TYPE_PXQ6:  return pxa_pxq6r_enabled() ? PXA_PXQ_FMT_P6R : PXA_PXQ_FMT_NONE;
        default:               return PXA_PXQ_FMT_NONE;
    }
}

typedef void (*pxq6_gateup_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        char *, size_t, size_t, const char *, size_t, size_t,
        const float *, size_t, const float *, size_t, int, int, int, int, float, float);
typedef void (*pxq6_mmv_fn)(const uint8_t *, const char *, size_t, size_t,
        char *, size_t, size_t, const char *, size_t, size_t, int, int, int);
typedef void (*pxq6_mmv_redfuse_fn)(const uint8_t *, const float *, char *, size_t, size_t,
        const char *, size_t, size_t, const float *, size_t, const float *, size_t,
        int, int, int, int, int, float, float);
typedef void (*pxq6_gateup_ks_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        float *, const char *, size_t, size_t, int, int, int, int);
typedef void (*pxq6_gateup_ksg_fn)(const uint8_t *, const uint8_t *, const char *, size_t,
        float *, const char *, size_t, size_t, int, int, int, int, int);
typedef void (*pxq6_gemm_fn)(const uint8_t *, const half *, float *, const float *, size_t,
        const pxq4_tile_info *, int, int);
typedef void (*pxq6_gufuse_h_fn)(const uint8_t *, const uint8_t *, const half *, half *,
        const float *, size_t, const float *, size_t, const pxq4_tile_info *, int, int, int, float, float);
typedef void (*pxq6_gufuse_f_fn)(const uint8_t *, const uint8_t *, const half *, float *,
        const float *, size_t, const float *, size_t, const pxq4_tile_info *, int, int, int, float, float);
typedef void (*pxq6_scat_fn)(const uint8_t *, const half *, char *, size_t, size_t,
        const pxq4_rowmap *, const pxq4_tile_info *, int, int);

#define PXQ6_PICK2(K, POL, b1, b2) \
    ((b1) ? ((b2) ? (K<POL, true, true>) : (K<POL, true, false>)) \
          : ((b2) ? (K<POL, false, true>) : (K<POL, false, false>)))
#define PXQ6_PICK_FMT(RET, NAME, K) \
    static inline RET NAME(int fmt, bool b1, bool b2) { \
        switch (fmt) { \
            case PXA_PXQ_FMT_P6: return PXQ6_PICK2(K, pxq6_pol_p6,   b1, b2); \
            case PXA_PXQ_FMT_P2: return PXQ6_PICK2(K, pxq6_pol_p2, false, b2); \
            case PXA_PXQ_FMT_P3: return PXQ6_PICK2(K, pxq6_pol_p3, false, b2); \
            case PXA_PXQ_FMT_P6R: return PXQ6_PICK2(K, pxq6_pol_p6r, false, b2); \
            default:             return PXQ6_PICK2(K, pxq6_pol_p6hq, b1, b2); \
        } \
    }

// decode-family pickers: runtime sourcing MODE (0 tab / 1 pairlut / 2 prmt / 3 tab+cs / 4 prmt+cs).
// P2/P3 (sub-nibble code packing) support only the tab paths — prmt/pairlut demote at compile time.
static inline int pxa_pxq6_decode_mode() {
    static const int m = [](){
        const bool cs = pxa_pxq6_ldcs();
        if (pxa_pxq6_prmt())    return cs ? PXQ6_MODE_PRMT_CS : PXQ6_MODE_PRMT;
        if (pxa_pxq6_pairlut()) return PXQ6_MODE_PAIRL;   // pairlut+cs not offered (lever is OFF-verdict)
        return cs ? PXQ6_MODE_TAB_CS : PXQ6_MODE_TAB;
    }();
    return m;
}
#define PXQ6_PICKM(K, POL, m, vx) \
    ((m) == PXQ6_MODE_PAIRL   ? ((vx) ? (K<POL, PXQ6_MODE_PAIRL,   true>) : (K<POL, PXQ6_MODE_PAIRL,   false>)) : \
     (m) == PXQ6_MODE_PRMT    ? ((vx) ? (K<POL, PXQ6_MODE_PRMT,    true>) : (K<POL, PXQ6_MODE_PRMT,    false>)) : \
     (m) == PXQ6_MODE_TAB_CS  ? ((vx) ? (K<POL, PXQ6_MODE_TAB_CS,  true>) : (K<POL, PXQ6_MODE_TAB_CS,  false>)) : \
     (m) == PXQ6_MODE_PRMT_CS ? ((vx) ? (K<POL, PXQ6_MODE_PRMT_CS, true>) : (K<POL, PXQ6_MODE_PRMT_CS, false>)) : \
                                ((vx) ? (K<POL, PXQ6_MODE_TAB,     true>) : (K<POL, PXQ6_MODE_TAB,     false>)))
#define PXQ6_PICKM23(K, POL, m, vx) /* P2/P3: tab paths only */ \
    (((m) == PXQ6_MODE_TAB_CS || (m) == PXQ6_MODE_PRMT_CS) \
        ? ((vx) ? (K<POL, PXQ6_MODE_TAB_CS, true>) : (K<POL, PXQ6_MODE_TAB_CS, false>)) \
        : ((vx) ? (K<POL, PXQ6_MODE_TAB,    true>) : (K<POL, PXQ6_MODE_TAB,    false>)))
#define PXQ6_PICKM_FMT(RET, NAME, K) \
    static inline RET NAME(int fmt, int m, bool vx) { \
        switch (fmt) { \
            case PXA_PXQ_FMT_P6: return PXQ6_PICKM(K, pxq6_pol_p6,   m, vx); \
            case PXA_PXQ_FMT_P2: return PXQ6_PICKM23(K, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P3: return PXQ6_PICKM23(K, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P6R: return PXQ6_PICKM23(K, pxq6_pol_p6r, m, vx); \
            default:             return PXQ6_PICKM(K, pxq6_pol_p6hq, m, vx); \
        } \
    }

#define PXQ6_PICKMU(K, PU, PG, m, vx) \
    ((m) == PXQ6_MODE_PAIRL   ? ((vx) ? (K<PU, PG, PXQ6_MODE_PAIRL,   true>) : (K<PU, PG, PXQ6_MODE_PAIRL,   false>)) : \
     (m) == PXQ6_MODE_PRMT    ? ((vx) ? (K<PU, PG, PXQ6_MODE_PRMT,    true>) : (K<PU, PG, PXQ6_MODE_PRMT,    false>)) : \
     (m) == PXQ6_MODE_TAB_CS  ? ((vx) ? (K<PU, PG, PXQ6_MODE_TAB_CS,  true>) : (K<PU, PG, PXQ6_MODE_TAB_CS,  false>)) : \
     (m) == PXQ6_MODE_PRMT_CS ? ((vx) ? (K<PU, PG, PXQ6_MODE_PRMT_CS, true>) : (K<PU, PG, PXQ6_MODE_PRMT_CS, false>)) : \
                                ((vx) ? (K<PU, PG, PXQ6_MODE_TAB,     true>) : (K<PU, PG, PXQ6_MODE_TAB,     false>)))
#define PXQ6_PICKMU23(K, PU, PG, m, vx) /* any P2/P3 operand: tab paths only */ \
    (((m) == PXQ6_MODE_TAB_CS || (m) == PXQ6_MODE_PRMT_CS) \
        ? ((vx) ? (K<PU, PG, PXQ6_MODE_TAB_CS, true>) : (K<PU, PG, PXQ6_MODE_TAB_CS, false>)) \
        : ((vx) ? (K<PU, PG, PXQ6_MODE_TAB,    true>) : (K<PU, PG, PXQ6_MODE_TAB,    false>)))
#define PXQ6_PICKM_FMT_GU(RET, NAME, K) \
    static inline RET NAME(int fu, int fg, int m, bool vx) { \
        if (fu == fg) switch (fu) { \
            case PXA_PXQ_FMT_P6: return PXQ6_PICKMU(K, pxq6_pol_p6, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P3: return PXQ6_PICKMU23(K, pxq6_pol_p3, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P6R: return PXQ6_PICKMU23(K, pxq6_pol_p6r, pxq6_pol_p6r, m, vx); \
            default:             return PXQ6_PICKMU(K, pxq6_pol_p6hq, pxq6_pol_p6hq, m, vx); \
        } \
        switch (fu*8 + fg) {   /* mixed pairs: universal files only (P2/P3/P6), tab modes only */ \
            case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P3: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p3, m, vx); \
            case PXA_PXQ_FMT_P2*8+PXA_PXQ_FMT_P6: return PXQ6_PICKMU23(K, pxq6_pol_p2, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p3, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P3*8+PXA_PXQ_FMT_P6: return PXQ6_PICKMU23(K, pxq6_pol_p3, pxq6_pol_p6, m, vx); \
            case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P2: return PXQ6_PICKMU23(K, pxq6_pol_p6, pxq6_pol_p2, m, vx); \
            case PXA_PXQ_FMT_P6*8+PXA_PXQ_FMT_P3: return PXQ6_PICKMU23(K, pxq6_pol_p6, pxq6_pol_p3, m, vx); \
            default: return nullptr;   /* unsupported mix -> driver declines (fallback) */ \
        } \
    }
PXQ6_PICKM_FMT_GU(pxq6_gateup_fn,     pxq6_pick_gateup,            k_pxq6_gateup_mmv)
PXQ6_PICKM_FMT(pxq6_mmv_fn,        pxq6_pick_mmv,               k_pxq6_mmv)
PXQ6_PICKM_FMT(pxq6_mmv_redfuse_fn, pxq6_pick_mmv_redfuse,      k_pxq6_mmv_redfuse)
PXQ6_PICKM_FMT_GU(pxq6_gateup_ks_fn,  pxq6_pick_gateup_ksplit,     k_pxq6_gateup_mmv_ksplit)
PXQ6_PICKM_FMT_GU(pxq6_gateup_ksg_fn, pxq6_pick_gateup_ksplit_gen, k_pxq6_gateup_mmv_ksplit_gen)
PXQ6_PICK_FMT(pxq6_gemm_fn,       pxq6_pick_gemm,              k_pxq6_gemm_grouped)
PXQ6_PICK_FMT(pxq6_scat_fn,       pxq6_pick_down_scat,         k_pxq6_gemm_down_scat)

#define PXQ6_PICK2T(K, POL, T, b1, b2) \
    ((b1) ? ((b2) ? (K<POL, T, true, true>) : (K<POL, T, true, false>)) \
          : ((b2) ? (K<POL, T, false, true>) : (K<POL, T, false, false>)))
static inline pxq6_gufuse_h_fn pxq6_pick_gufuse_h(int fmt, bool rag, bool pipe) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6,   half, rag, pipe);
        case PXA_PXQ_FMT_P2: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p2,   half, rag, pipe);
        case PXA_PXQ_FMT_P3: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p3,   half, rag, pipe);
        case PXA_PXQ_FMT_P6R: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6r, half, rag, pipe);
        default:             return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6hq, half, rag, pipe);
    }
}
static inline pxq6_gufuse_f_fn pxq6_pick_gufuse_f(int fmt, bool rag, bool pipe) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6,   float, rag, pipe);
        case PXA_PXQ_FMT_P2: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p2,   float, rag, pipe);
        case PXA_PXQ_FMT_P3: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p3,   float, rag, pipe);
        case PXA_PXQ_FMT_P6R: return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6r, float, rag, pipe);
        default:             return PXQ6_PICK2T(k_pxq6_gemm_gufuse, pxq6_pol_p6hq, float, rag, pipe);
    }
}
static inline pxq6_gemm_fn pxq6_pick_gemm_wmma(int fmt, bool f32acc) {
    switch (fmt) {
        case PXA_PXQ_FMT_P6: return f32acc ? (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6,   true> : (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6,   false>;
        default:             return f32acc ? (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6hq, true> : (pxq6_gemm_fn)k_pxq6_gemm_grouped_wmma<pxq6_pol_p6hq, false>;
    }
}
