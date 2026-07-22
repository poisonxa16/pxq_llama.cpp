// pxq2-quantize.inc.cpp — PXQ2 native quantizer (2-bit LM4 codes x E16-row scales;
// spec PXQ-UNIVERSAL-2026-07-17.md). Cloned from pxq6-quantize.inc.cpp (core tier only).
//
// SELF-CONTAINED functions; spliced into src/llama-quantize.cpp next to the PXQ6 block, and
// compiled standalone by pxa-bench/pxqu_ref.cpp (the correctness / wrel-reproduction tool).
//
// FORMAT (frozen):
//   values : 2-bit codes into the frozen LM4 book (4 entries, NO zero entry, absmax != 1)
//   scales : per-ROW fp16 anchor (64 anchors = 128 B header at the head of each 64-row panel)
//            x one 4-bit sub-scale per 16-elem block through the frozen PXQ6 SUB16 table
//            (two nibbles pack into the one scale byte per 32 elems -> slab scale SoA = 64 B,
//            IDENTICAL to PXQ6 core)
//   dequant: eff = fp32(anchor) * SUB16[s4]; w = eff * fp32(book[c])   (fp32 muls, parity-locked)
//   layout : slab = 64 B scale SoA + 64 x 8 B code rows; a code row = two LE uint32 words,
//            word h = elems 16h..16h+15, elem j at bits 2*(j&15) (4 codes/byte, low bits first)
//            panel = 128 B anchor header + kslabs x 576 B slabs; panels row-major; experts outer.
//
// QUANTIZE ALGORITHM — identical to PXQ6 (= pxqu_lab.py quant_cands, E16 scheme):
//   per row   : anchor = fp16_rn(row absmax)  [optional weighted-MSE anchor fit, env-gated]
//   per block : FULL search over all 16 sub-levels; codes = RTN against the sorted book
//               (midpoint rule = numpy searchsorted 'left'); keep argmin sum w_i (x_i - w_hat_i)^2
//               (double accumulation, imatrix column weights when present, per-expert slices).
//   zero rows (anchor snaps to 0): s4=0 / codes=PXQ2_ZIDX -> recon exactly 0 (eff==0), matching
//   the numpy oracle bit-exactly. Zero BLOCKS inside a nonzero row: s4=0 / codes=PXQ2_ZIDX;
//   ⚠ recon = anchor*SUB16[0]*book[ZIDX] != 0 here whereas the numpy oracle forces d=0 (the LM4
//   book has no zero entry, so an exact-zero recon is unrepresentable). This is the ONLY
//   quantizer/oracle divergence; all-zero 16-blocks inside nonzero rows do not occur in real
//   bf16 expert weights (verified none on Ornith-35B), and pxa-bench/pxqu_wrel.py measures the
//   gate with the same convention on both sides.
//
// GATES (Q-G1/Q-G2, see PXQ-UNIVERSAL-2026-07-17.md B1): byte-parity vs the numpy reference;
// wrel on the frozen 36-slice rng-42 protocol reproduces the lab number (0.3020488) per
// pxa-bench/pxqu_wrel.py (±1e-4, snapped-SUB16-adjusted oracle).
//
// Env:
//   PXA_PXQ2_ANCHOR_FIT=1  widen the row-anchor search (5 fp16 candidates around absmax,
//                          weighted-MSE argmin over the full row; default OFF = lab recipe).
//   PXA_PXQ2_BOOK / PXA_PXQ2_SUB  comma-separated table overrides (4 / 16 floats,
//                          fp16-snapped; MUST match the runtime kernels' tables).

#include "../ggml/include/ggml-pxq2-tables.h"
#include "../ggml/include/ggml-pxq6-tables.h"   // PXQ6_SUB16_INIT (frozen, shared verbatim)

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>

static const float pxq2_book_q_[4]   = PXQ2_BOOK_INIT;
static const float pxq2_sub16_q_[16] = PXQ6_SUB16_INIT;

#ifndef PXQ_TIE_BREAK_DEFINED
#define PXQ_TIE_BREAK_DEFINED
// deterministic tie-break for reproducible quantization (shared by all PXQ-family quantizers;
// applies ONLY when two candidates carry bit-identical weighted error — quality-neutral)
static inline bool pxq_tie_take_hi(int64_t row, int64_t blk) {
    uint64_t v = ((uint64_t)row ^ (uint64_t)blk) ^ 0x5Aull;
    v ^= v >> 32; v ^= v >> 16; v ^= v >> 8; v ^= v >> 4; v ^= v >> 2; v ^= v >> 1;
    return (v & 1) != 0;
}
#endif

static inline bool pxq2_parse_n(const char * e, float * out, int want) {
    int n = 0; float v[16];
    char * dup = strdup(e);
    for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
    free(dup);
    if (n != want) return false;
    for (int i = 0; i < want; ++i) out[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[i]));  // fp16-snap (spec)
    return true;
}

static inline const float * pxq2_book_q() {
    static float book[4];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(book, pxq2_book_q_, sizeof(book));
        if (const char * e = getenv("PXA_PXQ2_BOOK")) {
            if (pxq2_parse_n(e, book, 4)) fprintf(stderr, "PXQ2 quantize: custom codebook from PXA_PXQ2_BOOK\n");
            else fprintf(stderr, "PXA_PXQ2_BOOK: expected 4 floats — IGNORED\n");
        }
    }
    return book;
}

static inline const float * pxq2_sub_q() {
    static float sub[16];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(sub, pxq2_sub16_q_, sizeof(sub));
        if (const char * e = getenv("PXA_PXQ2_SUB")) {
            if (pxq2_parse_n(e, sub, 16)) fprintf(stderr, "PXQ2 quantize: custom SUB16 from PXA_PXQ2_SUB\n");
        }
    }
    return sub;
}

static inline const double * pxq2_mids_q() {
    static double mids[3];
    static bool init = false;
    if (!init) {
        init = true;
        const float * b = pxq2_book_q();
        for (int i = 0; i < 3; ++i) mids[i] = ((double)b[i] + (double)b[i+1]) * 0.5;
    }
    return mids;
}

static inline int pxq2_code(double xn, const double * mids) {
    int c = 0;
    for (int i = 0; i < 3; ++i) c += xn > mids[i];   // == numpy searchsorted(mids, xn, 'left')
    return c;
}

// quantize one 16-elem sub-block against a fixed effective scale d (fp32), double-accum weighted err.
static inline double pxq2_block_err(const float * x, const float * w, float d,
                                    const float * book, const double * mids, uint8_t * codes) {
    const double d64 = (double)d > 1e-30 ? (double)d : 1e-30;
    double err = 0.0;
    for (int i = 0; i < 16; ++i) {
        const int c = pxq2_code((double)x[i] / d64, mids);
        codes[i] = (uint8_t)c;
        const float rec = d * book[c];                    // fp32 product == kernel math
        const double e = (double)x[i] - (double)rec;
        err += (w ? (double)w[i] : 1.0) * e * e;
    }
    return err;
}

// pick the best 4-bit sub for one 16-elem block under a fixed row anchor (FULL 16-cand search).
static inline double pxq2_quant_subblock(const float * x, const float * w, float anchor,
                                         const float * book, const double * mids, const float * sub,
                                         uint8_t * s4_out, uint8_t * codes_out, int64_t row, int64_t blk) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f) || !(anchor > 0.f)) {
        *s4_out = 0;
        for (int i = 0; i < 16; ++i) codes_out[i] = PXQ2_ZIDX;   // min-|book| (no exact zero in LM4)
        return 0.0;
    }
    double best = 1e300;
    uint8_t codes[16];
    for (int j = 0; j < 16; ++j) {
        const float d = (float)((double)anchor * (double)sub[j]);   // fp64 product, single fp32 round
        const double err = pxq2_block_err(x, w, d, book, mids, codes);
        if (err < best || (err == best && pxq_tie_take_hi(row, blk))) {   // deterministic tie-break for reproducible quantization
            best = err;
            *s4_out = (uint8_t)j;
            memcpy(codes_out, codes, 16);
        }
    }
    return best;
}

static double pxq2_quant_row(const float * x, const float * w, int64_t K, float anchor,
                             const float * book, const double * mids, const float * sub,
                             uint8_t * s4_flat /*K/16*/, uint8_t * codes_flat /*K*/, int64_t row) {
    double err = 0.0;
    for (int64_t b = 0; b < K/16; ++b) {
        err += pxq2_quant_subblock(x + b*16, w ? w + b*16 : nullptr, anchor,
                                   book, mids, sub, &s4_flat[b], &codes_flat[b*16], row, b);
    }
    return err;
}

static inline bool pxq2_anchor_fit_enabled() {
    static const bool on = [](){ const char * e = getenv("PXA_PXQ2_ANCHOR_FIT"); return e && atoi(e) != 0; }();
    return on;
}

static float pxq2_pick_anchor(const float * x, const float * w, int64_t K,
                              const float * book, const double * mids, const float * sub,
                              uint8_t * s4_tmp, uint8_t * codes_tmp, int64_t row) {
    float amax = 0.f;
    for (int64_t i = 0; i < K; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax > 65504.f) amax = 65504.f;                    // fp16 ceiling (hardening)
    const float a0 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
    if (!pxq2_anchor_fit_enabled() || !(a0 > 0.f)) return a0;
    float best_a = a0; double best_e = -1.0;
    float prev[5]; int np = 0;
    for (int k = -2; k <= 2; ++k) {
        const float cand = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax * (float)exp2(k / 16.0)));
        if (!(cand > 0.f)) continue;
        bool dup = false;
        for (int i = 0; i < np; ++i) if (prev[i] == cand) { dup = true; break; }
        if (dup) continue;
        prev[np++] = cand;
        const double e = pxq2_quant_row(x, w, K, cand, book, mids, sub, s4_tmp, codes_tmp, row);
        if (best_e < 0.0 || e < best_e) { best_e = e; best_a = cand; }
    }
    return best_a;
}

// one [R,K] expert -> PXQ2 panels (bs16 subs, slab 576).
static void pxq2_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                 const float * imx /*K vals or null*/) {
    const float  * book = pxq2_book_q();
    const double * mids = pxq2_mids_q();
    const float  * sub  = pxq2_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ2_HDR_BYTES + KB*PXQ2_SLAB_BYTES;
    const int64_t nsub = K / 16;

    std::vector<uint8_t> s4(nsub), codes(K);
    for (int64_t p = 0; p < P; ++p) {
        uint8_t * panel = dst + p*panel_bytes;
        ggml_fp16_t * anchors = (ggml_fp16_t *)panel;      // 64 x fp16 header
        for (int64_t r = 0; r < 64; ++r) {
            const float * x = src + (p*64 + r)*K;
            const int64_t row = p*64 + r;
            const float anchor = pxq2_pick_anchor(x, imx, K, book, mids, sub, s4.data(), codes.data(), row);
            anchors[r] = ggml_fp32_to_fp16(anchor);
            pxq2_quant_row(x, imx, K, anchor, book, mids, sub, s4.data(), codes.data(), row);
            // scatter into slabs
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ2_HDR_BYTES + kb*PXQ2_SLAB_BYTES;
                const uint8_t * s = &s4[kb*2];             // 2 subs per 32 elems: lo nibble = elems 0-15
                slab[r] = (uint8_t)(s[0] | (s[1] << 4));
                uint8_t * out = slab + 64 + r*8;           // 8 code bytes / row
                const uint8_t * c = &codes[kb*32];
                for (int by = 0; by < 8; ++by) {           // 4 codes/byte, low bits first (LE words)
                    out[by] = (uint8_t)(c[4*by] | (c[4*by+1] << 2) | (c[4*by+2] << 4) | (c[4*by+3] << 6));
                }
            }
        }
    }
}

// reference dequant (CPU) — the parity-locked contract. Used by pxqu_ref + golden tests.
static void pxq2_dequant_expert(const uint8_t * src, float * dst, int64_t R, int64_t K) {
    const float * book = pxq2_book_q();
    const float * sub  = pxq2_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ2_HDR_BYTES + KB*PXQ2_SLAB_BYTES;
    for (int64_t p = 0; p < P; ++p) {
        const uint8_t * panel = src + p*panel_bytes;
        const ggml_fp16_t * anchors = (const ggml_fp16_t *)panel;
        for (int64_t r = 0; r < 64; ++r) {
            const float anchor = ggml_fp16_to_fp32(anchors[r]);
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t * slab = panel + PXQ2_HDR_BYTES + kb*PXQ2_SLAB_BYTES;
                float eff[2];
                eff[0] = anchor * sub[slab[r] & 0xf];      // elems 0-15
                eff[1] = anchor * sub[slab[r] >> 4];       // elems 16-31
                const uint8_t * q = slab + 64 + r*8;
                float * o = dst + (p*64 + r)*K + kb*32;
                for (int j = 0; j < 32; ++j) {
                    const int c = (q[j >> 2] >> (2*(j & 3))) & 3;
                    o[j] = eff[j >> 4] * book[c];
                }
            }
        }
    }
}

// full 3D expert tensor, threaded across experts (same pattern as pxq6_quantize_tensor).
// imatrix semantics: imx_size == K*E -> per-expert columns; == K -> shared; else ignored.
static void pxq2_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                 const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = (R/64)*(PXQ2_HDR_BYTES + (K/32)*(int64_t)PXQ2_SLAB_BYTES);
    auto imx_for = [&](int64_t e) -> const float * {
        if (!imx) return nullptr;
        if (imx_size == K*E) return imx + e*K;
        if (imx_size == K)   return imx;
        return nullptr;
    };
    (void)pxq2_book_q(); (void)pxq2_sub_q(); (void)pxq2_mids_q();   // init tables before threading
    if (nthread <= 1 || E <= 1) {
        for (int64_t e = 0; e < E; ++e) {
            pxq2_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
        return;
    }
    std::atomic<int64_t> counter{0};
    auto compute = [&]() {
        while (true) {
            const int64_t e = counter.fetch_add(1);
            if (e >= E) break;
            pxq2_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
    };
    std::vector<std::thread> th;
    const int n = (int) std::min<int64_t>(nthread, E);
    th.reserve(n);
    for (int i = 0; i < n; ++i) th.emplace_back(compute);
    for (auto & t : th) t.join();
}
