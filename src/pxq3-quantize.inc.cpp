// pxq3-quantize.inc.cpp — PXQ3 native quantizer (3-bit LM8 codes, BIT-PLANE packed, x
// E16-row scales; spec PXQ-UNIVERSAL-2026-07-17.md). Cloned from pxq6-quantize.inc.cpp.
//
// SELF-CONTAINED functions; spliced into src/llama-quantize.cpp next to the PXQ2 block, and
// compiled standalone by pxa-bench/pxqu_ref.cpp (the correctness / wrel-reproduction tool).
//
// FORMAT (frozen):
//   values : 3-bit codes into the frozen LM8 book (8 entries, NO zero entry, absmax != 1)
//   scales : per-ROW fp16 anchor (128 B header per 64-row panel) x one 4-bit sub-scale per
//            16-elem block through the frozen PXQ6 SUB16 table (= PXQ6 core; scale SoA 64 B)
//   dequant: eff = fp32(anchor) * SUB16[s4]; w = eff * fp32(book[c])   (fp32 muls, parity-locked)
//   layout : slab = 64 B scale SoA + 64 x 12 B code rows; a code row = three LE uint32 words:
//              w0 = LOW plane elems  0..15 (2 bits/elem, elem j at bits 2j)
//              w1 = LOW plane elems 16..31
//              w2 = HIGH plane (bit j = elem j's bit2, j = 0..31)
//            (bit-plane packing is LOCKED by the spec: fp16-LUT and int8-LUT extraction both
//            stay branch-free; every word 4-aligned for every row.)
//            panel = 128 B anchor header + kslabs x 832 B slabs; panels row-major; experts outer.
//
// QUANTIZE ALGORITHM — identical to PXQ6/PXQ2 (= pxqu_lab.py quant_cands, E16 scheme):
//   anchor = fp16_rn(row absmax); FULL 16-cand sub search per block; RTN codes against the
//   sorted book by the midpoint rule; double-accum imatrix-weighted SSE argmin.
//   zero rows: s4=0 / codes=PXQ3_ZIDX -> recon exactly 0 (eff==0) == numpy oracle. Zero blocks
//   inside a nonzero row: same convention; recon != 0 (LM8 has no zero entry) — the one
//   documented oracle divergence, unreachable on real weights (see pxq2-quantize.inc.cpp).
//
// GATES: byte-parity vs the numpy reference; wrel reproduces the lab 0.1435315 per
// pxa-bench/pxqu_wrel.py (±1e-4, snapped-SUB16-adjusted oracle).
//
// Env:
//   PXA_PXQ3_ANCHOR_FIT=1        widened row-anchor search (default OFF = lab recipe).
//   PXA_PXQ3_BOOK / PXA_PXQ3_SUB comma-separated overrides (8 / 16 floats, fp16-snapped).

#include "../ggml/include/ggml-pxq3-tables.h"
#include "../ggml/include/ggml-pxq6-tables.h"   // PXQ6_SUB16_INIT (frozen, shared verbatim)

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>

static const float pxq3_book_q_[8]   = PXQ3_BOOK_INIT;
static const float pxq3_sub16_q_[16] = PXQ6_SUB16_INIT;

static inline bool pxq3_parse_n(const char * e, float * out, int want) {
    int n = 0; float v[16];
    char * dup = strdup(e);
    for (char * t = strtok(dup, ","); t && n < want; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
    free(dup);
    if (n != want) return false;
    for (int i = 0; i < want; ++i) out[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[i]));  // fp16-snap
    return true;
}

static inline const float * pxq3_book_q() {
    static float book[8];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(book, pxq3_book_q_, sizeof(book));
        if (const char * e = getenv("PXA_PXQ3_BOOK")) {
            if (pxq3_parse_n(e, book, 8)) fprintf(stderr, "PXQ3 quantize: custom codebook from PXA_PXQ3_BOOK\n");
            else fprintf(stderr, "PXA_PXQ3_BOOK: expected 8 floats — IGNORED\n");
        }
    }
    return book;
}

static inline const float * pxq3_sub_q() {
    static float sub[16];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(sub, pxq3_sub16_q_, sizeof(sub));
        if (const char * e = getenv("PXA_PXQ3_SUB")) {
            if (pxq3_parse_n(e, sub, 16)) fprintf(stderr, "PXQ3 quantize: custom SUB16 from PXA_PXQ3_SUB\n");
        }
    }
    return sub;
}

static inline const double * pxq3_mids_q() {
    static double mids[7];
    static bool init = false;
    if (!init) {
        init = true;
        const float * b = pxq3_book_q();
        for (int i = 0; i < 7; ++i) mids[i] = ((double)b[i] + (double)b[i+1]) * 0.5;
    }
    return mids;
}

static inline int pxq3_code(double xn, const double * mids) {
    int c = 0;
    for (int i = 0; i < 7; ++i) c += xn > mids[i];   // == numpy searchsorted(mids, xn, 'left')
    return c;
}

static inline double pxq3_block_err(const float * x, const float * w, float d,
                                    const float * book, const double * mids, uint8_t * codes) {
    const double d64 = (double)d > 1e-30 ? (double)d : 1e-30;
    double err = 0.0;
    for (int i = 0; i < 16; ++i) {
        const int c = pxq3_code((double)x[i] / d64, mids);
        codes[i] = (uint8_t)c;
        const float rec = d * book[c];                    // fp32 product == kernel math
        const double e = (double)x[i] - (double)rec;
        err += (w ? (double)w[i] : 1.0) * e * e;
    }
    return err;
}

static inline double pxq3_quant_subblock(const float * x, const float * w, float anchor,
                                         const float * book, const double * mids, const float * sub,
                                         uint8_t * s4_out, uint8_t * codes_out) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f) || !(anchor > 0.f)) {
        *s4_out = 0;
        for (int i = 0; i < 16; ++i) codes_out[i] = PXQ3_ZIDX;   // min-|book| (no exact zero in LM8)
        return 0.0;
    }
    double best = 1e300;
    uint8_t codes[16];
    for (int j = 0; j < 16; ++j) {
        const float d = (float)((double)anchor * (double)sub[j]);
        const double err = pxq3_block_err(x, w, d, book, mids, codes);
        if (err < best) {
            best = err;
            *s4_out = (uint8_t)j;
            memcpy(codes_out, codes, 16);
        }
    }
    return best;
}

static double pxq3_quant_row(const float * x, const float * w, int64_t K, float anchor,
                             const float * book, const double * mids, const float * sub,
                             uint8_t * s4_flat /*K/16*/, uint8_t * codes_flat /*K*/) {
    double err = 0.0;
    for (int64_t b = 0; b < K/16; ++b) {
        err += pxq3_quant_subblock(x + b*16, w ? w + b*16 : nullptr, anchor,
                                   book, mids, sub, &s4_flat[b], &codes_flat[b*16]);
    }
    return err;
}

static inline bool pxq3_anchor_fit_enabled() {
    static const bool on = [](){ const char * e = getenv("PXA_PXQ3_ANCHOR_FIT"); return e && atoi(e) != 0; }();
    return on;
}

static float pxq3_pick_anchor(const float * x, const float * w, int64_t K,
                              const float * book, const double * mids, const float * sub,
                              uint8_t * s4_tmp, uint8_t * codes_tmp) {
    float amax = 0.f;
    for (int64_t i = 0; i < K; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax > 65504.f) amax = 65504.f;
    const float a0 = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
    if (!pxq3_anchor_fit_enabled() || !(a0 > 0.f)) return a0;
    float best_a = a0; double best_e = -1.0;
    float prev[5]; int np = 0;
    for (int k = -2; k <= 2; ++k) {
        const float cand = ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax * (float)exp2(k / 16.0)));
        if (!(cand > 0.f)) continue;
        bool dup = false;
        for (int i = 0; i < np; ++i) if (prev[i] == cand) { dup = true; break; }
        if (dup) continue;
        prev[np++] = cand;
        const double e = pxq3_quant_row(x, w, K, cand, book, mids, sub, s4_tmp, codes_tmp);
        if (best_e < 0.0 || e < best_e) { best_e = e; best_a = cand; }
    }
    return best_a;
}

// bit-plane pack/unpack for one 32-elem row-block (see header comment for the layout).
static inline void pxq3_pack32(const uint8_t * c /*32 codes 0-7*/, uint8_t * out /*12 B*/) {
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    for (int j = 0; j < 16; ++j) {
        w0 |= (uint32_t)(c[j]      & 3) << (2*j);
        w1 |= (uint32_t)(c[16 + j] & 3) << (2*j);
        w2 |= (uint32_t)(c[j]      >> 2) << j;
        w2 |= (uint32_t)(c[16 + j] >> 2) << (16 + j);
    }
    // LE byte emission (deterministic on any host endianness)
    for (int i = 0; i < 4; ++i) out[i]     = (uint8_t)(w0 >> (8*i));
    for (int i = 0; i < 4; ++i) out[4 + i] = (uint8_t)(w1 >> (8*i));
    for (int i = 0; i < 4; ++i) out[8 + i] = (uint8_t)(w2 >> (8*i));
}
static inline void pxq3_unpack32(const uint8_t * in /*12 B*/, uint8_t * c /*32 codes*/) {
    uint32_t w0 = 0, w1 = 0, w2 = 0;
    for (int i = 0; i < 4; ++i) { w0 |= (uint32_t)in[i] << (8*i); w1 |= (uint32_t)in[4+i] << (8*i); w2 |= (uint32_t)in[8+i] << (8*i); }
    for (int j = 0; j < 16; ++j) {
        c[j]      = (uint8_t)(((w0 >> (2*j)) & 3) | (((w2 >> j) & 1) << 2));
        c[16 + j] = (uint8_t)(((w1 >> (2*j)) & 3) | (((w2 >> (16 + j)) & 1) << 2));
    }
}

// one [R,K] expert -> PXQ3 panels (bs16 subs, slab 832).
static void pxq3_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                 const float * imx /*K vals or null*/) {
    const float  * book = pxq3_book_q();
    const double * mids = pxq3_mids_q();
    const float  * sub  = pxq3_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ3_HDR_BYTES + KB*PXQ3_SLAB_BYTES;
    const int64_t nsub = K / 16;

    std::vector<uint8_t> s4(nsub), codes(K);
    for (int64_t p = 0; p < P; ++p) {
        uint8_t * panel = dst + p*panel_bytes;
        ggml_fp16_t * anchors = (ggml_fp16_t *)panel;      // 64 x fp16 header
        for (int64_t r = 0; r < 64; ++r) {
            const float * x = src + (p*64 + r)*K;
            const float anchor = pxq3_pick_anchor(x, imx, K, book, mids, sub, s4.data(), codes.data());
            anchors[r] = ggml_fp32_to_fp16(anchor);
            pxq3_quant_row(x, imx, K, anchor, book, mids, sub, s4.data(), codes.data());
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ3_HDR_BYTES + kb*PXQ3_SLAB_BYTES;
                const uint8_t * s = &s4[kb*2];             // 2 subs per 32 elems: lo nibble = elems 0-15
                slab[r] = (uint8_t)(s[0] | (s[1] << 4));
                pxq3_pack32(&codes[kb*32], slab + 64 + r*12);
            }
        }
    }
}

// reference dequant (CPU) — the parity-locked contract. Used by pxqu_ref + golden tests.
static void pxq3_dequant_expert(const uint8_t * src, float * dst, int64_t R, int64_t K) {
    const float * book = pxq3_book_q();
    const float * sub  = pxq3_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ3_HDR_BYTES + KB*PXQ3_SLAB_BYTES;
    uint8_t c[32];
    for (int64_t p = 0; p < P; ++p) {
        const uint8_t * panel = src + p*panel_bytes;
        const ggml_fp16_t * anchors = (const ggml_fp16_t *)panel;
        for (int64_t r = 0; r < 64; ++r) {
            const float anchor = ggml_fp16_to_fp32(anchors[r]);
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t * slab = panel + PXQ3_HDR_BYTES + kb*PXQ3_SLAB_BYTES;
                float eff[2];
                eff[0] = anchor * sub[slab[r] & 0xf];      // elems 0-15
                eff[1] = anchor * sub[slab[r] >> 4];       // elems 16-31
                pxq3_unpack32(slab + 64 + r*12, c);
                float * o = dst + (p*64 + r)*K + kb*32;
                for (int j = 0; j < 32; ++j) o[j] = eff[j >> 4] * book[c[j]];
            }
        }
    }
}

// full 3D expert tensor, threaded across experts (same pattern as pxq6_quantize_tensor).
static void pxq3_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                 const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = (R/64)*(PXQ3_HDR_BYTES + (K/32)*(int64_t)PXQ3_SLAB_BYTES);
    auto imx_for = [&](int64_t e) -> const float * {
        if (!imx) return nullptr;
        if (imx_size == K*E) return imx + e*K;
        if (imx_size == K)   return imx;
        return nullptr;
    };
    (void)pxq3_book_q(); (void)pxq3_sub_q(); (void)pxq3_mids_q();   // init tables before threading
    if (nthread <= 1 || E <= 1) {
        for (int64_t e = 0; e < E; ++e) {
            pxq3_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
        return;
    }
    std::atomic<int64_t> counter{0};
    auto compute = [&]() {
        while (true) {
            const int64_t e = counter.fetch_add(1);
            if (e >= E) break;
            pxq3_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
    };
    std::vector<std::thread> th;
    const int n = (int) std::min<int64_t>(nthread, E);
    th.reserve(n);
    for (int i = 0; i < n; ++i) th.emplace_back(compute);
    for (auto & t : th) t.join();
}
