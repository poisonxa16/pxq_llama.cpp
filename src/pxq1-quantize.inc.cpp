// pxq1-quantize.inc.cpp — PXQ1 native quantizer (1-bit sign codes x E16-row scales).
// Cloned from pxq2-quantize.inc.cpp; the only differences are book size (2 vs 4), the
// 1-bit code packing (8/byte -> 4 B/row-block), and mids = {0}. Self-contained; spliced
// into src/llama-quantize.cpp next to the PXQ2/PXQ3 block, and compiled standalone by
// pxa-bench/pxq1_ref.cpp for the byte-parity gate.
//
// FORMAT: values 1-bit sign into the frozen 2-level book {-1,+1}; scales = per-ROW fp16
// anchor (128 B header/64-row panel) x one 4-bit SUB16 sub-scale per 16-elem block (two
// nibbles per scale byte, 64 B scale SoA per slab). dequant: eff = fp32(anchor)*SUB16[s4];
// w = eff * book[sign]  (fp32 muls, parity-locked). Code row = one LE uint32 (4 B), bit j
// = elem j sign. Zero blocks: s4=0 / code=PXQ1_ZIDX (recon != 0; symmetric book has no zero).

#include "../ggml/include/ggml-pxq1-tables.h"
#include "../ggml/include/ggml-pxq6-tables.h"   // PXQ6_SUB16_INIT (frozen, shared verbatim)

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>

static const float pxq1_book_q_[2]   = PXQ1_BOOK_INIT;
static const float pxq1_sub16_q_[16] = PXQ6_SUB16_INIT;

static inline const float * pxq1_book_q() { return pxq1_book_q_; }
static inline const float * pxq1_sub_q()  { return pxq1_sub16_q_; }

// one 16-elem sub-block against a fixed effective scale d (fp32), double-accum weighted err.
static inline double pxq1_block_err(const float * x, const float * w, float d,
                                    const float * book, uint8_t * codes) {
    const double d64 = (double)d > 1e-30 ? (double)d : 1e-30;
    double err = 0.0;
    for (int i = 0; i < 16; ++i) {
        const int c = ((double)x[i] / d64) > 0.0 ? 1 : 0;   // sign -> book index (mids = {0})
        codes[i] = (uint8_t)c;
        const float rec = d * book[c];                       // fp32 product == kernel math
        const double e = (double)x[i] - (double)rec;
        err += (w ? (double)w[i] : 1.0) * e * e;
    }
    return err;
}

// pick the best 4-bit sub for one 16-elem block under a fixed row anchor (FULL 16-cand search).
static inline double pxq1_quant_subblock(const float * x, const float * w, float anchor,
                                         const float * book, const float * sub,
                                         uint8_t * s4_out, uint8_t * codes_out) {
    float amax = 0.f;
    for (int i = 0; i < 16; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f) || !(anchor > 0.f)) {
        *s4_out = 0;
        for (int i = 0; i < 16; ++i) codes_out[i] = PXQ1_ZIDX;
        return 0.0;
    }
    double best = 1e300;
    uint8_t codes[16];
    for (int j = 0; j < 16; ++j) {
        const float d = (float)((double)anchor * (double)sub[j]);   // fp64 product, single fp32 round
        const double err = pxq1_block_err(x, w, d, book, codes);
        if (err < best) {
            best = err;
            *s4_out = (uint8_t)j;
            memcpy(codes_out, codes, 16);
        }
    }
    return best;
}

static double pxq1_quant_row(const float * x, const float * w, int64_t K, float anchor,
                             const float * book, const float * sub,
                             uint8_t * s4_flat /*K/16*/, uint8_t * codes_flat /*K*/) {
    double err = 0.0;
    for (int64_t b = 0; b < K/16; ++b) {
        err += pxq1_quant_subblock(x + b*16, w ? w + b*16 : nullptr, anchor,
                                   book, sub, &s4_flat[b], &codes_flat[b*16]);
    }
    return err;
}

static float pxq1_pick_anchor(const float * x, int64_t K) {
    float amax = 0.f;
    for (int64_t i = 0; i < K; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (amax > 65504.f) amax = 65504.f;                    // fp16 ceiling (hardening)
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(amax));
}

// one [R,K] expert -> PXQ1 panels (bs16 subs, slab 320, 1-bit codes).
static void pxq1_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K,
                                 const float * imx /*K vals or null*/) {
    const float * book = pxq1_book_q();
    const float * sub  = pxq1_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ1_HDR_BYTES + KB*PXQ1_SLAB_BYTES;
    const int64_t nsub = K / 16;

    std::vector<uint8_t> s4(nsub), codes(K);
    for (int64_t p = 0; p < P; ++p) {
        uint8_t * panel = dst + p*panel_bytes;
        ggml_fp16_t * anchors = (ggml_fp16_t *)panel;      // 64 x fp16 header
        for (int64_t r = 0; r < 64; ++r) {
            const float * x = src + (p*64 + r)*K;
            const float anchor = pxq1_pick_anchor(x, K);
            anchors[r] = ggml_fp32_to_fp16(anchor);
            pxq1_quant_row(x, imx, K, anchor, book, sub, s4.data(), codes.data());
            for (int64_t kb = 0; kb < KB; ++kb) {
                uint8_t * slab = panel + PXQ1_HDR_BYTES + kb*PXQ1_SLAB_BYTES;
                const uint8_t * s = &s4[kb*2];             // 2 subs per 32 elems: lo nibble = elems 0-15
                slab[r] = (uint8_t)(s[0] | (s[1] << 4));
                uint8_t * out = slab + 64 + r*4;           // 4 code bytes / row (8 codes/byte)
                const uint8_t * c = &codes[kb*32];
                for (int by = 0; by < 4; ++by) {
                    out[by] = (uint8_t)( c[8*by]   | (c[8*by+1] << 1) | (c[8*by+2] << 2) | (c[8*by+3] << 3)
                                       | (c[8*by+4] << 4) | (c[8*by+5] << 5) | (c[8*by+6] << 6) | (c[8*by+7] << 7));
                }
            }
        }
    }
}

// reference dequant (CPU) — the parity-locked contract. Used by pxq1_ref + golden tests.
static void pxq1_dequant_expert(const uint8_t * src, float * dst, int64_t R, int64_t K) {
    const float * book = pxq1_book_q();
    const float * sub  = pxq1_sub_q();
    const int64_t KB = K/32, P = R/64;
    const int64_t panel_bytes = PXQ1_HDR_BYTES + KB*PXQ1_SLAB_BYTES;
    for (int64_t p = 0; p < P; ++p) {
        const uint8_t * panel = src + p*panel_bytes;
        const ggml_fp16_t * anchors = (const ggml_fp16_t *)panel;
        for (int64_t r = 0; r < 64; ++r) {
            const float anchor = ggml_fp16_to_fp32(anchors[r]);
            for (int64_t kb = 0; kb < KB; ++kb) {
                const uint8_t * slab = panel + PXQ1_HDR_BYTES + kb*PXQ1_SLAB_BYTES;
                float eff[2];
                eff[0] = anchor * sub[slab[r] & 0xf];      // elems 0-15
                eff[1] = anchor * sub[slab[r] >> 4];       // elems 16-31
                const uint8_t * q = slab + 64 + r*4;
                float * o = dst + (p*64 + r)*K + kb*32;
                for (int j = 0; j < 32; ++j) {
                    const int c = (q[j >> 3] >> (j & 7)) & 1;
                    o[j] = eff[j >> 4] * book[c];
                }
            }
        }
    }
}

// full 3D expert tensor, threaded across experts.
// imatrix semantics: imx_size == K*E -> per-expert columns; == K -> shared; else ignored.
static void pxq1_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                 const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = (R/64)*(PXQ1_HDR_BYTES + (K/32)*(int64_t)PXQ1_SLAB_BYTES);
    auto imx_for = [&](int64_t e) -> const float * {
        if (!imx) return nullptr;
        if (imx_size == K*E) return imx + e*K;
        if (imx_size == K)   return imx;
        return nullptr;
    };
    if (nthread <= 1 || E <= 1) {
        for (int64_t e = 0; e < E; ++e)
            pxq1_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        return;
    }
    std::atomic<int64_t> counter{0};
    auto compute = [&]() {
        while (true) {
            const int64_t e = counter.fetch_add(1);
            if (e >= E) break;
            pxq1_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
    };
    std::vector<std::thread> th;
    const int n = (int) std::min<int64_t>(nthread, E);
    th.reserve(n);
    for (int i = 0; i < n; ++i) th.emplace_back(compute);
    for (auto & t : th) t.join();
}
