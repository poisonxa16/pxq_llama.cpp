// pxq5-quantize.inc.cpp — PXQ5 native quantizer for src/llama-quantize.cpp (spec v1).
//
// SELF-CONTAINED functions; splice into llama-quantize.cpp next to the PXQ4 block.
// WIRING (do on-box; anchors depend on the fork HEAD):
//   1. quantize.cpp QUANT_OPTIONS: { "PXQ5", LLAMA_FTYPE_MOSTLY_PXQ5,
//        " 4.25 bpw, PXA proprietary numerics (learned book + fine scale) in the slab layout", }
//   2. llama.h: LLAMA_FTYPE_MOSTLY_PXQ5 = 251
//   3. llama-model.cpp ftype_name: "PXQ5 - 4.25 bpw, PXA proprietary numerics, slab layout"
//   4. llama_model_quantize_internal:
//        case LLAMA_FTYPE_MOSTLY_PXQ5: default_type = GGML_TYPE_MXFP4; break;  // non-expert
//        const bool pxq5_out = ftype == LLAMA_FTYPE_MOSTLY_PXQ5;
//        if (pxq5_out) ftype = LLAMA_FTYPE_MOSTLY_MXFP4;   // ride the MXFP4 rules pipeline
//      Then, INSIDE the per-tensor quantize branch, IMMEDIATELY AFTER f32_data is prepared
//      (and BEFORE do_quantize), intercept eligible expert tensors:
//        if (pxq5_out && !params->dry_run && pxq4_tensor_eligible(name, tensor)) {
//            const int64_t K = tensor->ne[0], R = tensor->ne[1], E = tensor->ne[2]*tensor->ne[3];
//            new_size = (size_t)(E*R*(K/32))*17;
//            if (work.size() < new_size) work.resize(new_size);
//            new_data = work.data();
//            // imatrix: per-expert slice when sized ne0*E, shared when sized ne0 (see below)
//            pxq5_quantize_tensor(f32_data, (uint8_t *)new_data, R, K, E,
//                                 imatrix, imatrix ? imatrix_size : 0, nthread);
//            new_type = GGML_TYPE_PXQ5;
//            LLAMA_LOG_INFO("%s: PXQ5 native quantize (proprietary numerics) -> pxq5\n", name.c_str());
//            goto <the post-quantize bookkeeping>;   // skip do_quantize for this tensor
//        }
//      Requantize guard (same as PXQ4): tensors already PXQ5 in the input -> hard error
//      ("no CPU codec; PXQ5 is quantized from F32/BF16/Q8_0 sources only").
//   5. OPTIONAL (recommended): record the book in the output KVs:
//        gguf_set_arr_data(ctx_out, "pxa.pxq5.codebook", GGUF_TYPE_FLOAT32, pxq5_book_q(), 16);
//      v1 runtime reads the book from the compiled default / PXA_PXQ5_BOOK env — the KV is
//      provenance now, loader-consumed in v2.
//
// BIT-PARITY CONTRACT: this code must produce byte-identical output to the Python reference
// (pxa-bench/pxq5_quantize.py) on the same f32 input. Same tables (ggml-pxq5-tables.h), same
// double-precision error accumulation, same code assignment (midpoint RTN on the sorted book),
// same scale search window {fit-2 .. fit+1}. The reference tool's --verify mode asserts this.

#include "../ggml/include/ggml-pxq5-tables.h"

#include <cmath>
#include <cstring>

static const float pxq5_book_q_[16]    = PXQ5_BOOK_INIT;
static const float pxq5_scale_q_[256]  = PXQ5_SCALE_TAB_INIT;

static inline const float * pxq5_book_q() {
    // optional custom book (must match the runtime's PXA_PXQ5_BOOK — fp16-snapped)
    static float book[16];
    static bool init = false;
    if (!init) {
        init = true;
        memcpy(book, pxq5_book_q_, sizeof(book));
        if (const char * e = getenv("PXA_PXQ5_BOOK")) {
            int n = 0; float v[16];
            char * dup = strdup(e);
            for (char * t = strtok(dup, ","); t && n < 16; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
            free(dup);
            if (n == 16) {
                for (int i = 0; i < 16; ++i) book[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(v[i]));
                fprintf(stderr, "PXQ5 quantize: custom codebook from PXA_PXQ5_BOOK\n");
            }
        }
    }
    return book;
}

// midpoints of the sorted book, in double (code = #mids below x — identical to numpy searchsorted)
static inline const double * pxq5_mids_q() {
    static double mids[15];
    static bool init = false;
    if (!init) {
        init = true;
        const float * b = pxq5_book_q();
        for (int i = 0; i < 15; ++i) mids[i] = ((double)b[i] + (double)b[i+1]) * 0.5;
    }
    return mids;
}

static inline int pxq5_fit_scale(float amax) {
    // smallest s with scale >= amax (book absmax == 1); clamp [1,255]
    int s = (int)ceil(8.0 * log2((double)amax)) + PXQ5_SCALE_BIAS;
    return s < 1 ? 1 : (s > 255 ? 255 : s);
}

static inline int pxq5_code(double xn, const double * mids) {
    int c = 0;
    #pragma GCC unroll 15
    for (int i = 0; i < 15; ++i) c += xn > mids[i];   // searchsorted(left): count mids < x... see note
    return c;
}
// NOTE on ties: numpy searchsorted(mids, x) with default side='left' counts mids[i] < x is
// equivalent to `x > mids[i]` summation ONLY for x != mids[i]; at exact midpoints numpy 'left'
// gives the LOWER code while `>` gives the lower code too (x == mid -> not >, stays low). Match.

static void pxq5_quantize_block(const float * x, const float * w, uint8_t * sc_out, uint8_t * codes_out) {
    const float  * book = pxq5_book_q();
    const double * mids = pxq5_mids_q();
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) { float a = fabsf(x[i]); if (a > amax) amax = a; }
    if (!(amax > 0.f)) {
        *sc_out = 0;
        for (int i = 0; i < 32; ++i) codes_out[i] = 7;   // book[7] == 0
        return;
    }
    const int s0 = pxq5_fit_scale(amax);
    double best_err = 1e300;
    for (int k = -2; k <= 1; ++k) {
        int s = s0 + k; s = s < 1 ? 1 : (s > 255 ? 255 : s);
        const double d = (double)pxq5_scale_q_[s];
        double err = 0.0;
        uint8_t c[32];
        for (int i = 0; i < 32; ++i) {
            c[i] = (uint8_t)pxq5_code((double)x[i] / d, mids);
            const float rec = pxq5_scale_q_[s] * book[c[i]];      // fp32 product == kernel math
            const double e = (double)x[i] - (double)rec;
            err += (w ? (double)w[i] : 1.0) * e * e;
        }
        if (err < best_err) {
            best_err = err;
            *sc_out = (uint8_t)s;
            memcpy(codes_out, c, 32);
        }
    }
}

// one [R,K] expert -> slabs (layout identical to PXQ4: 64B scale SoA + 64 x 16B nibble rows,
// sequential pairs, K-major slabs in 64-row panels)
static void pxq5_quantize_expert(const float * src, uint8_t * dst, int64_t R, int64_t K, const float * imx /*K vals or null*/) {
    const int64_t KB = K/32, P = R/64;
    uint8_t codes[32];
    for (int64_t p = 0; p < P; ++p) {
        for (int64_t kb = 0; kb < KB; ++kb) {
            uint8_t * slab = dst + (p*KB + kb)*1088;
            const float * wblk = imx ? imx + kb*32 : nullptr;
            for (int64_t r = 0; r < 64; ++r) {
                const float * x = src + (p*64 + r)*K + kb*32;
                uint8_t sc;
                pxq5_quantize_block(x, wblk, &sc, codes);
                slab[r] = sc;
                uint8_t * out = slab + 64 + r*16;
                for (int b = 0; b < 16; ++b) out[b] = (uint8_t)(codes[2*b] | (codes[2*b+1] << 4));
            }
        }
    }
}

// full 3D expert tensor, threaded across experts (same pattern as pxq4_permute_from_mxfp4).
// imatrix semantics: imx_size == K*E -> per-expert columns; == K -> shared columns; else ignored.
static void pxq5_quantize_tensor(const float * src, uint8_t * dst, int64_t R, int64_t K, int64_t E,
                                 const float * imx, int64_t imx_size, int nthread) {
    const int64_t exp_elems = R*K;
    const int64_t exp_bytes = R*(K/32)*17;
    auto imx_for = [&](int64_t e) -> const float * {
        if (!imx) return nullptr;
        if (imx_size == K*E) return imx + e*K;
        if (imx_size == K)   return imx;
        return nullptr;
    };
    if (nthread <= 1 || E <= 1) {
        for (int64_t e = 0; e < E; ++e) {
            pxq5_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
        return;
    }
    std::atomic<int64_t> counter{0};
    auto compute = [&]() {
        while (true) {
            const int64_t e = counter.fetch_add(1);
            if (e >= E) break;
            pxq5_quantize_expert(src + e*exp_elems, dst + e*exp_bytes, R, K, imx_for(e));
        }
    };
    std::vector<std::thread> th;
    const int n = (int) std::min<int64_t>(nthread, E);
    th.reserve(n);
    for (int i = 0; i < n; ++i) th.emplace_back(compute);
    for (auto & t : th) t.join();
}
