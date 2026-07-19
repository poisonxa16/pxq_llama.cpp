// pxq6i8.cuh — N13: int8 DP4A MMQ-style PREFILL tile for the PXQ formats (sm_61 first).
//
// WHY THIS FORM (design justification, N13): the two alternatives were (a) repack PXQ ->
// q8_0 and reuse the existing mul_mat_q_id MMQ path, or (b) a PXQ trait inside mmq.cuh.
// (a) needs either a resident q8_0 shadow of the expert weights (~2x the 2-bit tier's VRAM —
// impossible next to a ~10 GiB model on the 11 GiB 1080Ti) or a per-invocation full-tensor
// repack through global memory, AND q8_0's per-32 scale cannot carry PXQ's per-16 (per-8 for
// HQ) sub-scales without a second requantization error on top of the s8 book snap. (b) wedges
// panel-interleaved, row-meta'd PXQ layouts into trait machinery built for standard ggml block
// quants — touching shared MMQ code used by every other type/arch. So: a THIRD form — an int8
// twin of k_pxq6_gemm_grouped inside the existing policy-templated family. It reuses the whole
// proven grouped-prefill driver structure (tile map, row mapping, GLU, scatter, bias folding,
// mixed up/gate/down formats via per-GEMM pickers) and confines every new line to this file
// plus one env-gated dispatch branch. The "repack tile in smem" idea survives as the W-stage:
// codes -> s8 via the snapped book, per-16 scale folded into a per-tile fp32 rescale.
//
// NUMERIC CONTRACT (q3-s8-snap.md): book values snap to s8 as q_i = rint(book_i * 127/absmax);
// dequant w = (anchor * SUB[s] * (absmax/127)) * q_s8 — the /127 and book absmax fold into the
// per-16-block fp32 eff scale; the stored fp16 anchor is untouched. For the PX16 book
// (PXQ5/PXQ6/PXQ6HQ, absmax = 1.0) this reproduces the frozen Q3 s8 book
// [-125,-93,-71,-53,-38,-25,-12,0,11,22,33,46,60,76,97,127] exactly (+0.073% wrel refit /
// +0.219% post-hoc table swap — this kernel is the post-hoc form). PXQ2 (LM4) and PXQ3 (LM8)
// books are snapped by the same rule at runtime; their 2/3-bit quantization floors dwarf the
// <=0.4%-of-absmax grid displacement. PXQ4's E2M1 book is NOT snap-validated -> declined.
// Activations are quantized q8_1-style (per-32 absmax/127, round-to-nearest) — the same
// class of activation quantization the stock MMQ path applies to every quantized type.
//
// NOT bit-exact vs the fused fp16 path (int8 x int8 -> fp32 rescale vs strict-k half2 chains):
// G3-class gates (temp-0 coherence + semantic equivalence + top-5 logit spot-check) are
// mandatory before any live use. Flag OFF (default) = byte-identical dispatch to the old build.
//
// ENV: PXA_PXQ_INT8_PREFILL=0 (default, OFF) | 1 (sm_61 only — the ship gate) | 2 (all archs,
// TEST: sm_70+ has real dp4a; sm_60 falls to the emulated ggml_cuda_dp4a — never ship that).
#pragma once

// requires: pxq6.cuh (policies incl. pxq6_pol_p2/p3 via pxq23.cuh, pxq6_panel, pxq6_ldcodes,
// pxq4_tile_info, pxq4_rowmap, pxq4_glu_apply, PXA_PXQ_FMT_*), common.cuh (ggml_cuda_dp4a).

static inline int pxa_pxq_int8_prefill() {
    static const int mode = [](){
        const char * e = getenv("PXA_PXQ_INT8_PREFILL");
        int m = e ? atoi(e) : 0;
        if (m < 0 || m > 2) m = 0;
        if (m) fprintf(stderr, "PXA_PXQ_INT8_PREFILL: mode %d (N13 dp4a int8 MMQ-tile prefill, %s; "
                       "s8 book snap — NOT bit-exact vs fused fp16, G3-gated)\n",
                       m, m == 1 ? "sm_61 only" : "ALL archs (TEST)");
        return m;
    }();
    return mode;
}

// ---------------------------------------------------------------------------------------------
// per-32-group s8 quantize of one staged f32 row (shared by the gather and GLU kernels).
// d = absmax/127, q = rint(x/d) in [-127,127]; d==0 group -> all-zero codes.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ void pxqi8_quant_row_groups(const float * __restrict__ xs,
        uint32_t * __restrict__ q_out, float * __restrict__ d_out, const int ngroups, const int nthr) {
    for (int g = threadIdx.x; g < ngroups; g += nthr) {
        const float * x = xs + g*PXQ4_QK;
        float amax = 0.f;
        #pragma unroll
        for (int t = 0; t < PXQ4_QK; ++t) amax = fmaxf(amax, fabsf(x[t]));
        const float inv = amax > 0.f ? 127.f/amax : 0.f;
        d_out[g] = amax / 127.f;
        #pragma unroll
        for (int w = 0; w < PXQ4_QK/4; ++w) {
            uint32_t pk = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const int v = (int)rintf(x[4*w + b]*inv);   // |x| <= amax -> v in [-127,127]
                pk |= (uint32_t)((uint8_t)v) << (8*b);
            }
            q_out[(size_t)g*(PXQ4_QK/4) + w] = pk;
        }
    }
}

// activation gather + q8 quantize: flat expert-grouped f32 rows -> s8 [total][K] (packed LE
// u32 words) + per-32-group scales f32 [total][K/32]. Same row mapping semantics as
// k_pxq4_gather_a_f16. dyn smem = K floats.
static __global__ void k_pxqi8_gather_quant(const char * __restrict__ src1,
        uint8_t * __restrict__ Aq, float * __restrict__ Ad,
        const pxq4_rowmap * __restrict__ map,
        const int64_t K, const int64_t ne11, const size_t nb11, const size_t nb12) {
    extern __shared__ float pxqi8_xs[];
    const int i = blockIdx.x;
    const int32_t i11 = map[i].i1 % ne11;
    const int32_t i12 = map[i].i2;
    const float * src = (const float *)(src1 + i11*nb11 + i12*nb12);
    for (int64_t j = threadIdx.x; j < K; j += blockDim.x) pxqi8_xs[j] = src[j];
    __syncthreads();
    pxqi8_quant_row_groups(pxqi8_xs, (uint32_t *)(Aq + (size_t)i*K), Ad + (size_t)i*(K/PXQ4_QK),
                           (int)(K/PXQ4_QK), blockDim.x);
}

// GLU epilogue + q8 quantize: C_gate/C_up f32 (flat [total][R]) -> H s8 + per-32 scales.
// The GLU arithmetic is pxq4_glu_apply verbatim. dyn smem = R floats.
static __global__ void k_pxqi8_glu_quant(const float * __restrict__ xg, const float * __restrict__ gu,
        uint8_t * __restrict__ Hq, float * __restrict__ Hd, const int R,
        const int unary, const float alpha, const float limit) {
    extern __shared__ float pxqi8_hs[];
    const int i = blockIdx.x;
    const float * g = xg + (size_t)i*R;
    const float * u = gu + (size_t)i*R;
    for (int j = threadIdx.x; j < R; j += blockDim.x)
        pxqi8_hs[j] = pxq4_glu_apply(g[j], u[j], unary, alpha, limit);
    __syncthreads();
    pxqi8_quant_row_groups(pxqi8_hs, (uint32_t *)(Hq + (size_t)i*R), Hd + (size_t)i*(R/PXQ4_QK),
                           R/PXQ4_QK, blockDim.x);
}

// ---------------------------------------------------------------------------------------------
// the int8 grouped prefill GEMM. Same grid/tile geometry as k_pxq6_gemm_grouped (64 threads,
// 64x64 C tile, one panel x one 64-token tile per block), but the inner product is dp4a on s8:
//   W-stage: thread = row; codes -> s8 via the snapped book (POL::pair over a smem table
//            holding the SNAPPED INTEGER values), packed 4/word into sWq[8][64]; the per-group
//            eff scale (anchor*SUB, POL::row_effs) x (book_absmax/127) goes to sEff[NG][64].
//   A-stage: thread = token; one aligned 32 B s8 read + the per-slab d scale.
//   FMA:     8 rows x 8 tokens per thread; per 32-k slab: NG integer sub-sums (dp4a chains of
//            8/NG words) -> fp32: acc += dA * sum_n eff[n]*sumi[n]. Token side is register-
//            cached in two 4-token panels to keep smem traffic ~3x below the naive form.
// NG = POL::NEFF (1 for P5's single per-32 scale, 2 for P2/P3/P6 per-16, 4 for P6HQ per-8).
// ---------------------------------------------------------------------------------------------
typedef void (*pxqi8_gemm_fn)(const uint8_t *, const uint8_t *, const float *, float *,
                              const float *, const size_t, const pxq4_tile_info *, const int, const int);

template <class POL>
static __global__ void __launch_bounds__(64)
k_pxqi8_gemm_grouped(const uint8_t * __restrict__ W,
                     const uint8_t * __restrict__ Aq, const float * __restrict__ Ad,
                     float * __restrict__ C,
                     const float * __restrict__ bias, const size_t bias_nb1,
                     const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    constexpr int NG  = POL::NEFF;    // integer sub-sum groups per 32-elem slab
    constexpr int WPG = 8 / NG;       // 4-byte k-words per group
    const int panels = R / PXQ4_BM, kslabs = K / PXQ4_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * pan = pxq6_panel<POL>(W, tile.e, panels, p, kslabs);
    const uint8_t * Aqt = Aq + (size_t)tile.row0*K;
    const float   * Adt = Ad + (size_t)tile.row0*kslabs;
    float         * Ct  = C  + (size_t)tile.row0*R + (size_t)p*PXQ4_BM;

    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float stab[16];               // s8-snapped book (integer values held as float)
    __shared__ uint32_t sWq[8][PXQ4_BM];     // packed s8 W tile, k-word major
    __shared__ uint32_t sAq[8][PXQ4_BN];     // packed s8 A tile, k-word major
    __shared__ float sEff[NG][PXQ4_BM];      // folded per-group W scales (anchor*SUB*absmax/127)
    __shared__ float sDa[PXQ4_BN];           // per-token per-slab A scale
    const int tid = threadIdx.x;
    POL::stage_tabs(tab, sub, tid);
    __syncthreads();
    float bmax = 0.f;                        // book absmax (P2/P3 pad entries are 0 -> harmless)
    #pragma unroll
    for (int t = 0; t < 16; ++t) bmax = fmaxf(bmax, fabsf(tab[t]));
    const float bfold = bmax / 127.f;        // folds into the per-group eff scale (contract)
    if (tid < 16) stab[tid] = rintf(tab[tid] * (127.f/bmax));
    __syncthreads();

    const float anch = POL::HDR ? POL::anchor(pan, tid) : 0.f;
    const int tx = tid & 7, ty = tid >> 3;   // rows 8*tx..+7, tokens 8*ty..+7

    float acc[8][8];
    #pragma unroll
    for (int r8 = 0; r8 < 8; ++r8)
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[r8][j] = 0.f;

    const bool a_valid = tid < tile.nrows;

    for (int kb = 0; kb < kslabs; ++kb) {
        const uint8_t * slab = pan + POL::HDR + (size_t)kb*POL::SLAB;
        {   // W-stage: this thread's row -> s8 words + folded eff scales
            uint32_t q[POL::CODE_WORDS];
            pxq6_ldcodes<POL>(slab + POL::CODE_OFF + tid*POL::CODE_BYTES, q);
            float eff[NG];
            POL::row_effs(slab, tid, anch, sub, eff);
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                const float2 v0 = POL::pair(q, 2*w,   stab);
                const float2 v1 = POL::pair(q, 2*w+1, stab);
                sWq[w][tid] = (uint32_t)((uint8_t)(int)v0.x)
                            | ((uint32_t)((uint8_t)(int)v0.y) << 8)
                            | ((uint32_t)((uint8_t)(int)v1.x) << 16)
                            | ((uint32_t)((uint8_t)(int)v1.y) << 24);
            }
            #pragma unroll
            for (int n = 0; n < NG; ++n) sEff[n][tid] = eff[n]*bfold;
        }
        if (a_valid) {   // A-stage: this thread's token — one aligned 32 B sector
            const uint8_t * arow = Aqt + (size_t)tid*K + kb*PXQ4_QK;
            const uint4 w0 = *(const uint4 *)arow;
            const uint4 w1 = *(const uint4 *)(arow + 16);
            sAq[0][tid] = w0.x; sAq[1][tid] = w0.y; sAq[2][tid] = w0.z; sAq[3][tid] = w0.w;
            sAq[4][tid] = w1.x; sAq[5][tid] = w1.y; sAq[6][tid] = w1.z; sAq[7][tid] = w1.w;
            sDa[tid] = Adt[(size_t)tid*kslabs + kb];
        } else {
            #pragma unroll
            for (int w = 0; w < 8; ++w) sAq[w][tid] = 0;
            sDa[tid] = 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int jb = 0; jb < 2; ++jb) {     // two 4-token register panels
            uint32_t aw[4][8]; float da[4];
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                const int t = 8*ty + 4*jb + jj;
                da[jj] = sDa[t];
                #pragma unroll
                for (int w = 0; w < 8; ++w) aw[jj][w] = sAq[w][t];
            }
            #pragma unroll
            for (int r8 = 0; r8 < 8; ++r8) {
                const int r = 8*tx + r8;
                uint32_t ww[8];
                #pragma unroll
                for (int w = 0; w < 8; ++w) ww[w] = sWq[w][r];
                float ef[NG];
                #pragma unroll
                for (int n = 0; n < NG; ++n) ef[n] = sEff[n][r];
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int n = 0; n < NG; ++n) {
                        int s = 0;
                        #pragma unroll
                        for (int w = 0; w < WPG; ++w)
                            s = ggml_cuda_dp4a((int)ww[n*WPG + w], (int)aw[jj][n*WPG + w], s);
                        sum += ef[n]*(float)s;
                    }
                    acc[r8][4*jb + jj] += da[jj]*sum;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r8 = 0; r8 < 8; ++r8) {
        const int row = 8*tx + r8;
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1
                                                + (size_t)(p*PXQ4_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int t = 8*ty + j;
            if (t < tile.nrows) Ct[(size_t)t*R + row] = acc[r8][j] + b;
        }
    }
}

// per-format picker. P4 (E2M1 book, not snap-validated) and unknown formats decline -> the
// caller falls back to the proven fp16/cublas paths for the WHOLE op.
static pxqi8_gemm_fn pxqi8_pick_gemm(int fmt) {
    switch (fmt) {
        case PXA_PXQ_FMT_P5:   return k_pxqi8_gemm_grouped<pxq6_pol_p5>;
        case PXA_PXQ_FMT_P6:   return k_pxqi8_gemm_grouped<pxq6_pol_p6>;
        case PXA_PXQ_FMT_P6HQ: return k_pxqi8_gemm_grouped<pxq6_pol_p6hq>;
        case PXA_PXQ_FMT_P2:   return k_pxqi8_gemm_grouped<pxq6_pol_p2>;
        case PXA_PXQ_FMT_P3:   return k_pxqi8_gemm_grouped<pxq6_pol_p3>;
        default:               return nullptr;
    }
}
