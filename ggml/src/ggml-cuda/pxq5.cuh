// pxq5.cuh — PXQ5: the PXA fully-proprietary 4-bit quant (learned numerics + PXQ4 slab layout).
//
// Design + measured CPU PoC: PXQ5 proprietary-quant design notes, 2026-07-16 (internal lab).
// On real MoE expert weights PXQ5 measures ~32% lower relative-L2 quantization error than MXFP4
// at the SAME 4.25 bpw (MSE −54%, max-abs −65%); imatrix-weighted quantize −37% on weighted error.
//
// NUMERICS (the ONLY deltas vs pxq4.cuh — layout, tiling, drivers are identical):
//   value(c, s) = pxq5_scale_tab[s] * pxq5_book[c]
//   - book: 16 learned fp16 levels (pooled Lloyd-Max on real expert weights), sorted asc,
//     book[7] == 0, absmax == 1. Default = PX16 (ggml-pxq5-tables.h). Optional per-model
//     override via env PXA_PXQ5_BOOK="v0,...,v15" (fp16-snapped; MUST match what the file
//     was quantized with — the quantizer records it in KV pxa.pxq5.codebook).
//   - scale byte: s==0 -> 0.0 (all-zero block); else 2^((s-160)/8) — log-uniform, 9% steps.
//     Decoded via a 256-entry fp32 table (frozen literals; NOT exp2f — bit-determinism).
//   Both tables land in smem at block start; the inner loops are byte-identical in shape to
//   PXQ4's (LUT16 lookup * per-row scale -> half2 HFMA2 / fp32 FMA).
//
// On-disk layout: IDENTICAL to PXQ4 (slab = 64 B scale SoA + 64 x 16 B nibble rows, sequential
// pairs; slabs K-major in 64-row panels; experts outermost; 17 B / 32 elems). PXQ5 is NOT
// transcodable from MXFP4/PXQ4 losslessly (different codebook) — it is quantized natively from
// F32/BF16/Q8_0 sources (llama-quantize PXQ5 / pxa-bench/pxq5_quantize.py).
//
// Env: PXA_PXQ5=0 disables the fused kernels (dequant->cublas fallback via convert.cu hooks,
// correct on every arch). PXA_PXQ5_BOOK overrides the codebook (see above). Default ON.
//
// PXA_PXQ5_FAST=1 (default OFF — proven path stays default until measured + parity-clean):
// speed-only fast variants of the fused kernels, PROVABLY BIT-IDENTICAL numerics:
//   (a) the 256-entry smem scale table is replaced by an exact bit-decode:
//         t = s-160;  d = 2^(t>>3) * frt8[t&7]      (s==0 -> 0.0f)
//       where 2^q is built by exponent-bit construction and frt8 holds the 8 fraction
//       values 2^(r/8), r=0..7 (= scale_tab[160..167]). A power-of-two multiply of an
//       fp32 value is EXACT (no rounding) while the result stays normal — q in [-20,11]
//       keeps everything normal, so d reproduces the frozen table bit-for-bit. Verified
//       exhaustively over all 255 scale bytes on the HOST at enable time (self-check in
//       pxa_pxq5_fast(); any mismatch self-disables fast mode) — numerics cannot drift.
//   (b) the book (+frt8) are staged into smem from __device__ GLOBALS with coalesced
//       loads instead of per-thread-divergent __constant__ reads (constant cache
//       serializes ~32-way per warp: the old startup paid ~272 serialized transactions
//       per block; the mmv blocks are short, so this was a real per-launch tax).
//   (c) static smem drops 1088 B -> 96 B per block, and the per-slab scale lookup moves
//       from stab[256] (up to 8-way bank conflicts: words s and s+32k share a bank) to
//       frt8[8] (8 words in 8 distinct banks -> conflict-free).
// The dot-product / accumulation / epilogue code is byte-identical to the proven kernels.
//
// WIRING NOTE (box fork): the drivers in ggml-cuda.cu are the PXQ4 drivers with the type check
// widened to (PXQ4 || PXQ5) and the kernel pointers selected by type — see the patch hunks.
// If the box pxq4 drivers have grown (e.g. the SILU glu generalization), mirror the SAME glu
// helper here; the on-device logit-parity gate catches any epilogue mismatch.
#pragma once

#include "pxq4.cuh"              // slab macros, tile structs, gather/swiglu/copy kernels (reused)
#include "../../include/ggml-pxq5-tables.h"
#include <cstring>               // memcmp/strdup (fast-mode host self-check + book env parse)

// per-TU constant tables (ggml-cuda.cu and convert.cu each get a copy; the env override is
// applied per-TU + per-device by pxq5_maybe_upload_book(), so all paths stay consistent).
static __device__ __constant__ float pxq5_book_c[16]      = PXQ5_BOOK_INIT;
static __device__ __constant__ float pxq5_scale_tab_c[256] = PXQ5_SCALE_TAB_INIT;

// PXA_PXQ5_FAST globals (device GLOBAL memory, not __constant__ — coalesced block-startup
// staging; see header comment). frt8[r] = 2^(r/8) = pxq5_scale_tab[160+r], frozen fp32 bits.
#define PXQ5_FRT8_INIT { \
    0x1.0000000000000p+0f, 0x1.172b840000000p+0f, 0x1.306fe00000000p+0f, 0x1.4bfdae0000000p+0f, \
    0x1.6a09e60000000p+0f, 0x1.8ace540000000p+0f, 0x1.ae89fa0000000p+0f, 0x1.d5818e0000000p+0f }
static __device__ float pxq5_book_g[16] = PXQ5_BOOK_INIT;
static __device__ float pxq5_frt8_g[8]  = PXQ5_FRT8_INIT;

static inline bool pxa_pxq5_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ5");
        bool v = !(e && atoi(e) == 0);
        fprintf(stderr, "PXA_PXQ5 fused kernels: %s\n", v ? "ON (PXA_PXQ5=0 disables)" : "OFF");
        return v;
    }();
    return on;
}

// K0 FASTON (PXQ6 spec, 2026-07-17): PXA_PXQ5_FAST now defaults ON — the fast kernels are
// PROVABLY BIT-IDENTICAL (exhaustive 255/255 host self-check below runs at every enable, and
// any mismatch self-disables back to the proven path). PXA_PXQ5_FAST=0 rolls back.
static inline bool pxa_pxq5_fast() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ5_FAST");
        bool v = !(e && atoi(e) == 0);
        if (v) {
            static const float tab[256] = PXQ5_SCALE_TAB_INIT;
            static const float fr8[8]   = PXQ5_FRT8_INIT;
            for (int r = 0; r < 8 && v; ++r) {
                if (memcmp(&tab[PXQ5_SCALE_BIAS + r], &fr8[r], 4) != 0) v = false;
            }
            for (int s = 1; s < 256 && v; ++s) {
                const int t = s - PXQ5_SCALE_BIAS;
                union { uint32_t u; float f; } p2; p2.u = (uint32_t)((t >> 3) + 127) << 23;
                const float d = p2.f * fr8[t & 7];
                if (memcmp(&d, &tab[s], 4) != 0) v = false;
            }
            union { float f; uint32_t u; } z; z.f = tab[0];
            if (z.u != 0) v = false;
            fprintf(stderr, v ? "PXA_PXQ5_FAST: ON (scale-decode bit-identity self-check PASS, 255/255)\n"
                              : "PXA_PXQ5_FAST: self-check FAILED — fast kernels DISABLED, proven path in use\n");
        }
        return v;
    }();
    return on;
}

// exact fast scale decode: d = 2^((s-160)/8) via 2^q * frt8[r] — bit-identical to
// pxq5_scale_tab_c[s] for ALL s (host-verified at enable time). s==0 -> 0.0f.
static __device__ __forceinline__ float pxq5_scale_fast(const int s, const float * __restrict__ fr8) {
    const int t = s - PXQ5_SCALE_BIAS;
    const float d = __int_as_float((int)((uint32_t)((t >> 3) + 127) << 23)) * fr8[t & 7];
    return s ? d : 0.0f;
}

// env-override upload (idempotent per device per TU). Call at every PXQ5 entry point.
static inline void pxq5_maybe_upload_book(int device) {
    static bool parsed = false;
    static bool have_env = false;
    static float envbook[16];
    static bool uploaded[64] = {false};
    if (!parsed) {
        parsed = true;
        if (const char * e = getenv("PXA_PXQ5_BOOK")) {
            int n = 0; float v[16];
            char * dup = strdup(e);
            for (char * t = strtok(dup, ","); t && n < 16; t = strtok(nullptr, ",")) v[n++] = strtof(t, nullptr);
            free(dup);
            if (n == 16) {
                for (int i = 0; i < 16; ++i) envbook[i] = __half2float(__float2half_rn(v[i]));  // fp16-snap (spec)
                have_env = true;
                fprintf(stderr, "PXA_PXQ5_BOOK: custom codebook active\n");
            } else {
                fprintf(stderr, "PXA_PXQ5_BOOK: expected 16 comma-separated floats, got %d — IGNORED\n", n);
            }
        }
    }
    if (have_env && device >= 0 && device < 64 && !uploaded[device]) {
        uploaded[device] = true;
        int cur = 0; cudaGetDevice(&cur);
        if (cur != device) cudaSetDevice(device);
        cudaMemcpyToSymbol(pxq5_book_c, envbook, sizeof(envbook));
        cudaMemcpyToSymbol(pxq5_book_g, envbook, sizeof(envbook));  // fast-kernel mirror (never diverges)
        if (cur != device) cudaSetDevice(cur);
    }
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant (PXQ5 -> f16/f32) — convert.cu fallback hook. Mirrors k_pxq4_dequant_matrix;
// numeric deltas: fp32 book LUT + scale-table decode.
// ---------------------------------------------------------------------------------------------
template <typename dst_t>
static __global__ void k_pxq5_dequant_matrix(const uint8_t * __restrict__ wq, dst_t * __restrict__ y,
                                             const int kslabs, const int64_t K) {
    __shared__ float tab[16];
    __shared__ float stab[256];
    if (threadIdx.x < 16) tab[threadIdx.x] = pxq5_book_c[threadIdx.x];
    #pragma unroll
    for (int i = threadIdx.x; i < 256; i += 64) stab[i] = pxq5_scale_tab_c[i];
    __syncthreads();
    const int64_t slab_id = blockIdx.x;
    const int64_t p  = slab_id / kslabs;
    const int     kb = (int)(slab_id % kslabs);
    const int     row = threadIdx.x;
    const uint8_t * slab = wq + slab_id*PXQ5_SLAB_BYTES;
    const float d = stab[slab[row]];
    dst_t * dst = y + (p*PXQ5_BM + row)*K + kb*PXQ5_QK;
    const uint4 q = *(const uint4 *)(slab + 64 + row*16);
    const uint8_t * qb = (const uint8_t *)&q;
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
        dst[2*b]   = (dst_t)(d * tab[qb[b] & 0xf]);
        dst[2*b+1] = (dst_t)(d * tab[qb[b] >> 4]);
    }
}

template <typename dst_t>
static void dequantize_row_pxq5_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    if (nrows % PXQ5_BM != 0 || n_per_row % PXQ5_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq5_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    int dev = -1; cudaGetDevice(&dev);
    pxq5_maybe_upload_book(dev);
    const int kslabs = (int)(n_per_row / PXQ5_QK);
    const int64_t nslabs = (nrows / PXQ5_BM) * (int64_t)kslabs;
    k_pxq5_dequant_matrix<dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

// ---------------------------------------------------------------------------------------------
// grouped fused prefill GEMM — identical tiling/accumulation to k_pxq4_gemm_grouped (PoC v2 shape:
// 64 thr, 8r x 8t/thread, strict k-order half2 accum). Numeric deltas only.
// ---------------------------------------------------------------------------------------------
static __global__ void __launch_bounds__(64)
k_pxq5_gemm_grouped(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                    const float * __restrict__ bias, const size_t bias_nb1,
                    const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ5_BM, kslabs = K / PXQ5_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * Wexp = W + ((size_t)(tile.e*panels + p)*kslabs)*PXQ5_SLAB_BYTES;
    const half    * At   = A + (size_t)tile.row0*K;
    float         * Ct   = C + (size_t)tile.row0*R + (size_t)p*PXQ5_BM;

    __shared__ float tab[16];
    __shared__ float stab[256];
    __shared__ half sW[PXQ5_QK][PXQ5_BM];
    __shared__ half sA[PXQ5_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    if (tid < 16) tab[tid] = pxq5_book_c[tid];
    #pragma unroll
    for (int i = tid; i < 256; i += 64) stab[i] = pxq5_scale_tab_c[i];

    const int tx = tid & 7, ty = tid >> 3;
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        {   // dequant W slab: 64 threads, one row each — scale via table, values via learned book
            const uint8_t * slab = Wexp + (size_t)kb*PXQ5_SLAB_BYTES;
            const float d = stab[slab[tid]];
            const uint4 q = *(const uint4 *)(slab + 64 + tid*16);
            const uint8_t * qb = (const uint8_t *)&q;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sW[2*b][tid]   = __float2half_rn(d * tab[qb[b] & 0xf]);
                sW[2*b+1][tid] = __float2half_rn(d * tab[qb[b] >> 4]);
            }
        }
        if (a_valid) {
            const half * src = At + (size_t)tid*K + kb*PXQ5_QK;
            uint4 v0 = *(const uint4 *)(src);
            uint4 v1 = *(const uint4 *)(src + 8);
            uint4 v2 = *(const uint4 *)(src + 16);
            uint4 v3 = *(const uint4 *)(src + 24);
            const half * h0 = (const half *)&v0; const half * h1 = (const half *)&v1;
            const half * h2 = (const half *)&v2; const half * h3 = (const half *)&v3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ5_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        #pragma unroll 4
        for (int kk = 0; kk < PXQ5_QK; ++kk) {
            half2 a2[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a2[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[2*i][j]   = __hfma2(wlo, a2[j], acc[2*i][j]);
                    acc[2*i+1][j] = __hfma2(whi, a2[j], acc[2*i+1][j]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ5_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows)     Ct[(size_t)t*R + row]     = __half2float(__low2half(acc[r][j]))  + b;
            if (t + 1 < tile.nrows) Ct[(size_t)(t+1)*R + row] = __half2float(__high2half(acc[r][j])) + b;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// decode (fast-TG) fused up+gate+glu mmv + down mmv — identical structure to the PXQ4 pair
// (grid (R/64, n_ids, Ny), 256 thr = 64 rows x 4 k-segs, x staged in smem, fp32 accum, no q8_1
// stage, no host syncs, graph-capturable). Numeric deltas only. NOTE: scale factoring is a plain
// fp32 multiply here (PXQ5 scales are not powers of two — the mmv already multiplies per-slab,
// so cost is identical to PXQ4's).
// ---------------------------------------------------------------------------------------------
static __global__ void __launch_bounds__(256)
k_pxq5_gateup_mmv(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
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

    extern __shared__ float pxq5_smem[];
    float * xs = pxq5_smem;
    float * red = pxq5_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[16];
    __shared__ float stab[256];
    if (threadIdx.x < 16) tab[threadIdx.x] = pxq5_book_c[threadIdx.x];
    if (threadIdx.x < 256) stab[threadIdx.x] = pxq5_scale_tab_c[threadIdx.x];
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ5_BM, kslabs = K / PXQ5_QK;
    const size_t slab0 = ((size_t)e*panels + p)*kslabs;

    float su = 0.f, sg = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ5_QK;
        {
            const uint8_t * slab = Wu + (slab0 + kb)*PXQ5_SLAB_BYTES;
            const float d = stab[slab[row]];
            const uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            float t = 0.f;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                t += tab[qb[b] & 0xf]*xk[2*b] + tab[qb[b] >> 4]*xk[2*b+1];
            }
            su += d*t;
        }
        {
            const uint8_t * slab = Wg + (slab0 + kb)*PXQ5_SLAB_BYTES;
            const float d = stab[slab[row]];
            const uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            float t = 0.f;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                t += tab[qb[b] & 0xf]*xk[2*b] + tab[qb[b] >> 4]*xk[2*b+1];
            }
            sg += d*t;
        }
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
        const int grow = p*PXQ5_BM + row;
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
        // glu epilogue: pxq4_glu_apply (pxq4.cuh) ported per the wiring note — handles both
        // SWIGLU_OAI (unary==1) and plain SILU-swiglu (unary==0, the qwen35moe/36 path).
        const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[grow] = r;
    }
}

static __global__ void __launch_bounds__(256)
k_pxq5_mmv(const uint8_t * __restrict__ W,
           const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
           char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
           const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
           const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq5_smem[];
    float * xs = pxq5_smem;
    float * red = pxq5_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[16];
    __shared__ float stab[256];
    if (threadIdx.x < 16) tab[threadIdx.x] = pxq5_book_c[threadIdx.x];
    if (threadIdx.x < 256) stab[threadIdx.x] = pxq5_scale_tab_c[threadIdx.x];
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ5_BM, kslabs = K / PXQ5_QK;
    const size_t slab0 = ((size_t)e*panels + p)*kslabs;

    float su = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ5_QK;
        const uint8_t * slab = W + (slab0 + kb)*PXQ5_SLAB_BYTES;
        const float d = stab[slab[row]];
        const uint4 q = *(const uint4 *)(slab + 64 + row*16);
        const uint8_t * qb = (const uint8_t *)&q;
        float t = 0.f;
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            t += tab[qb[b] & 0xf]*xk[2*b] + tab[qb[b] >> 4]*xk[2*b+1];
        }
        su += d*t;
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ5_BM + row] = u;
    }
}

// =============================================================================================
// PXA_PXQ5_FAST kernels — speed-only siblings of the proven set above. BIT-IDENTICAL numerics:
// the dot-product / accumulation / epilogue code is byte-for-byte the proven kernels'; the ONLY
// deltas are (a) the scale decode (exact 2^q * frt8[r] bit-construction, host-verified identical
// to the frozen table over all 255 values at enable time), (b) coalesced global staging of the
// 24 floats of tables instead of ~272 serialized divergent __constant__ reads per block, and
// (c) -1 KB static smem (+ conflict-free frt8[8] instead of stab[256] in the slab loop).
// Selected by the ggml-cuda.cu drivers when pxa_pxq5_fast(); default stays the proven path.
// =============================================================================================

static __global__ void __launch_bounds__(64)
k_pxq5_gemm_grouped_fast(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                         const float * __restrict__ bias, const size_t bias_nb1,
                         const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ5_BM, kslabs = K / PXQ5_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * Wexp = W + ((size_t)(tile.e*panels + p)*kslabs)*PXQ5_SLAB_BYTES;
    const half    * At   = A + (size_t)tile.row0*K;
    float         * Ct   = C + (size_t)tile.row0*R + (size_t)p*PXQ5_BM;

    __shared__ float tab[16];
    __shared__ float fr8[8];
    __shared__ half sW[PXQ5_QK][PXQ5_BM];
    __shared__ half sA[PXQ5_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    if      (tid < 16) tab[tid]      = pxq5_book_g[tid];
    else if (tid < 24) fr8[tid - 16] = pxq5_frt8_g[tid - 16];

    const int tx = tid & 7, ty = tid >> 3;
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        {   // dequant W slab: 64 threads, one row each — scale via exact bit-decode, values via book
            const uint8_t * slab = Wexp + (size_t)kb*PXQ5_SLAB_BYTES;
            const float d = pxq5_scale_fast(slab[tid], fr8);
            const uint4 q = *(const uint4 *)(slab + 64 + tid*16);
            const uint8_t * qb = (const uint8_t *)&q;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sW[2*b][tid]   = __float2half_rn(d * tab[qb[b] & 0xf]);
                sW[2*b+1][tid] = __float2half_rn(d * tab[qb[b] >> 4]);
            }
        }
        if (a_valid) {
            const half * src = At + (size_t)tid*K + kb*PXQ5_QK;
            uint4 v0 = *(const uint4 *)(src);
            uint4 v1 = *(const uint4 *)(src + 8);
            uint4 v2 = *(const uint4 *)(src + 16);
            uint4 v3 = *(const uint4 *)(src + 24);
            const half * h0 = (const half *)&v0; const half * h1 = (const half *)&v1;
            const half * h2 = (const half *)&v2; const half * h3 = (const half *)&v3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ5_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        #pragma unroll 4
        for (int kk = 0; kk < PXQ5_QK; ++kk) {
            half2 a2[4];
            #pragma unroll
            for (int j = 0; j < 4; ++j) a2[j] = *(const half2 *)&sA[kk][8*ty + 2*j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const half2 wp  = *(const half2 *)&sW[kk][8*tx + 2*i];
                const half2 wlo = __low2half2(wp), whi = __high2half2(wp);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[2*i][j]   = __hfma2(wlo, a2[j], acc[2*i][j]);
                    acc[2*i+1][j] = __hfma2(whi, a2[j], acc[2*i+1][j]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        const int row = 8*tx + r;
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ5_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows)     Ct[(size_t)t*R + row]     = __half2float(__low2half(acc[r][j]))  + b;
            if (t + 1 < tile.nrows) Ct[(size_t)(t+1)*R + row] = __half2float(__high2half(acc[r][j])) + b;
        }
    }
}

static __global__ void __launch_bounds__(256)
k_pxq5_gateup_mmv_fast(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
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

    extern __shared__ float pxq5_smem[];
    float * xs = pxq5_smem;
    float * red = pxq5_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[16];
    __shared__ float fr8[8];
    if      (threadIdx.x < 16) tab[threadIdx.x]      = pxq5_book_g[threadIdx.x];
    else if (threadIdx.x < 24) fr8[threadIdx.x - 16] = pxq5_frt8_g[threadIdx.x - 16];
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ5_BM, kslabs = K / PXQ5_QK;
    const size_t slab0 = ((size_t)e*panels + p)*kslabs;

    float su = 0.f, sg = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ5_QK;
        {
            const uint8_t * slab = Wu + (slab0 + kb)*PXQ5_SLAB_BYTES;
            const float d = pxq5_scale_fast(slab[row], fr8);
            const uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            float t = 0.f;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                t += tab[qb[b] & 0xf]*xk[2*b] + tab[qb[b] >> 4]*xk[2*b+1];
            }
            su += d*t;
        }
        {
            const uint8_t * slab = Wg + (slab0 + kb)*PXQ5_SLAB_BYTES;
            const float d = pxq5_scale_fast(slab[row], fr8);
            const uint4 q = *(const uint4 *)(slab + 64 + row*16);
            const uint8_t * qb = (const uint8_t *)&q;
            float t = 0.f;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                t += tab[qb[b] & 0xf]*xk[2*b] + tab[qb[b] >> 4]*xk[2*b+1];
            }
            sg += d*t;
        }
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
        const int grow = p*PXQ5_BM + row;
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
        const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[grow] = r;
    }
}

static __global__ void __launch_bounds__(256)
k_pxq5_mmv_fast(const uint8_t * __restrict__ W,
                const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
                char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
                const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
                const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq5_smem[];
    float * xs = pxq5_smem;
    float * red = pxq5_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[16];
    __shared__ float fr8[8];
    if      (threadIdx.x < 16) tab[threadIdx.x]      = pxq5_book_g[threadIdx.x];
    else if (threadIdx.x < 24) fr8[threadIdx.x - 16] = pxq5_frt8_g[threadIdx.x - 16];
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ5_BM, kslabs = K / PXQ5_QK;
    const size_t slab0 = ((size_t)e*panels + p)*kslabs;

    float su = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ5_QK;
        const uint8_t * slab = W + (slab0 + kb)*PXQ5_SLAB_BYTES;
        const float d = pxq5_scale_fast(slab[row], fr8);
        const uint4 q = *(const uint4 *)(slab + 64 + row*16);
        const uint8_t * qb = (const uint8_t *)&q;
        float t = 0.f;
        #pragma unroll
        for (int b = 0; b < 16; ++b) {
            t += tab[qb[b] & 0xf]*xk[2*b] + tab[qb[b] >> 4]*xk[2*b+1];
        }
        su += d*t;
    }
    red[kseg*64 + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s*64 + row];
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[p*PXQ5_BM + row] = u;
    }
}
