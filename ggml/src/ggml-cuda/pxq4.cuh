// pxq4.cuh — PXQ4: the PXA-native quant format (MXFP4 numerics, GEMM-tile-ordered layout).
//
// Design: pxa-llama-fork/PXA-QUANT-DESIGN-2026-07-15.md (PXACLAW repo). PoC: pxa-bench/pxq4_bench.cu
// (measured 2.53x vs the prod dequant+cublas16F path on P100 at live gpt-oss expert shapes).
//
// On-disk layout (v1, frozen):
//   slab = 64 weight rows x 32 K-values = 1088 B, 64 B aligned:
//     [64 B scales]   s[row] = E8M0 byte for (row, this 32-K block)      (SoA, coalesced)
//     [1024 B nibbles] row r = 16 B at offset 64 + 16*r
//                      byte b of a row = code(k=2b) lo | code(k=2b+1) hi (sequential pairs)
//   slabs K-major within a 64-row panel; panels row-major; experts outermost.
//   Constraints: rows % 64 == 0, K % 32 == 0 (the repack tool enforces; other tensors stay MXFP4).
//   Numerics are EXACTLY MXFP4 (E2M1 codes x E8M0 power-of-two scale): MXFP4 <-> PXQ4 is a lossless
//   bit permutation. type_size=17 / blck_size=32 keeps ggml_row_size/nbytes arithmetic identical.
//
// Env: PXA_PXQ4=0 disables the fused kernels (PXQ4 tensors then run dequant->cublas fallback,
// which is bit-identical f16 to the MXFP4 path). Default ON.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>
#include <cstdio>

#define PXQ4_QK         32
#define PXQ4_TYPE_SIZE  17
#define PXQ4_BM         64                 // rows per panel (= fused kernel tile height)
#define PXQ4_BN         64                 // token tile width
#define PXQ4_SLAB_BYTES 1088

// E2M1 code values (same table as kvalues_mxfp4)
static __device__ __constant__ int8_t pxq4_kvalues[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12};

static __device__ __forceinline__ float pxq4_e8m0_to_f32(uint8_t e) {
    union { float f; uint32_t u; } h;
    h.u = e >= 2 ? (uint32_t)(e - 1) << 23 : (e ? 0x00400000u : 0x00200000u);
    return h.f;   // always an exact power of two
}

static inline bool pxa_pxq4_enabled() {
    static const bool on = [](){
        const char * e = getenv("PXA_PXQ4");
        bool v = !(e && atoi(e) == 0);
        fprintf(stderr, "PXA_PXQ4 fused kernels: %s\n", v ? "ON (PXA_PXQ4=0 disables)" : "OFF");
        return v;
    }();
    return on;
}

// per-(expert, token-tile) descriptor for the grouped prefill GEMM
struct pxq4_tile_info {
    int32_t e;      // expert id
    int32_t row0;   // first flat activation row of this tile (expert-grouped order)
    int32_t nrows;  // 1..64 valid tokens in this tile
    int32_t _pad;
};

// layout-compatible with ggml-cuda.cu's mmid_row_mapping {int32 i1, i2}
struct pxq4_rowmap { int32_t i1, i2; };

// ---------------------------------------------------------------------------------------------
// full-matrix dequant (PXQ4 -> f16/f32), row-major output [nrows][K].
// grid.x = (nrows/64)*(K/32) slabs, 64 threads. Used by the convert.cu fallback hooks
// (dequant -> cublas keeps PXQ4 functional on every arch with the fused kernels off).
// Bit-identical to dequantize_block_mxfp4 on the same underlying MXFP4 codes.
// ---------------------------------------------------------------------------------------------
template <typename dst_t>
static __global__ void k_pxq4_dequant_matrix(const uint8_t * __restrict__ wq, dst_t * __restrict__ y,
                                             const int kslabs, const int64_t K) {
    __shared__ float tab[16];
    if (threadIdx.x < 16) tab[threadIdx.x] = (float)pxq4_kvalues[threadIdx.x];
    __syncthreads();
    const int64_t slab_id = blockIdx.x;
    const int64_t p  = slab_id / kslabs;
    const int     kb = (int)(slab_id % kslabs);
    const int     row = threadIdx.x;
    const uint8_t * slab = wq + slab_id*PXQ4_SLAB_BYTES;
    const float d = pxq4_e8m0_to_f32(slab[row]);
    dst_t * dst = y + (p*PXQ4_BM + row)*K + kb*PXQ4_QK;
    const uint4 q = *(const uint4 *)(slab + 64 + row*16);
    const uint8_t * qb = (const uint8_t *)&q;
    #pragma unroll
    for (int b = 0; b < 16; ++b) {
        dst[2*b]   = (dst_t)(d * tab[qb[b] & 0xf]);
        dst[2*b+1] = (dst_t)(d * tab[qb[b] >> 4]);
    }
}

template <typename dst_t>
static void dequantize_row_pxq4_cuda(const void * vx, dst_t * y, const int64_t nrows, const int64_t n_per_row,
                                     cudaStream_t stream) {
    // PXQ4 is panel-interleaved: only whole-matrix (nrows%64==0, base panel-aligned) dequant is defined.
    if (nrows % PXQ4_BM != 0 || n_per_row % PXQ4_QK != 0) {
        fprintf(stderr, "FATAL: dequantize_row_pxq4_cuda: nrows=%lld n_per_row=%lld not slab-aligned\n",
                (long long)nrows, (long long)n_per_row);
        abort();
    }
    const int kslabs = (int)(n_per_row / PXQ4_QK);
    const int64_t nslabs = (nrows / PXQ4_BM) * (int64_t)kslabs;
    k_pxq4_dequant_matrix<dst_t><<<nslabs, 64, 0, stream>>>((const uint8_t *)vx, y, kslabs, n_per_row);
}

// ---------------------------------------------------------------------------------------------
// activation gather: flat expert-grouped f32 rows -> contiguous f16 [total][K]
// (same mapping semantics as k_copy_src_to_contiguous, plus the f32->f16 convert the
//  cublas fp16 path does anyway — so the fused path sees the same f16 activations.)
// ---------------------------------------------------------------------------------------------
static __global__ void k_pxq4_gather_a_f16(const char * __restrict__ src1, half * __restrict__ A,
                                           const pxq4_rowmap * __restrict__ map,
                                           const int64_t K, const int64_t ne11,
                                           const size_t nb11, const size_t nb12) {
    const int i = blockIdx.x;
    const int32_t i11 = map[i].i1 % ne11;
    const int32_t i12 = map[i].i2;
    const float * src = (const float *)(src1 + i11*nb11 + i12*nb12);
    half * dst = A + (size_t)i*K;
    for (int64_t j = threadIdx.x; j < K; j += blockDim.x) {
        dst[j] = __float2half(src[j]);
    }
}

// ---------------------------------------------------------------------------------------------
// grouped fused prefill GEMM (the PoC's winning v2 shape: 64 threads, 8 rows x 8 tokens/thread,
// strict k-order half2 accumulation — same precision class as the prod CUBLAS_COMPUTE_16F path),
// extended for production: ragged token tiles (masked A loads / C stores), f32 output in the
// flat [token][R] layout the MoE epilogue expects, optional per-expert bias folded in.
// grid: (R/64 panels, n_tiles); block: 64 threads.
// ---------------------------------------------------------------------------------------------
static __global__ void __launch_bounds__(64)
k_pxq4_gemm_grouped(const uint8_t * __restrict__ W, const half * __restrict__ A, float * __restrict__ C,
                    const float * __restrict__ bias, const size_t bias_nb1,
                    const pxq4_tile_info * __restrict__ tiles, const int R, const int K) {
    const int panels = R / PXQ4_BM, kslabs = K / PXQ4_QK;
    const int p = blockIdx.x;
    const pxq4_tile_info tile = tiles[blockIdx.y];
    const uint8_t * Wexp = W + ((size_t)(tile.e*panels + p)*kslabs)*PXQ4_SLAB_BYTES;
    const half    * At   = A + (size_t)tile.row0*K;
    float         * Ct   = C + (size_t)tile.row0*R + (size_t)p*PXQ4_BM;

    __shared__ float tab[16];
    __shared__ half sW[PXQ4_QK][PXQ4_BM];
    __shared__ half sA[PXQ4_QK][PXQ4_BN];
    const int tid = threadIdx.x;
    if (tid < 16) tab[tid] = (float)pxq4_kvalues[tid];

    const int tx = tid & 7, ty = tid >> 3;      // 8 row-groups x 8 col-groups
    half2 acc[8][4];
    #pragma unroll
    for (int r = 0; r < 8; ++r)
        #pragma unroll
        for (int j = 0; j < 4; ++j) acc[r][j] = __floats2half2_rn(0.f, 0.f);

    const bool a_valid = tid < tile.nrows;

    for (int kb = 0; kb < kslabs; ++kb) {
        __syncthreads();
        {   // dequant W slab: 64 threads, one row each
            const uint8_t * slab = Wexp + (size_t)kb*PXQ4_SLAB_BYTES;
            const float d = pxq4_e8m0_to_f32(slab[tid]);
            const uint4 q = *(const uint4 *)(slab + 64 + tid*16);
            const uint8_t * qb = (const uint8_t *)&q;
            #pragma unroll
            for (int b = 0; b < 16; ++b) {
                sW[2*b][tid]   = __float2half_rn(d * tab[qb[b] & 0xf]);
                sW[2*b+1][tid] = __float2half_rn(d * tab[qb[b] >> 4]);
            }
        }
        if (a_valid) {   // A tile: one token per thread, 32 halves (64 B)
            const half * src = At + (size_t)tid*K + kb*PXQ4_QK;
            uint4 v0 = *(const uint4 *)(src);
            uint4 v1 = *(const uint4 *)(src + 8);
            uint4 v2 = *(const uint4 *)(src + 16);
            uint4 v3 = *(const uint4 *)(src + 24);
            const half * h0 = (const half *)&v0; const half * h1 = (const half *)&v1;
            const half * h2 = (const half *)&v2; const half * h3 = (const half *)&v3;
            #pragma unroll
            for (int i = 0; i < 8; ++i) { sA[i][tid] = h0[i]; sA[8+i][tid] = h1[i]; sA[16+i][tid] = h2[i]; sA[24+i][tid] = h3[i]; }
        } else {         // ragged tail: zero-fill so the FMA loop is mask-free
            const half hz = __float2half_rn(0.f);
            #pragma unroll
            for (int i = 0; i < PXQ4_QK; ++i) sA[i][tid] = hz;
        }
        __syncthreads();
        #pragma unroll 4
        for (int kk = 0; kk < PXQ4_QK; ++kk) {
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
        const float b = bias ? *(const float *)((const char *)bias + (size_t)tile.e*bias_nb1 + (size_t)(p*PXQ4_BM + row)*sizeof(float)) : 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int t = 8*ty + 2*j;
            if (t < tile.nrows)     Ct[(size_t)t*R + row]     = __half2float(__low2half(acc[r][j]))  + b;
            if (t + 1 < tile.nrows) Ct[(size_t)(t+1)*R + row] = __half2float(__high2half(acc[r][j])) + b;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// PXQ4 GLU epilogue: unary==1 => SWIGLU_OAI (gpt-oss: alpha=1.702, gate clamped/silu'd, up=(1+.)),
// unary==0 => plain SILU-swiglu (qwen: silu(gate)*up, with the exact limit<1e-6 branch prod uses).
// xg = GATE buffer, gu = UP buffer (biases already folded into the GEMM epilogues). Output f16/f32
// feeds the PXQ4 down-proj GEMM. Bit-faithful to unary.cu swiglu_oai_kernel / fused_mul_silu_f32.
// ---------------------------------------------------------------------------------------------
static __device__ __forceinline__ float pxq4_glu_apply(float g, float u, int unary, float alpha, float limit) {
    if (unary == 1) {                    // SWIGLU_OAI
        float gi = fminf(g, limit);
        float ui = fmaxf(fminf(u, limit), -limit);
        return gi / (1.0f + expf(-gi * alpha)) * (1.0f + ui);
    }
    // SILU-swiglu
    if (limit < 1e-6f) return (g / (1.0f + expf(-g))) * u;
    float gs = g / (1.0f + expf(-g));
    gs = fminf(gs, limit);
    return gs * fmaxf(-limit, fminf(limit, u));
}

template <typename dst_t>
static __global__ void k_pxq4_glu(const float * __restrict__ xg, const float * __restrict__ gu,
                                  dst_t * __restrict__ dst, const int64_t k,
                                  const int unary, const float alpha, const float limit) {
    const int64_t i = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = (dst_t)pxq4_glu_apply(xg[i], gu[i], unary, alpha, limit);
}

// ---------------------------------------------------------------------------------------------
// decode (fast-TG) fused up+gate+swiglu_oai mmv. No q8_1 stage, no host syncs: f32 activations
// in, ids read on-device, f32 out. Weights stream once at 4.25 bpw (coalesced by construction:
// 64 rows x 16 B = one contiguous 1 KiB nibble read + one 64 B scale read per slab).
// grid: (R/64 panels, n_ids, Ny); block: 256 = 64 rows x 4 k-segments; smem: K floats + reduce.
// Scale factoring is exact (E8M0 scales are powers of two).
// ---------------------------------------------------------------------------------------------
#define PXQ4_MMV_KSEG 4
static __global__ void __launch_bounds__(256)
k_pxq4_gateup_mmv(const uint8_t * __restrict__ Wu, const uint8_t * __restrict__ Wg,
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
    if (e < 0 || e >= n_as) return;   // SER slot: leave dst untouched (matches mul_mat_vec_q)

    extern __shared__ float pxq4_smem[];
    float * xs = pxq4_smem;                       // K floats
    float * red = pxq4_smem + K;                  // 2 * 64 * KSEG floats

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[16];
    if (threadIdx.x < 16) tab[threadIdx.x] = (float)pxq4_kvalues[threadIdx.x];
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ4_BM, kslabs = K / PXQ4_QK;
    const size_t slab0 = ((size_t)e*panels + p)*kslabs;

    float su = 0.f, sg = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ4_QK;
        {
            const uint8_t * slab = Wu + (slab0 + kb)*PXQ4_SLAB_BYTES;
            const float d = pxq4_e8m0_to_f32(slab[row]);
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
            const uint8_t * slab = Wg + (slab0 + kb)*PXQ4_SLAB_BYTES;
            const float d = pxq4_e8m0_to_f32(slab[row]);
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
    red[(kseg*64 + row)]                       = su;
    red[(PXQ4_MMV_KSEG*64) + (kseg*64 + row)]  = sg;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f, g = 0.f;
        #pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) {
            u += red[s*64 + row];
            g += red[PXQ4_MMV_KSEG*64 + s*64 + row];
        }
        const int grow = p*PXQ4_BM + row;
        if (bias_u) u += *(const float *)((const char *)bias_u + (size_t)e*bias_u_nb1 + (size_t)grow*sizeof(float));
        if (bias_g) g += *(const float *)((const char *)bias_g + (size_t)e*bias_g_nb1 + (size_t)grow*sizeof(float));
        // GLU epilogue: gate = silu side, up = linear side (both OAI and plain SILU handled)
        const float r = pxq4_glu_apply(g, u, unary, alpha, limit);
        float * out = (float *)(dst_base + (size_t)iy*dst_tok_stride + (size_t)j*dst_slot_stride);
        out[grow] = r;
    }
}

// decode down-proj mmv: out[(iy,j)] = W[e(iy,j)] . x[(iy,j)]   (x = the gateup output, f32)
static __global__ void __launch_bounds__(256)
k_pxq4_mmv(const uint8_t * __restrict__ W,
           const char * __restrict__ x_base, const size_t x_tok_stride, const size_t x_slot_stride,
           char * __restrict__ dst_base, const size_t dst_tok_stride, const size_t dst_slot_stride,
           const char * __restrict__ ids, const size_t ids_nb0, const size_t ids_nb1,
           const int R, const int K, const int n_as) {
    const int p  = blockIdx.x;
    const int j  = blockIdx.y;
    const int iy = blockIdx.z;
    const int e  = *(const int32_t *)(ids + (size_t)iy*ids_nb1 + (size_t)j*ids_nb0);
    if (e < 0 || e >= n_as) return;

    extern __shared__ float pxq4_smem[];
    float * xs = pxq4_smem;
    float * red = pxq4_smem + K;

    const float * x = (const float *)(x_base + (size_t)iy*x_tok_stride + (size_t)j*x_slot_stride);
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) xs[idx] = x[idx];

    __shared__ float tab[16];
    if (threadIdx.x < 16) tab[threadIdx.x] = (float)pxq4_kvalues[threadIdx.x];
    __syncthreads();

    const int row  = threadIdx.x & 63;
    const int kseg = threadIdx.x >> 6;
    const int panels = R / PXQ4_BM, kslabs = K / PXQ4_QK;
    const size_t slab0 = ((size_t)e*panels + p)*kslabs;

    float su = 0.f;
    for (int kb = kseg; kb < kslabs; kb += PXQ4_MMV_KSEG) {
        const float * xk = xs + kb*PXQ4_QK;
        const uint8_t * slab = W + (slab0 + kb)*PXQ4_SLAB_BYTES;
        const float d = pxq4_e8m0_to_f32(slab[row]);
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
        out[p*PXQ4_BM + row] = u;
    }
}
