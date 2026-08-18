// pxq4_kernel.cuh — PXQ4 (ggml type id 252) device decode primitives for sm_70, vendored from
// the pxq_llama engine and re-addressed for vLLM's two-tensor parameter split.
//
// PROVENANCE. Every device function below is a line-for-line copy of the corresponding
// function in <local-path> (read-only; nothing in that
// tree was modified). The ONLY permitted edits are ADDRESSING edits, listed exhaustively here:
//
//   1. The engine stores one contiguous blob per tensor and computes
//        panel = W + (e*panels + p) * (HDR + kslabs*SLAB)
//        slab  = panel + HDR + kb*SLAB
//        anchor= ((const half*)panel)[row]
//      vLLM's stock weight loaders can only narrow() a single declared dim, so the converter
//      splits the blob into two tensors (plan §5.3):
//        slabs  uint8   [P, S, 1088]   P = N/64 panels, S = K/32 slabs
//        anchor float16 [P, 64]
//      giving
//        slab   = slabs  + (p*kslabs + kb) * SLAB
//        anchor = anchor[p*64 + row]
//      This is a pure re-addressing of the SAME bytes in the SAME order — the split is a
//      memcpy partition performed offline, asserted round-trip-exact by the converter.
//
//   2. The expert axis (e, ids[], n_as) is deleted. This model has no MoE: the artifact has
//      866 tensors, zero *_exps, zero expert KVs. e is always 0.
//
//   3. The mmv stages its activation slice per CANONICAL CHUNK instead of staging the whole
//      K vector once. See k_pxq4_mmv for why this is bit-exact and why it matters.
//
//   4. Activations arrive fp16 and results leave fp16 (vLLM's linear dtype) instead of fp32.
//      See k_pxq4_mmv for the exactness argument.
//
// NOT vendored (deliberately dropped): every MoE kernel (gufuse / down_scat / grouped / WMMA),
// the HQ policy pxq6_pol_p6hq, the PXQ2/PXQ3/PXQ6R policies, the int8/MMVQ family, the K-split
// workspace kernels (they cudaMalloc and decline under stream capture — pxq6.cuh:2480-2494 —
// which is disqualifying under vLLM's FULL_AND_PIECEWISE cuda graphs), and every decode MODE
// variant except MODE_TAB. All engine env gates that select the other modes default to OFF
// (pxq6.cuh:140-152), so MODE_TAB + VECX is exactly what the shipping engine runs on sm_70
// (selection logic: pxq6.cuh:3393-3400, VECX default true at :141).

#pragma once

#include <cuda_fp16.h>
#include <stdint.h>
#include "pxq4_kernel_tables.h"

#ifndef PXQ4_EXTERN_SHARED
#define PXQ4_EXTERN_SHARED extern __shared__ __align__(16)
#endif

// ---------------------------------------------------------------------------------------------
// device-resident tables. One copy per translation unit, as in the engine (pxq6.cuh:79-81).
// Initialised from the frozen literals; pxq4_upload_tables() may overwrite them from the values
// recorded in the checkpoint's gguf KVs pxa.pxq6.book / pxa.pxq6.sub, which is what the engine
// does when PXA_PXQ6_BOOK / PXA_PXQ6_SUB were set at quantize time.
// ---------------------------------------------------------------------------------------------
static __device__ float pxq4_book_g[16]  = PXQ4_BOOK_INIT;
static __device__ float pxq4_sub16_g[16] = PXQ4_SUB16_INIT;

// ---------------------------------------------------------------------------------------------
// format policy — pxq6_pol_p6, pxq6.cuh:317-346, with the panel-relative anchor() accessor
// removed (the anchor now arrives as its own tensor; see addressing edit 1).
// ---------------------------------------------------------------------------------------------
struct pxq4_pol {
    static constexpr int SLAB       = PXQ4_SLAB_BYTES;
    static constexpr int CODE_OFF   = PXQ4_CODE_OFF;
    static constexpr int CODE_BYTES = PXQ4_CODE_BYTES;
    static constexpr int CODE_WORDS = 4;
    static constexpr int NEFF       = PXQ4_NEFF;

    // tid < 16 stages the book, 16 <= tid < 32 stages the sublevels. Callers must have at
    // least 32 threads and must __syncthreads() before reading tab/sub.
    __device__ static void stage_tabs(float * tab, float * sub, int tid) {
        if      (tid < 16) tab[tid]      = pxq4_book_g[tid];
        else if (tid < 32) sub[tid - 16] = pxq4_sub16_g[tid - 16];
    }

    // The two effective scales of one (row, 32-column block). The engine's parity-locked
    // dequant contract is eff = fp32(anchor_fp16) * SUB16[s4], then w = eff * fp32(book[c]);
    // the multiply ORDER is load-bearing for bit-exactness and must not be reassociated
    // (pxq-cpu.h:16-18, ggml-pxq6-tables.h:7-9).
    __device__ static void row_effs(const uint8_t * slab, int row, float anch,
                                    const float * sub, float * eff) {
        const int sb = slab[row];
        eff[0] = anch * sub[sb & 0xf];    // elements  0-15 of this 32-element block
        eff[1] = anch * sub[sb >> 4];     // elements 16-31
    }

    // one code byte -> (book[low nibble], book[high nibble]). b indexes the LE byte of the
    // 16-byte code row held in q[0..3]; element 2b takes the low nibble, 2b+1 the high.
    __device__ static float2 pair(const uint32_t * q, int b, const float * tab) {
        const int byte = (q[b >> 2] >> (8 * (b & 3))) & 0xff;
        return make_float2(tab[byte & 0xf], tab[byte >> 4]);
    }
};

// per-row code load: one 16-byte vector load. The address is
//   slab_base + 64 + row*16, and slab_base is a multiple of 1088 = 68*16 from a
// torch-allocated (>=256 B aligned) tensor base, so the uint4 load is always 16-B aligned.
// The launcher asserts contiguity, which is what makes that hold. (pxq6.cuh:436-441)
static __device__ __forceinline__ void pxq4_ldcodes(const uint8_t * p, uint32_t * q) {
    *(uint4 *)q = *(const uint4 *)p;
}

// The inner accumulation shape. PXQ_CANON_V2 == 0 is the shipping form; it emits
// FMUL+FFMA+FADD per pair, and BOTH forms are deterministic, but they are NOT bit-identical
// to each other, so this constant is a build-time re-baselining switch, never a runtime flag.
// (pxq6.cuh:575-608)
static __device__ __forceinline__ float pxq4_acc2(float acc, float a0, float x0,
                                                  float a1, float x1) {
#if PXQ4_CANON_V2
    return __fmaf_rn(a1, x1, __fmaf_rn(a0, x0, acc));
#else
    return acc + (a0 * x0 + a1 * x1);
#endif
}

// dot product of one weight row's 32-element block against 32 activations.
// (pxq6.cuh:634-674, MODE_TAB arm only)
template <bool VECX>
static __device__ __forceinline__ float pxq4_dot32(const uint8_t * __restrict__ slab, int row,
                                                   float anch,
                                                   const float * __restrict__ xk,
                                                   const float * __restrict__ tab,
                                                   const float * __restrict__ sub) {
    float eff[PXQ4_NEFF];
    pxq4_pol::row_effs(slab, row, anch, sub, eff);
    uint32_t q[4];
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + row * pxq4_pol::CODE_BYTES, q);

    // Two partial sums: t[0] accumulates the eff[0] half (elements 0-15, i.e. byte pairs
    // b = 0..7), t[1] the eff[1] half. `(b*NEFF) >> 4` is the engine's index expression;
    // with NEFF == 2 it is exactly (b >= 8).
    float t[PXQ4_NEFF];
#pragma unroll
    for (int i = 0; i < PXQ4_NEFF; ++i) t[i] = 0.f;

    if (VECX) {
        // float4 activation loads. Requires &xk[0] to be 16-B aligned, which holds because
        // every caller passes a 32-float-aligned offset into a 16-B-aligned base.
#pragma unroll
        for (int b = 0; b < 16; b += 2) {
            const float4 xv = *(const float4 *)&xk[2 * b];
            const float2 p0 = pxq4_pol::pair(q, b,     tab);
            const float2 p1 = pxq4_pol::pair(q, b + 1, tab);
            t[(b * PXQ4_NEFF) >> 4]       = pxq4_acc2(t[(b * PXQ4_NEFF) >> 4],       p0.x, xv.x, p0.y, xv.y);
            t[((b + 1) * PXQ4_NEFF) >> 4] = pxq4_acc2(t[((b + 1) * PXQ4_NEFF) >> 4], p1.x, xv.z, p1.y, xv.w);
        }
    } else {
#pragma unroll
        for (int b = 0; b < 16; ++b) {
            const float2 p = pxq4_pol::pair(q, b, tab);
            t[(b * PXQ4_NEFF) >> 4] = pxq4_acc2(t[(b * PXQ4_NEFF) >> 4], p.x, xk[2 * b], p.y, xk[2 * b + 1]);
        }
    }
    return eff[0] * t[0] + eff[1] * t[1];
}

// Canonical chunk count: a function of SHAPE ONLY. That is the property that makes a K-split
// launch bit-identical to an unsplit one in the engine, and it is why we reproduce it here —
// it is the only way our mmv output matches the shipping engine's byte-for-byte.
// (pxq6.cuh:826-833)
static __host__ __device__ __forceinline__ int pxq4_canon_nfix(int kslabs, int cmax) {
    int lim = kslabs / PXQ4_MMV_KSEG;   // >= KSEG slabs per chunk so every lane stays busy
    if (lim < 1)    lim = 1;
    if (lim > cmax) lim = cmax;
    int n = 1;
    while (n * 2 <= lim) n *= 2;        // largest power of two <= lim
    return n;
}

// Upper bound on the slabs in any one canonical chunk; the mmv's dynamic smem is sized from it.
static __host__ __forceinline__ int pxq4_canon_max_chunk(int kslabs) {
    const int nfix = pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    return (kslabs + nfix - 1) / nfix;
}

// ---------------------------------------------------------------------------------------------
// full-matrix dequant. One block per slab, 64 threads, one row each.
// (pxq6.cuh:680-726, with addressing edit 1 and the expert axis removed)
//
// out is [N, K] row-major fp16 — vLLM's weight layout for torch.mm(x, w.t()).
// ---------------------------------------------------------------------------------------------
template <typename dst_t>
static __global__ void k_pxq4_dequant_matrix(const uint8_t * __restrict__ slabs,
                                             const __half  * __restrict__ anchor,
                                             dst_t * __restrict__ y,
                                             const int kslabs, const int64_t K) {
    __shared__ float tab[16];
    __shared__ float sub[16];

    // STORE COALESCING (engine change of 2026-07-27, kept verbatim). Decoding is naturally
    // row-major — one thread owns one row and produces 32 consecutive outputs — but storing
    // that straight to y has 32 threads writing addresses K elements apart, so each store
    // instruction touches 32 sectors to deliver 64 useful bytes. Stage the 64x32 tile in smem
    // and write it back along K instead. Same values, same addresses, different instruction
    // mapping => bit-identical.
    //
    // The +2 pad makes the row stride 34, i.e. 17 four-byte banks for a 2-byte dst_t;
    // gcd(17, 32) == 1, so the column-major fill below is bank-conflict-free.
    __shared__ dst_t tile[PXQ4_BM][PXQ4_QK + 2];

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);
    __syncthreads();

    const int64_t slab_id = blockIdx.x;
    const int64_t p       = slab_id / kslabs;
    const int     kb      = (int)(slab_id % kslabs);
    const int     row     = threadIdx.x;

    const uint8_t * slab = slabs + ((size_t)p * kslabs + kb) * pxq4_pol::SLAB;
    const float     anch = __half2float(anchor[(size_t)p * PXQ4_BM + row]);

    float eff[PXQ4_NEFF];
    pxq4_pol::row_effs(slab, row, anch, sub, eff);
    uint32_t q[4];
    pxq4_ldcodes(slab + pxq4_pol::CODE_OFF + row * pxq4_pol::CODE_BYTES, q);

#pragma unroll
    for (int b = 0; b < 16; ++b) {                 // b = element-pair index
        const float  e = eff[(b * PXQ4_NEFF) >> 4];
        const float2 v = pxq4_pol::pair(q, b, tab);
        tile[row][2 * b]     = (dst_t)(e * v.x);
        tile[row][2 * b + 1] = (dst_t)(e * v.y);
    }
    __syncthreads();

    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    const int nwarp = blockDim.x  >> 5;
    for (int r = warp; r < PXQ4_BM; r += nwarp) {
        y[(p * PXQ4_BM + r) * K + kb * PXQ4_QK + lane] = tile[r][lane];
    }
}

// ---------------------------------------------------------------------------------------------
// decode matrix-vector: out[M, N] = x[M, K] * W[N, K]^T, for small M.
// (pxq6.cuh:914-971, k_pxq6_mmv, with addressing edits 1-4)
//
// grid  = (N/64, M)      block = PXQ4_MMV_KSEG * 64 = 256 threads
// dynamic smem = pxq4_canon_max_chunk(K/32) * 32 floats
//
// EDIT 3 — CHUNKED ACTIVATION STAGING, and why it is bit-exact.
// The engine stages the whole K-vector in smem, which caps K at (46 KiB - 1 KiB)/4 = 11520
// floats and forces every wider node (a dense ffn_down is K = 17408) onto a K-split kernel
// that allocates a workspace with a raw cudaMalloc — which declines under stream capture and
// is therefore unusable inside vLLM's FULL_AND_PIECEWISE cuda graphs. We stage per CANONICAL
// CHUNK instead. The canonical chunk boundaries b0/b1 already exist in the engine's fold
// (they are what makes split == unsplit); staging inside that loop changes only WHEN an
// activation is copied to smem, never which activations are multiplied by which weights, in
// what order, or with what rounding. The fold is therefore byte-identical to the engine's
// unsplit kernel, and the smem bound drops to ceil(kslabs/nfix)*128 bytes — 4352 B at
// K = 17408, i.e. the cliff disappears entirely rather than being worked around.
//
// EDIT 4 — fp16 activation intake and fp16 result.
// The engine's mmv takes fp32 x and writes fp32. vLLM hands us fp16 activations and wants an
// fp16 result. Converting half->float on the smem staging write is EXACT (every fp16 is
// representable in fp32), so the values fed to the fold are identical to what an fp32 x
// holding the same numbers would have supplied. The result is folded in fp32 exactly as in
// the engine and rounded once, at the end, with __float2half_rn — the same single rounding
// vLLM would apply anyway when storing an fp32 result into an fp16 output tensor. Net effect:
// the fp16 x also halves the activation bytes read per block, which matters because every
// block re-reads the whole activation vector.
//
// NOTE ON M: the weight is re-read once per token (grid.y), so this kernel is only a win for
// very small M; the caller must cap M (PXQ4_MMV_MAX_M, default 8, matching the engine's
// PXA_PXQ4_2D_MAX_NY at ggml-cuda.cu:4019-4021) and use dequant + GEMM above that.
// ---------------------------------------------------------------------------------------------
template <bool VECX>
static __global__ void __launch_bounds__(256)
k_pxq4_mmv(const uint8_t * __restrict__ slabs,
           const __half  * __restrict__ anchor,
           const __half  * __restrict__ x,          // [M, K]
           __half        * __restrict__ out,        // [M, N]
           const int R, const int K) {
    const int p  = blockIdx.x;                      // panel = 64 output rows
    const int iy = blockIdx.y;                      // token

    // Dynamic shared memory. Routed through a macro so the host simulator
    // (pxq4_kernel_hostsim.cpp, a test-only artifact) can bind the same name to a global
    // arena; under nvcc this expands to the ordinary CUDA declaration.
    PXQ4_EXTERN_SHARED float pxq4_xs[];

    __shared__ float tab[16];
    __shared__ float sub[16];
    __shared__ float red[PXQ4_MMV_KSEG * PXQ4_BM];

    pxq4_pol::stage_tabs(tab, sub, threadIdx.x);

    const int row    = threadIdx.x & 63;
    const int kseg   = threadIdx.x >> 6;
    const int kslabs = K / PXQ4_QK;

    const uint8_t * pan_slabs = slabs + (size_t)p * kslabs * pxq4_pol::SLAB;
    const float     anch      = __half2float(anchor[(size_t)p * PXQ4_BM + row]);
    const __half  * xt        = x + (size_t)iy * K;

    // PXQ_CANON_v1: two-level fixed-chunk fold per lane.
    const int nfix = pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX);
    float su = 0.f;
    for (int c = 0; c < nfix; ++c) {
        const int b0 = (kslabs * c) / nfix;
        const int b1 = (kslabs * (c + 1)) / nfix;
        const int n  = (b1 - b0) * PXQ4_QK;

        // barrier 1 also covers the stage_tabs writes on the first iteration, and protects
        // the previous chunk's readers from this chunk's writers on every later iteration.
        __syncthreads();
        for (int idx = threadIdx.x; idx < n; idx += blockDim.x) {
            pxq4_xs[idx] = __half2float(xt[b0 * PXQ4_QK + idx]);
        }
        __syncthreads();

        float t = 0.f;
        for (int kb = b0 + kseg; kb < b1; kb += PXQ4_MMV_KSEG) {
            t += pxq4_dot32<VECX>(pan_slabs + (size_t)kb * pxq4_pol::SLAB, row, anch,
                                  pxq4_xs + (size_t)(kb - b0) * PXQ4_QK, tab, sub);
        }
        su += t;
    }

    red[kseg * PXQ4_BM + row] = su;
    __syncthreads();
    if (kseg == 0) {
        float u = 0.f;
#pragma unroll
        for (int s = 0; s < PXQ4_MMV_KSEG; ++s) u += red[s * PXQ4_BM + row];
        out[(size_t)iy * R + p * PXQ4_BM + row] = __float2half_rn(u);
    }
}
