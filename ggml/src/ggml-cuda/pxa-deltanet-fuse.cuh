// PXA_FUSE_DELTANET — R2 DeltaNet decode glue-kernel fusion (2026-07-07)
//
// The qwen35moe/qwen3next DeltaNet decode path (bs=1) runs a chain of small glue kernels
// around the fused delta_net_recurrent_f32 core, each paying a ~3-10us execution-latency
// floor on Pascal/Volta:
//
//   SILU(conv_out) -> L2_NORM(q) -> L2_NORM(k)          3 kernels -> 1  (pxa_dn_silu_qknorm_f32)
//   CONT(conv tail) + CONCAT(state writeback, 4.2MB x2) 2 kernels -> 1  (pxa_dn_conv_tail_f32)
//     + delta_net writes its new ssm state DIRECTLY into the cache row (state_dst override),
//       eliminating the CONCAT's 4.15MB read + 4.15MB write per layer per token entirely.
//   FUSED_RMS_NORM(ssm_norm) + FUSED_MUL_UNARY(silu z)  2 kernels -> 1  (pxa_dn_rms_silu_gate_f32)
//
// Env gate: PXA_FUSE_DELTANET (bitmask, default 3 = both fusions ON; =0 restores the eager path)
//   bit0 (1): the qk-norm/state-writeback cluster (anchored at the SILU node; consumes the
//             23-node SILU..CONCAT window incl. the already-fused ADD+SOFTPLUS+MUL beta-gate)
//   bit1 (2): the out-gate rms+silu fusion (anchored at FUSED_RMS_NORM)
//   Default ON: measured +3.7% P100 decode on the published U16 config (docs/LEVERS.md §2),
//   bit-exact vs the eager kernels; pattern mismatch still falls through to eager per-node.
//
// Correctness notes:
//  - Pure eval-time pattern fusion: the ggml graph is UNCHANGED; only the launched kernel
//    sequence differs. Any structural mismatch falls through to the eager per-node path.
//  - Bit-exact math vs the eager kernels: silu x/(1+e^-x); l2_norm rsqrtf(fmaxf(sumsq,eps^2));
//    fused_rms_norm rsqrtf(sumsq/ncols+eps) * w; fused_mul_unary(SILU, no-limit) silu(z)*y.
//  - The state-row conv-tail write (kernel B) is ordered AFTER the SSM_CONV read of the same
//    region by stream order; the delta-net in-place ssm-state update is race-free because every
//    state element is read-at-start/written-at-end by the SAME thread (block-disjoint ownership).
//  - CUDA-graph capture safe: no host syncs; all pointers are graph-node pointers covered by
//    ggml_graph_node_has_matching_properties (slot/pointer changes force re-capture).
//  - Gated to n_tokens==1 && n_seqs==1 && no per-step ckpt (src6==NULL): plain decode only.
//    Prefill/MTP-verify/mixed-seq batches keep the eager path (their node order differs anyway).

#pragma once

#include "pxa-enhance.cuh"   // level default: REFERENCE -> 0 (eager path); env always wins

static inline int pxa_fuse_deltanet_mask() {
    static const int mask = getenv("PXA_FUSE_DELTANET") ? atoi(getenv("PXA_FUSE_DELTANET")) : pxa_fuse_deltanet_default();
    return mask;
}

// ---------------------------------------------------------------------------------------------
// Kernel A: silu over the full conv output + per-head l2-norm of the q/k regions.
//   raw      [total]          pre-activation conv output (token columns of conv_output_raw)
//   silu_out [total]          silu(raw) — the full tensor is written (v region consumed by delta-net)
//   qn/kn    [hd*nh] each     l2-normalized silu'd q/k head blocks (the L2_NORM node dsts)
// Blocks [0, 2*nh): one per q/k head (block reduction). Blocks >= 2*nh: elementwise tail.
static __global__ void pxa_dn_silu_qknorm_f32(
        const float * __restrict__ raw, float * __restrict__ silu_out,
        float * __restrict__ qn, float * __restrict__ kn,
        const int hd, const int nh, const int total, const float eps) {
    const int nqk = 2*nh;
    if ((int)blockIdx.x < nqk) {
        const int base = blockIdx.x * hd;
        const int tid  = threadIdx.x;
        float vals[2]; // hd <= 2*blockDim.x (hd in {64,128}, blockDim.x = 128)
        int nv = 0;
        float sumsq = 0.0f;
        for (int c = tid; c < hd; c += blockDim.x) {
            const float x  = raw[base + c];
            const float sv = x / (1.0f + expf(-x));
            silu_out[base + c] = sv;
            vals[nv++] = sv;
            sumsq += sv*sv;
        }
        sumsq = warp_reduce_sum(sumsq);
        __shared__ float smem[8];
        if (blockDim.x > WARP_SIZE) {
            const int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
            if (lid == 0) smem[wid] = sumsq;
            __syncthreads();
            sumsq = tid < blockDim.x/WARP_SIZE ? smem[tid] : 0.0f;
            sumsq = warp_reduce_sum(sumsq);
            if (tid == 0) smem[0] = sumsq;
            __syncthreads();
            sumsq = smem[0];
        }
        const float scale = rsqrtf(fmaxf(sumsq, eps*eps));
        float * dstp = (int)blockIdx.x < nh ? qn + blockIdx.x*hd : kn + (blockIdx.x - nh)*hd;
        nv = 0;
        for (int c = tid; c < hd; c += blockDim.x) {
            dstp[c] = vals[nv++] * scale;
        }
    } else {
        const int idx = nqk*hd + (blockIdx.x - nqk)*blockDim.x + threadIdx.x;
        if (idx < total) {
            const float x = raw[idx];
            silu_out[idx] = x / (1.0f + expf(-x));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B: gather the strided conv-tail view [dc, cd] straight into the state cache row,
// replacing CONT + the conv half of the CONCAT. dst layout = the CONT layout [t + c*dc].
static __global__ void pxa_dn_conv_tail_f32(
        const float * __restrict__ src, float * __restrict__ dst,
        const int dc, const int64_t cd, const int64_t se0, const int64_t se1) {
    const int64_t idx = (int64_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= dc*cd) return;
    const int64_t t = idx % dc;
    const int64_t c = idx / dc;
    dst[idx] = src[t*se0 + c*se1];
}

// ---------------------------------------------------------------------------------------------
// Kernel C: fused rms-norm(x)*w * silu(z) — the DeltaNet gated-output epilogue.
static __global__ void pxa_dn_rms_silu_gate_f32(
        const float * __restrict__ x, const float * __restrict__ w,
        const float * __restrict__ z, float * __restrict__ dst,
        const int ncols, const float eps, void * __restrict__ vq8 = nullptr) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * xr = x + (int64_t)row*ncols;
    const float * zr = z + (int64_t)row*ncols;
    float       * dr = dst + (int64_t)row*ncols;
    float sumsq = 0.0f;
    for (int c = tid; c < ncols; c += blockDim.x) {
        const float v = xr[c];
        sumsq += v*v;
    }
    sumsq = warp_reduce_sum(sumsq);
    __shared__ float smem[8];
    if (blockDim.x > WARP_SIZE) {
        const int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
        if (lid == 0) smem[wid] = sumsq;
        __syncthreads();
        sumsq = tid < blockDim.x/WARP_SIZE ? smem[tid] : 0.0f;
        sumsq = warp_reduce_sum(sumsq);
        if (tid == 0) smem[0] = sumsq;
        __syncthreads();
        sumsq = smem[0];
    }
    const float scale = rsqrtf(sumsq/ncols + eps);
    for (int c = tid; c < ncols; c += blockDim.x) {
        const float zi = zr[c];
        dr[c] = xr[c]*scale*w[c] * (zi / (1.0f + expf(-zi)));
    }
    // G2-F2 QUANTFOLD epilogue: emit the FLAT q8_1 sidecar of dst (bit-identical to
    // quantize_q8_1 of the flattened output; requires ncols % 32 == 0, driver-guarded).
    // This row owns flat chunks [row*ncols/32, (row+1)*ncols/32), one warp per chunk.
    if (vq8) {
        block_q8_1 * q8 = (block_q8_1 *)vq8;
        const int wid = tid / WARP_SIZE, lid = tid % WARP_SIZE;
        for (int cb = wid; cb < ncols/WARP_SIZE; cb += blockDim.x/WARP_SIZE) {
            const int c = cb*WARP_SIZE + lid;
            const float zi = zr[c];
            const float xi = xr[c]*scale*w[c] * (zi / (1.0f + expf(-zi)));
            float amax = fabsf(xi);
            float sum  = xi;
            amax = warp_reduce_max(amax);
            sum  = warp_reduce_sum(sum);
            const float d = amax / 127;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
            const int64_t ib = (int64_t)row*(ncols/WARP_SIZE) + cb;
            q8[ib].qs[lid] = q;
            if (lid == 0) {
                reinterpret_cast<half&>(q8[ib].ds.x) = d;
                reinterpret_cast<half&>(q8[ib].ds.y) = sum;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Cluster handler. Anchored at the UNARY(SILU) node of a DeltaNet layer at bs=1 decode.
// Matches the fixed 23-node window (verified against the live decode graph, both the
// steady-state and the pos-0 state-reset variants have identical structure from the anchor):
//   i+0  UNARY SILU   conv_output_silu       i+12 ADD    alpha_biased
//   i+1  VIEW         conv tail (of raw)     i+13 UNARY  SOFTPLUS
//   i+2  CONT         new_conv_states_cont   i+14 MUL    g_in
//   i+3  RESHAPE                             i+15 PERMUTE g_fused
//   i+4  VIEW         q_conv                 i+16 PERMUTE beta_fused
//   i+5  L2_NORM                             i+17 RESHAPE state_fused
//   i+6  PERMUTE      q_fused                i+18 DELTA_NET
//   i+7  VIEW         k_conv                 i+19 VIEW    (new ssm state)
//   i+8  L2_NORM                             i+20 RESHAPE
//   i+9  PERMUTE      k_fused                i+21 RESHAPE
//   i+10 VIEW         v_in                   i+22 CONCAT  state_cpy (dst = cache row)
//   i+11 PERMUTE      v_fused
// Returns the number of EXTRA nodes consumed (22) or 0 for no-match (eager fallback).
static int pxa_try_deltanet_cluster(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!(pxa_fuse_deltanet_mask() & 1)) return 0;
    if (i + 22 >= cgraph->n_nodes) return 0;

    ggml_tensor * S   = cgraph->nodes[i];
    ggml_tensor * T   = cgraph->nodes[i+1];
    ggml_tensor * C   = cgraph->nodes[i+2];
    ggml_tensor * R1  = cgraph->nodes[i+3];
    ggml_tensor * Q   = cgraph->nodes[i+4];
    ggml_tensor * LQ  = cgraph->nodes[i+5];
    ggml_tensor * PQ  = cgraph->nodes[i+6];
    ggml_tensor * K   = cgraph->nodes[i+7];
    ggml_tensor * LK  = cgraph->nodes[i+8];
    ggml_tensor * PK  = cgraph->nodes[i+9];
    ggml_tensor * Vv  = cgraph->nodes[i+10];
    ggml_tensor * PV  = cgraph->nodes[i+11];
    ggml_tensor * AD  = cgraph->nodes[i+12];
    ggml_tensor * SP  = cgraph->nodes[i+13];
    ggml_tensor * MU  = cgraph->nodes[i+14];
    ggml_tensor * PG  = cgraph->nodes[i+15];
    ggml_tensor * PB  = cgraph->nodes[i+16];
    ggml_tensor * RS  = cgraph->nodes[i+17];
    ggml_tensor * D   = cgraph->nodes[i+18];
    ggml_tensor * NV  = cgraph->nodes[i+19];
    ggml_tensor * NR1 = cgraph->nodes[i+20];
    ggml_tensor * NR2 = cgraph->nodes[i+21];
    ggml_tensor * CC  = cgraph->nodes[i+22];

    // --- op skeleton ---
    if (T->op   != GGML_OP_VIEW      || C->op   != GGML_OP_CONT      || R1->op  != GGML_OP_RESHAPE ||
        Q->op   != GGML_OP_VIEW      || LQ->op  != GGML_OP_L2_NORM   || PQ->op  != GGML_OP_PERMUTE ||
        K->op   != GGML_OP_VIEW      || LK->op  != GGML_OP_L2_NORM   || PK->op  != GGML_OP_PERMUTE ||
        Vv->op  != GGML_OP_VIEW      || PV->op  != GGML_OP_PERMUTE   ||
        AD->op  != GGML_OP_ADD       || SP->op  != GGML_OP_UNARY     || MU->op  != GGML_OP_MUL     ||
        PG->op  != GGML_OP_PERMUTE   || PB->op  != GGML_OP_PERMUTE   || RS->op  != GGML_OP_RESHAPE ||
        D->op   != GGML_OP_DELTA_NET || NV->op  != GGML_OP_VIEW      || NR1->op != GGML_OP_RESHAPE ||
        NR2->op != GGML_OP_RESHAPE   || CC->op  != GGML_OP_CONCAT) {
        return 0;
    }
    if ((ggml_unary_op)SP->op_params[0] != GGML_UNARY_OP_SOFTPLUS) return 0;

    // --- SILU anchor: bs=1 slice of an SSM_CONV output ---
    if (S->type != GGML_TYPE_F32 || S->ne[1] != 1 || S->ne[2] != 1 || S->ne[3] != 1) return 0;
    if (!S->src[0] || S->src[0]->op != GGML_OP_VIEW) return 0;
    ggml_tensor * RAW = S->src[0]->src[0];
    if (!RAW || RAW->op != GGML_OP_SSM_CONV || RAW->type != GGML_TYPE_F32) return 0;
    if (S->src[0]->data != RAW->data) return 0;         // token slice at offset 0
    if (!ggml_is_contiguous(S)) return 0;
    const int64_t cd = S->ne[0];                        // conv_dim

    // --- conv-tail view + CONT + reshape ---
    if (T->src[0] != RAW || T->type != GGML_TYPE_F32) return 0;
    const int64_t dc = T->ne[0];                        // d_conv - 1
    if (dc < 1 || dc > 8 || T->ne[1] != cd || T->ne[2] != 1 || T->ne[3] != 1) return 0;
    if (C->src[0] != T || !ggml_is_contiguous(C) || C->ne[0] != dc || C->ne[1] != cd) return 0;
    if (R1->src[0] != C) return 0;
    if (T->nb[0] % sizeof(float) || T->nb[1] % sizeof(float)) return 0;
    const int64_t tail_off = ((const char *)T->data - (const char *)RAW->data);
    if (tail_off < 0 || tail_off % sizeof(float)) return 0;

    // --- q/k views + l2 norms + permutes ---
    if (Q->src[0] != S || K->src[0] != S || Vv->src[0] != S) return 0;
    const int64_t hd = Q->ne[0], nh = Q->ne[1];
    if ((hd != 64 && hd != 128) || nh < 1 || nh > 256) return 0;
    if (Q->ne[2] != 1 || Q->ne[3] != 1) return 0;
    if (K->ne[0] != hd || K->ne[1] != nh || K->ne[2] != 1 || K->ne[3] != 1) return 0;
    if (Q->nb[1] != hd*sizeof(float) || K->nb[1] != hd*sizeof(float)) return 0;
    if (Q->data != S->data) return 0;                                   // q at offset 0
    if ((const char *)K->data - (const char *)S->data != (int64_t)(nh*hd*sizeof(float))) return 0;
    if ((const char *)Vv->data - (const char *)S->data != (int64_t)(2*nh*hd*sizeof(float))) return 0;
    if (2*nh*hd + Vv->ne[0]*Vv->ne[1] != cd) return 0;                  // q+k+v tile the conv row
    if (LQ->src[0] != Q || LK->src[0] != K) return 0;
    if (LQ->type != GGML_TYPE_F32 || LK->type != GGML_TYPE_F32) return 0;
    if (!ggml_is_contiguous(LQ) || !ggml_is_contiguous(LK)) return 0;
    float epsq, epsk;
    memcpy(&epsq, LQ->op_params, sizeof(float));
    memcpy(&epsk, LK->op_params, sizeof(float));
    if (epsq != epsk) return 0;
    if (PQ->src[0] != LQ || PK->src[0] != LK || PV->src[0] != Vv) return 0;

    // --- beta-gate chain (delegated to the existing fused kernel) ---
    if (SP->src[0] != AD || MU->src[0] != SP) return 0;
    if (!AD->src[1] || !MU->src[1]) return 0;
    if (ggml_nrows(AD->src[1]) != 1 || ggml_nrows(MU->src[1]) != 1) return 0;
    if (AD->src[1]->ne[0] != AD->src[0]->ne[0] || MU->src[1]->ne[0] != MU->src[0]->ne[0]) return 0;
    if (AD->type != GGML_TYPE_F32 || MU->type != GGML_TYPE_F32 ||
        AD->src[0]->type != GGML_TYPE_F32 || AD->src[1]->type != GGML_TYPE_F32) return 0;
    if (!ggml_is_contiguous(AD->src[0])) return 0;
    if (PG->src[0] != MU) return 0;

    // --- delta-net core: exactly the tensors produced above, plain decode only ---
    if (D->src[0] != PQ || D->src[1] != PK || D->src[2] != PV ||
        D->src[3] != PG || D->src[4] != PB || D->src[5] != RS || D->src[6] != nullptr) {
        return 0;
    }
    if (D->src[0]->ne[1] != 1 || D->src[0]->ne[3] != 1) return 0;       // n_tokens==1, n_seqs==1

    // --- state writeback: CONCAT(conv_tail_cont, new_ssm_state) into the cache row ---
    if (NV->src[0] != D || NR1->src[0] != NV || NR2->src[0] != NR1) return 0;
    if (CC->op_params[0] != 0) return 0;                                // concat along dim 0
    if (CC->src[0] != R1 || CC->src[1] != NR2) return 0;
    if (CC->type != GGML_TYPE_F32 || !ggml_is_contiguous(CC)) return 0;
    const int64_t conv_elems = ggml_nelements(R1);
    const int64_t ssm_elems  = ggml_nelements(NR2);
    if (conv_elems != dc*cd) return 0;
    if (ggml_nelements(CC) != conv_elems + ssm_elems) return 0;

    if (!ops_are_same_device(cgraph, i, i+22)) return 0;

    // ------------------------------------ execute ------------------------------------
    cudaStream_t stream = ctx.stream();

    { // A: silu + q/k l2-norm (replaces SILU, L2_NORM, L2_NORM)
        const int nqk = 2*(int)nh;
        const int block = 128;
        const int tail_blocks = (int)(((cd - (int64_t)nqk*hd) + block - 1)/block);
        const int nblocks = nqk + tail_blocks;
        pxa_dn_silu_qknorm_f32<<<nblocks, block, 0, stream>>>(
                (const float *)S->src[0]->data, (float *)S->data,
                (float *)LQ->data, (float *)LK->data,
                (int)hd, (int)nh, (int)cd, epsq);
        CUDA_CHECK(cudaGetLastError());
    }

    { // B: conv tail straight into the state cache row (replaces CONT + the conv half of CONCAT)
        const int block = 256;
        const int nblocks = (int)((dc*cd + block - 1)/block);
        pxa_dn_conv_tail_f32<<<nblocks, block, 0, stream>>>(
                (const float *)((const char *)RAW->data + tail_off), (float *)CC->data,
                (int)dc, cd, (int64_t)(T->nb[0]/sizeof(float)), (int64_t)(T->nb[1]/sizeof(float)));
        CUDA_CHECK(cudaGetLastError());
    }

    // beta-gate: the fork's existing fused ADD+SOFTPLUS+MUL kernel
    ggml_cuda_fused_softplus(ctx, MU);

    // delta-net core with the new ssm state redirected into the cache row (replaces the
    // ssm half of CONCAT — and its 2x ~4MB of pure copy traffic)
    ggml_cuda_op_delta_net_ex(ctx, D, (float *)CC->data + conv_elems);

    // CONCAT (i+22) is fully covered by B + the redirect; all other nodes in the window are views.
    return 22;
}

// ---------------------------------------------------------------------------------------------
// Out-gate handler: FUSED_RMS_NORM(delta-net output, ssm_norm) + FUSED_MUL_UNARY(z, ., SILU) -> 1.
// Returns true if fused (caller then skips one node).
static bool pxa_try_deltanet_outgate(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (!(pxa_fuse_deltanet_mask() & 2)) return false;
    if (i + 1 >= cgraph->n_nodes) return false;

    ggml_tensor * F = cgraph->nodes[i];
    ggml_tensor * M = cgraph->nodes[i+1];
    if (M->op != GGML_OP_FUSED_MUL_UNARY) return false;
    if ((ggml_unary_op)M->op_params[0] != GGML_UNARY_OP_SILU) return false;
    float limit;
    memcpy(&limit, (const float *)M->op_params + 1, sizeof(float));
    if (limit >= 1e-6f) return false;                    // limited variant not replicated
    if (M->src[1] != F || !M->src[0]) return false;
    // anchor strictly to the DeltaNet out-gate: rms input is a reshape of a view of DELTA_NET
    if (!F->src[0] || F->src[0]->op != GGML_OP_RESHAPE ||
        !F->src[0]->src[0] || F->src[0]->src[0]->op != GGML_OP_VIEW ||
        !F->src[0]->src[0]->src[0] || F->src[0]->src[0]->src[0]->op != GGML_OP_DELTA_NET) {
        return false;
    }
    if (F->type != GGML_TYPE_F32 || M->type != GGML_TYPE_F32) return false;
    if (F->src[0]->type != GGML_TYPE_F32 || M->src[0]->type != GGML_TYPE_F32) return false;
    if (F->ne[2] != 1 || F->ne[3] != 1) return false;
    if (!F->src[1] || ggml_nrows(F->src[1]) != 1 || F->src[1]->ne[0] != F->ne[0]) return false;
    if (F->src[1]->type != GGML_TYPE_F32) return false;
    if (!ggml_are_same_shape(M->src[0], M) || !ggml_are_same_shape(F, M)) return false;
    if (!ggml_is_contiguous(F->src[0]) || !ggml_is_contiguous(M->src[0]) || !ggml_is_contiguous(M)) return false;
    if (!ops_are_same_device(cgraph, i, i+1)) return false;

    float eps;
    memcpy(&eps, F->op_params, sizeof(float));

    const int ncols = (int)F->ne[0];
    const int nrows = (int)ggml_nrows(F);
    // G2-F2 QUANTFOLD: if a q8_1-GEMV consumer of M follows, emit the sidecar in the same launch
    void * g2_q8 = nullptr;
    int64_t g2_padded = 0;
    if (pxa_g2_quantfold() && (ncols % WARP_SIZE) == 0 &&
        pxa_g2_normfuse_wanted(ctx, cgraph, i + 1, M, g2_padded)) {
        g2_q8 = pxa_g2_q8_buf(ctx.device, ctx.stream(), (size_t)(g2_padded/QK8_1)*sizeof(block_q8_1));
    }
    pxa_dn_rms_silu_gate_f32<<<nrows, 128, 0, ctx.stream()>>>(
            (const float *)F->src[0]->data, (const float *)F->src[1]->data,
            (const float *)M->src[0]->data, (float *)M->data, ncols, eps, g2_q8);
    CUDA_CHECK(cudaGetLastError());
    if (g2_q8) {
        auto & sc = pxa_g2_q8sc[ctx.device];
        sc.t = M; sc.data = M->data; sc.padded = g2_padded; sc.eval = pxa_g2_eval_serial;
    }
    // F's own dst is deliberately left unwritten: its only consumer is M (verified by the
    // strict delta-net anchor — this exact graph site).
    return true;
}
