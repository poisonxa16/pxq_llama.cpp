//
// DeepSeek-V4 (deepseek_v4 / DeepseekV4ForCausalLM) graph.
//
// Adapted from llama.cpp `src/models/deepseek4.cpp` @ upstream commit 82dbc4f01
// (originating PR #24162 and follow-ups #25202, #25370, #25521, #25588, #25702,
// #25787, #25325, #25945, #25871).
// Copyright (c) 2023-2026 The ggml authors. MIT license.
// SPDX-License-Identifier: MIT
//
// Upstream carries no per-file copyright header on src/models/deepseek4.cpp; the
// project relies on its root LICENSE. This block is the explicit provenance note
// required for a file taken substantially from that source. Follows the local
// precedent in src/graphs/build_hy3.cpp.
//
// ---------------------------------------------------------------------------
// STRUCTURAL DIFFERENCES vs upstream (deliberate; each one is load-bearing)
// ---------------------------------------------------------------------------
//  1. Upstream is a `llm_graph_context` subclass with an input-object system
//     (llm_graph_input_dsv4 / _dsv4_raw). This tree has neither. The graph is a
//     method on llm_build_context; the DS4 input tensors are created here and
//     their pointers stashed in `llama_dsv4_get_inputs(lctx)` for the host-side
//     filler `llama_dsv4_set_inputs()` (owned by the memory chunk).
//  2. All four fused ops are OFF (LIGHTNING_INDEXER, DSV4_HC_{PRE,COMB,POST}).
//     Only upstream's primitive fallback branches are ported. There is no
//     `res->add_fused_node()` equivalent here, so those calls are dropped.
//  3. `ggml_rope_ext_back` -> `ggml_rope_back` (upstream renamed the symbol; the
//     parameter list is byte-identical). Used in the FORWARD graph to de-rope the
//     attention output before the o-LoRA.
//  4. `llm_graph_context::build_attn_mha` has no equivalent here (our
//     llm_build_kqv is welded to the POD llama_kv_cache), so it is ported as the
//     private member build_dsv4_attn_mha() below, restricted to the two arguments
//     DS4 actually uses (kq_b == nullptr, v_mla == nullptr).
//  5. ALL DS4 mask inputs are GGML_TYPE_F32 and UNPADDED [n_kv, n_tokens, 1, 1].
//     Upstream builds the CSA/HCA masks as F16 under -fa. We cannot: this tree's
//     `ggml_fill` asserts F32 (ggml.c) and `ggml_set_rows` asserts a F32 source,
//     and build_top_k_mask needs both. The graph pads to GGML_KQ_MASK_PAD and
//     casts to F16 itself, immediately before handing a mask to flash-attention.
//     -> The memory chunk must write FLOATS into all four mask tensors.
//  6. `dsv4_with_zero_dep` uses ggml_sum_rows over one row instead of ggml_sum:
//     GGML_OP_SUM has no CUDA backend in this tree and would split the graph to
//     the CPU on every compressed layer. The node exists purely to order the
//     persistent-state overwrite after the state-backed commit has read it; the
//     substitute carries the same dependency edge.
//  7. Single stream only (`-np 1`). This tree's ggml_flash_attn_ext asserts
//     mask->ne[2] == 1 && mask->ne[3] == 1, so a multi-stream mask cannot even be
//     expressed here. Asserted explicitly below rather than silently mis-shaped.
//
// ---------------------------------------------------------------------------
// INTERFACE CONTRACT — declarations this file requires from
// `src/llama-kv-cache-dsv4.h` (owned by the memory chunk).
// Reproduced verbatim so the owner can paste it; if any name changes, this file
// must change with it.
// ---------------------------------------------------------------------------
//
//   enum llama_dsv4_stream {
//       LLAMA_DSV4_CSA = 0,   // compress ratio 4  (+ lightning indexer)
//       LLAMA_DSV4_HCA = 1,   // compress ratio 128
//       LLAMA_DSV4_LID = 2,   // lightning-indexer key stream (ratio 4)
//   };
//
//   // verbatim from upstream llama_kv_cache_dsv4_context::comp_plan
//   struct llama_dsv4_comp_plan {
//       std::vector<int32_t> state_pos;
//       std::vector<int32_t> state_persist_src_idxs;
//       std::vector<int32_t> state_persist_dst_idxs;
//       std::vector<int32_t> state_read_idxs;
//       std::vector<int64_t> state_write_idxs;
//       std::vector<int32_t> state_write_pos;
//       std::vector<int32_t> n_visible;
//       int64_t n_stream = 1;
//       int64_t n_kv     = 0;
//   };
//
//   struct llama_dsv4_comp_inputs {
//       ggml_tensor * state_pos              = nullptr; // I32 [n_state]
//       ggml_tensor * state_persist_src_idxs = nullptr; // I32 [n_persist_src]
//       ggml_tensor * state_persist_dst_idxs = nullptr; // I32 [n_persist_dst]
//       ggml_tensor * state_read_idxs        = nullptr; // I32 [ratio*n_write] (overlap: 2*ratio*n_write)
//       ggml_tensor * state_write_idxs       = nullptr; // I64 [n_write]
//       ggml_tensor * state_write_pos        = nullptr; // I32 [n_write]
//       ggml_tensor * kq_mask                = nullptr; // F32 [n_kv, n_tokens, 1, 1]
//       ggml_tensor * k_rot                  = nullptr; // F32 [nrot, nrot] or null
//   };
//
//   struct llama_dsv4_inputs {
//       ggml_tensor * raw_k_idxs  = nullptr;            // I64 [n_tokens]
//       ggml_tensor * raw_kq_mask = nullptr;            // F32 [n_kv_raw, n_tokens, 1, 1]
//       ggml_tensor * raw_k_rot   = nullptr;            // F32 [nrot, nrot] or null
//       llama_dsv4_comp_inputs comp[3];                 // indexed by llama_dsv4_stream
//   };
//
//   llama_dsv4_inputs & llama_dsv4_get_inputs(llama_context & lctx);
//   const llama_dsv4_comp_plan & llama_dsv4_get_plan(const llama_context & lctx, llama_dsv4_stream s);
//   uint32_t llama_dsv4_get_raw_n_kv(const llama_context & lctx);
//   int64_t  llama_dsv4_get_raw_nrot(const llama_context & lctx);  // 0 => no raw k_rot
//   int64_t  llama_dsv4_get_comp_nrot(const llama_context & lctx, llama_dsv4_stream s);
//
//   ggml_tensor * llama_dsv4_get_raw_k(const llama_context & lctx, ggml_context * ctx, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_raw_k(const llama_context & lctx, ggml_context * ctx,
//                                      ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il);
//   ggml_tensor * llama_dsv4_get_comp_k(const llama_context & lctx, ggml_context * ctx,
//                                       llama_dsv4_stream s, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_comp_k(const llama_context & lctx, ggml_context * ctx,
//                                       llama_dsv4_stream s, ggml_tensor * k_cur,
//                                       ggml_tensor * k_idxs, int32_t il);
//   ggml_tensor * llama_dsv4_get_state_kv(const llama_context & lctx, ggml_context * ctx,
//                                         llama_dsv4_stream s, int32_t il);
//   ggml_tensor * llama_dsv4_get_state_score(const llama_context & lctx, ggml_context * ctx,
//                                            llama_dsv4_stream s, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_state_kv(const llama_context & lctx, ggml_context * ctx,
//                                         llama_dsv4_stream s, ggml_tensor * cur,
//                                         ggml_tensor * idxs, int32_t il);
//   ggml_tensor * llama_dsv4_cpy_state_score(const llama_context & lctx, ggml_context * ctx,
//                                            llama_dsv4_stream s, ggml_tensor * cur,
//                                            ggml_tensor * idxs, int32_t il);
//
// ---------------------------------------------------------------------------
// hparams / tensor fields required from the arch chunk (spec §1.2, §1.6):
//   hparams: dsv4_hc_mult, dsv4_hc_sinkhorn_iters, dsv4_hc_eps, dsv4_hash_layer_count,
//            dsv4_o_group_count, dsv4_o_lora_rank, dsv4_compress_rope_base,
//            dsv4_compress_ratios[LLAMA_MAX_LAYERS],
//            swiglu_limits[LLAMA_MAX_LAYERS], swiglu_limits_shared[LLAMA_MAX_LAYERS]
//   model:   hc_head_fn, hc_head_base, hc_head_scale
//   layer:   wq_b, wo_a, wo_b, hc_attn_{fn,base,scale}, hc_ffn_{fn,base,scale},
//            attn_comp_{wkv,wgate,ape,norm}, indexer_comp_{wkv,wgate,ape,norm},
//            ffn_gate_tid2eid
//            (attn_norm, attn_sinks, wq_a, attn_q_a_norm, wkv, attn_kv_a_norm,
//             indexer_proj, indexer_attn_q_b, ffn_norm, ffn_gate_inp,
//             ffn_exp_probs_b, ffn_*_exps, ffn_*_shexp already exist here)
//   NOTE: this tree calls the compressed-KV norm `attn_kv_a_norm`, upstream calls
//   it `attn_kv_norm`. We use the local name -- in BOTH the graph and the loader.
//

#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-kv-cache-dsv4.h"

#include <cmath>
#include <cstdio>

// compression ratios that select the three per-layer attention regimes
static constexpr int64_t DSV4_CSA_RATIO = 4;    // compressed sparse attn + lightning indexer
static constexpr int64_t DSV4_HCA_RATIO = 128;  // hyper-compressed attn

// upstream: dsv4_rope_attn_factor()
static float dsv4_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

static size_t dsv4_elem_offset(const ggml_tensor * t, int64_t i) {
    return ggml_row_size(t->type, i);
}

static ggml_tensor * dsv4_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, dsv4_elem_offset(t, i0));
}

static ggml_tensor * dsv4_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], dsv4_elem_offset(t, i0));
}

// upstream: dsv4_append_zero_row(). Adds one synthetic trailing row that
// state_read_idxs may address for the first block's "previous" half.
static ggml_tensor * dsv4_append_zero_row(ggml_context * ctx, ggml_tensor * t, bool neg_inf) {
    ggml_tensor * row = ggml_view_1d(ctx, t, t->ne[0], 0);
    row = neg_inf ? ggml_scale_bias(ctx, row, 0.0f, -INFINITY) : ggml_scale(ctx, row, 0.0f);
    row = ggml_reshape_2d(ctx, row, t->ne[0], 1);

    return ggml_concat(ctx, t, row, 1);
}

// upstream: dsv4_with_zero_dep(), but built on sum_rows (see note 6 at the top).
// Adds a numerically-zero edge t <- dep so the persistent-state overwrite cannot
// be scheduled before the state-backed commit that reads the old state.
static ggml_tensor * dsv4_with_zero_dep(ggml_context * ctx, ggml_tensor * t, ggml_tensor * dep) {
    if (dep == nullptr) {
        return t;
    }

    ggml_tensor * row  = ggml_reshape_2d(ctx, ggml_view_1d(ctx, dep, dep->ne[0], 0), dep->ne[0], 1);
    ggml_tensor * zero = ggml_scale(ctx, ggml_sum_rows(ctx, row), 0.0f); // [1,1]

    return ggml_add(ctx, t, zero);
}

static ggml_tensor * dsv4_hc_affine(ggml_context * ctx, ggml_tensor * x, ggml_tensor * scale, ggml_tensor * base) {
    x = ggml_mul(ctx, x, scale);
    x = ggml_add(ctx, x, base);
    return x;
}

// upstream: llama_mul_mat_hadamard() from src/llama-impl.h. `rot` is an explicit
// [n,n] rotation matrix supplied by the cache; this tree's ggml_hadamard() is a
// different (matrix-free) op and is NOT a substitute.
static ggml_tensor * dsv4_mul_mat_rot(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * rot) {
    const int64_t n = rot->ne[0];

    ggml_tensor * res;
    if (!ggml_is_contiguous(cur)) {
        res = ggml_cont_2d(ctx, cur, n, ggml_nelements(cur)/n);
    } else {
        res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    }
    res = ggml_mul_mat(ctx, rot, res);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

// Pad a DS4 mask along dim 1 to GGML_KQ_MASK_PAD (flash-attention requirement in
// this tree) and cast it to the type the attention kernel wants. All DS4 masks
// arrive here as F32 [n_kv, n_tokens, 1, 1] - see note 5 at the top.
static ggml_tensor * dsv4_finalize_kq_mask(ggml_context * ctx, ggml_tensor * m, bool flash_attn) {
    GGML_ASSERT(m->ne[2] == 1 && m->ne[3] == 1 && "DSV4 first cut is single-stream (-np 1)");

    const int64_t n_pad = GGML_PAD(m->ne[1], GGML_KQ_MASK_PAD);
    if (n_pad > m->ne[1]) {
        ggml_tensor * pad = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, m->ne[0], n_pad - m->ne[1], 1, 1);
        pad = ggml_fill(ctx, pad, -INFINITY);
        m   = ggml_concat(ctx, m, pad, 1);
    }

    if (flash_attn && m->type != GGML_TYPE_F16) {
        m = ggml_cast(ctx, m, GGML_TYPE_F16);
    }

    return m;
}

static ggml_tensor * dsv4_build_input_1d(ggml_context * ctx, ggml_type type, int64_t n, const char * name) {
    if (n <= 0) {
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, type, n);
    ggml_set_input(t);
    ggml_set_name(t, name);

    return t;
}

static ggml_tensor * dsv4_build_input_mask(ggml_context * ctx, int64_t n_kv, int64_t n_tokens, const char * name) {
    if (n_kv <= 0) {
        return nullptr;
    }

    // always F32, always unpadded - see note 5 at the top of this file
    ggml_tensor * t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_tokens, 1, 1);
    ggml_set_input(t);
    ggml_set_name(t, name);

    return t;
}

static ggml_tensor * dsv4_build_input_rot(ggml_context * ctx, int64_t nrot, const char * name) {
    if (nrot <= 0) {
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
    ggml_set_input(t);
    ggml_set_name(t, name);

    return t;
}

//
// input construction (upstream: llm_graph_context::build_inp_dsv4 +
// dsv4_build_comp_inputs + dsv4_build_raw_kq_mask in llama-graph.cpp)
//

void llm_build_context::build_dsv4_inputs() {
    // Derive this batch's compression plans FIRST: every plan-sized tensor below must be
    // built from the same plan llama_dsv4_set_inputs() will fill it from.
    llama_dsv4_build_plans(lctx, batch);

    llama_dsv4_inputs & inp = llama_dsv4_get_inputs(lctx);

    inp = llama_dsv4_inputs();

    const int64_t n_kv_raw = (int64_t) llama_dsv4_get_raw_n_kv(lctx);

    inp.raw_k_idxs  = dsv4_build_input_1d(ctx0, GGML_TYPE_I64, n_tokens, "dsv4_raw_k_idxs");
    inp.raw_kq_mask = dsv4_build_input_mask(ctx0, n_kv_raw, n_tokens, "dsv4_raw_kq_mask");
    inp.raw_k_rot   = dsv4_build_input_rot(ctx0, llama_dsv4_get_raw_nrot(lctx), "dsv4_raw_k_rot");

    static const char * const s_name[3] = { "csa", "hca", "lid" };

    for (int s = 0; s < 3; ++s) {
        const llama_dsv4_stream    st   = (llama_dsv4_stream) s;
        const llama_dsv4_comp_plan & pl = llama_dsv4_get_plan(lctx, st);
        llama_dsv4_comp_inputs     & ci = inp.comp[s];

        GGML_ASSERT(pl.n_stream == 1 && "DSV4 first cut is single-stream (-np 1)");

        char name[64];

        snprintf(name, sizeof(name), "dsv4_%s_state_pos", s_name[s]);
        ci.state_pos = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_pos.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_persist_src_idxs", s_name[s]);
        ci.state_persist_src_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_persist_src_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_persist_dst_idxs", s_name[s]);
        ci.state_persist_dst_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_persist_dst_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_read_idxs", s_name[s]);
        ci.state_read_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_read_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_write_idxs", s_name[s]);
        ci.state_write_idxs = dsv4_build_input_1d(ctx0, GGML_TYPE_I64, (int64_t) pl.state_write_idxs.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_state_write_pos", s_name[s]);
        ci.state_write_pos = dsv4_build_input_1d(ctx0, GGML_TYPE_I32, (int64_t) pl.state_write_pos.size(), name);

        snprintf(name, sizeof(name), "dsv4_%s_kq_mask", s_name[s]);
        ci.kq_mask = dsv4_build_input_mask(ctx0, pl.n_kv, n_tokens, name);

        snprintf(name, sizeof(name), "dsv4_%s_k_rot", s_name[s]);
        ci.k_rot = dsv4_build_input_rot(ctx0, llama_dsv4_get_comp_nrot(lctx, st), name);
    }
}

//
// hyper-connections
//

// upstream: build_hc_pre(x, weights, il) - the mixing half. Collapses the
// [n_embd, hc, n_tokens] residual stream to [n_embd, n_tokens] with per-branch
// weights. Fused path (ggml_dsv4_hc_pre) deliberately not ported.
ggml_tensor * llm_build_context::build_dsv4_hc_mix(ggml_tensor * x, ggml_tensor * weights, int il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(x->ne[1] == (int64_t) hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];

    ggml_tensor * result = nullptr;
    for (int64_t ih = 0; ih < hc; ++ih) {
        ggml_tensor * xh  = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], ih*x->nb[1]);
        ggml_tensor * wh  = ggml_view_2d(ctx0, weights, 1, nt, weights->nb[1], ih*weights->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, xh, wh);
        result = result ? ggml_add(ctx0, result, cur) : cur;
    }

    GGML_UNUSED(il);

    return result;
}

// upstream: build_hc_sinkhorn(). Row softmax over dst, one column
// normalization, then (iters-1) alternating row/column normalizations.
ggml_tensor * llm_build_context::build_dsv4_hc_sinkhorn(ggml_tensor * comb, int il) const {
    GGML_UNUSED(il);

    // comb is [dst_hc, src_hc, n_tokens]
    comb = ggml_soft_max(ctx0, comb);

    ggml_tensor * eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    eps = ggml_fill(ctx0, eps, hparams.dsv4_hc_eps);

    comb = ggml_add(ctx0, comb, eps);

    auto norm_cols = [&]() {
        ggml_tensor * comb_src_dst = ggml_cont(ctx0, ggml_permute(ctx0, comb, 1, 0, 2, 3));
        ggml_tensor * col_sum = ggml_sum_rows(ctx0, comb_src_dst);
        col_sum = ggml_add(ctx0, col_sum, eps);
        col_sum = ggml_permute(ctx0, col_sum, 1, 0, 2, 3);
        comb = ggml_div(ctx0, comb, col_sum);
    };

    auto norm_rows = [&]() {
        ggml_tensor * row_sum = ggml_sum_rows(ctx0, comb);
        row_sum = ggml_add(ctx0, row_sum, eps);
        comb = ggml_div(ctx0, comb, row_sum);
    };

    norm_cols();
    for (uint32_t i = 1; i < hparams.dsv4_hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }

    return comb;
}

// upstream: build_hc_pre(x, hc_fn, hc_scale, hc_base, &post, &comb, il).
// Produces the layer input, plus the post weights and the Sinkhorn-normalised
// hc x hc combination matrix consumed by build_dsv4_hc_post().
ggml_tensor * llm_build_context::build_dsv4_hc_pre(
        ggml_tensor  * x,
        ggml_tensor  * hc_fn,
        ggml_tensor  * hc_scale,
        ggml_tensor  * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int            il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * scale_pre  = dsv4_view_1d(ctx0, hc_scale, 1, 0);
    ggml_tensor * scale_post = dsv4_view_1d(ctx0, hc_scale, 1, 1);

    ggml_tensor * base_pre  = dsv4_view_1d(ctx0, hc_base, hc, 0);
    ggml_tensor * base_post = dsv4_view_1d(ctx0, hc_base, hc, hc);

    ggml_tensor * pre = dsv4_view_2d(ctx0, mixes, hc, nt, 0);
    pre = dsv4_hc_affine(ctx0, pre, scale_pre, base_pre);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dsv4_view_2d(ctx0, mixes, hc, nt, hc);
    *post = dsv4_hc_affine(ctx0, *post, scale_post, base_post);
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    {
        ggml_tensor * scale_comb = dsv4_view_1d(ctx0, hc_scale, 1, 2);
        ggml_tensor * base_comb  = dsv4_view_1d(ctx0, hc_base, hc*hc, 2*hc);

        *comb = dsv4_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = dsv4_hc_affine(ctx0, *comb, scale_comb, base_comb);
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_dsv4_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    return build_dsv4_hc_mix(x, pre, il);
}

// upstream: build_hc_post(). Scatters the sublayer output back across the hc
// branches and mixes in the (Sinkhorn-weighted) incoming residual.
ggml_tensor * llm_build_context::build_dsv4_hc_post(
        ggml_tensor * x,
        ggml_tensor * residual,
        ggml_tensor * post,
        ggml_tensor * comb,
        int           il) const {
    GGML_ASSERT(x->ne[0] == n_embd);
    GGML_ASSERT(residual->ne[1] == (int64_t) hparams.dsv4_hc_mult);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[1];

    ggml_tensor * out = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * post_dst = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        ggml_tensor * cur = ggml_mul(ctx0, x, post_dst);

        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * res_src = ggml_view_2d(ctx0, residual, n_embd, nt, residual->nb[2], src*residual->nb[1]);
            ggml_tensor * comb_src_dst = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2],
                    dst*comb->nb[0] + src*comb->nb[1]);
            cur = ggml_add(ctx0, cur, ggml_mul(ctx0, res_src, comb_src_dst));
        }

        cur = ggml_reshape_3d(ctx0, cur, n_embd, 1, nt);
        out = out ? ggml_concat(ctx0, out, cur, 1) : cur;
    }

    GGML_UNUSED(il);

    return out;
}

// upstream: build_hc_head(). Final collapse of the hc-wide stream to n_embd.
ggml_tensor * llm_build_context::build_dsv4_hc_head(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base) const {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    ggml_tensor * flat      = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes     = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_head_mixes", -1);

    ggml_tensor * pre = dsv4_hc_affine(ctx0, mixes, hc_scale, hc_base);
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_head_pre", -1);

    return build_dsv4_hc_mix(x, pre, -1);
}

//
// attention core (upstream: llm_graph_context::build_attn_mha, restricted to
// kq_b == nullptr and v_mla == nullptr, which is all DS4 uses)
//

ggml_tensor * llm_build_context::build_dsv4_attn_mha(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * kq_mask,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const bool v_trans = v->nb[1] > v->nb[2];

    const int64_t n_stream = k->ne[3];
    GGML_ASSERT(n_stream == 1 && "DSV4 first cut is single-stream (-np 1)");

    q = ggml_view_4d(ctx0, q, q->ne[0], q->ne[1], q->ne[2]/n_stream, n_stream,
            q->nb[1], q->nb[2], q->nb[3]/n_stream, 0);

    q = ggml_permute(ctx0, q, 0, 2, 1, 3);
    k = ggml_permute(ctx0, k, 0, 2, 1, 3);
    v = ggml_permute(ctx0, v, 0, 2, 1, 3);

    ggml_tensor * cur;

    if (cparams.flash_attn) {
        if (v_trans) {
            v = ggml_transpose(ctx0, v);
        }

        if (k->type == GGML_TYPE_F32) {
            k = ggml_cast(ctx0, k, GGML_TYPE_F16);
        }
        if (v->type == GGML_TYPE_F32) {
            v = ggml_cast(ctx0, v, GGML_TYPE_F16);
        }

        cur = ggml_flash_attn_ext(ctx0, q, k, v, kq_mask, kq_scale, hparams.f_max_alibi_bias,
                hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f);
        ggml_flash_attn_ext_add_sinks(cur, sinks);
        ggml_flash_attn_ext_set_prec(cur, GGML_PREC_F32);

        cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    } else {
        ggml_tensor * kq = ggml_mul_mat(ctx0, k, q);
        cb(kq, "kq", il);

        // DS4 scores need the F32 accumulator (long context, sinks, -inf masks)
        ggml_mul_mat_set_prec(kq, GGML_PREC_F32);

        kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, hparams.f_max_alibi_bias);
        ggml_soft_max_add_sinks(kq, sinks);
        cb(kq, "kq_soft_max", il);

        if (!v_trans) {
            v = ggml_cont(ctx0, ggml_transpose(ctx0, v));
            cb(v, "v_cont", il);
        }

        ggml_tensor * kqv = ggml_mul_mat(ctx0, v, kq);
        cb(kqv, "kqv", il);

        cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);
        cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
    }

    ggml_build_forward_expand(gf, cur);

    return cur;
}

//
// compressed-KV reconstruction from the compressor ring state
//

// upstream: build_hca_compressed_kv_from_state(). ratio 128, no overlap.
ggml_tensor * llm_build_context::build_dsv4_hca_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t       n_embd_head,
        const char *  name,
        int           il) const {
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == DSV4_HCA_RATIO*n_blocks);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    ggml_tensor * kv = ggml_get_rows(ctx0, kv_state, state_read_idxs);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(kv, name, il);

    ggml_tensor * score = ggml_get_rows(ctx0, score_state, state_read_idxs);
    score = ggml_reshape_3d(ctx0, score, n_embd_head, DSV4_HCA_RATIO, n_blocks);
    cb(score, name, il);

    ggml_tensor * values = ggml_cont(ctx0, ggml_permute(ctx0, kv,    1, 0, 2, 3));
    ggml_tensor * scores = ggml_cont(ctx0, ggml_permute(ctx0, score, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = llm_build_norm(ctx0, comp, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

// upstream: build_overlap_compressed_kv_from_state(). ratio 4 with the
// previous/current overlapped halves (CSA and LID).
ggml_tensor * llm_build_context::build_dsv4_overlap_compressed_kv_from_state(
        ggml_tensor * kv_state,
        ggml_tensor * score_state,
        ggml_tensor * state_read_idxs,
        ggml_tensor * comp_pos,
        ggml_tensor * norm,
        int64_t       ratio,
        int64_t       n_embd_head,
        const char *  name,
        int           il) const {
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_blocks         = comp_pos ? comp_pos->ne[0] : 0;

    GGML_ASSERT(n_blocks > 0);
    GGML_ASSERT(state_read_idxs);
    GGML_ASSERT(state_read_idxs->ne[0] == 2*ratio*n_blocks);
    GGML_ASSERT(kv_state->ne[0]    == 2*n_embd_head);
    GGML_ASSERT(score_state->ne[0] == 2*n_embd_head);
    GGML_ASSERT(n_embd_head >= n_embd_head_rope);

    kv_state    = dsv4_append_zero_row(ctx0, kv_state,    false);
    score_state = dsv4_append_zero_row(ctx0, score_state, true);

    const int64_t n_read = ratio*n_blocks;

    ggml_tensor * kv_rows    = ggml_get_rows(ctx0, kv_state,    state_read_idxs);
    ggml_tensor * score_rows = ggml_get_rows(ctx0, score_state, state_read_idxs);

    ggml_tensor * kv_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1], 0));
    kv_prev = ggml_reshape_3d(ctx0, kv_prev, n_embd_head, ratio, n_blocks);
    cb(kv_prev, name, il);

    ggml_tensor * score_prev = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1], 0));
    score_prev = ggml_reshape_3d(ctx0, score_prev, n_embd_head, ratio, n_blocks);
    cb(score_prev, name, il);

    ggml_tensor * kv_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, kv_rows, n_embd_head, n_read, kv_rows->nb[1],
                n_read*kv_rows->nb[1] + ggml_row_size(kv_rows->type, n_embd_head)));
    kv_cur = ggml_reshape_3d(ctx0, kv_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * score_cur = ggml_cont(ctx0,
            ggml_view_2d(ctx0, score_rows, n_embd_head, n_read, score_rows->nb[1],
                n_read*score_rows->nb[1] + ggml_row_size(score_rows->type, n_embd_head)));
    score_cur = ggml_reshape_3d(ctx0, score_cur, n_embd_head, ratio, n_blocks);

    ggml_tensor * values = ggml_concat(ctx0, kv_prev,    kv_cur,    1);
    ggml_tensor * scores = ggml_concat(ctx0, score_prev, score_cur, 1);

    values = ggml_cont(ctx0, ggml_permute(ctx0, values, 1, 0, 2, 3));
    scores = ggml_cont(ctx0, ggml_permute(ctx0, scores, 1, 0, 2, 3));

    ggml_tensor * weights = ggml_soft_max(ctx0, scores);
    ggml_tensor * comp = ggml_mul(ctx0, values, weights);
    comp = ggml_sum_rows(ctx0, comp);
    comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));
    cb(comp, name, il);

    comp = llm_build_norm(ctx0, comp, hparams, norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(comp, name, il);

    ggml_tensor * comp_nope = ggml_view_3d(ctx0, comp, n_embd_head_nope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            0);
    ggml_tensor * comp_pe = ggml_view_3d(ctx0, comp, n_embd_head_rope, 1, n_blocks,
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head),
            ggml_row_size(comp->type, n_embd_head_nope));

    comp_pe = ggml_rope_ext(ctx0, comp_pe, comp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig,
            hparams.dsv4_compress_rope_base, freq_scale, ext_factor,
            dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(comp_pe, name, il);

    comp = ggml_concat(ctx0, comp_nope, comp_pe, 0);
    cb(comp, name, il);

    return comp;
}

//
// lightning indexer
//

// upstream: build_lid_top_k(). Fused path (ggml_lightning_indexer) not ported.
ggml_tensor * llm_build_context::build_dsv4_lid_top_k(
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int           il) const {
    const auto & layer   = model.layers[il];
    const auto & inp_lid = llama_dsv4_get_inputs(lctx).comp[LLAMA_DSV4_LID];

    const int64_t n_embd_indexer_head      = hparams.indexer_head_size;
    const int64_t n_embd_indexer_head_rope = hparams.n_rot;
    const int64_t n_embd_indexer_head_nope = n_embd_indexer_head - n_embd_indexer_head_rope;
    const int64_t n_indexer_head           = hparams.indexer_n_head;
    const int64_t nt                       = cur->ne[1];

    GGML_ASSERT(inp_lid.kq_mask);
    GGML_ASSERT(n_embd_indexer_head >= n_embd_indexer_head_rope);

    ggml_tensor * indexer_q = llm_build_lora_mm(lctx, ctx0, layer.indexer_attn_q_b, qr);
    indexer_q = ggml_reshape_3d(ctx0, indexer_q, n_embd_indexer_head, n_indexer_head, nt);
    cb(indexer_q, "lid_q", il);

    ggml_tensor * indexer_q_nope = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_nope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            0);
    ggml_tensor * indexer_q_pe = ggml_view_3d(ctx0, indexer_q, n_embd_indexer_head_rope, n_indexer_head, nt,
            ggml_row_size(indexer_q->type, n_embd_indexer_head),
            ggml_row_size(indexer_q->type, n_embd_indexer_head)*n_indexer_head,
            ggml_row_size(indexer_q->type, n_embd_indexer_head_nope));

    indexer_q_pe = ggml_rope_ext(ctx0, indexer_q_pe, inp_pos, nullptr, n_embd_indexer_head_rope,
            rope_type, n_ctx_orig, hparams.dsv4_compress_rope_base, freq_scale,
            ext_factor, dsv4_rope_attn_factor(freq_scale, ext_factor), beta_fast, beta_slow);
    cb(indexer_q_pe, "lid_q_pe", il);

    indexer_q = ggml_concat(ctx0, indexer_q_nope, indexer_q_pe, 0);
    if (inp_lid.k_rot) {
        indexer_q = dsv4_mul_mat_rot(ctx0, indexer_q, inp_lid.k_rot);
        cb(indexer_q, "lid_q_rot", il);
    }

    ggml_tensor * indexer_weights = llm_build_lora_mm(lctx, ctx0, layer.indexer_proj, cur);
    indexer_weights = ggml_scale(ctx0, indexer_weights, 1.0f/sqrtf(float(n_embd_indexer_head*n_indexer_head)));
    cb(indexer_weights, "lid_weights", il);

    ggml_tensor * indexer_k = llama_dsv4_get_comp_k(lctx, ctx0, LLAMA_DSV4_LID, il);
    const int64_t n_lid = inp_lid.kq_mask->ne[0];
    GGML_ASSERT(n_lid > 0);
    GGML_ASSERT(n_lid <= indexer_k->ne[2]);

    indexer_k = ggml_view_4d(ctx0, indexer_k,
            indexer_k->ne[0], indexer_k->ne[1], n_lid, indexer_k->ne[3],
            indexer_k->nb[1], indexer_k->nb[2], indexer_k->nb[3], 0);
    cb(indexer_k, "lid_k", il);

    const int64_t n_stream = indexer_k->ne[3];
    GGML_ASSERT(n_stream == 1 && "DSV4 first cut is single-stream (-np 1)");

    indexer_q = ggml_view_4d(ctx0, indexer_q,
            indexer_q->ne[0], indexer_q->ne[1], indexer_q->ne[2]/n_stream, n_stream,
            indexer_q->nb[1], indexer_q->nb[2], indexer_q->nb[3]/n_stream, 0);
    indexer_weights = ggml_view_4d(ctx0, indexer_weights,
            indexer_weights->ne[0], indexer_weights->ne[1]/n_stream, indexer_weights->ne[2], n_stream,
            indexer_weights->nb[1], indexer_weights->nb[2]/n_stream, indexer_weights->nb[3]/n_stream, 0);

    indexer_q = ggml_permute(ctx0, indexer_q, 0, 2, 1, 3);
    cb(indexer_q, "lid_q", il);
    indexer_k = ggml_permute(ctx0, indexer_k, 0, 2, 1, 3);
    cb(indexer_k, "lid_k", il);

    ggml_tensor * indexer_kq = ggml_mul_mat(ctx0, indexer_k, indexer_q);
    cb(indexer_kq, "lid_kq", il);

    indexer_kq = ggml_cont(ctx0, ggml_permute(ctx0, indexer_kq, 2, 1, 0, 3));
    cb(indexer_kq, "lid_kq", il);

    ggml_tensor * indexer_score = ggml_relu(ctx0, indexer_kq);
    indexer_score = ggml_mul(ctx0, indexer_score, indexer_weights);
    indexer_score = ggml_sum_rows(ctx0, indexer_score);
    indexer_score = ggml_cont(ctx0, ggml_permute(ctx0, indexer_score, 2, 1, 0, 3));
    cb(indexer_score, "lid_score", il);

    indexer_score = ggml_add(ctx0, indexer_score, inp_lid.kq_mask);
    cb(indexer_score, "lid_score_masked", il);

    const int64_t n_top_k = indexer_score->ne[0] < (int64_t) hparams.indexer_top_k
        ? indexer_score->ne[0] : (int64_t) hparams.indexer_top_k;

    // NOTE: ggml_top_k() returns a STRIDED VIEW in this tree (argsort + view_4d),
    // where upstream returns a contiguous I32 tensor. The ggml_cont() below is
    // what makes the downstream set_rows legal - do not "optimize" it away.
    ggml_tensor * top_k = ggml_cont(ctx0, ggml_top_k(ctx0, indexer_score, (int) n_top_k));
    cb(top_k, "lid_top_k", il);

    return top_k;
}

// upstream: build_top_k_mask(). Turns the top-k index list into an additive
// -inf/0 mask and folds in the base visibility mask.
ggml_tensor * llm_build_context::build_dsv4_top_k_mask(
        ggml_tensor * kq_mask,
        ggml_tensor * top_k,
        const char *  name,
        int           il) const {
    GGML_ASSERT(kq_mask);
    GGML_ASSERT(top_k);
    GGML_ASSERT(kq_mask->type == GGML_TYPE_F32 && "DSV4 comp masks are F32 in this tree (ggml_fill/ggml_set_rows)");

    ggml_tensor * kq_mask_all = ggml_fill(ctx0, kq_mask, -INFINITY);
    kq_mask_all = ggml_view_4d(ctx0, kq_mask_all, 1, kq_mask_all->ne[0], kq_mask_all->ne[1], kq_mask_all->ne[3],
            kq_mask_all->nb[0], kq_mask_all->nb[1], kq_mask_all->nb[2], 0);

    ggml_tensor * top_k_3d = ggml_view_4d(ctx0, top_k, top_k->ne[0], top_k->ne[1], top_k->ne[3], 1,
            top_k->nb[1], top_k->nb[2], top_k->ne[3]*top_k->nb[3], 0);

    // upstream picks F16 here under -fa; this tree's ggml_set_rows asserts a F32
    // source, and the CUDA backend supports the F32 -> F16/F32 store, so F32 it is.
    ggml_tensor * zeros = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, top_k_3d->ne[0], top_k_3d->ne[1], top_k_3d->ne[2]);
    zeros = ggml_fill(ctx0, zeros, 0.0f);

    ggml_tensor * kq_mask_top_k = ggml_set_rows(ctx0, kq_mask_all, zeros, top_k_3d);
    kq_mask_top_k = ggml_view_4d(ctx0, kq_mask_top_k,
            kq_mask_top_k->ne[1], kq_mask_top_k->ne[2], 1, kq_mask_top_k->ne[3],
            kq_mask_top_k->nb[2], kq_mask_top_k->nb[3], kq_mask_top_k->nb[3], 0);

    kq_mask_top_k = ggml_add(ctx0, kq_mask_top_k, kq_mask);
    cb(kq_mask_top_k, name, il);

    return kq_mask_top_k;
}

//
// the three per-layer attention regimes
//

// upstream: build_csa_lid_attention(). raw SWA window + top-k(512) sparse
// compressed rows, selected by the lightning indexer.
ggml_tensor * llm_build_context::build_dsv4_csa_lid_attention(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * qr,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const llama_dsv4_inputs & inp     = llama_dsv4_get_inputs(lctx);
    const auto              & inp_csa = inp.comp[LLAMA_DSV4_CSA];

    GGML_ASSERT(inp_csa.kq_mask);

    ggml_tensor * top_k = build_dsv4_lid_top_k(qr, cur, inp_pos, il);

    ggml_tensor * k_rot = inp.raw_k_rot;
    if (k_rot) {
        q  = dsv4_mul_mat_rot(ctx0, q,  k_rot);
        kv = dsv4_mul_mat_rot(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    ggml_build_forward_expand(gf, llama_dsv4_cpy_raw_k(lctx, ctx0, kv, inp.raw_k_idxs, il));

    ggml_tensor * raw_k = llama_dsv4_get_raw_k(lctx, ctx0, il);
    cb(raw_k, "csa_raw_k", il);

    ggml_tensor * csa_k = llama_dsv4_get_comp_k(lctx, ctx0, LLAMA_DSV4_CSA, il);
    const int64_t n_csa = inp_csa.kq_mask->ne[0];
    GGML_ASSERT(n_csa > 0);
    GGML_ASSERT(n_csa <= csa_k->ne[2]);

    csa_k = ggml_view_4d(ctx0, csa_k,
            csa_k->ne[0], csa_k->ne[1], n_csa, csa_k->ne[3],
            csa_k->nb[1], csa_k->nb[2], csa_k->nb[3], 0);
    cb(csa_k, "csa_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, csa_k, 2);
    cb(k_all, "csa_k_all", il);

    ggml_tensor * raw_mask = inp.raw_kq_mask;
    ggml_tensor * csa_mask = build_dsv4_top_k_mask(inp_csa.kq_mask, top_k, "csa_top_k_mask", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, raw_mask, csa_mask, 0);
    kq_mask = dsv4_finalize_kq_mask(ctx0, kq_mask, cparams.flash_attn);
    cb(kq_mask, "csa_lid_kq_mask", il);

    ggml_tensor * out = build_dsv4_attn_mha(gf, q, k_all, k_all, kq_mask, sinks, kq_scale, il);
    if (k_rot) {
        out = dsv4_mul_mat_rot(ctx0, out, k_rot);
    }
    cb(out, "attn_csa_lid", il);

    return out;
}

// upstream: build_hca_attention(). raw SWA window + every completed ratio-128
// compressed row (no sparsity).
ggml_tensor * llm_build_context::build_dsv4_hca_attention(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const llama_dsv4_inputs & inp     = llama_dsv4_get_inputs(lctx);
    const auto              & inp_hca = inp.comp[LLAMA_DSV4_HCA];

    GGML_ASSERT(inp_hca.kq_mask);

    ggml_tensor * k_rot = inp.raw_k_rot;
    if (k_rot) {
        q  = dsv4_mul_mat_rot(ctx0, q,  k_rot);
        kv = dsv4_mul_mat_rot(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    ggml_build_forward_expand(gf, llama_dsv4_cpy_raw_k(lctx, ctx0, kv, inp.raw_k_idxs, il));

    ggml_tensor * raw_k = llama_dsv4_get_raw_k(lctx, ctx0, il);
    cb(raw_k, "hca_raw_k", il);

    ggml_tensor * hca_k = llama_dsv4_get_comp_k(lctx, ctx0, LLAMA_DSV4_HCA, il);
    const int64_t n_hca = inp_hca.kq_mask->ne[0];
    GGML_ASSERT(n_hca > 0);
    GGML_ASSERT(n_hca <= hca_k->ne[2]);

    hca_k = ggml_view_4d(ctx0, hca_k,
            hca_k->ne[0], hca_k->ne[1], n_hca, hca_k->ne[3],
            hca_k->nb[1], hca_k->nb[2], hca_k->nb[3], 0);
    cb(hca_k, "hca_comp_k", il);

    ggml_tensor * k_all = ggml_concat(ctx0, raw_k, hca_k, 2);
    cb(k_all, "hca_k_all", il);

    ggml_tensor * kq_mask = ggml_concat(ctx0, inp.raw_kq_mask, inp_hca.kq_mask, 0);
    kq_mask = dsv4_finalize_kq_mask(ctx0, kq_mask, cparams.flash_attn);
    cb(kq_mask, "hca_kq_mask", il);

    ggml_tensor * out = build_dsv4_attn_mha(gf, q, k_all, k_all, kq_mask, sinks, kq_scale, il);
    if (k_rot) {
        out = dsv4_mul_mat_rot(ctx0, out, k_rot);
    }
    cb(out, "attn_hca", il);

    return out;
}

// upstream: build_raw_attention(). ratio 0 layers: sliding window only.
ggml_tensor * llm_build_context::build_dsv4_raw_attention(
        ggml_cgraph * gf,
        ggml_tensor * q,
        ggml_tensor * kv,
        ggml_tensor * sinks,
        float         kq_scale,
        int           il) const {
    const llama_dsv4_inputs & inp = llama_dsv4_get_inputs(lctx);

    ggml_tensor * k_rot = inp.raw_k_rot;
    if (k_rot) {
        q  = dsv4_mul_mat_rot(ctx0, q,  k_rot);
        kv = dsv4_mul_mat_rot(ctx0, kv, k_rot);
    }

    ggml_build_forward_expand(gf, q);
    ggml_build_forward_expand(gf, kv);

    ggml_build_forward_expand(gf, llama_dsv4_cpy_raw_k(lctx, ctx0, kv, inp.raw_k_idxs, il));

    ggml_tensor * k = llama_dsv4_get_raw_k(lctx, ctx0, il);

    ggml_tensor * kq_mask = dsv4_finalize_kq_mask(ctx0, inp.raw_kq_mask, cparams.flash_attn);

    ggml_tensor * out = build_dsv4_attn_mha(gf, q, k, k, kq_mask, sinks, kq_scale, il);
    if (k_rot) {
        out = dsv4_mul_mat_rot(ctx0, out, k_rot);
    }
    cb(out, "attn_raw", il);

    return out;
}

//
// full per-layer attention (upstream: build_attention)
//

ggml_tensor * llm_build_context::build_dsv4_attention(
        ggml_cgraph * gf,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        int           il) const {
    const auto & layer = model.layers[il];
    llama_dsv4_inputs & inp = llama_dsv4_get_inputs(lctx);

    const int64_t n_embd_head      = hparams.n_embd_head_k_full;
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t nt               = cur->ne[1];

    GGML_ASSERT(n_embd_head == (int64_t) hparams.n_embd_head_v_full);
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];

    const bool    use_compress_rope = ratio != 0;
    const float   freq_base_l       = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float   freq_scale_l      = use_compress_rope ? freq_scale : 1.0f;
    const float   ext_factor_l      = use_compress_rope ? ext_factor : 0.0f;
    const float   attn_factor_l     = dsv4_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float   beta_fast_l       = use_compress_rope ? beta_fast : 0.0f;
    const float   beta_slow_l       = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l      = use_compress_rope ? n_ctx_orig : 0;

    ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, layer.wq_a, cur);
    cb(qr, "qr", il);

    qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = ggml_rms_norm(ctx0, q, norm_rms_eps);
    cb(q, "q_norm", il);

    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            0);
    ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, nt,
            ggml_row_size(q->type, n_embd_head),
            ggml_row_size(q->type, n_embd_head)*n_head,
            ggml_row_size(q->type, n_embd_head_nope));
    q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(q_pe, "q_pe", il);
    q = ggml_concat(ctx0, q_nope, q_pe, 0);
    cb(q, "q", il);

    ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, layer.wkv, cur);
    kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    cb(kv, "kv_norm", il);

    ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            0);
    ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, nt,
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head),
            ggml_row_size(kv->type, n_embd_head_nope));
    kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    cb(kv_pe, "kv_pe", il);
    kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
    cb(kv, "kv", il);

    auto & inp_csa = inp.comp[LLAMA_DSV4_CSA];
    auto & inp_hca = inp.comp[LLAMA_DSV4_HCA];
    auto & inp_lid = inp.comp[LLAMA_DSV4_LID];

    // ---- HCA (ratio 128) compressor-state projections ----
    ggml_tensor * hca_state_kv    = nullptr;
    ggml_tensor * hca_state_score = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_hca.state_pos) {
        hca_state_kv = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wkv, cur);
        cb(hca_state_kv, "hca_state_kv", il);

        hca_state_score = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wgate, cur);
        cb(hca_state_score, "hca_state_score", il);

        ggml_tensor * ape_rows = ggml_get_rows(ctx0, layer.attn_comp_ape, inp_hca.state_pos);
        hca_state_score = ggml_add(ctx0, hca_state_score, ape_rows);
        cb(hca_state_score, "hca_state_score_ape", il);
    }

    // ---- CSA + LID (ratio 4) compressor-state update and commit ----
    if (ratio == DSV4_CSA_RATIO && inp_csa.state_pos) {
        ggml_tensor * csa_state_kv = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wkv, cur);
        cb(csa_state_kv, "csa_state_kv", il);

        ggml_tensor * csa_state_score = llm_build_lora_mm(lctx, ctx0, layer.attn_comp_wgate, cur);
        cb(csa_state_score, "csa_state_score", il);

        ggml_tensor * csa_ape_rows = ggml_get_rows(ctx0, layer.attn_comp_ape, inp_csa.state_pos);
        csa_state_score = ggml_add(ctx0, csa_state_score, csa_ape_rows);
        cb(csa_state_score, "csa_state_score_ape", il);

        GGML_ASSERT(inp_csa.state_write_idxs);

        ggml_tensor * csa_source_kv = ggml_concat(ctx0,
                llama_dsv4_get_state_kv(lctx, ctx0, LLAMA_DSV4_CSA, il), csa_state_kv, 1);
        ggml_tensor * csa_source_score = ggml_concat(ctx0,
                llama_dsv4_get_state_score(lctx, ctx0, LLAMA_DSV4_CSA, il), csa_state_score, 1);

        ggml_tensor * kv_comp_csa_state = build_dsv4_overlap_compressed_kv_from_state(
                csa_source_kv,
                csa_source_score,
                inp_csa.state_read_idxs,
                inp_csa.state_write_pos,
                layer.attn_comp_norm,
                DSV4_CSA_RATIO,
                n_embd_head,
                "csa_state_compress",
                il);

        if (inp_csa.k_rot) {
            kv_comp_csa_state = dsv4_mul_mat_rot(ctx0, kv_comp_csa_state, inp_csa.k_rot);
            cb(kv_comp_csa_state, "csa_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, llama_dsv4_cpy_comp_k(lctx, ctx0, LLAMA_DSV4_CSA,
                    kv_comp_csa_state, inp_csa.state_write_idxs, il));

        csa_state_kv    = dsv4_with_zero_dep(ctx0, csa_state_kv,    kv_comp_csa_state);
        csa_state_score = dsv4_with_zero_dep(ctx0, csa_state_score, kv_comp_csa_state);

        ggml_tensor * csa_persist_kv    = ggml_get_rows(ctx0, csa_state_kv,    inp_csa.state_persist_src_idxs);
        ggml_tensor * csa_persist_score = ggml_get_rows(ctx0, csa_state_score, inp_csa.state_persist_src_idxs);

        csa_state_kv = llama_dsv4_cpy_state_kv(lctx, ctx0, LLAMA_DSV4_CSA,
                csa_persist_kv, inp_csa.state_persist_dst_idxs, il);
        csa_state_score = llama_dsv4_cpy_state_score(lctx, ctx0, LLAMA_DSV4_CSA,
                csa_persist_score, inp_csa.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, csa_state_kv);
        ggml_build_forward_expand(gf, csa_state_score);

        ggml_tensor * lid_state_kv = llm_build_lora_mm(lctx, ctx0, layer.indexer_comp_wkv, cur);
        cb(lid_state_kv, "lid_state_kv", il);

        ggml_tensor * lid_state_score = llm_build_lora_mm(lctx, ctx0, layer.indexer_comp_wgate, cur);
        cb(lid_state_score, "lid_state_score", il);

        ggml_tensor * lid_ape_rows = ggml_get_rows(ctx0, layer.indexer_comp_ape, inp_lid.state_pos);
        lid_state_score = ggml_add(ctx0, lid_state_score, lid_ape_rows);
        cb(lid_state_score, "lid_state_score_ape", il);

        GGML_ASSERT(inp_lid.state_write_idxs);

        ggml_tensor * lid_source_kv = ggml_concat(ctx0,
                llama_dsv4_get_state_kv(lctx, ctx0, LLAMA_DSV4_LID, il), lid_state_kv, 1);
        ggml_tensor * lid_source_score = ggml_concat(ctx0,
                llama_dsv4_get_state_score(lctx, ctx0, LLAMA_DSV4_LID, il), lid_state_score, 1);

        ggml_tensor * kv_comp_lid_state = build_dsv4_overlap_compressed_kv_from_state(
                lid_source_kv,
                lid_source_score,
                inp_lid.state_read_idxs,
                inp_lid.state_write_pos,
                layer.indexer_comp_norm,
                DSV4_CSA_RATIO,
                hparams.indexer_head_size,
                "lid_state_compress",
                il);

        if (inp_lid.k_rot) {
            kv_comp_lid_state = dsv4_mul_mat_rot(ctx0, kv_comp_lid_state, inp_lid.k_rot);
            cb(kv_comp_lid_state, "lid_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, llama_dsv4_cpy_comp_k(lctx, ctx0, LLAMA_DSV4_LID,
                    kv_comp_lid_state, inp_lid.state_write_idxs, il));

        lid_state_kv    = dsv4_with_zero_dep(ctx0, lid_state_kv,    kv_comp_lid_state);
        lid_state_score = dsv4_with_zero_dep(ctx0, lid_state_score, kv_comp_lid_state);

        ggml_tensor * lid_persist_kv    = ggml_get_rows(ctx0, lid_state_kv,    inp_lid.state_persist_src_idxs);
        ggml_tensor * lid_persist_score = ggml_get_rows(ctx0, lid_state_score, inp_lid.state_persist_src_idxs);

        lid_state_kv = llama_dsv4_cpy_state_kv(lctx, ctx0, LLAMA_DSV4_LID,
                lid_persist_kv, inp_lid.state_persist_dst_idxs, il);
        lid_state_score = llama_dsv4_cpy_state_score(lctx, ctx0, LLAMA_DSV4_LID,
                lid_persist_score, inp_lid.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, lid_state_kv);
        ggml_build_forward_expand(gf, lid_state_score);
    }

    // ---- HCA commit ----
    ggml_tensor * hca_state_dep = nullptr;
    if (ratio == DSV4_HCA_RATIO && inp_hca.state_write_idxs) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        ggml_tensor * hca_source_kv = ggml_concat(ctx0,
                llama_dsv4_get_state_kv(lctx, ctx0, LLAMA_DSV4_HCA, il), hca_state_kv, 1);
        ggml_tensor * hca_source_score = ggml_concat(ctx0,
                llama_dsv4_get_state_score(lctx, ctx0, LLAMA_DSV4_HCA, il), hca_state_score, 1);

        ggml_tensor * kv_comp_hca = build_dsv4_hca_compressed_kv_from_state(
                hca_source_kv,
                hca_source_score,
                inp_hca.state_read_idxs,
                inp_hca.state_write_pos,
                layer.attn_comp_norm,
                n_embd_head,
                "hca_state_compress",
                il);

        if (inp_hca.k_rot) {
            kv_comp_hca = dsv4_mul_mat_rot(ctx0, kv_comp_hca, inp_hca.k_rot);
            cb(kv_comp_hca, "hca_state_compress_rot", il);
        }

        ggml_build_forward_expand(gf, llama_dsv4_cpy_comp_k(lctx, ctx0, LLAMA_DSV4_HCA,
                    kv_comp_hca, inp_hca.state_write_idxs, il));
        hca_state_dep = kv_comp_hca;
    }

    if (ratio == DSV4_HCA_RATIO && inp_hca.state_pos) {
        GGML_ASSERT(hca_state_kv);
        GGML_ASSERT(hca_state_score);

        hca_state_kv    = dsv4_with_zero_dep(ctx0, hca_state_kv,    hca_state_dep);
        hca_state_score = dsv4_with_zero_dep(ctx0, hca_state_score, hca_state_dep);

        ggml_tensor * hca_persist_kv    = ggml_get_rows(ctx0, hca_state_kv,    inp_hca.state_persist_src_idxs);
        ggml_tensor * hca_persist_score = ggml_get_rows(ctx0, hca_state_score, inp_hca.state_persist_src_idxs);

        hca_state_kv = llama_dsv4_cpy_state_kv(lctx, ctx0, LLAMA_DSV4_HCA,
                hca_persist_kv, inp_hca.state_persist_dst_idxs, il);
        hca_state_score = llama_dsv4_cpy_state_score(lctx, ctx0, LLAMA_DSV4_HCA,
                hca_persist_score, inp_hca.state_persist_dst_idxs, il);

        ggml_build_forward_expand(gf, hca_state_kv);
        ggml_build_forward_expand(gf, hca_state_score);
    }

    // ---- the attention itself ----
    // NOTE (deviation from upstream): upstream additionally requires
    // `inp_lid.k_rot != nullptr` to take the CSA/LID branch, because for its LID
    // cache attn_rot_k is force-enabled (llama-kv-cache.cpp: the DEEPSEEK4 /
    // GLM_DSA indexer special case), so it is never null there. Requiring it here
    // would silently demote every ratio-4 layer to plain SWA if the memory chunk
    // does not build a rotation matrix. The rotation is orthogonal and applied to
    // both indexer q and indexer k, so the scores - and therefore the top-k - are
    // identical with or without it; it exists to condition quantized indexer-K
    // storage. So: select on the ratio and the masks, and rotate only if k_rot
    // was actually provided.
    ggml_tensor * out = nullptr;
    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    if (ratio == DSV4_CSA_RATIO && inp_csa.kq_mask && inp_lid.kq_mask) {
        out = build_dsv4_csa_lid_attention(gf, q, kv, qr, cur, inp_pos, layer.attn_sinks, kq_scale, il);
    } else if (ratio == DSV4_HCA_RATIO && inp_hca.kq_mask) {
        out = build_dsv4_hca_attention(gf, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        out = build_dsv4_raw_attention(gf, q, kv, layer.attn_sinks, kq_scale, il);
    }

    // ---- de-rope, then the grouped output LoRA ----
    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            0);
    ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, nt,
            ggml_row_size(out->type, n_embd_head),
            ggml_row_size(out->type, n_embd_head)*n_head,
            ggml_row_size(out->type, n_embd_head_nope));

    // upstream calls this ggml_rope_ext_back; this tree still has the original
    // name ggml_rope_back with a byte-identical parameter list.
    out_pe = ggml_rope_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_concat(ctx0, out_nope, out_pe, 0);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0,
            ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

//
// the graph
//

ggml_cgraph * llm_build_context::build_deepseek4() {
    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc = hparams.dsv4_hc_mult;

    GGML_ASSERT(hc == 4 && "DSV4 loader/graph assume hyper_connection.count == 4");
    GGML_ASSERT(hparams.n_embd_head_k_full == hparams.n_embd_head_v_full);

    ggml_tensor * cur;

    ggml_tensor * inp         = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_tokens  = lctx.inp_tokens;   // needed by the hash-routed layers
    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    build_dsv4_inputs();

    const llama_dsv4_inputs & dsv4 = llama_dsv4_get_inputs(lctx);
    if (dsv4.raw_kq_mask) {
        ggml_build_forward_expand(gf, dsv4.raw_kq_mask);
    }

    // widen the residual stream to hc branches: [n_embd, hc, n_tokens]
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_dsv4_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "attn_norm", il);

        cur = build_dsv4_attention(gf, cur, inp_pos, il);

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_dsv4_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "ffn_norm", il);

        // The first dsv4_hash_layer_count blocks route by a token-id -> expert-id hash
        // table instead of by the router top-k. exp_probs_b does not exist there.
        // Skipped on the warm-up graph: that build forces n_expert_used = n_expert so
        // every expert tensor is touched, and the hash table is only n_expert_used wide
        // (upstream guards this the same way, with !cparams.warmup).
        ggml_tensor * selected_experts = nullptr;
        ggml_tensor * exp_probs_b      = layer.ffn_exp_probs_b;
        if ((uint32_t) il < hparams.dsv4_hash_layer_count && !is_warmup) {
            GGML_ASSERT(layer.ffn_gate_tid2eid);
            GGML_ASSERT(inp_tokens && "DSV4 hash-routed layers need token ids (embedding input is unsupported)");
            // NOTE: ffn_gate_tid2eid is I32; this tree's CUDA get_rows has no I32
            // path, so this one node falls back to the CPU backend (correct, but
            // it splits the graph on the first dsv4_hash_layer_count blocks).
            selected_experts = ggml_get_rows(ctx0, layer.ffn_gate_tid2eid, inp_tokens);
            cb(selected_experts, "ffn_moe_topk_hash", il);
            exp_probs_b = nullptr;
        }

        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                hparams.expert_weights_norm,
                hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, /*add_input*/ false,
                /*up_gate_exps*/ nullptr, /*up_gate_exps_b*/ nullptr,
                /*input_logits*/ nullptr, /*down_exps_s*/ nullptr,
                /*selected_experts_in*/ selected_experts);
        cb(moe_out, "ffn_moe_out", il);

        // shared expert. DS4's SwiGLU clamps the GATE to [-inf, limit] and the UP
        // to [-limit, limit] and only then multiplies - it is NOT silu-then-clamp.
        // Built inline instead of through llm_build_ffn() so the shared helper's
        // fused SILU path stays byte-untouched for every other arch.
        ggml_tensor * ffn_shexp;
        {
            ggml_tensor * up   = llm_build_lora_mm(lctx, ctx0, layer.ffn_up_shexp,   cur);
            cb(up, "ffn_up_shexp", il);
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_shexp, cur);
            cb(gate, "ffn_gate_shexp", il);

            const float limit = hparams.swiglu_limits_shared[il];
            if (limit > 1e-6f) {
                up   = ggml_clamp(ctx0, up,   -limit,     limit);
                gate = ggml_clamp(ctx0, gate, -INFINITY,  limit);
                cb(gate, "ffn_gate_shexp_clamped", il);
            }

            ggml_tensor * act = ggml_swiglu_split(ctx0, gate, up);
            cb(act, "ffn_swiglu_shexp", il);

            ffn_shexp = llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp, act);
        }
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        inpL = lctx.cvec.apply_to(ctx0, inpL, il);
        cb(inpL, "l_last", il);
    }

    if (inp_out_ids && n_tokens > 1) {
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        flat = ggml_get_rows(ctx0, flat, inp_out_ids);
        inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, n_outputs);
    }

    cur = build_dsv4_hc_head(inpL, model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "hc_head", -1);

    cur = build_output(lctx, ctx0, cur, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
