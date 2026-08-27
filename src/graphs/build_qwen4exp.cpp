#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
//
// The block structure is not the usual norm -> op -> residual. Instead the residual is WIDE:
// hc parallel streams of n_embd carried as [n_embd, hc, n_tokens]. A low-rank mixer collapses
// the streams into the single [n_embd, n_tokens] the token mixer and the MoE consume, and a
// scatter puts each block output back with a per-stream weight. The mixer IS the layer norm --
// this arch has no attn_norm, no ffn_norm and no output_norm at all.
//
// Reused from the rest of this fork rather than rewritten:
//   * the fused Gated DeltaNet kernel (delta_net, hc_mode) on the linear-attention layers
//   * build_std_attention (gated Q projection + q/k norms + IMRoPE) on the full-attention layers
//   * llm_build_std_moe_ffn for the expert FFN
// Both helpers normally fold in their own norm and residual add; hc_mode / a null norm weight /
// add_input=false turn those off so the hyper-connection owns them.

ggml_tensor * llm_build_context::build_qwen4exp_hc_mix(
        ggml_tensor *  x,
        ggml_tensor *  w_norm,
        ggml_tensor *  w_down,
        ggml_tensor *  w_up,
        ggml_tensor *  w_inject,
        ggml_tensor ** inject,
        int            il) {
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t hc_dim = hc*n_embd;
    const int64_t nt     = x->ne[2];

    // Grouped RMSNorm: the reduction is over ONE stream (ne[0] == n_embd), then a single
    // [hc_dim] gamma scales all of them. The converter folded each gamma to (1 + w).
    ggml_tensor * xn = ggml_rms_norm(ctx0, x, hparams.f_norm_rms_eps);
    xn = ggml_reshape_2d(ctx0, xn, hc_dim, nt);
    xn = ggml_mul(ctx0, xn, w_norm);
    cb(xn, "hc_norm", il);

    ggml_tensor * lo   = llm_build_lora_mm(lctx, ctx0, w_down, xn);
    lo = ggml_silu(ctx0, ggml_scale(ctx0, lo, 1.0f/(float) hc));
    ggml_tensor * gate = ggml_sigmoid(ctx0, llm_build_lora_mm(lctx, ctx0, w_up, lo));
    cb(gate, "hc_gate", il);

    ggml_tensor * gated = ggml_mul(ctx0, xn, gate);
    gated = ggml_reshape_3d(ctx0, gated, n_embd, hc, nt);

    // Collapse the streams by their mean. Summing hc strided views beats a permute + sum_rows
    // because hc is 4.
    ggml_tensor * mixed = ggml_cont(ctx0,
            ggml_view_2d(ctx0, gated, n_embd, nt, ggml_row_size(gated->type, n_embd)*hc, 0));
    for (int64_t c = 1; c < hc; ++c) {
        ggml_tensor * s = ggml_view_2d(ctx0, gated, n_embd, nt,
                ggml_row_size(gated->type, n_embd)*hc,
                ggml_row_size(gated->type, n_embd)*c);
        mixed = ggml_add(ctx0, mixed, s);
    }
    mixed = ggml_scale(ctx0, mixed, 1.0f/(float) hc);
    cb(mixed, "hc_mixed", il);

    if (inject) {
        *inject = llm_build_lora_mm(lctx, ctx0, w_inject, xn);
        cb(*inject, "hc_inject", il);
    }

    return mixed;
}

ggml_tensor * llm_build_context::build_qwen4exp_hc_combine(
        ggml_tensor * residual,
        ggml_tensor * block_out,
        ggml_tensor * inject,
        int           il) {
    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = residual->ne[2];

    // 2*sigmoid centres the scatter weights on 1, so a zero injection is a plain residual add.
    ggml_tensor * w = ggml_sigmoid(ctx0, ggml_scale(ctx0, inject, 1.0f/(float) hc));
    w = ggml_scale(ctx0, w, 2.0f);
    w = ggml_reshape_3d(ctx0, w, 1, hc, nt);

    ggml_tensor * b = ggml_reshape_3d(ctx0, block_out, n_embd, 1, nt);
    b = ggml_repeat_4d(ctx0, b, n_embd, hc, nt, 1);

    ggml_tensor * cur = ggml_add(ctx0, residual, ggml_mul(ctx0, b, w));
    cb(cur, "hc_combine", il);

    return cur;
}

ggml_cgraph * llm_build_context::build_qwen4exp() {

    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc          = hparams.dsv4_hc_mult;
    const int64_t n_embd_head = hparams.n_embd_head_v(0);
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(0));
    GGML_ASSERT(hc > 0 && "qwen4exp needs a hyper-connection count");

    delta_net delta(lctx, batch);

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    ggml_tensor * inpL        = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;

    // the recurrent-path inputs the fused delta-net kernel reads (same set qwen35moe builds)
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    lctx.inp_conv_seq_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_tokens, n_tokens);
    cb(lctx.inp_conv_seq_map, "inp_conv_seq_map", -1);
    ggml_set_input(lctx.inp_conv_seq_map);

    lctx.inp_qnext_state_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    cb(lctx.inp_qnext_state_mask, "inp_qnext_state_mask", -1);
    ggml_set_input(lctx.inp_qnext_state_mask);

    const float KQ_scale = hparams.f_attention_scale == 0.0f ? 1.0f/sqrtf(float(n_embd_head))
                                                             : hparams.f_attention_scale;

    if (hparams.ple_n_heads > 0) {
        // The PLE n-gram side path needs a host-filled hash-index input and a slice of recurrent
        // state for its dilated causal conv; neither exists in this fork yet. Refuse rather than
        // return logits that quietly leave the side path out.
        GGML_ABORT("qwen4exp: this GGUF carries a PLE key group, which the graph does not build yet");
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inject = nullptr;
        ggml_tensor * cur = build_qwen4exp_hc_mix(res_hc,
                layer.hc_attn_norm, layer.hc_attn_down, layer.hc_attn_up, layer.hc_attn_inject,
                &inject, il);
        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recurrent(il)) {
            // hc_mode: no input norm (the mixer above was it), no residual add (the combine
            // below is it), sigmoid output gate instead of Qwen3.5 silu.
            cur = delta.build_layer_attn_linear(ctx0, gf, cur, nullptr, il, cb, /*hc_mode*/ true);
        } else {
            // a null norm weight and add_input=false reduce build_std_attention to the block
            // itself: gated Q projection, q/k norms, IMRoPE (is_multi), attention, wo.
            cur = build_std_attention(gf, /*attn_norm*/ nullptr, cur, inp_pos, /*inp_out_ids*/ nullptr,
                    /*rope_factors*/ nullptr, KQ_mask, /*sinks*/ nullptr, /*inp_attn_scale*/ nullptr,
                    KQ_scale, 0.0f, /*n_swa*/ 0, il,
                    /*do_rope*/ true, /*add_graph_split*/ false, /*add_input*/ false,
                    /*is_norm*/ false, /*is_multi*/ true);
        }
        cb(cur, "attn_block_out", il);

        res_hc = build_qwen4exp_hc_combine(res_hc, cur, inject, il);

        cur = build_qwen4exp_hc_mix(res_hc,
                layer.hc_ffn_norm, layer.hc_ffn_down, layer.hc_ffn_up, layer.hc_ffn_inject,
                &inject, il);

        cur = llm_build_std_moe_ffn(ctx0, lctx, /*ffn_norm*/ nullptr, cur,
                layer.ffn_gate_inp,   nullptr,
                layer.ffn_up_exps,    nullptr,
                layer.ffn_gate_exps,  nullptr,
                layer.ffn_down_exps,  nullptr,
                nullptr,
                layer.ffn_up_shexp,   nullptr,
                layer.ffn_gate_shexp, nullptr,
                layer.ffn_down_shexp, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, /*add_input*/ false,
                layer.ffn_up_gate_exps, nullptr, layer.ffn_gate_inp_shexp);
        cb(cur, "ffn_out", il);

        res_hc = build_qwen4exp_hc_combine(res_hc, cur, inject, il);

        res_hc = lctx.cvec.apply_to(ctx0, res_hc, il);
        cb(res_hc, "l_out", il);
    }

    // The head mixer is the output norm; this arch ships no separate one. It takes no
    // injection because nothing is scattered back after it.
    ggml_tensor * cur = build_qwen4exp_hc_mix(res_hc,
            model.hc_head_norm, model.hc_head_down, model.hc_head_up, nullptr, nullptr, -1);

    // one gather at the very end: the wide residual has to stay whole through every layer,
    // so tokens are not dropped per-layer the way the dense graphs do it
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }
    cb(cur, "result_norm", -1);

    cur = build_output(lctx, ctx0, cur, model.output, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
