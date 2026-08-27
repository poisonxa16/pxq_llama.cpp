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


// PLE: per-layer n-gram hash embeddings, a side path that runs on the layers named by
// <arch>.ple.layers (layer 1 alone in the shipped model) and adds into the wide residual.
//
// The gather table per_layer_token_embd is 95.4 GiB in the shipped model - larger than all
// six cards put together - so it is meant to be pinned to host RAM with
//   -ot per_layer_token_embd=CPU
// The row indices come from inp_ple_rows, hashed host-side in llama_set_inputs.
ggml_tensor * llm_build_context::build_qwen4exp_ple(ggml_cgraph * gf, ggml_tensor * hidden, int il) {
    const int64_t hc      = hparams.dsv4_hc_mult;
    const int64_t hc_dim  = hc*n_embd;
    const int64_t n_heads = hparams.ple_n_heads;
    const int64_t kern    = hparams.ple_conv_kernel;
    const int64_t dil     = hparams.ple_ngram_size;
    const int64_t hist    = (kern - 1)*dil;

    GGML_ASSERT(model.tok_embd_per_layer != nullptr);

    // gather, then flatten the heads: get_rows lays the head dimension out slowest
    ggml_tensor * emb = ggml_get_rows(ctx0, model.tok_embd_per_layer, lctx.inp_ple_rows);
    emb = ggml_reshape_2d(ctx0, emb, hparams.ple_head_dim*n_heads, n_tokens);
    cb(emb, "ple_embd", il);

    ggml_tensor * key   = llm_build_lora_mm(lctx, ctx0, model.layers[il].ple_key,   emb);
    ggml_tensor * value = llm_build_lora_mm(lctx, ctx0, model.layers[il].ple_value, emb);

    // both norms reduce over ONE hc stream and scale with a weight over the whole hc*n_embd
    auto grouped_norm = [&](ggml_tensor * x, ggml_tensor * w) {
        ggml_tensor * t = ggml_reshape_3d(ctx0, x, n_embd, hc, n_tokens);
        t = ggml_rms_norm(ctx0, t, hparams.f_norm_rms_eps);
        t = ggml_reshape_2d(ctx0, t, hc_dim, n_tokens);
        t = ggml_mul(ctx0, t, w);
        return ggml_reshape_3d(ctx0, t, n_embd, hc, n_tokens);
    };

    key = grouped_norm(key, model.layers[il].ple_norm_key);
    ggml_tensor * query = grouped_norm(hidden, model.layers[il].ple_norm_query);

    // per-stream dot product, then a SIGNED square root before the sigmoid
    ggml_tensor * sc = ggml_sum_rows(ctx0, ggml_mul(ctx0, key, query));
    sc = ggml_scale(ctx0, sc, 1.0f/sqrtf((float) n_embd));
    ggml_tensor * mag  = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, sc), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, ggml_sgn(ctx0, sc), mag));
    cb(gate, "ple_gate", il);

    ggml_tensor * v3 = ggml_reshape_3d(ctx0, value, n_embd, 1, n_tokens);
    v3 = ggml_repeat_4d(ctx0, v3, n_embd, hc, n_tokens, 1);
    ggml_tensor * gated = ggml_mul(ctx0, v3, gate);
    cb(gated, "ple_gated_value", il);

    ggml_tensor * normalized = grouped_norm(
            ggml_reshape_2d(ctx0, gated, hc_dim, n_tokens), model.layers[il].ple_norm_conv);
    normalized = ggml_reshape_2d(ctx0, normalized, hc_dim, n_tokens);

    // The conv input of the PREVIOUS `hist` positions is not recomputable here: it depends on
    // the residual at those positions, which is gone. So it is carried across calls explicitly -
    // read as an input, written back host-side after the eval from this named output.
    ggml_set_name(normalized, "ple_conv_in");
    ggml_set_output(normalized);
    ggml_build_forward_expand(gf, normalized);

    // [hc_dim, hist + n_tokens] with the carried history in front
    ggml_tensor * padded = ggml_concat(ctx0, lctx.inp_ple_conv_hist, normalized, 1);
    cb(padded, "ple_conv_padded", il);

    // Depthwise causal conv DILATED by the n-gram size, as a sum of shifted copies:
    //   out[c, t] = sum_k w[k, c] * x[c, t - (K-1-k)*dilation]
    // written this way because ggml_conv_1d_dw is documented as unreliable.
    ggml_tensor * conv_out = nullptr;
    for (int64_t k = 0; k < kern; ++k) {
        const int64_t start = hist - (kern - 1 - k)*dil;
        ggml_tensor * shifted = ggml_view_2d(ctx0, padded, hc_dim, n_tokens,
                padded->nb[1], start*padded->nb[1]);

        // column k of the [kern, hc_dim] kernel is one weight per channel
        ggml_tensor * wk = ggml_cont(ctx0,
                ggml_view_2d(ctx0, model.layers[il].ple_conv1d, 1, hc_dim,
                        model.layers[il].ple_conv1d->nb[1], k*model.layers[il].ple_conv1d->nb[0]));
        wk = ggml_reshape_1d(ctx0, wk, hc_dim);
        if (wk->type != GGML_TYPE_F32) {
            wk = ggml_cast(ctx0, wk, GGML_TYPE_F32);
        }

        ggml_tensor * term = ggml_mul(ctx0, ggml_cont(ctx0, shifted), wk);
        conv_out = conv_out ? ggml_add(ctx0, conv_out, term) : term;
    }

    // A/B switch for attribution. The PLE conv is the one path here whose history is
    // carried across calls by hand (read back from a named node after the graph runs),
    // and with 11 graph splits across 6 GPUs that readback is the least-proven code in
    // this port - the single-GPU fixture could never exercise it. Setting
    // PXA_QWEN4EXP_NO_PLE_CONV=1 drops the conv branch entirely, leaving the rest of the
    // PLE side path intact, so a run with and without it isolates the carry.
    static const bool pxa_no_ple_conv = getenv("PXA_QWEN4EXP_NO_PLE_CONV") != nullptr;
    if (pxa_no_ple_conv) {
        conv_out = ggml_scale(ctx0, conv_out, 0.0f);
    }

    conv_out = ggml_silu(ctx0, conv_out);
    conv_out = ggml_reshape_3d(ctx0, ggml_cont(ctx0, conv_out), n_embd, hc, n_tokens);
    cb(conv_out, "ple_conv_out", il);

    return ggml_add(ctx0, hidden, ggml_add(ctx0, ggml_reshape_3d(ctx0, gated, n_embd, hc, n_tokens), conv_out));
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

    const bool has_ple = hparams.ple_n_heads > 0;
    if (has_ple) {
        const int64_t hc_dim = hc*n_embd;
        const int64_t hist   = (hparams.ple_conv_kernel - 1)*hparams.ple_ngram_size;

        lctx.inp_ple_rows = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, hparams.ple_n_heads*n_tokens);
        cb(lctx.inp_ple_rows, "inp_ple_rows", -1);
        ggml_set_input(lctx.inp_ple_rows);

        lctx.inp_ple_conv_hist = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hc_dim, hist);
        cb(lctx.inp_ple_conv_hist, "inp_ple_conv_hist", -1);
        ggml_set_input(lctx.inp_ple_conv_hist);
    }

    // the wide residual starts as hc identical copies of the embedding
    ggml_tensor * res_hc = ggml_repeat_4d(ctx0,
            ggml_reshape_3d(ctx0, inpL, n_embd, 1, n_tokens),
            n_embd, hc, n_tokens, 1);
    cb(res_hc, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // Second A/B rung for attribution: PXA_QWEN4EXP_NO_PLE=1 skips the whole PLE side
        // path, not just its conv. If the output collapse survives BOTH this and
        // PXA_QWEN4EXP_NO_PLE_CONV, then PLE is exonerated and the fault is in the other
        // state-carrying path - the delta-net recurrent state running under hc_mode on 36
        // of the 48 layers. Diagnostic only: the logits are wrong with PLE off, since the
        // shipped model genuinely has the side path.
        static const bool pxa_no_ple = getenv("PXA_QWEN4EXP_NO_PLE") != nullptr;
        if (has_ple && hparams.is_ple(il) && !pxa_no_ple) {
            res_hc = build_qwen4exp_ple(gf, res_hc, il);
        }

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
