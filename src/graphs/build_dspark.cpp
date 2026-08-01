#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

#include <cmath>
#include <cstdio>

// ============================================================================
// DSpark block drafter (M3).
//
// Three DS4 blocks in the RAW-SWA regime, run over a block of block_size+1 rows in ONE
// forward pass. Established from the reference implementation (antirez ds4, HEAD
// 54b36ed), not guessed; the file:line citations below are into that tree.
//
// The five properties that are load-bearing for acceptance (do not "simplify" any of
// them - each is a silent quality bug, not a crash):
//
//   1. ROW 0 IS INVIOLATE ACROSS ALL THREE STAGES (ds4.c:30785-30806). It carries the
//      target signal into every stage. Here it is enforced explicitly: after each
//      stage's FFN hc_post the row-0 slice is restored from the stage-entry stream, so
//      no arithmetic anywhere in the stage can leak into it.
//   2. ATTENTION IS NON-CAUSAL WITHIN THE BLOCK (ds4_cuda.cu:13466-13530). Every draft
//      row sees every other draft row INCLUDING later ones. Without it, block drafting
//      does not work at all. This is a MASK property, not an op: the mask is a graph
//      input and llama_set_inputs fills the intra-block square open in both directions.
//   3. The Markov chain is genuinely sequential (M4).
//   4. Confidence is evaluated BEFORE the Markov bias and short-circuits (M4).
//   5. The scheduler is part of the algorithm (M5).
//
// Row 0 shares its position with row 1: positions are
//   [pos, pos, pos+1, pos+2, pos+3, pos+4]                        (ds4.c:30261-30264)
// and the draft rows are seeded from the TARGET's token embeddings of
//   [last_real_token, NOISE, NOISE, NOISE, NOISE]                 (ds4.c:30266-30270)
// where NOISE = dspark.noise_token_id is a LEARNED placeholder embedding for the
// not-yet-known positions. That learned placeholder is what makes one-pass block
// drafting possible at all.
//
// DELIBERATE DEVIATION, numerically inert: the reference computes Q for rows 1..k only
// while computing KV for all k+1 rows. Here Q is computed for all k+1 rows and row 0's
// attention output is then discarded by the row-0 restore in (1). That keeps every
// shape rectangular and constant, which is what `can_reuse_graph` needs (a variable row
// count would evict the cached drafter graph every cycle - see the plan's 4.2). The cost
// is 1/(k+1) of one attention; the benefit is graph reuse on every cycle.
//
// WHY THE DS4 ATTENTION HELPERS ARE NOT REUSED HERE (verified, and it is not a style
// choice): llama_dsv4_get_inputs() DISCARDS its context argument and returns the
// process-global module (llama-kv-cache-dsv4.cpp:670), and build_dsv4_raw_attention is
// welded to it - it calls llama_dsv4_cpy_raw_k(), which would write the DRAFTER's latent
// K rows into the TARGET model's layer-0/1/2 raw SWA ring. Not a compile error, not a
// crash: silent target-KV corruption presenting as low acceptance. The drafter therefore
// runs its attention against its OWN standard llama_kv_cache. What IS reused is
// everything verified clean of global state: build_dsv4_attn_mha, build_dsv4_hc_pre /
// _post / _head (the 20-iteration Sinkhorn is already one fused node), and the
// deepseek4 MoE path.
//
// A DSpark stage carries no compressor, no lightning indexer and no ffn_gate_tid2eid -
// its tensor set is byte-for-byte DS4's blk.0, i.e. compress ratio 0. So there is no
// CSA/HCA branch here and no hash routing, and llm_load_hparams_dspark() forces
// dsv4_hash_layer_count to 0 so a stage can never take the target's hash-routing branch
// and read a table that does not exist in the file.
// ============================================================================

ggml_cgraph * llm_build_context::build_dspark() {
    const auto & tgt = model.dspark_target;
    GGML_ASSERT(tgt && "DSpark drafter used without llama_dspark_bind_target()");
    GGML_ASSERT(tgt->tok_embd && tgt->output && "DSpark borrows the target's embd/lm_head");

    ggml_cgraph * gf = new_graph_custom();

    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t n_block    = hparams.dspark_block_size;      // k candidate positions
    const int64_t n_rows     = n_block + 1;                    // + the target row
    const int64_t n_capture  = hparams.dspark_n_target_layers;

    GGML_ASSERT(hc == 4 && "DSpark graph assumes hyper_connection.count == 4");
    GGML_ASSERT(n_block > 0);
    GGML_ASSERT(n_tokens == n_rows && "the drafter batch is exactly block_size+1 rows");
    GGML_ASSERT(hparams.n_embd_head_k_full == hparams.n_embd_head_v_full);

    // The grouped output LoRA reshapes wo_a to {ne[0], o_lora_rank, o_groups}. Reading
    // o_lora_rank straight off attn_output_a.ne[1] yields o_lora_rank*o_groups (8192 on
    // this checkpoint, not 1024) and this reshape is where that lands. Assert the
    // relation the loader derived so a regression cannot pass silently.
    GGML_ASSERT(hparams.dsv4_o_group_count > 0 && hparams.dsv4_o_lora_rank > 0);
    GGML_ASSERT(model.layers[0].wo_a->ne[1] ==
            (int64_t) hparams.dsv4_o_lora_rank * (int64_t) hparams.dsv4_o_group_count &&
            "wo_a rank must be o_lora_rank*o_groups - see llm_load_hparams_dspark()");

    // ---- inputs -------------------------------------------------------------
    // The captured target hidden states, [n_capture*n_embd, 1], host-fed by the spec loop.
    lctx.inp_dspark_cap = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_capture*n_embd, 1);
    ggml_set_input(lctx.inp_dspark_cap);
    cb(lctx.inp_dspark_cap, "dspark_cap", -1);

    lctx.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_rows);
    ggml_set_input(lctx.inp_tokens);
    cb(lctx.inp_tokens, "dspark_inp_tokens", -1);

    ggml_tensor * inp_pos = build_inp_pos();

    // Non-causal within the block. Filled by llama_set_inputs; NOT build_inp_KQ_mask(),
    // whose fill is driven by cparams.causal_attn and would hide row 3 from row 1.
    const auto & kv_self = lctx.kv_self;
    const int64_t n_kv   = kv_self.n;
    {
        const int64_t n_mask_rows = GGML_PAD(n_rows, GGML_KQ_MASK_PAD);
        lctx.inp_KQ_mask = ggml_new_tensor_2d(ctx0,
                cparams.flash_attn ? GGML_TYPE_F16 : GGML_TYPE_F32, n_kv, n_mask_rows);
        ggml_set_input(lctx.inp_KQ_mask);
        cb(lctx.inp_KQ_mask, "dspark_kq_mask", -1);
    }

    // ---- 3a: main_x = RMSNorm(main_proj(concat(h40,h41,h42)), main_norm) ----
    //                                                          ds4.c:29981-30012
    ggml_tensor * main_x = llm_build_lora_mm(lctx, ctx0,
            model.layers[0].dspark_main_proj, lctx.inp_dspark_cap);
    main_x = llm_build_norm(ctx0, main_x, hparams,
            model.layers[0].dspark_main_norm, nullptr, LLM_NORM_RMS, cb, -1);
    cb(main_x, "dspark_main_x", -1);                                    // [n_embd, 1]

    // ---- 3b: assemble the block -------------------------------------------
    // rows 1..k from the TARGET's token embeddings (borrowed, not owned):
    ggml_tensor * tok = ggml_get_rows(ctx0, tgt->tok_embd, lctx.inp_tokens);
    cb(tok, "dspark_tok_embd", -1);                                 // [n_embd, n_rows]

    // row 0 := main_x, rows 1..k := the seeded embeddings.
    ggml_tensor * rows_1k = ggml_view_2d(ctx0, tok, n_embd, n_block,
            tok->nb[1], tok->nb[1]);
    ggml_tensor * block = ggml_concat(ctx0, main_x, ggml_cont(ctx0, rows_1k), 1);
    cb(block, "dspark_block_in", -1);                               // [n_embd, n_rows]

    // widen to the hc branches: [n_embd, hc, n_rows]
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, block, n_embd, 1, n_rows);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_rows, 1);
    cb(inpL, "dspark_hc_init", -1);

    const int64_t n_embd_head      = hparams.n_embd_head_k_full;
    const int64_t n_embd_head_rope = hparams.n_rot;
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const float   kq_scale         = 1.0f/sqrtf(float(n_embd_head));

    // ratio-0 layers take the PLAIN rope basis, never the compressed one
    // (build_dsv4_attention: use_compress_rope = ratio != 0).
    const float   freq_base_l   = freq_base;
    const float   freq_scale_l  = 1.0f;
    const float   ext_factor_l  = 0.0f;
    // dsv4_rope_attn_factor() returns exactly 1.0f when ext_factor == 0, which is the
    // ratio-0 case unconditionally. Inlined because the helper is file-static in
    // build_deepseek4.cpp and duplicating it would be a second thing to keep in sync.
    const float   attn_factor_l = 1.0f;
    const float   beta_fast_l   = 0.0f;
    const float   beta_slow_l   = 0.0f;
    const int32_t n_ctx_orig_l  = 0;

    // ---- 3c: the three stages ---------------------------------------------
    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * stage_in = inpL;          // for the row-0 restore below
        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;
        ggml_tensor * cur  = nullptr;

        cur = build_dsv4_hc_pre(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base,
                &post, &comb, il);
        cur = llm_build_norm(ctx0, cur, hparams, layer.attn_norm, nullptr, LLM_NORM_RMS, cb, il);
        cb(cur, "dspark_attn_norm", il);

        // ---- attention, against the DRAFTER's own cache -------------------
        {
            ggml_tensor * qr = llm_build_lora_mm(lctx, ctx0, layer.wq_a, cur);
            qr = llm_build_norm(ctx0, qr, hparams, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, cb, il);

            ggml_tensor * q = llm_build_lora_mm(lctx, ctx0, layer.wq_b, qr);
            q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, n_rows);
            q = ggml_rms_norm(ctx0, q, norm_rms_eps);

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_nope, n_head, n_rows,
                    ggml_row_size(q->type, n_embd_head),
                    ggml_row_size(q->type, n_embd_head)*n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_rope, n_head, n_rows,
                    ggml_row_size(q->type, n_embd_head),
                    ggml_row_size(q->type, n_embd_head)*n_head,
                    ggml_row_size(q->type, n_embd_head_nope));
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_embd_head_rope, rope_type,
                    n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                    beta_fast_l, beta_slow_l);
            q = ggml_concat(ctx0, q_nope, q_pe, 0);
            cb(q, "dspark_q", il);

            // one shared 512-wide latent for all 64 heads
            ggml_tensor * kv = llm_build_lora_mm(lctx, ctx0, layer.wkv, cur);
            kv = llm_build_norm(ctx0, kv, hparams, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, cb, il);
            kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, n_rows);

            ggml_tensor * kv_nope = ggml_view_3d(ctx0, kv, n_embd_head_nope, 1, n_rows,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head), 0);
            ggml_tensor * kv_pe = ggml_view_3d(ctx0, kv, n_embd_head_rope, 1, n_rows,
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head),
                    ggml_row_size(kv->type, n_embd_head_nope));
            kv_pe = ggml_rope_ext(ctx0, kv_pe, inp_pos, nullptr, n_embd_head_rope, rope_type,
                    n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                    beta_fast_l, beta_slow_l);
            kv = ggml_concat(ctx0, kv_nope, kv_pe, 0);
            cb(kv, "dspark_kv", il);

            // ALL k+1 rows are written, including row 0 (ds4: "KV for all 6 rows").
            // MLA shares one latent, so the same tensor serves as K and as V.
            llm_build_kv_store(lctx, ctx0, hparams, cparams, kv_self, gf,
                    kv, /*v_cur*/ nullptr, n_rows, kv_head, cb, il);

            ggml_tensor * k = ggml_view_3d(ctx0, kv_self.k_l[il],
                    n_embd_head, 1, n_kv,
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head),
                    ggml_row_size(kv_self.k_l[il]->type, n_embd_head), 0);
            cb(k, "dspark_k_all", il);

            // the window ++ the k+1 block rows, with the intra-block square open in
            // BOTH directions - non-causal (property 2).
            ggml_tensor * out = build_dsv4_attn_mha(gf, q, k, k,
                    lctx.inp_KQ_mask, layer.attn_sinks, kq_scale, il);

            // de-rope, then the grouped output LoRA
            out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, n_rows);
            ggml_tensor * out_nope = ggml_view_3d(ctx0, out, n_embd_head_nope, n_head, n_rows,
                    ggml_row_size(out->type, n_embd_head),
                    ggml_row_size(out->type, n_embd_head)*n_head, 0);
            ggml_tensor * out_pe = ggml_view_3d(ctx0, out, n_embd_head_rope, n_head, n_rows,
                    ggml_row_size(out->type, n_embd_head),
                    ggml_row_size(out->type, n_embd_head)*n_head,
                    ggml_row_size(out->type, n_embd_head_nope));
            out_pe = ggml_rope_back(ctx0, out_pe, inp_pos, nullptr, n_embd_head_rope, rope_type,
                    n_ctx_orig_l, freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l,
                    beta_fast_l, beta_slow_l);
            out = ggml_concat(ctx0, out_nope, out_pe, 0);

            out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, n_rows);
            out = ggml_permute(ctx0, out, 0, 2, 1, 3);
            ggml_tensor * oa = ggml_mul_mat(ctx0,
                    ggml_reshape_3d(ctx0, layer.wo_a, layer.wo_a->ne[0], o_lora_rank, n_groups), out);
            oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
            oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, n_rows);

            cur = llm_build_lora_mm(lctx, ctx0, layer.wo_b, oa);
            cb(cur, "dspark_attn_out", il);
        }

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        cb(inpL, "dspark_hc_attn_post", il);

        // ---- FFN: 256-expert top-k MoE + one shared expert ----------------
        residual = inpL;
        cur = build_dsv4_hc_pre(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                &post, &comb, il);
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = llm_build_norm(ctx0, cur, hparams, layer.ffn_norm, nullptr, LLM_NORM_RMS, cb, il);

        ggml_tensor * moe_out = llm_build_moe_ffn(ctx0, lctx, cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU,
                hparams.expert_weights_norm,
                hparams.expert_weights_scale != 0.0f,
                hparams.expert_weights_scale,
                (llm_expert_gating_func_type) hparams.expert_gating_func,
                cb, il, gf, /*add_input*/ false,
                nullptr, nullptr, nullptr, nullptr, /*selected_experts_in*/ nullptr);
        cb(moe_out, "dspark_moe_out", il);

        ggml_tensor * ffn_shexp;
        {
            ggml_tensor * up   = llm_build_lora_mm(lctx, ctx0, layer.ffn_up_shexp,   cur);
            ggml_tensor * gate = llm_build_lora_mm(lctx, ctx0, layer.ffn_gate_shexp, cur);
            const float limit = hparams.swiglu_limits_shared[il];
            if (limit > 1e-6f) {
                up   = ggml_clamp(ctx0, up,   -limit,    limit);
                gate = ggml_clamp(ctx0, gate, -INFINITY, limit);
            }
            ffn_shexp = llm_build_lora_mm(lctx, ctx0, layer.ffn_down_shexp,
                    ggml_swiglu_split(ctx0, gate, up));
        }
        cur = ggml_add(ctx0, moe_out, ffn_shexp);

        inpL = build_dsv4_hc_post(cur, residual, post, comb, il);
        cb(inpL, "dspark_hc_ffn_post", il);

        // ---- PROPERTY 1: row 0 is restored, bit-for-bit, from the stage entry ----
        // The hand-off replaces rows 1..k ONLY (ds4.c:30785-30806). Enforced here as an
        // explicit splice rather than trusted to the arithmetic, so a future change to
        // any stage internal cannot leak into the target row.
        {
            ggml_tensor * row0 = ggml_view_3d(ctx0, stage_in, n_embd, hc, 1,
                    stage_in->nb[1], stage_in->nb[2], 0);
            ggml_tensor * rows = ggml_view_3d(ctx0, inpL, n_embd, hc, n_block,
                    inpL->nb[1], inpL->nb[2], inpL->nb[2]);
            inpL = ggml_concat(ctx0, ggml_cont(ctx0, row0), ggml_cont(ctx0, rows), 2);
            cb(inpL, "dspark_row0_preserved", il);
        }
    }

    // ---- 3d: hc_head -> RMSNorm(norm) -> the TARGET's lm_head --------------
    //                                                          ds4.c:31548-31714
    ggml_tensor * cur = build_dsv4_hc_head(inpL,
            model.hc_head_fn, model.hc_head_scale, model.hc_head_base);
    cb(cur, "dspark_hc_head", -1);

    cur = llm_build_norm(ctx0, cur, hparams,
            model.layers[n_layer-1].dspark_norm, nullptr, LLM_NORM_RMS, cb, -1);

    // drop the target row: only the k draft positions get logits
    cur = ggml_view_2d(ctx0, cur, n_embd, n_block, cur->nb[1], cur->nb[1]);
    cur = ggml_cont(ctx0, cur);
    cb(cur, "dspark_head_in", -1);

    cur = llm_build_lora_mm(lctx, ctx0, tgt->output, cur);
    cb(cur, "result_output", -1);                                 // [n_vocab, n_block]

    ggml_build_forward_expand(gf, cur);

    // 3e (M4): the rank-256 Markov chain and the confidence head are applied to these
    // logits, sequentially over k, with confidence evaluated BEFORE the bias at each
    // position and short-circuiting (ds4.c:31941-31986, :32060-32227). Composed from
    // existing ops first, as the bit-exactness oracle for any later fused kernel.

    return gf;
}
