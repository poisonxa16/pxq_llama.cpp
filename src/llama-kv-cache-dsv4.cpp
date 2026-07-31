// DeepSeek-V4 memory module — IMPLEMENTATION PLACEHOLDER.
//
// Adapted from llama.cpp `src/llama-kv-cache-dsv4.cpp` @ upstream commit 82dbc4f01
// (PR #24162 and follow-ups). Copyright (c) 2023-2026 The ggml authors. MIT.
//
// #####################################################################
// #                                                                   #
// #   ⛔ THIS MODULE IS NOT IMPLEMENTED. IT IS A LINKABLE SHELL.       #
// #                                                                   #
// #   The DeepSeek-V4 port spec calls this chunk C, an 800-1100 line  #
// #   REWRITE of upstream's 2,359-line llama-kv-cache-dsv4.{h,cpp}    #
// #   against this tree's flat POD `struct llama_kv_cache`. That      #
// #   rewrite WAS NEVER DELIVERED — no C-memory.patch was produced.   #
// #                                                                   #
// #   What is missing, concretely:                                    #
// #     * allocation of the three append-only compressed block        #
// #       streams (csa ratio 4, hca ratio 128, lid ratio 4) and the   #
// #       compressor ring state, per layer;                           #
// #     * dsv4_build_comp_plan() — upstream lines 418-600 — which     #
// #       computes state_read_idxs / state_write_idxs / n_visible.    #
// #       The spec is explicit that this arithmetic must be           #
// #       TRANSCRIBED LITERALLY from upstream and never re-derived:   #
// #       getting it wrong makes the model load and emit FLUENT       #
// #       GARBAGE with no loud signal;                                #
// #     * the get_/cpy_ view+set_rows accessors below;                #
// #     * upstream's four post-launch cache bugfixes 024c46ae4 /      #
// #       13f2b28b0 / 7f575c39d / 91d2fc387.                          #
// #                                                                   #
// #   Because a silently-wrong cache is the single worst failure      #
// #   mode for this port, this shell is FAIL-CLOSED:                  #
// #   llama_dsv4_memory_init() logs and returns false, so             #
// #   llama_new_context_with_model() REFUSES to instantiate a         #
// #   DeepSeek-V4 model. Every graph-side accessor GGML_ABORTs and    #
// #   is unreachable while that holds. Nothing here can produce a     #
// #   plausible-looking wrong answer.                                 #
// #                                                                   #
// #   DO NOT "fix" this by making init return true.                   #
// #                                                                   #
// #####################################################################
//
// Known integration gap for whoever writes chunk C, found during this
// integration and NOT yet addressed anywhere in the tree:
//
//   The graph builder sizes its input tensors from the plan
//   (build_dsv4_inputs() reads pl.state_pos.size(), pl.n_kv, ...), so the
//   plan for a ubatch must already exist when llama_build_graph() runs.
//   Chunk B only hooked llama_dsv4_set_inputs(), which llama.cpp calls at
//   :5823 — AFTER llama_build_graph() at :5727. A fourth hook is required,
//   e.g. `void llama_dsv4_plan_ubatch(llama_context &, const llama_batch &)`
//   called immediately before llama_build_graph() at :5727 and :6112.
//   Without it llama_dsv4_get_plan() returns a stale/empty plan and every
//   DSV4 input tensor is built with the wrong extent.

#include "llama-kv-cache-dsv4.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstdint>

// The only reason this exists is so llama_dsv4_get_inputs() can return a
// reference. It is never read: init fails before any graph is built.
static llama_dsv4_inputs g_dsv4_inputs_unimplemented;

static const llama_dsv4_comp_plan g_dsv4_plan_unimplemented;

[[noreturn]] static void dsv4_not_implemented(const char * fn) {
    GGML_ABORT("DeepSeek-V4: %s() is not implemented — the DSV4 memory module (chunk C) "
               "was never written. This call is unreachable while llama_dsv4_memory_init() "
               "returns false; reaching it means someone made init succeed without "
               "implementing the cache.", fn);
}

//
// lifecycle
//

bool llama_dsv4_memory_init(llama_context & /*lctx*/, ggml_type /*type_k*/, ggml_type /*type_v*/,
                            uint32_t /*kv_size*/, bool /*offload*/) {
    LLAMA_LOG_ERROR(
        "%s: DeepSeek-V4 is NOT RUNNABLE in this build.\n"
        "%s:   The arch metadata, tensor load path, graph and quantizer landed, but the DSV4\n"
        "%s:   memory module (the three compressed block streams and the per-ubatch\n"
        "%s:   compression plan) is not implemented — see src/llama-kv-cache-dsv4.cpp.\n"
        "%s:   Refusing to create a context rather than running on an empty cache and\n"
        "%s:   emitting fluent garbage. Conversion and quantization are unaffected.\n",
        __func__, __func__, __func__, __func__, __func__, __func__);
    return false;
}

void llama_dsv4_set_inputs(llama_context & /*lctx*/, const llama_batch & /*batch*/) {
    dsv4_not_implemented(__func__);
}

void llama_dsv4_memory_free(llama_context & /*lctx*/) {
    // Reachable: llama_free() calls this unconditionally for DEEPSEEK4, including
    // on the cleanup path after llama_dsv4_memory_init() failed. Nothing to free.
}

//
// graph-side accessors — all unreachable while init fails
//

llama_dsv4_inputs & llama_dsv4_get_inputs(llama_context & /*lctx*/) {
    dsv4_not_implemented(__func__);
    return g_dsv4_inputs_unimplemented;
}

const llama_dsv4_comp_plan & llama_dsv4_get_plan(const llama_context & /*lctx*/, llama_dsv4_stream /*s*/) {
    dsv4_not_implemented(__func__);
    return g_dsv4_plan_unimplemented;
}

uint32_t llama_dsv4_get_raw_n_kv(const llama_context & /*lctx*/) {
    dsv4_not_implemented(__func__);
}

int64_t llama_dsv4_get_raw_nrot(const llama_context & /*lctx*/) {
    dsv4_not_implemented(__func__);
}

int64_t llama_dsv4_get_comp_nrot(const llama_context & /*lctx*/, llama_dsv4_stream /*s*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_get_raw_k(const llama_context & /*lctx*/, ggml_context * /*ctx*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_cpy_raw_k(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                   ggml_tensor * /*k_cur*/, ggml_tensor * /*k_idxs*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_get_comp_k(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                    llama_dsv4_stream /*s*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_cpy_comp_k(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                    llama_dsv4_stream /*s*/, ggml_tensor * /*k_cur*/,
                                    ggml_tensor * /*k_idxs*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_get_state_kv(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                      llama_dsv4_stream /*s*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_get_state_score(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                         llama_dsv4_stream /*s*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_cpy_state_kv(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                      llama_dsv4_stream /*s*/, ggml_tensor * /*cur*/,
                                      ggml_tensor * /*idxs*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}

ggml_tensor * llama_dsv4_cpy_state_score(const llama_context & /*lctx*/, ggml_context * /*ctx*/,
                                         llama_dsv4_stream /*s*/, ggml_tensor * /*cur*/,
                                         ggml_tensor * /*idxs*/, int32_t /*il*/) {
    dsv4_not_implemented(__func__);
}
