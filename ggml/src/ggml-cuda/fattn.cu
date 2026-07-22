//
// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//

#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16-interface.cuh"
#include "fattn-vec-f32-interface.cuh"
#include "fattn-wmma-f16-interface.cuh"
#include "fattn-mma-f16-interface.cuh"
#include "fattn-new-mma.cuh"
#include "fattn.cuh"
#include "convert.cuh"

#include <cstdint>
#include <cstdlib>

#define FATTN_KQ_STRIDE 256

static inline bool mma_better_than_turing(const int cc) {
    return GGML_CUDA_CC_IS_NVIDIA(cc) && ggml_cuda_highest_compiled_arch(cc) > CC_TURING;
}

// PXQ port of upstream ik_llama.cpp PR #2144 (merged 2026-07-17):
// on sm_60 (P100/CC_PASCAL) the fp16 FA vec kernel accumulates the online-softmax
// denominator and the P.V product in fp16, flipping ~3-4% of decode top-1 tokens vs an
// all-fp32 reference. P100 decode is memory-bandwidth-bound, so routing decode
// (batch <= 8) to the fp32 vec kernel is measured-free upstream (tg128 96.79 vs 96.61
// t/s; neutral-or-better at long context). Prefill and the D=256 vec path stay on
// vec_f16. sm_61 (1080Ti) is unaffected (no fast fp16 -> already vec_f32).
// Env kill-switch: PXQ_SM60_FA_VEC_F32=0 restores the old fp16-accumulating route.
static bool pxq_sm60_fa_vec_f32_enabled() {
    static const bool enabled = [] {
        const char * v = getenv("PXQ_SM60_FA_VEC_F32");
        return !(v && v[0] == '0');
    }();
    return enabled;
}

static inline bool pxq_use_sm60_vec_f32(const int cc, const ggml_tensor * Q) {
    return cc == CC_PASCAL && Q->ne[1] <= 8 && pxq_sm60_fa_vec_f32_enabled();
}

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];

    ggml_tensor local_dst, Kl, Vl, Ml;
    if (n_swa > 0) {
        int ntokens = std::max(FATTN_KQ_STRIDE, int(Q->ne[1]));
        int nton = FATTN_KQ_STRIDE*((ntokens + n_swa + FATTN_KQ_STRIDE - 1)/FATTN_KQ_STRIDE);
        int first = K->ne[1] - nton;
        local_dst = *dst;
        local_dst.op_params[4] = 0;
        if (first > 0) { // PXA: windowed SWA slice RE-ENABLED (tile nb31 stride fix makes it correct + fast)
            local_dst = *dst;
            Kl = *K; Kl.ne[1] = nton; Kl.data = (char *)K->data + K->nb[1]*first;
            Vl = *V; Vl.ne[1] = nton; Vl.data = (char *)V->data + V->nb[1]*first;
            Ml = *mask; Ml.ne[0] = nton; Ml.data = (char *)mask->data + mask->nb[0]*first;
            local_dst.src[1] = &Kl;
            local_dst.src[2] = &Vl;
            local_dst.src[3] = &Ml;
            local_dst.op_params[4] = 0;
            dst = &local_dst;
        }
        dst = &local_dst;
    }

    // On AMD the tile kernels perform poorly, use the vec kernel instead:
    if (cc >= CC_OFFSET_AMD) {
        if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                if (pxq_use_sm60_vec_f32(cc, Q)) { // PR #2144: sm_60 decode -> fp32 accumulation
                    ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
                } else {
                    ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
                }
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 128 && V->ne[0] == 128 && Q->ne[0] == 128 && Q->ne[1] == 1 &&
            (Q->ne[2] / K->ne[2] == 12 || Q->ne[2] / K->ne[2] == 6 || Q->ne[2] / K->ne[2] == 10)) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    if (new_mma_available(cc) && K->ne[0] == 256 && V->ne[0] == 256 && Q->ne[0] == 256 && Q->ne[1] == 1 && Q->ne[2] / K->ne[2] == 6) {
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        return;
    }

    //
    // It turns out the new new MMA implementation is slower than the
    // previous MMA implementation.
    // Hence, we use it only for DeepSeek with MLA enabled, where head sizes are 576, 512,
    // so no other implementation works.
    //

    if (new_mma_available(cc) &&
            ((K->ne[0] == 576 && V->ne[0] == 512) ||
             (K->ne[0] == 320 && V->ne[0] == 256) ||
             (K->ne[0] == 512 && V->ne[0] == 512) ||
             (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        //printf("Using ggml_cuda_flash_attn_ext_mma_new\n");
        ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
        return;
    }

    //
    // We need this because I haven't adapted new MMA kernels to work for different
    // K and V head sizes.
    // We also need it if the new MMA is not available
    //
    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        // Attention-sink models (e.g. gpt-oss): the wmma kernel does not implement attention
        // sinks, so on sm_70 (Volta/V100) the prefill path would otherwise drop them and corrupt
        // the context. The tile kernel is now sink-aware for head sizes 64/128 -> route there.
        if (dst->src[4] != nullptr && K->ne[0] == V->ne[0] && (K->ne[0] == 64 || K->ne[0] == 128)) {
            if (precision == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
            return;
        }
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
        return;
    }

    // As mentioned above, the new-new MMA is slower then the new MMA.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
    //ggml_cuda_flash_attn_ext_mma_new(ctx, dst);
}

bool ggml_cuda_fattn_is_supported(ggml_backend_cuda_context & ctx, const ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int32_t precision = KQV->op_params[3];
    const int32_t n_swa = KQV->op_params[4];
    if (cc >= CC_OFFSET_AMD) {
        return precision == GGML_PREC_DEFAULT ? ggml_cuda_fattn_vec_f16_is_supported(ctx, dst)
                                              : ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
        } else {
            return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
        }
    }

    if (!fp16_mma_available(cc)) {
        if (precision == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                if (pxq_use_sm60_vec_f32(cc, Q)) { // PR #2144: keep supported-check in lockstep
                    return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
                }
                return ggml_cuda_fattn_vec_f16_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f16_is_supported(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
            } else {
                return ggml_cuda_fattn_tile_f32_is_supported(ctx, dst);
            }
        }
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    // So, not sure why in mainline they thought that for CC_ADA_LOVELACE or when KV cache is not f16 the vector kernels are faster.
    // On my GPU (RTX-4080) MMA is efinitely faster for GQA, both for f16 and for quantized KV cache.
    //const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    //const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && !(Q->ne[1] == 1 && n_swa > 0 && K->ne[0] == V->ne[0]);
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && K->ne[0] == V->ne[0] && Q->ne[0] % (2*WARP_SIZE) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1 && !ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
        return ggml_cuda_fattn_vec_f32_is_supported(ctx, dst);
    }

    if (new_mma_available(cc) &&
            (Q->ne[0] == 576 || Q->ne[0] == 320 || Q->ne[0] == 512 || (K->ne[0] == 192 && V->ne[0] == 128 && mma_better_than_turing(cc)))) {
        if (Q->ne[0] == 576 || Q->ne[0] == 512 || Q->ne[0] == 320) {
            int gqa_ratio = Q->ne[2]/K->ne[2];
            return (gqa_ratio % 4) == 0;
        }
        return true;
    }

    if (!new_mma_available(cc) || K->ne[0] != V->ne[0]) {
        return ggml_cuda_fattn_wmma_f16_is_supported(ctx, dst);
    }

    return ggml_cuda_fattn_mma_f16_is_supported(ctx, dst);
}
