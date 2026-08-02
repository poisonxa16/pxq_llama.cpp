//
// GGML_OP_MASK_TO_IDX: additive attention mask -> per-row list of visible KV columns.
//
// Ported from ik_llama.cpp (k_mask_to_index in ggml-cuda/indexer_topk.cu, PR #2165),
// with two deliberate changes:
//
//   * the selection predicate is "not -inf" rather than "== 0", so a mask carrying a
//     finite bias (ALiBi) still yields a SUPERSET of the visible set instead of an
//     empty one. See the contract comment on ggml_mask_to_index() in ggml.h.
//   * the append is bounds-checked. The upstream kernel writes idx_r[start++] with no
//     clamp, so a row whose visible count exceeds dst->ne[0] corrupts neighbouring
//     memory. The bound is supposed to hold by construction; clamping turns a caller
//     bug into dropped KV instead of heap corruption.
//
// One warp per row. counts[] holds each lane's hit count; every lane then recomputes
// the same exclusive prefix sum so no second barrier is needed.
//

#include "mask_to_idx.cuh"

#include <type_traits>

static __device__ __forceinline__ float mask_to_idx_f32(const half  v) { return __half2float(v); }
static __device__ __forceinline__ float mask_to_idx_f32(const float v) { return v; }

template <typename mask_t>
static __global__ void k_mask_to_idx(
        const int ne00, const int ne0,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb1,  const size_t nb2,  const size_t nb3,
        const mask_t * __restrict__ mask, int * __restrict__ idx) {

    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    __shared__ int counts[WARP_SIZE];

    const mask_t * mask_r = (const mask_t *)((const char *) mask + i1*nb01 + i2*nb02 + i3*nb03);
    int          * idx_r  = (int          *)((      char *) idx  + i1*nb1  + i2*nb2  + i3*nb3);

    for (int j = threadIdx.x; j < ne0; j += WARP_SIZE) {
        idx_r[j] = -1;
    }

    // Keep everything the mask does not hard-exclude. Written as !(v <= -inf) so that
    // NaN (which fails every ordered compare) is KEPT: a superset stays correct, a
    // subset silently drops KV.
    int n_on = 0;
    for (int j = threadIdx.x; j < ne00; j += WARP_SIZE) {
        const float v = mask_to_idx_f32(mask_r[j]);
        n_on += !(v <= -INFINITY) ? 1 : 0;
    }
    counts[threadIdx.x] = n_on;
    __syncthreads();

    int start = 0;
    for (int i = 0; i < threadIdx.x; ++i) {
        start += counts[i];
    }

    for (int j = threadIdx.x; j < ne00; j += WARP_SIZE) {
        const float v = mask_to_idx_f32(mask_r[j]);
        if (!(v <= -INFINITY)) {
            if (start < ne0) {
                idx_r[start] = j;
            }
            ++start;
        }
    }
}

void ggml_cuda_op_mask_to_idx(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(src->type == GGML_TYPE_F32 || src->type == GGML_TYPE_F16);
    GGML_ASSERT(src->ne[1] == dst->ne[1] && src->ne[2] == dst->ne[2] && src->ne[3] == dst->ne[3]);
    GGML_ASSERT(src->ne[0] >= dst->ne[0]);
    GGML_ASSERT(dst->nb[0] == sizeof(int32_t));
    GGML_ASSERT(src->nb[0] == ggml_type_size(src->type));

    const dim3 grid(dst->ne[1], dst->ne[2], dst->ne[3]);

    if (src->type == GGML_TYPE_F16) {
        k_mask_to_idx<half><<<grid, WARP_SIZE, 0, ctx.stream()>>>(
                src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const half *) src->data, (int *) dst->data);
    } else {
        k_mask_to_idx<float><<<grid, WARP_SIZE, 0, ctx.stream()>>>(
                src->ne[0], dst->ne[0],
                src->nb[1], src->nb[2], src->nb[3],
                dst->nb[1], dst->nb[2], dst->nb[3],
                (const float *) src->data, (int *) dst->data);
    }
    CUDA_CHECK(cudaGetLastError());
}
