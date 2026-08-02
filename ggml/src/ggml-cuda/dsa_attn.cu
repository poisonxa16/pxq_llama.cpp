//
// DSA sparse attention. Ported from ik_llama.cpp ggml-cuda/dsa_attn.cu (PR #2165).
//
// Why this file matters here: on sm_70 every CUDA flash-attention predicate rejects
// head dim 512 (the wmma kernel takes {64,80,96,112,128,256}; the 512 branch needs
// new_mma_available, i.e. Ampere+). So DeepSeek-V4's FLASH_ATTN_EXT nodes were all
// placed on the CPU backend while the server still printed flash_attn = 1. This path
// is arch-neutral -- cuBLAS fp16 GEMM plus plain CUDA kernels, no wmma/mma/cp.async
// intrinsics and no __CUDA_ARCH__ branches -- so it runs on Volta as written.
//
// Instead of a dense [n_kv x n_tokens] score matrix it gathers only the rows named by
// the src[5] index list (GGML_OP_MASK_TO_IDX), so cost scales with the number of
// VISIBLE rows rather than the padded ring width.
//
// Deviations from the ik original, all deliberate:
//
//   * the precondition set is factored into ggml_cuda_dsa_attn_supported() and shared
//     verbatim between dispatch and the scheduler gate, so the two cannot drift.
//   * added preconditions the original omits, each of which is a real abort or a wrong
//     answer rather than a slow path:
//       - max_bias != 0 (ALiBi) and logit_softcap != 0 are REJECTED; this kernel reads
//         only op_params[0] (scale) and would silently ignore both.
//       - the softmax shared-memory request is checked against the device limit. The
//         original asserts it inside the kernel launcher, which aborts the process for
//         a large index list instead of falling back.
//       - dst must be contiguous (k_copy_dst writes a dense block), the index list must
//         have a dense dim 0 and at least Q->ne[1] rows, K/V row counts must match, and
//         a sink tensor must be F32 with one entry per head.
//   * AMD is excluded: the path is untestable here, so it is not claimed.
//

#include "dsa_attn.cuh"

#include <algorithm>
#include <cstring>

#define DSA_ATTN_MAX_ROWS       32
#define DSA_SOFT_MAX_BLOCK_SIZE 1024

// True when V is the same memory as K (possibly a within-row sub-range). All three
// DeepSeek-V4 call sites pass the same tensor as K and V, so this is the normal case
// and it makes the V gather free.
static inline bool v_is_k_view(const ggml_tensor * K, const ggml_tensor * V) {
    if (!V || !V->data) return false;
    const char * k_data = (const char *) K->data;
    const char * v_data = (const char *) V->data;
    const size_t k_row_size = ggml_row_size(K->type, K->ne[0]);
    const size_t v_row_size = ggml_row_size(V->type, V->ne[0]);
    return v_data >= k_data && v_data + v_row_size <= k_data + k_row_size;
}

static __global__ void k_prepare_mask(int nidx, const int * __restrict__ idx, const half * __restrict__ m_in,
        half * __restrict__ m_out, size_t stride_idx, size_t stride_m) {
    const int row = blockIdx.x;
    const int col = blockIdx.y*blockDim.x + threadIdx.x;
    idx += row*stride_idx;
    const int ii = idx[col];
    // A -1 slot is padding: force it fully masked so softmax gives it zero weight.
    m_out[row*nidx + col] = ii >= 0 ? m_in[row*stride_m + ii] : __float2half(-INFINITY);
}

static __global__ void k_prepare_one_batch_kv(int nk, int ncol, const int * idx, const char * k_in,
        half * k_out, size_t stride_k, size_t stride_idx) {
    const int row = blockIdx.y;
    const int col = blockIdx.x;
    int i = idx[row*stride_idx + col];
    if (i < 0) {
        // Padding slot. Gather row 0 so the read stays in bounds; k_prepare_mask has
        // already set this column's mask to -inf, so the value never contributes.
        i = 0;
    }
    const half * k_row = (const half *)(k_in + stride_k * i);
    k_out += (row*ncol + col)*nk;
    for (int j = threadIdx.x; j < nk; j += blockDim.x) {
        k_out[j] = k_row[j];
    }
}

static __global__ void k_prepare_one_batch_q(int ne0, int ne1, size_t nb1, size_t nb2,
        const float * q_in, half * q_out) {
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    if (i0 >= ne0) {
        return;
    }
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z;
    q_out[i0 + (i2 + i1*ne1)*ne0] = __float2half(q_in[i0 + i1*nb1 + i2*nb2]);
}

static __global__ void k_copy_dst(int nelem, const half * kqv16, float * dst) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nelem) {
        return;
    }
    dst[i] = __half2float(kqv16[i]);
}

template <int ncols_template, int block_size_template>
static __global__ void soft_max_f16_simple(half * x, const half * mask, const float * sinks,
        const int ncols_par, const int nrows_y, const float scale) {
    const int ncols = ncols_template == 0 ? ncols_par : ncols_template;

    const int tid  = threadIdx.x;
    const int rowx = blockIdx.x;
    const int rowy = rowx / nrows_y;

    const int block_size = block_size_template == 0 ? blockDim.x : block_size_template;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float data_soft_max_f16_simple[];
    float * buf_iw = data_soft_max_f16_simple; // inter-warp communication
    float * vals   = buf_iw + WARP_SIZE;       // cached values between passes

    float max_val = sinks ? sinks[rowx % nrows_y] : -INFINITY;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        const int64_t iy = (int64_t)rowy*ncols + col;

        const float val = scale*__half2float(x[ix]) + __half2float(mask[iy]);

        vals[col] = val;
        max_val = max(max_val, val);
    }

    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -INFINITY;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f; // partial sum

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            break;
        }

        const float val = expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    if (sinks) {
        tmp += expf(sinks[rowx % nrows_y] - max_val);
    }

    const float inv_sum = 1.0f / tmp;

#pragma unroll
    for (int col0 = 0; col0 < ncols; col0 += block_size) {
        const int col = col0 + tid;

        if (ncols_template == 0 && col >= ncols) {
            return;
        }

        const int64_t ix = (int64_t)rowx*ncols + col;
        x[ix] = __float2half(vals[col] * inv_sum);
    }
}

static size_t dsa_soft_max_shmem(int ncols_x) {
    return (GGML_PAD((size_t) ncols_x, (size_t) WARP_SIZE) + WARP_SIZE)*sizeof(float);
}

static void soft_max_f16_cuda_simple(half * x, const half * mask, const float * sinks, const int ncols_x,
        const int nrows_x, const int nrows_y, const float scale, cudaStream_t stream) {
    int nth = WARP_SIZE;
    while (nth < ncols_x && nth < DSA_SOFT_MAX_BLOCK_SIZE) nth *= 2;
    const dim3 block_dims(nth,     1, 1);
    const dim3 block_nums(nrows_x, 1, 1);
    const size_t shmem = dsa_soft_max_shmem(ncols_x);
    static_assert(DSA_SOFT_MAX_BLOCK_SIZE == 1024, "These values need to be adjusted.");

    // Guaranteed by ggml_cuda_dsa_attn_supported(); re-stated so a future caller that
    // bypasses the gate fails loudly instead of launching with too much shared memory.
    GGML_ASSERT(shmem < ggml_cuda_info().devices[ggml_cuda_get_device()].smpb);

    switch (ncols_x) {
        case   32: soft_max_f16_simple<  32,   32><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case   64: soft_max_f16_simple<  64,   64><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case  128: soft_max_f16_simple< 128,  128><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case  256: soft_max_f16_simple< 256,  256><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case  512: soft_max_f16_simple< 512,  512><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case 1024: soft_max_f16_simple<1024, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case 2048: soft_max_f16_simple<2048, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        case 4096: soft_max_f16_simple<4096, 1024><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
        default:   soft_max_f16_simple<   0,    0><<<block_nums, block_dims, shmem, stream>>>(x, mask, sinks, ncols_x, nrows_y, scale); break;
    }
}

bool ggml_cuda_dsa_attn_supported(const ggml_tensor * dst, int cc) {
    if (!dst || dst->op != GGML_OP_FLASH_ATTN_EXT) return false;

    // Untestable here; do not claim it.
    if (cc >= CC_OFFSET_AMD) return false;

    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K       = dst->src[1];
    const ggml_tensor * V       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sink    = dst->src[4];
    const ggml_tensor * indexer = dst->src[5];

    if (!Q || !K || !V || !mask || !indexer) return false;

    // --- op semantics this kernel implements ---------------------------------
    // Only op_params[0] (scale) is honoured. ALiBi and logit softcapping would be
    // silently ignored, which is a wrong answer rather than a slow one.
    float max_bias, logit_softcap;
    memcpy(&max_bias,      (const char *) dst->op_params + 1*sizeof(float), sizeof(float));
    memcpy(&logit_softcap, (const char *) dst->op_params + 2*sizeof(float), sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) return false;

    // --- dtypes --------------------------------------------------------------
    if (Q->type != GGML_TYPE_F32) return false;
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16) return false;
    if (mask->type != GGML_TYPE_F16) return false;
    if (indexer->type != GGML_TYPE_I32) return false;
    if (dst->type != GGML_TYPE_F32) return false;
    if (sink && sink->type != GGML_TYPE_F32) return false;

    // --- shapes --------------------------------------------------------------
    if (K->ne[0] != Q->ne[0]) return false;              // QK^T contracts over ne[0]
    if (K->ne[1] != V->ne[1]) return false;              // one index list gathers both
    if (K->ne[2] > 1 || K->ne[3] > 1) return false;      // single stream, no GQA broadcast
    if (V->ne[2] > 1 || V->ne[3] > 1) return false;
    if (mask->ne[2] > 1 || mask->ne[3] > 1) return false;
    if (Q->ne[3] > 1) return false;
    if (mask->ne[0] < K->ne[1]) return false;            // mask must cover every KV row
    if (mask->ne[1] < Q->ne[1]) return false;            // ... and every query row
    if (sink && sink->ne[0] < Q->ne[2]) return false;    // one sink per head

    // Index list: one row per query, dim 0 dense, and a 256-multiple width because
    // k_prepare_mask launches exactly indexer->ne[0]/256 blocks of 256 threads.
    if (indexer->ne[0] <= 0 || indexer->ne[0] % 256 != 0) return false;
    if (indexer->ne[1] < Q->ne[1]) return false;
    if (indexer->ne[2] > 1 || indexer->ne[3] > 1) return false;
    if (indexer->nb[0] != sizeof(int32_t)) return false;
    if (indexer->nb[1] % sizeof(int32_t) != 0) return false;

    // --- memory layout the kernels assume ------------------------------------
    if (Q->nb[0] != sizeof(float)) return false;
    if (Q->nb[1] % sizeof(float) != 0 || Q->nb[2] % sizeof(float) != 0) return false;
    if (K->nb[0] != sizeof(half) || V->nb[0] != sizeof(half)) return false;
    if (mask->nb[0] != sizeof(half) || mask->nb[1] % sizeof(half) != 0) return false;
    if (!ggml_is_contiguous(dst)) return false;          // k_copy_dst writes a dense block

    // --- worth it? -----------------------------------------------------------
    // Gathering costs indexer->ne[0] rows per query, so the sparse path only wins if
    // the index list is meaningfully narrower than the dense KV.
    if (Q->ne[1] <= 16) {
        if (indexer->ne[0] >= K->ne[1]) return false;
    } else {
        if (K->ne[1] < 4*indexer->ne[0]) return false;
    }

    // --- launchability -------------------------------------------------------
    // The softmax caches one float per column in shared memory. Without this check a
    // wide index list aborts the process inside the launcher instead of falling back.
    if (dsa_soft_max_shmem(indexer->ne[0]) >= ggml_cuda_info().devices[ggml_cuda_get_device()].smpb) {
        return false;
    }

    return true;
}

void ggml_cuda_dsa_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q       = dst->src[0];
    const ggml_tensor * K       = dst->src[1];
    const ggml_tensor * V       = dst->src[2];
    const ggml_tensor * mask    = dst->src[3];
    const ggml_tensor * sink    = dst->src[4];
    const ggml_tensor * indexer = dst->src[5];

    // The caller must have consulted the shared predicate; assert rather than
    // re-deciding, so dispatch and gate can never diverge.
    GGML_ASSERT(ggml_cuda_dsa_attn_supported(dst, ggml_cuda_info().devices[ggml_cuda_get_device()].cc));

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    const half alpha = 1.0f;
    const half beta  = 0.0f;

    const int  n_idx      = indexer->ne[0];
    const int  max_rows   = std::min<int>(Q->ne[1], DSA_ATTN_MAX_ROWS);
    const bool is_k_view  = v_is_k_view(K, V);
    const size_t stride_idx = indexer->nb[1]/sizeof(int);

    ggml_cuda_pool_alloc<half> q16   (ctx.pool(), (size_t) Q->ne[0]*Q->ne[2]*max_rows);
    ggml_cuda_pool_alloc<half> kq16  (ctx.pool(), (size_t) n_idx*Q->ne[2]*max_rows);
    ggml_cuda_pool_alloc<half> kqv16 (ctx.pool(), (size_t) V->ne[0]*Q->ne[2]*max_rows);
    ggml_cuda_pool_alloc<half> mask16(ctx.pool(), (size_t) n_idx*Q->ne[1]);
    ggml_cuda_pool_alloc<half> k16   (ctx.pool(), (size_t) n_idx*K->ne[0]*max_rows);
    ggml_cuda_pool_alloc<half> v16   (ctx.pool());

    size_t v_offset = 0;
    if (is_k_view) {
        v_offset = (const half *)V->data - (const half *)K->data;
    } else {
        v16.alloc((size_t) n_idx*V->ne[0]*max_rows);
    }

    // The gathered mask is small; build it once for the whole call.
    {
        const dim3 grid(Q->ne[1], n_idx/256, 1);
        k_prepare_mask<<<grid, 256, 0, ctx.stream()>>>(n_idx, (const int *) indexer->data,
                (const half *) mask->data, mask16.get(), stride_idx, mask->nb[1]/sizeof(half));
        CUDA_CHECK(cudaGetLastError());
    }

    const int nstep = (Q->ne[1] + max_rows - 1)/max_rows;

    for (int istep = 0; istep < nstep; ++istep) {
        const int first = istep*max_rows;
        const int last  = std::min<int>(first + max_rows, Q->ne[1]);
        const int nrows = last - first;

        {
            const dim3 grid(n_idx, nrows, 1);
            k_prepare_one_batch_kv<<<grid, 256, 0, ctx.stream()>>>(K->ne[0], n_idx,
                    (const int *) indexer->data + stride_idx*first,
                    (const char *) K->data, k16.get(), K->nb[1], stride_idx);
            if (!is_k_view) {
                k_prepare_one_batch_kv<<<grid, 256, 0, ctx.stream()>>>(V->ne[0], n_idx,
                        (const int *) indexer->data + stride_idx*first,
                        (const char *) V->data, v16.get(), V->nb[1], stride_idx);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        {
            const int nblock = (Q->ne[0] + 255)/256;
            const dim3 grid(nblock, nrows, Q->ne[2]);
            k_prepare_one_batch_q<<<grid, 256, 0, ctx.stream()>>>(Q->ne[0], Q->ne[2],
                    Q->nb[1]/sizeof(float), Q->nb[2]/sizeof(float),
                    (const float *)((const char *) Q->data + first*Q->nb[1]), q16.get());
            CUDA_CHECK(cudaGetLastError());
        }

        // KQ^T over the gathered rows: [n_idx, n_head] per query row.
        CUBLAS_CHECK(cublasHgemmStridedBatched(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                    n_idx, Q->ne[2], Q->ne[0],
                    &alpha, k16.get(), K->ne[0], (long long) K->ne[0]*n_idx,
                    q16.get(), Q->ne[0], (long long) Q->ne[0]*Q->ne[2],
                    &beta, kq16.get(), n_idx, (long long) n_idx*Q->ne[2], nrows));

        soft_max_f16_cuda_simple(kq16.get(), mask16.get() + (size_t) first*n_idx,
                sink ? (const float *) sink->data : nullptr,
                n_idx, Q->ne[2]*nrows, Q->ne[2], scale, ctx.stream());
        CUDA_CHECK(cudaGetLastError());

        const half * v_src = is_k_view ? k16.get() + v_offset : v16.get();
        const int    v_ld  = is_k_view ? K->ne[0] : V->ne[0];
        const long long v_stride = (long long) v_ld*n_idx;

        CUBLAS_CHECK(cublasHgemmStridedBatched(ctx.cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                    V->ne[0], Q->ne[2], n_idx,
                    &alpha, v_src, v_ld, v_stride,
                    kq16.get(), n_idx, (long long) n_idx*Q->ne[2],
                    &beta, kqv16.get(), V->ne[0], (long long) V->ne[0]*Q->ne[2], nrows));

        {
            const int nelem  = V->ne[0]*Q->ne[2]*nrows;
            const int nblock = (nelem + 255)/256;
            k_copy_dst<<<nblock, 256, 0, ctx.stream()>>>(nelem, kqv16.get(),
                    (float *)((char *) dst->data + dst->nb[2]*first));
            CUDA_CHECK(cudaGetLastError());
        }
    }
}
