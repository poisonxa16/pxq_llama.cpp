// pxq4_kernel.cu — host launchers for the vendored PXQ4 sm_70 kernels.
//
// This TU owns the only copies of pxq4_book_g / pxq4_sub16_g, so the table upload/download
// helpers must live here (they are `static __device__` and therefore TU-local, exactly as in
// the engine, pxq6.cuh:79-81).

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "pxq4_kernel.cuh"
#include "pxq4_kernel_launch.h"

#define PXQ4_CUDA_CHECK(expr)                                                                 \
    do {                                                                                      \
        cudaError_t err_ = (expr);                                                            \
        if (err_ != cudaSuccess) {                                                            \
            fprintf(stderr, "pxq4: %s failed at %s:%d: %s\n", #expr, __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                                \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

int pxq4_mmv_smem_bytes(int kslabs) {
    return pxq4_canon_max_chunk(kslabs) * PXQ4_QK * (int)sizeof(float);
}

bool pxq4_mmv_supported(int kslabs) {
    // 48 KiB is the sm_70 per-block static+dynamic shared budget available WITHOUT the
    // cudaFuncSetAttribute opt-in. We deliberately stay under it: opting in to the 96 KiB
    // Volta maximum would drop the kernel to one block per SM, and chunked staging (see
    // k_pxq4_mmv) means no real shape ever needs it — K = 17408 costs 4352 B.
    const int stat_smem = (int)(2 * 16 * sizeof(float) + PXQ4_MMV_KSEG * PXQ4_BM * sizeof(float));
    return pxq4_mmv_smem_bytes(kslabs) + stat_smem <= 48 * 1024;
}

void pxq4_launch_dequant_f16(const uint8_t * slabs, const void * anchor, void * out,
                             int panels, int kslabs, cudaStream_t stream) {
    const int64_t nslabs = (int64_t)panels * kslabs;
    const int64_t K      = (int64_t)kslabs * PXQ4_QK;
    // one block per slab, 64 threads (one weight row each)
    k_pxq4_dequant_matrix<__half><<<(unsigned)nslabs, PXQ4_BM, 0, stream>>>(
        slabs, (const __half *)anchor, (__half *)out, kslabs, K);
    PXQ4_CUDA_CHECK(cudaGetLastError());
}

void pxq4_launch_mmv_f16(const uint8_t * slabs, const void * anchor, const void * x, void * out,
                         int M, int panels, int kslabs, bool vecx, cudaStream_t stream) {
    const int    R     = panels * PXQ4_BM;
    const int    K     = kslabs * PXQ4_QK;
    const size_t smem  = (size_t)pxq4_mmv_smem_bytes(kslabs);
    const dim3   grid((unsigned)panels, (unsigned)M, 1u);
    if (vecx) {
        k_pxq4_mmv<true><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    } else {
        k_pxq4_mmv<false><<<grid, 256, smem, stream>>>(
            slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
    }
    PXQ4_CUDA_CHECK(cudaGetLastError());
}

void pxq4_upload_tables(const float * book16, const float * sub16) {
    PXQ4_CUDA_CHECK(cudaMemcpyToSymbol(pxq4_book_g,  book16, 16 * sizeof(float)));
    PXQ4_CUDA_CHECK(cudaMemcpyToSymbol(pxq4_sub16_g, sub16,  16 * sizeof(float)));
}

void pxq4_download_tables(float * book16, float * sub16) {
    PXQ4_CUDA_CHECK(cudaMemcpyFromSymbol(book16, pxq4_book_g,  16 * sizeof(float)));
    PXQ4_CUDA_CHECK(cudaMemcpyFromSymbol(sub16,  pxq4_sub16_g, 16 * sizeof(float)));
}
