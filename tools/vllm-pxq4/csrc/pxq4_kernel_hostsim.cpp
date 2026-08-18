// pxq4_kernel_hostsim.cpp — execute the REAL PXQ4 device kernels on the CPU, with no GPU and
// no CUDA toolkit, so that bit-exactness against the numpy reference can be gated before the
// GPU lease is taken.
//
// This compiles pxq4_kernel.cuh unmodified against hostsim/cuda_fp16.h and emulates a CUDA
// launch: blocks run sequentially, the threads of a block run concurrently on std::threads,
// __shared__ is a function-local static (valid precisely because blocks are sequential), and
// __syncthreads() is a sense-reversing barrier sized to the block.
//
// It is a TEST artifact. It is not part of the shipped extension and must never be linked into
// libpxq4_sm70.so.
//
// Build: g++ -O2 -std=c++17 -shared -fPIC -Ihostsim -I. -pthread \
//            pxq4_kernel_hostsim.cpp -o libpxq4_hostsim.so
// (build_hostsim.sh does this and then runs test_pxq4_kernel_ref.py against it.)

#include <cstdint>
#include <cstdlib>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "pxq4_kernel.cuh"

// ---------------------------------------------------------------------------------------------
// launch context + barrier
// ---------------------------------------------------------------------------------------------
thread_local dim3 threadIdx(0, 0, 0);
thread_local dim3 blockIdx(0, 0, 0);
dim3 blockDim(1, 1, 1);
dim3 gridDim(1, 1, 1);

namespace {
std::mutex              g_bar_mtx;
std::condition_variable g_bar_cv;
unsigned                g_bar_n     = 1;   // threads per block
unsigned                g_bar_count = 0;
unsigned                g_bar_gen   = 0;
}  // namespace

void __syncthreads() {
    std::unique_lock<std::mutex> lk(g_bar_mtx);
    const unsigned gen = g_bar_gen;
    if (++g_bar_count == g_bar_n) {
        g_bar_count = 0;
        ++g_bar_gen;
        g_bar_cv.notify_all();
    } else {
        g_bar_cv.wait(lk, [&] { return g_bar_gen != gen; });
    }
}

// The dynamic shared-memory arena the mmv declares as `extern __shared__ float pxq4_xs[]`.
// Sized well past anything a real layer needs: the chunked staging (see k_pxq4_mmv) tops out
// at ceil(kslabs/nfix)*32 floats, which is 1088 floats for a K = 17408 ffn_down.
#define PXQ4_HOSTSIM_SMEM_FLOATS 16384
alignas(16) float pxq4_xs[PXQ4_HOSTSIM_SMEM_FLOATS];

namespace {

template <typename Body>
void launch(unsigned gx, unsigned gy, unsigned nthreads, Body body) {
    blockDim = dim3(nthreads, 1, 1);
    gridDim  = dim3(gx, gy, 1);
    g_bar_n  = nthreads;
    for (unsigned by = 0; by < gy; ++by) {
        for (unsigned bx = 0; bx < gx; ++bx) {
            {
                std::lock_guard<std::mutex> lk(g_bar_mtx);
                g_bar_count = 0;
            }
            std::vector<std::thread> ts;
            ts.reserve(nthreads);
            for (unsigned t = 0; t < nthreads; ++t) {
                ts.emplace_back([=] {
                    threadIdx = dim3(t, 0, 0);
                    blockIdx  = dim3(bx, by, 0);
                    body();
                });
            }
            for (auto & th : ts) th.join();
        }
    }
}

}  // namespace

extern "C" {

// fp32 dequant — this is the PARITY-LOCKED contract (pxq-cpu.h:16-18): the fp32 products
// eff = anchor*sub, w = eff*book are what the numpy reference must reproduce exactly.
void pxq4_hostsim_dequant_f32(const uint8_t * slabs, const uint16_t * anchor, float * out,
                              int panels, int kslabs) {
    const int64_t K = (int64_t)kslabs * PXQ4_QK;
    launch((unsigned)(panels * kslabs), 1u, PXQ4_BM, [=] {
        k_pxq4_dequant_matrix<float>(slabs, (const __half *)anchor, out, kslabs, K);
    });
}

// fp16 dequant — what torch.ops.pxq4.dequant_out actually writes. It is the fp32 result above
// with ONE extra round-to-nearest-even, so a GPU gate must compare against
// reference.dequant(...).astype(np.float16), never against the fp32 array directly.
void pxq4_hostsim_dequant_f16(const uint8_t * slabs, const uint16_t * anchor, uint16_t * out,
                              int panels, int kslabs) {
    const int64_t K = (int64_t)kslabs * PXQ4_QK;
    launch((unsigned)(panels * kslabs), 1u, PXQ4_BM, [=] {
        k_pxq4_dequant_matrix<__half>(slabs, (const __half *)anchor, (__half *)out, kslabs, K);
    });
}

void pxq4_hostsim_mmv_f16(const uint8_t * slabs, const uint16_t * anchor, const uint16_t * x,
                          uint16_t * out, int M, int panels, int kslabs, int vecx) {
    const int R = panels * PXQ4_BM;
    const int K = kslabs * PXQ4_QK;
    if (pxq4_canon_max_chunk(kslabs) * PXQ4_QK > PXQ4_HOSTSIM_SMEM_FLOATS) abort();
    if (vecx) {
        launch((unsigned)panels, (unsigned)M, 256u, [=] {
            k_pxq4_mmv<true>(slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
        });
    } else {
        launch((unsigned)panels, (unsigned)M, 256u, [=] {
            k_pxq4_mmv<false>(slabs, (const __half *)anchor, (const __half *)x, (__half *)out, R, K);
        });
    }
}

int pxq4_hostsim_canon_nfix(int kslabs)      { return pxq4_canon_nfix(kslabs, PXQ4_CANON_CMAX); }
int pxq4_hostsim_canon_max_chunk(int kslabs) { return pxq4_canon_max_chunk(kslabs); }

// the frozen literals this TU was compiled with, for a transcription check against the
// checkpoint's gguf KVs and against the numpy reference's own copy
void pxq4_hostsim_builtin_tables(float * book16, float * sub16) {
    static const float b[16] = PXQ4_BOOK_INIT;
    static const float s[16] = PXQ4_SUB16_INIT;
    for (int i = 0; i < 16; ++i) { book16[i] = b[i]; sub16[i] = s[i]; }
}

}  // extern "C"
