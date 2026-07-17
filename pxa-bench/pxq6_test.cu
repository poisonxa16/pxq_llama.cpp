// pxq6_test.cu — PXQ6 kernel-family correctness + bench harness (standalone, any CUDA arch).
//
// CORRECTNESS (runs on the 1080Ti while the Teslas are busy — bit-exactness is arch-independent
// for identical fp32 op sequences on IEEE hardware):
//   decode : new-family baseline vs {PAIRLUT, VECX, PAIR+VECX, KSPLIT(bit-exact), KSPLIT_GEN}
//            per format (PXQ4/PXQ5/PXQ6/PXQ6HQ) — memcmp; KSPLIT_GEN reports maxdiff (expected
//            tiny nonzero — G3-gated form). Old proven PXQ4/PXQ5 kernels vs new-family baseline
//            — memcmp (proves the policy family preserved the exact chains).
//   prefill: gemm baseline vs {RAGTAIL, PIPE, RAG+PIPE} memcmp; GUFUSE vs 3-kernel pipeline
//            memcmp; SCATFUSE vs gemm+copy memcmp; old PXQ4/PXQ5 gemm vs new baseline memcmp.
//   dequant: k_pxq6_dequant_matrix output vs the CPU reference contract (bitwise).
//   WMMA   : compiled in; executes only on cc==700 (prints SKIP elsewhere) — correctness =
//            maxdiff vs fp32 CPU reference (NOT bit-exact by design) + kernel-t/s A/B.
// BENCH (--bench, for the V100/P100 window): decode gateup+down kernel timings per variant,
//   prefill GEMM timings per variant (incl. WMMA on sm_70) at the live 35B shapes.
//
// Build (in the CUDA devel container):
//   nvcc -O3 -std=c++17 -arch=sm_61 [-gencode arch=compute_70,code=sm_70 ...] \
//        -I. -o pxq6_test pxq6_test.cu
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <thread>
#include <atomic>
#include <random>
#include <immintrin.h>
#include <cuda_fp16.h>

typedef uint16_t ggml_fp16_t;
static inline float ggml_fp16_to_fp32(ggml_fp16_t h) { return _cvtsh_ss(h); }
static inline ggml_fp16_t ggml_fp32_to_fp16(float f) { return _cvtss_sh(f, 0); }

// minimal ggml type shim so pxq6.cuh's host helpers compile standalone
enum ggml_type { GGML_TYPE_PXQ4 = 250, GGML_TYPE_PXQ5 = 251, GGML_TYPE_PXQ6 = 252, GGML_TYPE_PXQ6HQ = 253 };

#include "pxq6-quantize.inc.cpp"        // CPU quantizer + reference dequant (the contract)
#include "pxq6.cuh"                      // the kernel family under test

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA ERR %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static std::mt19937_64 grng(42);

// build a random-but-realistic expert tensor and quantize it with the real converter
static std::vector<uint8_t> make_wq(int fmt, int64_t R, int64_t K, int64_t E, std::vector<float> * wf32 = nullptr) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> W(R*K*E);
    for (auto & v : W) v = nd(grng);
    // sprinkle outliers + a zero row + a zero block per expert for edge coverage
    for (int64_t e = 0; e < E; ++e) {
        float * we = W.data() + e*R*K;
        we[(e*7) % (R*K)] = 0.9f;
        for (int64_t k = 0; k < K; ++k) we[(e % R)*K + k] = 0.f;
        for (int64_t k = 0; k < 16 && K >= 32; ++k) we[((e+3) % R)*K + 32 + k] = 0.f;
    }
    if (wf32) *wf32 = W;
    if (fmt == PXA_PXQ_FMT_P6 || fmt == PXA_PXQ_FMT_P6HQ) {
        const int tier = fmt == PXA_PXQ_FMT_P6HQ ? 1 : 0;
        const int64_t eb = (R/64)*(PXQ6_HDR_BYTES + (K/32)*(int64_t)(tier ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES));
        std::vector<uint8_t> q(eb*E);
        pxq6_quantize_tensor(W.data(), q.data(), R, K, E, nullptr, 0, 8, tier);
        return q;
    }
    // PXQ4/PXQ5: random VALID slab bytes (codes uniform; scales in a sane range)
    const int64_t eb = R*(K/32)*17;
    std::vector<uint8_t> q(eb*E);
    std::uniform_int_distribution<int> byt(0, 255);
    std::uniform_int_distribution<int> sc4(118, 132);    // e8m0 around 2^-9..2^5
    std::uniform_int_distribution<int> sc5(120, 200);    // pxq5 log grid
    for (int64_t s = 0; s < (int64_t)q.size(); s += 1088) {
        for (int i = 0; i < 64; ++i)  q[s + i] = fmt == PXA_PXQ_FMT_P4 ? (uint8_t)sc4(grng) : (uint8_t)sc5(grng);
        for (int i = 64; i < 1088; ++i) q[s + i] = (uint8_t)byt(grng);
    }
    return q;
}

struct DevBuf { void * p = nullptr; size_t n = 0;
    void up(const void * h, size_t bytes) { if (bytes > n) { if (p) cudaFree(p); CK(cudaMalloc(&p, bytes)); n = bytes; } CK(cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice)); }
    void alloc(size_t bytes) { if (bytes > n) { if (p) cudaFree(p); CK(cudaMalloc(&p, bytes)); n = bytes; } }
};

static const char * fmtname(int f) {
    return f == 1 ? "PXQ4" : f == 2 ? "PXQ5" : f == 3 ? "PXQ6" : "PXQ6HQ";
}

// ---------------- decode test ----------------
static int g_fail = 0;
static void check(const char * what, bool ok) {
    printf("  %-46s %s\n", what, ok ? "BIT-EXACT" : "*** MISMATCH ***");
    if (!ok) ++g_fail;
}

static void test_decode(int fmt, bool bench) {
    const int64_t R = 512, K = 2048, E = 32, n_ids = 8, Ny = 2;
    const int64_t Rd = 2048, Kd = 512;
    std::vector<uint8_t> hu = make_wq(fmt, R, K, E), hg = make_wq(fmt, R, K, E), hd = make_wq(fmt, Rd, Kd, E);
    std::vector<float> hx(Ny*K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : hx) v = nd(grng);
    std::vector<int32_t> hids(Ny*n_ids);
    for (auto & v : hids) v = (int32_t)(grng() % E);
    hids[1] = -1;   // SER slot coverage
    std::vector<float> hbu(E*R), hbg(E*R);
    for (auto & v : hbu) v = nd(grng)*0.01f;
    for (auto & v : hbg) v = nd(grng)*0.01f;

    DevBuf du, dg, dd, dx, dids, dbu, dbg, dout, dout2, dws, ddown, ddown2;
    du.up(hu.data(), hu.size()); dg.up(hg.data(), hg.size()); dd.up(hd.data(), hd.size());
    dx.up(hx.data(), hx.size()*4); dids.up(hids.data(), hids.size()*4);
    dbu.up(hbu.data(), hbu.size()*4); dbg.up(hbg.data(), hbg.size()*4);
    const size_t outb = Ny*n_ids*R*4, downb = Ny*n_ids*Rd*4;
    dout.alloc(outb); dout2.alloc(outb); ddown.alloc(downb); ddown2.alloc(downb);
    dws.alloc(Ny*n_ids*2*8*R*4);

    const size_t x_tok = K*4, dst_tok = n_ids*R*4, dst_slot = R*4;
    const size_t ids_nb0 = 4, ids_nb1 = n_ids*4;
    const int unary = 0; const float alpha = 1.702f, limit = 0.f;
    const size_t smem_gu = K*4 + 2*PXQ4_MMV_KSEG*64*4;
    dim3 grid((unsigned)(R/64), (unsigned)n_ids, (unsigned)Ny);

    auto run_gu = [&](pxq6_gateup_fn fn, void * out) {
        CK(cudaMemset(out, 0xee, outb));
        fn<<<grid, 256, smem_gu>>>((const uint8_t *)du.p, (const uint8_t *)dg.p,
            (const char *)dx.p, x_tok, (char *)out, dst_tok, dst_slot,
            (const char *)dids.p, ids_nb0, ids_nb1,
            (const float *)dbu.p, R*4, (const float *)dbg.p, R*4,
            (int)R, (int)K, (int)E, unary, alpha, limit);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    };
    std::vector<uint8_t> ref(outb), got(outb);

    printf("[decode %s]\n", fmtname(fmt));
    run_gu(pxq6_pick_gateup(fmt, false, false), dout.p);
    CK(cudaMemcpy(ref.data(), dout.p, outb, cudaMemcpyDeviceToHost));

    // old proven kernels vs new-family baseline
    if (fmt == PXA_PXQ_FMT_P4 || fmt == PXA_PXQ_FMT_P5) {
        auto * old_gu = fmt == PXA_PXQ_FMT_P5 ? k_pxq5_gateup_mmv_fast : k_pxq4_gateup_mmv;
        run_gu(old_gu, dout2.p);
        CK(cudaMemcpy(got.data(), dout2.p, outb, cudaMemcpyDeviceToHost));
        check("old proven kernel == new-family baseline", got == ref);
        if (fmt == PXA_PXQ_FMT_P5) {
            run_gu(k_pxq5_gateup_mmv, dout2.p);   // proven (non-fast) twin
            CK(cudaMemcpy(got.data(), dout2.p, outb, cudaMemcpyDeviceToHost));
            check("old PROVEN (slow-table) == new baseline", got == ref);
        }
    }
    // K2 variants
    for (int m = 1; m < 4; ++m) {
        run_gu(pxq6_pick_gateup(fmt, m & 1, m & 2), dout2.p);
        CK(cudaMemcpy(got.data(), dout2.p, outb, cudaMemcpyDeviceToHost));
        char buf[64]; snprintf(buf, 64, "PAIRLUT=%d VECX=%d == baseline", m & 1, (m >> 1) & 1);
        check(buf, got == ref);
    }
    // K1 bit-exact ksplit
    {
        CK(cudaMemset(dout2.p, 0xee, outb));
        dim3 grids((unsigned)(R/64*PXQ4_MMV_KSEG), (unsigned)n_ids, (unsigned)Ny);
        pxq6_pick_gateup_ksplit(fmt, false, false)<<<grids, 64, K*4>>>(
            (const uint8_t *)du.p, (const uint8_t *)dg.p, (const char *)dx.p, x_tok,
            (float *)dws.p, (const char *)dids.p, ids_nb0, ids_nb1, (int)R, (int)K, (int)E, (int)n_ids);
        CK(cudaGetLastError());
        dim3 gridr((unsigned)((R + 255)/256), (unsigned)n_ids, (unsigned)Ny);
        k_pxq6_gateup_reduce<<<gridr, 256>>>((const float *)dws.p, (char *)dout2.p, dst_tok, dst_slot,
            (const char *)dids.p, ids_nb0, ids_nb1,
            (const float *)dbu.p, R*4, (const float *)dbg.p, R*4,
            (int)R, (int)E, (int)n_ids, unary, alpha, limit);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dout2.p, outb, cudaMemcpyDeviceToHost));
        check("K1 KSPLIT (kseg form) == baseline", got == ref);
    }
    // K1b generic split (NOT expected bit-exact; report maxdiff)
    {
        const int S = 4;
        CK(cudaMemset(dout2.p, 0xee, outb));
        const int kslabs = (int)(K/32);
        const int kcmax = ((kslabs + S - 1)/S)*32;
        dim3 grids((unsigned)(R/64*S), (unsigned)n_ids, (unsigned)Ny);
        pxq6_pick_gateup_ksplit_gen(fmt, false, false)<<<grids, 256, kcmax*4 + 2*PXQ4_MMV_KSEG*64*4>>>(
            (const uint8_t *)du.p, (const uint8_t *)dg.p, (const char *)dx.p, x_tok,
            (float *)dws.p, (const char *)dids.p, ids_nb0, ids_nb1, (int)R, (int)K, (int)E, (int)n_ids, S);
        CK(cudaGetLastError());
        dim3 gridr((unsigned)((R + 255)/256), (unsigned)n_ids, (unsigned)Ny);
        k_pxq6_gateup_reduce_gen<<<gridr, 256>>>((const float *)dws.p, (char *)dout2.p, dst_tok, dst_slot,
            (const char *)dids.p, ids_nb0, ids_nb1,
            (const float *)dbu.p, R*4, (const float *)dbg.p, R*4,
            (int)R, (int)E, (int)n_ids, unary, alpha, limit, S);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(got.data(), dout2.p, outb, cudaMemcpyDeviceToHost));
        double md = 0; const float * a = (const float *)ref.data(), * b = (const float *)got.data();
        for (size_t idx = 0; idx < outb/4; ++idx) {
            if (((const uint32_t *)a)[idx] == 0xeeeeeeee) continue;   // SER slots untouched
            md = fmaxf(md, fabsf(a[idx] - b[idx]));
        }
        printf("  K1b KSPLIT_GEN S=4 vs baseline: maxdiff %.3e (deterministic, G3-gated; %s)\n",
               md, got == ref ? "bit-exact here" : "expected non-exact");
    }
    // down mmv: baseline vs variants + old kernels
    {
        dim3 gridd((unsigned)(Rd/64), (unsigned)n_ids, (unsigned)Ny);
        const size_t smem_d = Kd*4 + PXQ4_MMV_KSEG*64*4;
        // x for down = the gateup out layout [iy][j][R]; use dout (baseline)
        auto run_d = [&](pxq6_mmv_fn fn, void * out) {
            CK(cudaMemset(out, 0xdd, downb));
            fn<<<gridd, 256, smem_d>>>((const uint8_t *)dd.p, (const char *)dout.p, dst_tok, dst_slot,
                (char *)out, n_ids*Rd*4, Rd*4, (const char *)dids.p, ids_nb0, ids_nb1, (int)Rd, (int)Kd, (int)E);
            CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        };
        std::vector<uint8_t> dref(downb), dgot(downb);
        run_d(pxq6_pick_mmv(fmt, false, false), ddown.p);
        CK(cudaMemcpy(dref.data(), ddown.p, downb, cudaMemcpyDeviceToHost));
        if (fmt == PXA_PXQ_FMT_P4 || fmt == PXA_PXQ_FMT_P5) {
            auto * old_d = fmt == PXA_PXQ_FMT_P5 ? k_pxq5_mmv_fast : k_pxq4_mmv;
            run_d(old_d, ddown2.p);
            CK(cudaMemcpy(dgot.data(), ddown2.p, downb, cudaMemcpyDeviceToHost));
            check("down: old proven == new baseline", dgot == dref);
        }
        for (int m = 1; m < 4; ++m) {
            run_d(pxq6_pick_mmv(fmt, m & 1, m & 2), ddown2.p);
            CK(cudaMemcpy(dgot.data(), ddown2.p, downb, cudaMemcpyDeviceToHost));
            char buf[64]; snprintf(buf, 64, "down: PAIRLUT=%d VECX=%d == baseline", m & 1, (m >> 1) & 1);
            check(buf, dgot == dref);
        }
    }
    if (bench) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto t_gu = [&](pxq6_gateup_fn fn, const char * lbl) {
            run_gu(fn, dout2.p);
            cudaEventRecord(e0);
            for (int r = 0; r < 200; ++r) run_gu(fn, dout2.p);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            printf("  BENCH gateup %-28s %8.2f us/launch\n", lbl, ms*1000/200);
        };
        t_gu(pxq6_pick_gateup(fmt, false, false), "baseline");
        t_gu(pxq6_pick_gateup(fmt, true, true), "pairlut+vecx");
        // ksplit bench
        dim3 grids((unsigned)(R/64*PXQ4_MMV_KSEG), (unsigned)n_ids, (unsigned)Ny);
        dim3 gridr((unsigned)((R + 255)/256), (unsigned)n_ids, (unsigned)Ny);
        auto run_ks = [&]() {
            pxq6_pick_gateup_ksplit(fmt, false, false)<<<grids, 64, K*4>>>(
                (const uint8_t *)du.p, (const uint8_t *)dg.p, (const char *)dx.p, x_tok,
                (float *)dws.p, (const char *)dids.p, ids_nb0, ids_nb1, (int)R, (int)K, (int)E, (int)n_ids);
            k_pxq6_gateup_reduce<<<gridr, 256>>>((const float *)dws.p, (char *)dout2.p, dst_tok, dst_slot,
                (const char *)dids.p, ids_nb0, ids_nb1,
                (const float *)dbu.p, R*4, (const float *)dbg.p, R*4,
                (int)R, (int)E, (int)n_ids, unary, alpha, limit);
        };
        run_ks(); CK(cudaDeviceSynchronize());
        cudaEventRecord(e0);
        for (int r = 0; r < 200; ++r) run_ks();
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        printf("  BENCH gateup %-28s %8.2f us/launch (incl. reducer)\n", "ksplit-s4", ms*1000/200);
    }
}

// ---------------- prefill test ----------------
static void test_prefill(int fmt, bool bench, int cc) {
    const int64_t R = 512, K = 2048, E = 8;
    const int ntile = 6;
    std::vector<float> wf;
    std::vector<uint8_t> hu = make_wq(fmt, R, K, E, &wf), hg = make_wq(fmt, R, K, E);
    // tiles: mixed full/ragged
    std::vector<pxq4_tile_info> tiles;
    int row0 = 0;
    const int nr[ntile] = {64, 64, 17, 64, 5, 33};
    for (int t = 0; t < ntile; ++t) { tiles.push_back({(int32_t)(t % E), row0, nr[t], 0}); row0 += nr[t]; }
    const int total = row0;
    std::vector<uint16_t> hA(total*K);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto & v : hA) v = ggml_fp32_to_fp16(nd(grng));
    std::vector<float> hbu(E*R);
    for (auto & v : hbu) v = nd(grng)*0.01f;

    DevBuf dwu, dwg, dA, dC, dC2, dCg, dH, dH2, dtl, dbu, dmap, dscat, dscat2;
    dwu.up(hu.data(), hu.size()); dwg.up(hg.data(), hg.size());
    dA.up(hA.data(), hA.size()*2); dtl.up(tiles.data(), tiles.size()*sizeof(pxq4_tile_info));
    dbu.up(hbu.data(), hbu.size()*4);
    dC.alloc(total*R*4); dC2.alloc(total*R*4); dCg.alloc(total*R*4);
    dH.alloc(total*R*4); dH2.alloc(total*R*4);

    dim3 grid((unsigned)(R/64), (unsigned)ntile);
    auto run_g = [&](pxq6_gemm_fn fn, const void * w, void * out) {
        CK(cudaMemset(out, 0xcc, total*R*4));
        fn<<<grid, 64>>>((const uint8_t *)w, (const half *)dA.p, (float *)out,
                         (const float *)dbu.p, R*4, (const pxq4_tile_info *)dtl.p, (int)R, (int)K);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    };
    std::vector<uint8_t> ref(total*R*4), got(total*R*4);

    printf("[prefill %s]\n", fmtname(fmt));
    run_g(pxq6_pick_gemm(fmt, false, false), dwu.p, dC.p);
    CK(cudaMemcpy(ref.data(), dC.p, ref.size(), cudaMemcpyDeviceToHost));

    if (fmt == PXA_PXQ_FMT_P4 || fmt == PXA_PXQ_FMT_P5) {
        auto * old_g = fmt == PXA_PXQ_FMT_P5 ? k_pxq5_gemm_grouped_fast : k_pxq4_gemm_grouped;
        run_g(old_g, dwu.p, dC2.p);
        CK(cudaMemcpy(got.data(), dC2.p, got.size(), cudaMemcpyDeviceToHost));
        check("gemm: old proven == new baseline", got == ref);
    }
    for (int m = 1; m < 4; ++m) {
        run_g(pxq6_pick_gemm(fmt, m & 1, m & 2), dwu.p, dC2.p);
        CK(cudaMemcpy(got.data(), dC2.p, got.size(), cudaMemcpyDeviceToHost));
        char buf[64]; snprintf(buf, 64, "gemm: RAGTAIL=%d PIPE=%d == baseline", m & 1, (m >> 1) & 1);
        // ragged columns beyond nrows are UNWRITTEN (0xcc both sides) => memcmp still valid
        check(buf, got == ref);
    }
    // GUFUSE vs 3-kernel pipeline
    {
        run_g(pxq6_pick_gemm(fmt, false, false), dwu.p, dC.p);
        run_g(pxq6_pick_gemm(fmt, false, false), dwg.p, dCg.p);
        // note: gate GEMM uses the same bias buffer here (bg == bu) for simplicity
        const int64_t k = (int64_t)total*R;
        CK(cudaMemset(dH.p, 0xaa, total*R*4));
        k_pxq4_glu<float><<<(unsigned)((k + 255)/256), 256>>>((const float *)dCg.p, (const float *)dC.p,
                (float *)dH.p, k, 0, 1.702f, 0.f);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemset(dH2.p, 0xaa, total*R*4));
        pxq6_pick_gufuse_f(fmt, false, false)<<<grid, 64>>>((const uint8_t *)dwu.p, (const uint8_t *)dwg.p,
                (const half *)dA.p, (float *)dH2.p, (const float *)dbu.p, R*4, (const float *)dbu.p, R*4,
                (const pxq4_tile_info *)dtl.p, (int)R, (int)K, 0, 1.702f, 0.f);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> h1(total*R), h2(total*R);
        CK(cudaMemcpy(h1.data(), dH.p, total*R*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), dH2.p, total*R*4, cudaMemcpyDeviceToHost));
        // pipeline GLU also processed the unwritten ragged lanes (0xcc garbage) -> compare only valid rows
        bool ok = true;
        for (int t = 0; t < ntile && ok; ++t)
            for (int r = 0; r < tiles[t].nrows && ok; ++r)
                for (int64_t c = 0; c < R && ok; ++c) {
                    const int64_t idx = (int64_t)(tiles[t].row0 + r)*R + c;
                    ok = memcmp(&h1[idx], &h2[idx], 4) == 0;
                }
        check("K3 GUFUSE == up+gate+GLU pipeline (valid rows)", ok);
    }
    // SCATFUSE vs gemm + copy (identity mapping)
    {
        std::vector<pxq4_rowmap> map(total);
        for (int t = 0; t < total; ++t) map[t] = {t, 0};
        dmap.up(map.data(), map.size()*8);
        dscat.alloc(total*R*4); dscat2.alloc(total*R*4);
        run_g(pxq6_pick_gemm(fmt, false, false), dwu.p, dC.p);
        // emulate copy: dst[i1*nb1] = C[i*R..]; identity => same layout
        CK(cudaMemset(dscat.p, 0xbb, total*R*4));
        CK(cudaMemcpy(dscat.p, dC.p, total*R*4, cudaMemcpyDeviceToDevice));
        CK(cudaMemset(dscat2.p, 0xbb, total*R*4));
        pxq6_pick_down_scat(fmt, false, false)<<<grid, 64>>>((const uint8_t *)dwu.p, (const half *)dA.p,
                (char *)dscat2.p, R*4, 0, (const pxq4_rowmap *)dmap.p, (const pxq4_tile_info *)dtl.p, (int)R, (int)K);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        std::vector<float> h1(total*R), h2(total*R);
        CK(cudaMemcpy(h1.data(), dscat.p, total*R*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), dscat2.p, total*R*4, cudaMemcpyDeviceToHost));
        bool ok = true;   // scat kernel writes WITHOUT bias; C was computed WITH bias -> redo baseline without bias
        run_g(pxq6_pick_gemm(fmt, false, false), dwu.p, dC2.p);   // placeholder to keep flow obvious
        // recompute reference without bias:
        CK(cudaMemset(dC2.p, 0xbb, total*R*4));
        pxq6_pick_gemm(fmt, false, false)<<<grid, 64>>>((const uint8_t *)dwu.p, (const half *)dA.p,
                (float *)dC2.p, nullptr, 0, (const pxq4_tile_info *)dtl.p, (int)R, (int)K);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h1.data(), dC2.p, total*R*4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < ntile && ok; ++t)
            for (int r = 0; r < tiles[t].nrows && ok; ++r)
                for (int64_t c = 0; c < R && ok; ++c) {
                    const int64_t idx = (int64_t)(tiles[t].row0 + r)*R + c;
                    ok = memcmp(&h1[idx], &h2[idx], 4) == 0;
                }
        check("K3 SCATFUSE == gemm+copy (valid rows)", ok);
    }
    // dequant matrix vs CPU contract (PXQ6 formats only; PXQ4/5 already covered by prior gates)
    if (fmt >= PXA_PXQ_FMT_P6) {
        const int tier = fmt == PXA_PXQ_FMT_P6HQ ? 1 : 0;
        std::vector<float> cpu(R*K), gpu(R*K);
        pxq6_dequant_expert(hu.data(), cpu.data(), R, K, tier);
        DevBuf dq, dy; dq.up(hu.data(), (R/64)*(PXQ6_HDR_BYTES + (K/32)*(tier ? PXQ6HQ_SLAB_BYTES : PXQ6_SLAB_BYTES)));
        dy.alloc(R*K*4);
        const int kslabs = (int)(K/32);
        if (tier) k_pxq6_dequant_matrix<pxq6_pol_p6hq, float><<<(R/64)*kslabs, 64>>>((const uint8_t *)dq.p, (float *)dy.p, kslabs, K);
        else      k_pxq6_dequant_matrix<pxq6_pol_p6,   float><<<(R/64)*kslabs, 64>>>((const uint8_t *)dq.p, (float *)dy.p, kslabs, K);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gpu.data(), dy.p, R*K*4, cudaMemcpyDeviceToHost));
        check("dequant_matrix == CPU reference contract", memcmp(cpu.data(), gpu.data(), R*K*4) == 0);
    }
    // WMMA (sm_70 only)
    if (cc == 700) {
        run_g(pxq6_pick_gemm_wmma(fmt, true), dwu.p, dC2.p);
        std::vector<float> h1(total*R), h2(total*R);
        CK(cudaMemcpy(h1.data(), dC.p, total*R*4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h2.data(), dC2.p, total*R*4, cudaMemcpyDeviceToHost));
        double md = 0, rel = 0, den = 0;
        for (int t = 0; t < ntile; ++t)
            for (int r = 0; r < tiles[t].nrows; ++r)
                for (int64_t c = 0; c < R; ++c) {
                    const int64_t idx = (int64_t)(tiles[t].row0 + r)*R + c;
                    md = fmax(md, fabs((double)h1[idx] - h2[idx]));
                    rel += ((double)h1[idx] - h2[idx])*((double)h1[idx] - h2[idx]);
                    den += (double)h1[idx]*h1[idx];
                }
        printf("  K6 WMMA fp32-acc vs half2 baseline: maxdiff %.4e relL2 %.4e (NOT bit-exact by design; G3+G4 gate)\n",
               md, sqrt(rel/fmax(den, 1e-30)));
    } else {
        printf("  K6 WMMA: SKIP (cc=%d != 700 — staged for the V100 window)\n", cc);
    }
    if (bench) {
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        auto t_g = [&](pxq6_gemm_fn fn, const char * lbl) {
            run_g(fn, dwu.p, dC2.p);
            cudaEventRecord(e0);
            for (int r = 0; r < 100; ++r)
                fn<<<grid, 64>>>((const uint8_t *)dwu.p, (const half *)dA.p, (float *)dC2.p,
                                 (const float *)dbu.p, R*4, (const pxq4_tile_info *)dtl.p, (int)R, (int)K);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1); CK(cudaGetLastError());
            printf("  BENCH gemm %-28s %8.2f us/launch\n", lbl, ms*1000/100);
        };
        t_g(pxq6_pick_gemm(fmt, false, false), "baseline");
        t_g(pxq6_pick_gemm(fmt, true, false), "ragtail");
        t_g(pxq6_pick_gemm(fmt, false, true), "pipe");
        t_g(pxq6_pick_gemm(fmt, true, true), "ragtail+pipe");
        if (cc == 700) {
            t_g(pxq6_pick_gemm_wmma(fmt, true),  "WMMA fp32-acc");
            t_g(pxq6_pick_gemm_wmma(fmt, false), "WMMA fp16-acc");
        }
    }
}

int main(int argc, char ** argv) {
    bool bench = argc > 1 && !strcmp(argv[1], "--bench");
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    const int cc = prop.major*100 + prop.minor*10;
    printf("pxq6_test on %s (cc %d)%s\n", prop.name, cc, bench ? " [BENCH]" : "");
    for (int fmt = 1; fmt <= 4; ++fmt) test_decode(fmt, bench);
    for (int fmt = 1; fmt <= 4; ++fmt) test_prefill(fmt, bench, cc);
    printf(g_fail ? "RESULT: %d MISMATCHES\n" : "RESULT: ALL BIT-EXACT CHECKS PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
