//
// PXA_ROPE_FLIPPED_v1 correctness harness.
//
// The port's whole claim is: "rotating the TRAILING n_dims channels in place is
// the same function as view/rope/concat on the leading-nope layout, only
// cheaper." So the primary test is not a tolerance comparison against a
// reference, it is an EQUALITY comparison against the code path being replaced:
//
//   arm CLASSIC  : nope=view(x,0) ; pe=view(x,off) ; pe=rope(pe) ; concat(nope,pe)
//   arm FLIPPED  : rope_inplace(x) with op_params[15]=1
//
// Both arms see the same bytes and run on the same backend, so cos/sin are
// computed identically and the two must agree BIT-EXACTLY. Anything less means
// the index arithmetic is wrong. Cross-backend (CPU vs CUDA) uses a tolerance,
// because the CPU builds theta incrementally and CUDA uses powf.
//
// Arms and controls:
//   1. classic == flipped-inplace, bit exact, CPU and CUDA, norm/neox, fwd/back
//   2. classic == flipped NON-inplace (the kernel path the model never takes;
//      upstream's version of it walks off the end of the row, so it is tested
//      here rather than trusted)
//   3. independent double-precision reference, to catch both arms being wrong
//      the same way
//   4. DISCRIMINATION: flipped vs not-flipped on the same input are reported so
//      "they matched" cannot mean "the comparison cannot tell them apart"
//   5. FUSION: two adjacent flipped ROPE nodes, run with fusion=1 and fusion=0.
//      ggml_cuda_op_rope_rope_impl compares only op_params[0..14], so without
//      the flipped guard the fused kernel would silently accept the pair and
//      rotate the leading channels.
//   6. abort probes (argv-selected, run as separate processes by the runner):
//      mrope+flipped, vision+flipped and f16-CPU+flipped must all die.
//
// Tiny on purpose: a few hundred MB, one card. The DGX is on loan.
//

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int g_fail = 0;

static void report(const char * name, bool ok, const char * detail = "") {
    printf("%-62s %s %s\n", name, ok ? "PASS" : "**FAIL**", detail);
    if (!ok) g_fail++;
}

struct rope_cfg {
    const char * name;
    int64_t ne0;      // full row width
    int64_t n_dims;   // rotated channels
    int64_t n_head;
    int64_t nt;       // tokens (== positions)
    int     mode;     // 0 = norm, GGML_ROPE_TYPE_NEOX
    bool    back;     // inverse rotation
    float   ext_factor;
};

static const float FREQ_BASE  = 10000.0f;
static const float FREQ_SCALE = 1.0f;
static const float ATTN_FAC   = 1.0f;
static const float BETA_FAST  = 32.0f;
static const float BETA_SLOW  = 1.0f;
static const int   N_CTX_ORIG = 4096;

static void gen_inputs(const rope_cfg & c, uint32_t seed,
                       std::vector<float> & x, std::vector<int32_t> & pos) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    x.resize((size_t) c.ne0*c.n_head*c.nt);
    for (auto & v : x) v = nd(rng);
    pos.resize((size_t) c.nt);
    for (int64_t i = 0; i < c.nt; ++i) pos[i] = (int32_t) (i*3 + 7);
}

// ---------------------------------------------------------------------------
// arm builders
// ---------------------------------------------------------------------------

enum arm_kind { ARM_CLASSIC, ARM_FLIPPED_INPLACE, ARM_FLIPPED_COPY, ARM_UNFLIPPED_INPLACE };

// Returns false only on an infrastructure failure (alloc/compute), never on a
// numeric one - the caller decides what "wrong" means.
static bool run_arm(ggml_backend_t be, const rope_cfg & c, arm_kind arm,
                    const std::vector<float> & xh, const std::vector<int32_t> & posh,
                    std::vector<float> & out) {
    const int64_t off = c.ne0 - c.n_dims;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * x   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.ne0, c.n_head, c.nt);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.nt);

    ggml_tensor * o = nullptr;

    if (arm == ARM_CLASSIC) {
        const size_t rs = ggml_row_size(x->type, c.ne0);
        ggml_tensor * pe = off == 0
            ? x
            : ggml_view_3d(ctx, x, c.n_dims, c.n_head, c.nt, rs, rs*c.n_head,
                           ggml_row_size(x->type, off));
        pe = c.back
            ? ggml_rope_back(ctx, pe, pos, nullptr, c.n_dims, c.mode, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, c.ext_factor, ATTN_FAC, BETA_FAST, BETA_SLOW)
            : ggml_rope_ext (ctx, pe, pos, nullptr, c.n_dims, c.mode, N_CTX_ORIG,
                             FREQ_BASE, FREQ_SCALE, c.ext_factor, ATTN_FAC, BETA_FAST, BETA_SLOW);
        if (off == 0) {
            o = pe;   // ne0 == n_dims: there is no nope half to splice back on
        } else {
            ggml_tensor * nope = ggml_view_3d(ctx, x, off, c.n_head, c.nt, rs, rs*c.n_head, 0);
            o = ggml_concat(ctx, nope, pe, 0);
        }
    } else {
        const bool inplace = arm != ARM_FLIPPED_COPY;
        o = inplace
            ? ggml_rope_ext_inplace(ctx, x, pos, nullptr, c.n_dims, c.mode, N_CTX_ORIG,
                                    FREQ_BASE, FREQ_SCALE, c.ext_factor, ATTN_FAC, BETA_FAST, BETA_SLOW)
            : ggml_rope_ext        (ctx, x, pos, nullptr, c.n_dims, c.mode, N_CTX_ORIG,
                                    FREQ_BASE, FREQ_SCALE, c.ext_factor, ATTN_FAC, BETA_FAST, BETA_SLOW);
        if (arm != ARM_UNFLIPPED_INPLACE) {
            o->op_params[15] = 1;
        }
        if (c.back) {
            o->op = GGML_OP_ROPE_BACK;
        }
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(x,   xh.data(),   0, xh.size()*sizeof(float));
    ggml_backend_tensor_set(pos, posh.data(), 0, posh.size()*sizeof(int32_t));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out.resize(ggml_nelements(o));
    ggml_backend_tensor_get(o, out.data(), 0, out.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Two adjacent flipped ROPE nodes in one graph - the shape that trips
// ggml_cuda_op_rope_rope fusion.
static bool run_pair(ggml_backend_t be, const rope_cfg & c, bool flipped,
                     const std::vector<float> & xh, const std::vector<int32_t> & posh,
                     std::vector<float> & out_a, std::vector<float> & out_b) {
    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * xa  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.ne0, c.n_head, c.nt);
    ggml_tensor * xb  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, c.ne0, c.n_head, c.nt);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.nt);

    ggml_tensor * oa = ggml_rope_ext_inplace(ctx, xa, pos, nullptr, c.n_dims, c.mode, N_CTX_ORIG,
                                             FREQ_BASE, FREQ_SCALE, c.ext_factor, ATTN_FAC, BETA_FAST, BETA_SLOW);
    ggml_tensor * ob = ggml_rope_ext_inplace(ctx, xb, pos, nullptr, c.n_dims, c.mode, N_CTX_ORIG,
                                             FREQ_BASE, FREQ_SCALE, c.ext_factor, ATTN_FAC, BETA_FAST, BETA_SLOW);
    if (flipped) { oa->op_params[15] = 1; ob->op_params[15] = 1; }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { ggml_free(ctx); return false; }

    ggml_backend_tensor_set(xa,  xh.data(),   0, xh.size()*sizeof(float));
    ggml_backend_tensor_set(xb,  xh.data(),   0, xh.size()*sizeof(float));
    ggml_backend_tensor_set(pos, posh.data(), 0, posh.size()*sizeof(int32_t));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, oa);
    ggml_build_forward_expand(gf, ob);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) {
        ggml_backend_buffer_free(buf); ggml_free(ctx); return false;
    }

    out_a.resize(ggml_nelements(oa)); ggml_backend_tensor_get(oa, out_a.data(), 0, out_a.size()*sizeof(float));
    out_b.resize(ggml_nelements(ob)); ggml_backend_tensor_get(ob, out_b.data(), 0, out_b.size()*sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// ---------------------------------------------------------------------------
// independent double-precision reference (ext_factor == 0 only)
// ---------------------------------------------------------------------------

static void reference(const rope_cfg & c, const std::vector<float> & xh,
                      const std::vector<int32_t> & posh, bool flipped,
                      std::vector<float> & out) {
    const int64_t off  = flipped ? c.ne0 - c.n_dims : 0;
    const bool    neox = (c.mode & GGML_ROPE_TYPE_NEOX) != 0;
    const double  ts   = pow((double) FREQ_BASE, -2.0/(double) c.n_dims);
    const double  ssgn = c.back ? -1.0 : 1.0;

    out = xh;   // untouched channels pass through
    for (int64_t t = 0; t < c.nt; ++t) {
        for (int64_t h = 0; h < c.n_head; ++h) {
            const size_t base = ((size_t) t*c.n_head + h)*c.ne0;
            for (int64_t i0 = 0; i0 < c.n_dims; i0 += 2) {
                const double theta = (double) posh[t]*pow(ts, (double) (i0/2));
                const double cs = cos(theta)*ATTN_FAC;
                const double sn = sin(theta)*ATTN_FAC*ssgn;

                const size_t i_a = neox ? base + off + i0/2              : base + off + i0;
                const size_t i_b = neox ? base + off + i0/2 + c.n_dims/2 : base + off + i0 + 1;

                const double x0 = xh[i_a];
                const double x1 = xh[i_b];
                out[i_a] = (float) (x0*cs - x1*sn);
                out[i_b] = (float) (x0*sn + x1*cs);
            }
        }
    }
}

// ---------------------------------------------------------------------------

static size_t ndiff_exact(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return (size_t) -1;
    size_t n = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) n++;
    }
    return n;
}

static double rms_pct(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return INFINITY;
    double se = 0.0, sr = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        const double d = (double) a[i] - (double) b[i];
        se += d*d; sr += (double) b[i]*(double) b[i];
    }
    if (sr <= 0.0) return se > 0.0 ? INFINITY : 0.0;
    return 100.0*sqrt(se/a.size())/sqrt(sr/a.size());
}

// ---------------------------------------------------------------------------
// abort probes - each is expected to kill the process
// ---------------------------------------------------------------------------

static int abort_probe(const std::string & which) {
    ggml_backend_t be = ggml_backend_cpu_init();
    const int64_t ne0 = 64, n_dims = 32, nh = 2, nt = 4;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), NULL, true };
    ggml_context * ctx = ggml_init(ip);

    const bool f16 = which == "f16";
    ggml_tensor * x = ggml_new_tensor_3d(ctx, f16 ? GGML_TYPE_F16 : GGML_TYPE_F32, ne0, nh, nt);

    int mode = 0;
    int64_t npos = nt;
    if (which == "mrope")  { mode = GGML_ROPE_TYPE_MROPE;  npos = nt*4; }
    if (which == "vision") { mode = GGML_ROPE_TYPE_VISION; npos = nt*4; }

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, npos);

    ggml_tensor * o;
    if (mode) {
        int sections[4] = { 8, 8, 8, 8 };
        o = ggml_rope_multi(ctx, x, pos, nullptr, which == "vision" ? (int) (ne0/2) : (int) n_dims,
                            sections, mode, N_CTX_ORIG,
                            FREQ_BASE, FREQ_SCALE, 0.0f, ATTN_FAC, BETA_FAST, BETA_SLOW);
    } else {
        o = ggml_rope_ext_inplace(ctx, x, pos, nullptr, (int) n_dims, 0, N_CTX_ORIG,
                                  FREQ_BASE, FREQ_SCALE, 0.0f, ATTN_FAC, BETA_FAST, BETA_SLOW);
    }
    o->op_params[15] = 1;   // the thing that must be rejected

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<char> zero(ggml_nbytes(x), 0);
    ggml_backend_tensor_set(x, zero.data(), 0, zero.size());
    std::vector<int32_t> ph((size_t) npos, 1);
    ggml_backend_tensor_set(pos, ph.data(), 0, ph.size()*sizeof(int32_t));

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, o);
    ggml_backend_graph_compute(be, gf);   // must not return

    printf("abort probe '%s' RETURNED - the guard did not fire\n", which.c_str());
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return 0;   // 0 == the probe FAILED to abort; the runner treats that as a failure
}

// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    if (argc > 1 && strncmp(argv[1], "abort:", 6) == 0) {
        return abort_probe(argv[1] + 6);
    }

    ggml_backend_t cpu  = ggml_backend_cpu_init();
    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr);
    if (!cuda) { printf("no CUDA device -- cannot run\n"); return 2; }

    // DS4's real geometry first (attention key_length 512 / rope.dimension_count 64;
    // indexer key_length 128 / n_rot 64), then shapes that stress the arithmetic.
    // ne0 == n_dims is the degenerate case where flipped and classic coincide.
    // 576/192 are deliberately non-power-of-two offsets.
    const rope_cfg cases[] = {
        { "norm fwd, ne0 512 n_dims 64 (DS4 attn/kv)", 512, 64,  8,  4, 0,                    false, 0.0f },
        { "norm fwd, ne0 512 n_dims 64, 1 head",       512, 64,  1,  4, 0,                    false, 0.0f },
        { "norm fwd, ne0 512 n_dims 64, 1 token",      512, 64,  8,  1, 0,                    false, 0.0f },
        { "norm fwd, ne0 512 n_dims 64, 64 tok",       512, 64,  4, 64, 0,                    false, 0.0f },
        { "norm fwd, ne0 128 n_dims 64 (DS4 indexer)", 128, 64,  4,  8, 0,                    false, 0.0f },
        { "norm fwd, ne0 64  n_dims 64 (no nope half)", 64, 64,  4,  8, 0,                    false, 0.0f },
        { "norm fwd, ne0 576 n_dims 64 (odd offset)",  576, 64,  4,  8, 0,                    false, 0.0f },
        { "norm fwd, ne0 512 n_dims 64, yarn ext",     512, 64,  4,  8, 0,                    false, 0.5f },
        { "norm BACK, ne0 512 n_dims 64 (derope)",     512, 64,  8,  4, 0,                    true,  0.0f },
        { "norm BACK, ne0 512 n_dims 64, yarn ext",    512, 64,  4,  8, 0,                    true,  0.5f },
        { "neox fwd, ne0 512 n_dims 64",               512, 64,  8,  4, GGML_ROPE_TYPE_NEOX,  false, 0.0f },
        { "neox BACK, ne0 512 n_dims 64",              512, 64,  8,  4, GGML_ROPE_TYPE_NEOX,  true,  0.0f },
        { "neox fwd, ne0 192 n_dims 64 (odd offset)",  192, 64,  4,  8, GGML_ROPE_TYPE_NEOX,  false, 0.0f },
    };

    printf("=== 1. classic view/rope/concat  ==  flipped in-place (BIT EXACT, same backend) ===\n");
    for (const auto & c : cases) {
        std::vector<float> xh; std::vector<int32_t> ph;
        gen_inputs(c, 4242, xh, ph);
        for (int pass = 0; pass < 2; ++pass) {
            ggml_backend_t be = pass == 0 ? cpu : cuda;
            std::vector<float> a, b;
            const bool ok1 = run_arm(be, c, ARM_CLASSIC,         xh, ph, a);
            const bool ok2 = run_arm(be, c, ARM_FLIPPED_INPLACE, xh, ph, b);
            char nm[160]; snprintf(nm, sizeof nm, "[%s] %s", pass == 0 ? "cpu " : "cuda", c.name);
            if (!ok1 || !ok2) { report(nm, false, "(compute failed)"); continue; }
            const size_t nd = ndiff_exact(a, b);
            char d[128]; snprintf(d, sizeof d, "(%zu/%zu floats differ)", nd, a.size());
            report(nm, nd == 0, d);
        }
    }

    printf("\n=== 2. classic == flipped NON-in-place (the kernel the model never takes) ===\n");
    for (const auto & c : cases) {
        std::vector<float> xh; std::vector<int32_t> ph;
        gen_inputs(c, 4242, xh, ph);
        for (int pass = 0; pass < 2; ++pass) {
            ggml_backend_t be = pass == 0 ? cpu : cuda;
            std::vector<float> a, b;
            const bool ok1 = run_arm(be, c, ARM_CLASSIC,      xh, ph, a);
            const bool ok2 = run_arm(be, c, ARM_FLIPPED_COPY, xh, ph, b);
            char nm[160]; snprintf(nm, sizeof nm, "[%s] %s", pass == 0 ? "cpu " : "cuda", c.name);
            if (!ok1 || !ok2) { report(nm, false, "(compute failed)"); continue; }
            const size_t nd = ndiff_exact(a, b);
            char d[128]; snprintf(d, sizeof d, "(%zu/%zu floats differ)", nd, a.size());
            report(nm, nd == 0, d);
        }
    }

    printf("\n=== 3. flipped in-place vs independent double reference (ext_factor 0 only) ===\n");
    for (const auto & c : cases) {
        if (c.ext_factor != 0.0f) continue;
        std::vector<float> xh, ref; std::vector<int32_t> ph;
        gen_inputs(c, 4242, xh, ph);
        reference(c, xh, ph, /*flipped*/ true, ref);
        for (int pass = 0; pass < 2; ++pass) {
            ggml_backend_t be = pass == 0 ? cpu : cuda;
            std::vector<float> b;
            if (!run_arm(be, c, ARM_FLIPPED_INPLACE, xh, ph, b)) { report(c.name, false, "(compute failed)"); continue; }
            const double e = rms_pct(b, ref);
            char nm[160]; snprintf(nm, sizeof nm, "[%s] %s", pass == 0 ? "cpu " : "cuda", c.name);
            char d[128]; snprintf(d, sizeof d, "(rms %.6f%% vs ref)", e);
            report(nm, e < 0.02, d);
        }
    }

    printf("\n=== 4. DISCRIMINATION: does the comparison in test 1 have any power? ===\n");
    printf("%-62s %s\n", "", "(flipped vs UNflipped on the same input; must be FAR apart)");
    {
        double worst = INFINITY;
        for (const auto & c : cases) {
            if (c.ne0 == c.n_dims) continue;   // no nope half: flipped == unflipped by definition
            std::vector<float> xh; std::vector<int32_t> ph;
            gen_inputs(c, 4242, xh, ph);
            std::vector<float> f, u;
            if (!run_arm(cuda, c, ARM_FLIPPED_INPLACE,   xh, ph, f)) continue;
            if (!run_arm(cuda, c, ARM_UNFLIPPED_INPLACE, xh, ph, u)) continue;
            const double e = rms_pct(f, u);
            if (e < worst) worst = e;
        }
        char d[128]; snprintf(d, sizeof d, "(closest pair: %.2f%% rms apart)", worst);
        report("flipped and unflipped are distinguishable", worst > 1.0, d);
    }

    printf("\n=== 5. FUSION: adjacent flipped ROPE pair must not be fused away ===\n");
    printf("%-62s %s\n", "", "(rope_rope memcmp covers op_params[0..14] only)");
    {
        ggml_backend_t cuda_f1 = ggml_backend_cuda_init(0, "fusion=1");
        ggml_backend_t cuda_f0 = ggml_backend_cuda_init(0, "fusion=0");
        if (!cuda_f1 || !cuda_f0) {
            report("fusion A/B backends", false, "(init failed)");
        } else {
            const rope_cfg c = cases[0];
            std::vector<float> xh; std::vector<int32_t> ph;
            gen_inputs(c, 4242, xh, ph);

            std::vector<float> classic;
            run_arm(cuda_f0, c, ARM_CLASSIC, xh, ph, classic);

            for (int f = 0; f < 2; ++f) {
                ggml_backend_t be = f ? cuda_f1 : cuda_f0;
                std::vector<float> a, b;
                if (!run_pair(be, c, /*flipped*/ true, xh, ph, a, b)) {
                    report(f ? "flipped pair, fusion=1" : "flipped pair, fusion=0", false, "(compute failed)");
                    continue;
                }
                const size_t d1 = ndiff_exact(a, classic);
                const size_t d2 = ndiff_exact(b, classic);
                char d[160]; snprintf(d, sizeof d, "(node A %zu differ, node B %zu differ vs classic)", d1, d2);
                report(f ? "flipped pair, fusion=1 == classic" : "flipped pair, fusion=0 == classic",
                       d1 == 0 && d2 == 0, d);
            }
            ggml_backend_free(cuda_f1);
            ggml_backend_free(cuda_f0);
        }
    }

    printf("\n%s  (%d failure%s)\n", g_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED",
           g_fail, g_fail == 1 ? "" : "s");

    ggml_backend_free(cuda);
    ggml_backend_free(cpu);
    return g_fail ? 1 : 0;
}
