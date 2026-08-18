// pxq4_kernel_torch.cpp — torch operator bindings for the PXQ4 sm_70 kernels.
//
// NAMESPACE. The library is `pxq4`, deliberately NOT `_C`: the host vLLM fork
// (github.com/KewaiiGamer/1Cat-vLLM) already owns `torch.ops._C` with 54 registered sm70 ops,
// and a second TORCH_LIBRARY(_C, ...) in the same process is a hard registration conflict.
//
// FROZEN ABI (plan §7.1) — components B and C agree on exactly this and nothing else:
//     pxq4::dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()
//     pxq4::mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()
//     pxq4::version() -> int
// The remaining entry points below are ADDITIVE setup/introspection helpers. They are eager-
// mode only and are never called from inside a captured region, so adding them cannot disturb
// the frozen contract.
//
// META / FAKE KERNELS ARE NOT REGISTERED HERE. Per plan §6.7 the fake implementations live in
// the runtime package (`src/pxq4_vllm/ops.py`, agent B) via torch.library.register_fake.
// Registering a Meta kernel here as well would make that call raise on a duplicate
// registration, so this TU registers CUDA implementations only. The two ops mutate their
// first argument and return nothing, so the fakes are shape checks with no return value:
//
//     torch.ops.load_library(_LIB)          # must happen before register_fake
//
//     @torch.library.register_fake("pxq4::dequant_out")
//     def _(out, slabs, anchor):
//         torch._check(out.shape == (slabs.shape[0] * 64, slabs.shape[1] * 32))
//         return None
//
//     @torch.library.register_fake("pxq4::mmv_out")
//     def _(out, x, slabs, anchor):
//         torch._check(x.shape[1] == slabs.shape[1] * 32)
//         torch._check(out.shape == (x.shape[0], slabs.shape[0] * 64))
//         return None
//
// The `Tensor(a!)` annotations in the schemas below are what tell the functionalisation pass
// that `out` is written in place; without them torch.compile silently drops the call.
//
// CAPTURE SAFETY. dequant_out and mmv_out allocate nothing, launch one kernel each, use only
// preallocated caller memory and static/dynamic shared memory, and never synchronise or read
// a device value on the host. They are cuda-graph-capture safe. set_tables is NOT: it is a
// cudaMemcpyToSymbol and must be called once, eagerly, at load time.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <cstdlib>
#include <cstring>

#include "pxq4_kernel_launch.h"
#include "pxq4_kernel_tables.h"

namespace {

// Default token-count ceiling for the mmv path. Mirrors the engine's PXA_PXQ4_2D_MAX_NY
// (ggml-cuda.cu:4019-4021, default 8). Above it the weight-reread cost of the mmv (grid.y is
// the token axis, so each block reads its whole panel once per token) loses to
// dequant + cuBLAS. The Python side owns the actual policy; this is the vendored default.
constexpr int64_t kPxq4MmvMaxM = 8;

struct Geom {
    int panels;
    int kslabs;
    int64_t N;
    int64_t K;
};

Geom check_weight(const at::Tensor & slabs, const at::Tensor & anchor) {
    TORCH_CHECK(slabs.is_cuda() && anchor.is_cuda(), "pxq4: slabs/anchor must be CUDA tensors");
    TORCH_CHECK(slabs.scalar_type() == at::kByte, "pxq4: slabs must be uint8, got ", slabs.scalar_type());
    TORCH_CHECK(anchor.scalar_type() == at::kHalf, "pxq4: anchor must be float16, got ", anchor.scalar_type());
    TORCH_CHECK(slabs.dim() == 3, "pxq4: slabs must be [panels, kslabs, 1088], got dim ", slabs.dim());
    TORCH_CHECK(anchor.dim() == 2, "pxq4: anchor must be [panels, 64], got dim ", anchor.dim());
    TORCH_CHECK(slabs.size(2) == PXQ4_SLAB_BYTES, "pxq4: slab stride must be ", PXQ4_SLAB_BYTES,
                ", got ", slabs.size(2));
    TORCH_CHECK(anchor.size(1) == PXQ4_BM, "pxq4: anchor row must be ", PXQ4_BM, ", got ", anchor.size(1));
    TORCH_CHECK(slabs.size(0) == anchor.size(0),
                "pxq4: panel count mismatch: slabs ", slabs.size(0), " vs anchor ", anchor.size(0));
    // Contiguity is not cosmetic: the vendored code does a 16-byte uint4 load at
    // slab_base + 64 + 16*row, which is only guaranteed aligned when the slab stride is the
    // natural 1088 and the base comes from a torch allocation. A narrow()ed, non-contiguous
    // view would silently misalign it.
    TORCH_CHECK(slabs.is_contiguous(), "pxq4: slabs must be contiguous");
    TORCH_CHECK(anchor.is_contiguous(), "pxq4: anchor must be contiguous");
    TORCH_CHECK(slabs.size(0) > 0 && slabs.size(1) > 0, "pxq4: empty weight");

    Geom g;
    g.panels = (int)slabs.size(0);
    g.kslabs = (int)slabs.size(1);
    g.N = (int64_t)g.panels * PXQ4_BM;
    g.K = (int64_t)g.kslabs * PXQ4_QK;
    return g;
}

void dequant_out(at::Tensor & out, const at::Tensor & slabs, const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4: out must be a CUDA float16 tensor");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == g.N && out.size(1) == g.K,
                "pxq4: out must be [", g.N, ", ", g.K, "], got [", out.size(0), ", ",
                out.dim() == 2 ? out.size(1) : -1, "]");
    TORCH_CHECK(out.is_contiguous(), "pxq4: out must be contiguous");
    TORCH_CHECK(out.device() == slabs.device(), "pxq4: out and slabs on different devices");

    pxq4_launch_dequant_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), out.data_ptr(),
                            g.panels, g.kslabs, at::cuda::getCurrentCUDAStream());
}

void mmv_out(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
             const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf, "pxq4: x must be a CUDA float16 tensor");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kHalf, "pxq4: out must be a CUDA float16 tensor");
    TORCH_CHECK(x.dim() == 2 && out.dim() == 2, "pxq4: x and out must be 2D");
    TORCH_CHECK(x.size(1) == g.K, "pxq4: x K=", x.size(1), " does not match weight K=", g.K);
    TORCH_CHECK(out.size(0) == x.size(0) && out.size(1) == g.N,
                "pxq4: out must be [", x.size(0), ", ", g.N, "]");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    TORCH_CHECK(x.device() == slabs.device() && out.device() == slabs.device(),
                "pxq4: tensors on different devices");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs),
                "pxq4: K=", g.K, " needs ", pxq4_mmv_smem_bytes(g.kslabs),
                " B of dynamic shared memory, which exceeds the 48 KiB sm_70 budget; "
                "use the dequant + GEMM path for this layer");

    const int64_t M = x.size(0);
    if (M == 0) return;
    TORCH_CHECK(M <= 65535, "pxq4: mmv grid.y limit exceeded, M=", M);

    // Deliberately NOT enforcing M <= kPxq4MmvMaxM here: the ceiling is a performance policy
    // owned by the Python caller, and a hard check would make the op unusable for the
    // crossover sweep that decides where the ceiling actually belongs (plan risk 4).

    // Split-path threshold. The split mmv pays a fixed two-launch + partials round-trip
    // cost (~10 us); below ~8 MB of slab bytes the monolithic kernel's launch economy wins
    // even at 12% occupancy (measured: TP4 o_proj, 4.2 MB, mono 21 us vs split 32 us, while
    // every shape >= 8 MB gains 1.5-2.3x from the split). Env-tunable for re-baselining on
    // a different part.
    static const int64_t kSplitMinBytes = [] {
        const char * e = getenv("PXQ4_MMV_SPLIT_MIN_BYTES");
        return e && *e ? (int64_t)atoll(e) : (int64_t)(8 << 20);
    }();
    const int64_t slab_bytes = (int64_t)g.panels * g.kslabs * PXQ4_SLAB_BYTES;
    const int nfix = pxq4_mmv_nfix(g.kslabs);
    if (nfix > 1 && slab_bytes >= kSplitMinBytes) {
        // K-chunk-split fast path: bit-identical to the monolithic kernel (see
        // k_pxq4_mmv_part) but with nfix-times the blocks, which is what keeps an 80-SM
        // V100 busy at decode on this model's small-N TP shapes. The fp32 partials come
        // from the caching allocator: under torch.cuda.graph capture that allocation is
        // served from the graph-private pool exactly like the `out = torch.empty(...)`
        // the Python caller already performs per apply(), so capture safety is unchanged.
        at::Tensor part = at::empty({M * (int64_t)g.panels * nfix * 256},
                                    x.options().dtype(at::kFloat));
        pxq4_launch_mmv_split_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                                  part.data_ptr<float>(), out.data_ptr(),
                                  (int)M, g.panels, g.kslabs, /*vecx=*/true,
                                  at::cuda::getCurrentCUDAStream());
    } else {
        pxq4_launch_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                            out.data_ptr(), (int)M, g.panels, g.kslabs, /*vecx=*/true,
                            at::cuda::getCurrentCUDAStream());
    }
}

// The pre-split monolithic mmv, kept callable so the parity gate can assert the split path
// bit-identical against it on device, and as a one-line fallback if a future shape ever
// misbehaves. Same ABI as mmv_out.
void mmv_out_mono(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                  const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == g.K && x.scalar_type() == at::kHalf, "pxq4: bad x");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == g.N &&
                out.scalar_type() == at::kHalf, "pxq4: bad out");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    TORCH_CHECK(pxq4_mmv_supported(g.kslabs), "pxq4: K too large for mmv smem budget");
    if (x.size(0) == 0) return;
    pxq4_launch_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(),
                        out.data_ptr(), (int)x.size(0), g.panels, g.kslabs, /*vecx=*/true,
                        at::cuda::getCurrentCUDAStream());
}

// Debug twin of mmv_out with the float4 activation loads disabled. Same values, different
// load width; used to isolate an alignment fault from an arithmetic fault at G6/G8.
void mmv_out_scalar(at::Tensor & out, const at::Tensor & x, const at::Tensor & slabs,
                    const at::Tensor & anchor) {
    const Geom g = check_weight(slabs, anchor);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == g.K && x.scalar_type() == at::kHalf, "pxq4: bad x");
    TORCH_CHECK(out.dim() == 2 && out.size(0) == x.size(0) && out.size(1) == g.N &&
                out.scalar_type() == at::kHalf, "pxq4: bad out");
    TORCH_CHECK(x.is_contiguous() && out.is_contiguous(), "pxq4: x and out must be contiguous");
    if (x.size(0) == 0) return;
    pxq4_launch_mmv_f16(slabs.data_ptr<uint8_t>(), anchor.data_ptr(), x.data_ptr(), out.data_ptr(),
                        (int)x.size(0), g.panels, g.kslabs, /*vecx=*/false,
                        at::cuda::getCurrentCUDAStream());
}

int64_t version() { return 2; }  // 2 = K-chunk-split mmv decode path

int64_t mmv_max_m() { return kPxq4MmvMaxM; }

bool mmv_supported(int64_t K) {
    if (K <= 0 || K % PXQ4_QK != 0) return false;
    return pxq4_mmv_supported((int)(K / PXQ4_QK));
}

int64_t mmv_smem_bytes(int64_t K) {
    TORCH_CHECK(K > 0 && K % PXQ4_QK == 0, "pxq4: K must be a positive multiple of ", PXQ4_QK);
    return pxq4_mmv_smem_bytes((int)(K / PXQ4_QK));
}

// Overwrite the device book / sublevel tables from the checkpoint's recorded values
// (gguf KVs pxa.pxq6.book / pxa.pxq6.sub, mirrored into config.json's quantization_config).
// The engine allows PXA_PXQ6_BOOK / PXA_PXQ6_SUB to override the frozen literals at quantize
// time, so a checkpoint is only self-describing if we honour what it recorded.
// EAGER ONLY: this is a cudaMemcpyToSymbol and must happen before any cuda-graph capture.
void set_tables(const at::Tensor & book, const at::Tensor & sub) {
    TORCH_CHECK(book.numel() == 16 && sub.numel() == 16, "pxq4: book and sub must have 16 entries");
    const at::Tensor b = book.to(at::kCPU).to(at::kFloat).contiguous();
    const at::Tensor s = sub.to(at::kCPU).to(at::kFloat).contiguous();
    pxq4_upload_tables(b.data_ptr<float>(), s.data_ptr<float>());
}

// Read the tables back as [2, 16] float32 on CPU: row 0 = book, row 1 = sub.
at::Tensor get_tables() {
    at::Tensor t = at::empty({2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    pxq4_download_tables(t.data_ptr<float>(), t.data_ptr<float>() + 16);
    return t;
}

// The frozen compile-time literals, so a caller can verify the checkpoint's tables against
// the values this .so was built with WITHOUT touching the device (plan §5.6 check 3).
at::Tensor builtin_tables() {
    static const float book[16] = PXQ4_BOOK_INIT;
    static const float sub[16]  = PXQ4_SUB16_INIT;
    at::Tensor t = at::empty({2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCPU));
    std::memcpy(t.data_ptr<float>(),      book, sizeof(book));
    std::memcpy(t.data_ptr<float>() + 16, sub,  sizeof(sub));
    return t;
}

}  // namespace

TORCH_LIBRARY(pxq4, m) {
    m.def("dequant_out(Tensor(a!) out, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_scalar(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("mmv_out_mono(Tensor(a!) out, Tensor x, Tensor slabs, Tensor anchor) -> ()");
    m.def("version() -> int");
    m.def("mmv_max_m() -> int");
    m.def("mmv_supported(int K) -> bool");
    m.def("mmv_smem_bytes(int K) -> int");
    m.def("set_tables(Tensor book, Tensor sub) -> ()");
    m.def("get_tables() -> Tensor");
    m.def("builtin_tables() -> Tensor");
}

TORCH_LIBRARY_IMPL(pxq4, CUDA, m) {
    m.impl("dequant_out", &dequant_out);
    m.impl("mmv_out", &mmv_out);
    m.impl("mmv_out_scalar", &mmv_out_scalar);
    m.impl("mmv_out_mono", &mmv_out_mono);
}

// Device-independent entry points. CompositeExplicitAutograd puts them on every backend
// without claiming an autograd formula (these ops have no gradient and never will).
TORCH_LIBRARY_IMPL(pxq4, CompositeExplicitAutograd, m) {
    m.impl("version", &version);
    m.impl("mmv_max_m", &mmv_max_m);
    m.impl("mmv_supported", &mmv_supported);
    m.impl("mmv_smem_bytes", &mmv_smem_bytes);
    m.impl("builtin_tables", &builtin_tables);
    // set_tables/get_tables take CPU tensors (or none at all), so they cannot be dispatched
    // on the CUDA key by argument device -- they must be registered device-independently even
    // though their bodies touch the current CUDA device.
    m.impl("set_tables", &set_tables);
    m.impl("get_tables", &get_tables);
}
