//

// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license

// PXA provenance canary -- kept in the binary for authorship forensics.
#ifdef __GNUC__
__attribute__((used))
#endif
__attribute__((used, visibility("default"))) const char pxa_provenance[] =
    "pxq_llama :: authored by PXA Network (pxanetwork.com) :: "
    "creator=PXANetwork :: origin-canary=PXA-7Q6LM32E16-ORIGIN :: forensic-watermark";
namespace {
// Keep the authorship canary in the linked binary (forensic provenance) --
// referenced at load so --gc-sections cannot drop it.
struct pxa_prov_keeper_t { pxa_prov_keeper_t() { volatile const char * volatile s = pxa_provenance; (void)s; } };
__attribute__((used)) static pxa_prov_keeper_t pxa_prov_keeper_instance;
} // namespace
// SPDX-License-Identifier: MIT
//
#include "ggml-cuda.h"
#include "ggml.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/pxa-enhance.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/dmmv.cuh"
#include "ggml-cuda/pxa-smalln.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/mmq_id.cuh"
#include "ggml-cuda/quantize_id.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/multiadd.cuh"
#include "ggml-cuda/hadamard.cuh"
#include "ggml-cuda/reduce.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/delta-net.cuh"

#include <algorithm>
#include <array>
#include <atomic>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>

#define IK_PRINT_TIMING 0

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void ggml_cuda_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fprintf(stderr, "%s", msg);
}

ggml_log_callback ggml_cuda_log_callback = ggml_cuda_default_log_callback;
void * ggml_cuda_log_user_data = NULL;

GGML_API void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_cuda_log_callback = log_callback;
    ggml_cuda_log_user_data = user_data;
}

#define GGML_CUDA_LOG_INFO(...) ggml_cuda_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_CUDA_LOG_WARN(...) ggml_cuda_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_CUDA_LOG_ERROR(...) ggml_cuda_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_CUDA_LOG_DEBUG(...) ggml_cuda_log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_cuda_log(enum ggml_log_level level, const char * format, ...) {
    if (ggml_cuda_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_cuda_log_callback(level, buffer, ggml_cuda_log_user_data);
        } else {
            std::vector<char> buffer2(len + 1);  // vsnprintf adds a null terminator
            va_end(args);
            va_start(args, format);
            vsnprintf(&buffer2[0], buffer2.size(), format, args);
            ggml_cuda_log_callback(level, buffer2.data(), ggml_cuda_log_user_data);
        }
        va_end(args);
    }
}

// PXA_PROFILE: per-op-type GPU-time profiler (env PXA_PROFILE=1). Sync-per-op perturbs the total but
// gives an accurate per-op-TYPE breakdown of where decode time goes. Dumps every 20000 op-computes.
static double g_pxa_op_us[128] = {0};
static long   g_pxa_op_cnt[128] = {0};
static long   g_pxa_total_ops = 0;
static int    g_pxa_prof_on = -1;
// PXA_NODE_PROF (2026-07-15): name-bucketed accumulation (digit runs -> #) + per-graph summaries.
static std::map<std::string, std::pair<double,long>> g_pxa_name_us;
static long g_pxa_prof_every = 20000;
static std::string pxa_name_bucket(const char * name) {
    std::string b; b.reserve(48);
    bool indig = false;
    for (const char * c = name; *c && b.size() < 46; ++c) {
        if (*c >= '0' && *c <= '9') { if (!indig) b.push_back('#'); indig = true; }
        else { indig = false; b.push_back(*c); }
    }
    return b;
}
static int    g_pxa_op_check_on = -1; // PXA_WAVE4_DIAG
static void pxa_prof_dump() {
    double tot=0; for (int i=0;i<128;++i) tot+=g_pxa_op_us[i];
    int idx[128]; for (int i=0;i<128;++i) idx[i]=i;
    for (int i=0;i<128;++i) for (int j=i+1;j<128;++j) if (g_pxa_op_us[idx[j]]>g_pxa_op_us[idx[i]]) { int t=idx[i]; idx[i]=idx[j]; idx[j]=t; }
    fprintf(stderr, "==== PXA_PROFILE (op-computes=%ld, total_gpu_us=%.0f) ====\n", g_pxa_total_ops, tot);
    for (int k=0;k<128;++k){ int o=idx[k]; if (g_pxa_op_cnt[o]==0) continue; fprintf(stderr,"  %-24s us=%10.0f (%5.1f%%) cnt=%8ld avg=%6.1f\n", ggml_op_name((ggml_op)o), g_pxa_op_us[o], 100.0*g_pxa_op_us[o]/(tot>0?tot:1), g_pxa_op_cnt[o], g_pxa_op_us[o]/g_pxa_op_cnt[o]); }
    {
        std::vector<std::pair<double, const std::string *>> v;
        v.reserve(g_pxa_name_us.size());
        for (auto & kv : g_pxa_name_us) v.push_back({kv.second.first, &kv.first});
        std::sort(v.begin(), v.end(), [](const auto & a, const auto & b){ return a.first > b.first; });
        int n = 0;
        fprintf(stderr, "---- top name buckets ----\n");
        for (auto & e : v) {
            if (++n > 48) break;
            auto & pr = g_pxa_name_us[*e.second];
            fprintf(stderr, "  %-46s us=%10.0f (%5.1f%%) cnt=%8ld avg=%6.1f\n", e.second->c_str(), pr.first, 100.0*pr.first/(tot>0?tot:1), pr.second, pr.first/pr.second);
        }
    }
    fflush(stderr);
}
[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    GGML_CUDA_LOG_ERROR("CUDA error: %s\n", msg);
    GGML_CUDA_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_CUDA_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CUDA error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
#if defined(GGML_USE_HIPBLAS) && defined(GGML_HIP_UMA)
    auto res = hipMallocManaged(ptr, size);
    if (res == hipSuccess) {
        // if error we "need" to know why...
        CUDA_CHECK(hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
    }
    return res;
#else

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr)
    {
        err = cudaMallocManaged(ptr, size);
    }
    else
    {
        err = cudaMalloc(ptr, size);
    }
    return err;
#else
    return cudaMalloc(ptr, size);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

#endif
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    yes\n", __func__);
#else
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ:    no\n", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: yes\n", __func__);
#else
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS: no\n", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    int  pxa_ccs[GGML_CUDA_MAX_DEVICES] = {0};
    char pxa_names[GGML_CUDA_MAX_DEVICES][256] = {{0}};
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        GGML_CUDA_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                prop.totalGlobalMem/(1024*1024));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm   = prop.multiProcessorCount;
        info.devices[id].smpb  = prop.sharedMemPerBlock;
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpbo = prop.sharedMemPerBlock;
        info.devices[id].cc = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        pxa_ccs[id] = info.devices[id].cc;
        snprintf(pxa_names[id], sizeof(pxa_names[id]), "%s", prop.name);
    }

    // PXA master config tiers (PXA_REFERENCE / default / PXA_ENHANCE): one-time per-device report
    pxa_enhance_log_startup(info.device_count, pxa_ccs, pxa_names);

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

#ifdef GGML_USE_NCCL
    info.have_nccl = false;
    if (info.device_count > 1) {
        int gpu_list[GGML_CUDA_MAX_DEVICES];
        for(int i = 0; i < info.device_count; ++i) gpu_list[i] = i;
        auto status = ncclCommInitAll(info.nccl_coms, info.device_count, gpu_list);
        if (status != ncclSuccess) {
            printf("=============================== NCCL initialization failed with status %d\n", int(status));
        } else {
            printf("=============================== NCCL main communicator initialized\n");
            info.have_nccl = true;
            auto com = info.nccl_coms + info.device_count;
            if (info.device_count == 4) {
                int devs[8] = {0,1, 2,3, 0,2, 1,3};
                auto com = info.nccl_coms + info.device_count;
                for (int ip = 0; ip < 4; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            } else if (info.device_count == 3) {
                int devs[4] = {0,1, 0,2};
                for (int ip = 0; ip < 2; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            }
        }
    }
#endif
    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;
    uint64_t gen = 0; // PXA_CUDA_GRAPH_V2 P3: bumped on real cudaFree (see ggml_cuda_pool::generation)

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    uint64_t generation() const override { return gen; }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_CUDA_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_CUDA_LOG_WARN("Cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
        ++gen; // PXA_CUDA_GRAPH_V2 P3: pool memory was returned to the driver; captured graphs may hold it
    }
};

// pool with virtual memory
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) (pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) (pool_addr + pool_used));
    }
};
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
//static std::atomic<int> ggml_cuda_lock_counter;
static int ggml_cuda_lock_counter = 0;

ggml_backend_cuda_context::ggml_backend_cuda_context(int device) :
    device(device), name(GGML_CUDA_NAME + std::to_string(device)) {
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    if (info->all_ctx[device]) {
        GGML_CUDA_LOG_WARN("%s: a context for device %d already exists?\n", __func__, device);
    } else{
        info->all_ctx[device] = this;
    }
    // PXA_CUBLAS_EAGER_INIT (default ON, =0 rollback): create this device's cuBLAS handle (and
    // its PXA_CUBLAS_EAGER_WS workspace) at backend init, BEFORE weights fill VRAM. Some decode
    // configs first touch cuBLAS mid-inference via a rare fallback shape; lazy handle+workspace
    // creation at that point is exactly the near-full-card alloc that fails intermittently.
    static const bool pxa_cublas_eager_init =
        !(getenv("PXA_CUBLAS_EAGER_INIT") && atoi(getenv("PXA_CUBLAS_EAGER_INIT")) == 0);
    if (pxa_cublas_eager_init) {
        cublas_handle(device);
    }
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {

#ifdef USE_CUDA_GRAPH
    // Let's leave this debug log in for now, so we have a trace in case
    // number of CUDA graphs goes crazy
    GGML_CUDA_LOG_INFO("%s: have %d graphs\n", __func__, int(cuda_graphs.size()));
#endif

    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (compute_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(compute_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
        if (cublas_workspaces[i] != nullptr) {
            CUDA_CHECK(cudaFree(cublas_workspaces[i]));
        }
    }
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    if (info->all_ctx[device] == this) {
        info->all_ctx[device] = nullptr;
    }

}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

GGML_CALL static const char * ggml_backend_cuda_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_buffer_get_name;
}

GGML_CALL static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
#ifndef NDEBUG
    printf("%s(%s -> %s)\n", __func__, src->name, dst->name);
#endif
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_cuda_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

struct ggml_backend_cuda_split_buffer_type_context {
    //std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        //for (ggml_tensor_extra_gpu * extra : tensor_extras) {
        //    for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
        //        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
        //            if (extra->events[id][is] != nullptr) {
        //                CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
        //            }
        //        }
        //        if (extra->data_device[id] != nullptr) {
        //            CUDA_CHECK(cudaFree(extra->data_device[id]));
        //        }
        //    }
        //    delete extra;
        //}
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

GGML_CALL static const char * ggml_backend_cuda_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buffer);
}

static bool ggml_backend_buffer_is_cuda_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_cuda_split); // only used in debug builds currently, avoid unused function warning in release builds
}

GGML_CALL static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_split_buffer_init_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (!tensor->extra) return;
    //printf("%s(%s, %p)\n", __func__, tensor->name, tensor->extra);
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        auto split = extra->splits[i];
        auto ne0 = split->ne[0];
        auto size = ggml_nbytes(split);
        auto padded_size = size;
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            int nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            auto row_size = ggml_row_size(split->type, ne0);
            padded_size += padded_row_size - row_size;
        }
        ggml_cuda_set_device(i);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, padded_size, i));
        if (padded_size > size) {
            CUDA_CHECK(cudaMemset(buf + size, 0, padded_size - size));
        }
        //printf("    allocated %zu bytes for tensor %s of type %s, dim = %ld x %ld x %ld. padding: %zu\n", padded_size, split->name, ggml_type_name(split->type),
        //        split->ne[0], split->ne[1], split->ne[2], padded_size - size);
        split->data = buf;
        auto ctx = new ggml_backend_cuda_buffer_context(i, buf);
        auto buft = ggml_backend_cuda_buffer_type(i);
        split->buffer = ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, padded_size);
        ggml_backend_buffer_set_usage(split->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return;

}

GGML_CALL static void ggml_backend_cuda_split_buffer_set_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!tensor->extra && tensor->view_src && tensor->view_src->extra) {
        // OK, this is an ugly hack, but I don't really see a way to trick the machine into correctly
        // loading non-contiguous merged split tensors.
        auto view_src = tensor->view_src;
        auto extra = (ggml_split_tensor_t *)view_src->extra;
        void * extra_ptr;
        memcpy(&extra_ptr, view_src->op_params, sizeof(extra_ptr));
        if (extra_ptr) {
            std::string merged_name = view_src->name;
            if (auto pos = merged_name.find("ffn_gate_up_exps.weight"); pos != std::string::npos) {
                std::string name = tensor->name;
                auto pos_u = name.find("ffn_up_exps.weight");
                auto pos_g = name.find("ffn_gate_exps.weight");
                if (pos_u != std::string::npos || pos_g != std::string::npos) {
                    GGML_ASSERT(extra->split_dim == 1);
                    auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                    int ne = 0;
                    for (int is = 0; is < int(ranges.size()); ++is) {
                        auto & r = ranges[is];
                        GGML_ASSERT((extra->splits[is] && !r.empty()) || (!extra->splits[is] && r.empty()));
                        if (r.empty()) continue;
                        GGML_ASSERT(r.size() == 2);
                        auto split = extra->splits[is];
                        ggml_cuda_set_device(is);
                        int ir = pos_g != std::string::npos ? 0 : 1;
                        auto p = r[ir];
                        size_t offset = 0;
                        if (ir == 1) {
                            p.first -= tensor->ne[1];
                            GGML_ASSERT(p.first >= 0);
                            offset = split->ne[1]/2 * split->nb[1];
                        }
                        for (int i02 = 0; i02 < split->ne[2]; ++i02) {
                            auto dst = (char *)split->data + i02*split->nb[2] + offset;
                            auto src = (const char *)data + i02*tensor->nb[2] + ne*tensor->nb[1];
                            CUDA_CHECK(cudaMemcpyAsync(dst, src, p.second*tensor->nb[1], cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        ne += p.second;
                        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
                return;
            }
            if (auto pos = merged_name.find("ffn_gate_up_exps.bias"); pos != std::string::npos) {
                std::string name = tensor->name;
                auto pos_u = name.find("ffn_up_exps.bias");
                auto pos_g = name.find("ffn_gate_exps.bias");
                if (pos_u != std::string::npos || pos_g != std::string::npos) {
                    GGML_ASSERT(extra->split_dim == 0);
                    auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                    int ne = 0;
                    for (int is = 0; is < int(ranges.size()); ++is) {
                        auto & r = ranges[is];
                        GGML_ASSERT((extra->splits[is] && !r.empty()) || (!extra->splits[is] && r.empty()));
                        if (r.empty()) continue;
                        GGML_ASSERT(r.size() == 2);
                        auto split = extra->splits[is];
                        ggml_cuda_set_device(is);
                        int ir = pos_g != std::string::npos ? 0 : 1;
                        auto p = r[ir];
                        size_t offset = 0;
                        if (ir == 1) {
                            p.first -= tensor->ne[0];
                            GGML_ASSERT(p.first >= 0);
                            offset = split->ne[0]/2 * split->nb[0];
                        }
                        for (int i01 = 0; i01 < split->ne[1]; ++i01) {
                            auto dst = (char *)split->data + i01*split->nb[1] + offset;
                            auto src = (const char *)data + i01*tensor->nb[1] + ne*tensor->nb[0];
                            CUDA_CHECK(cudaMemcpyAsync(dst, src, p.second*tensor->nb[0], cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        ne += p.second;
                        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
                return;
            }
        }
    }
    if (!tensor->extra) return;
    static std::map<ggml_type, int> k_map = {
        { GGML_TYPE_Q4_0_R8   , 8},
        { GGML_TYPE_Q5_0_R4   , 4},
        { GGML_TYPE_Q8_0_R8   , 8},
        { GGML_TYPE_Q2_K_R4   , 4},
        { GGML_TYPE_Q3_K_R4   , 4},
        { GGML_TYPE_Q4_K_R4   , 4},
        { GGML_TYPE_Q5_K_R4   , 4},
        { GGML_TYPE_Q6_K_R4   , 4},
        { GGML_TYPE_IQ2_XXS_R4, 4},
        { GGML_TYPE_IQ2_XS_R4 , 4},
        { GGML_TYPE_IQ3_XXS_R4, 4},
        { GGML_TYPE_IQ1_S_R4  , 4},
        { GGML_TYPE_IQ4_NL_R4 , 4},
        { GGML_TYPE_IQ3_S_R4  , 4},
        { GGML_TYPE_IQ2_S_R4  , 4},
        { GGML_TYPE_IQ4_XS_R8 , 8},
        { GGML_TYPE_IQ1_M_R4  , 4},
        { GGML_TYPE_BF16_R16  , 16},
        { GGML_TYPE_Q6_0_R4   , 4},
        { GGML_TYPE_IQ2_BN_R4 , 4},
        { GGML_TYPE_IQ2_K_R4  , 4},
        { GGML_TYPE_IQ3_K_R4  , 4},
        { GGML_TYPE_IQ4_K_R4  , 4},
        { GGML_TYPE_IQ5_K_R4  , 4},
        { GGML_TYPE_IQ4_KS_R4 , 4},
        { GGML_TYPE_IQ5_KS_R4 , 4},
        { GGML_TYPE_Q8_K_R16  , 4},
        { GGML_TYPE_Q8_KV_R8  , 4},
        { GGML_TYPE_Q8_K_R8   , 8},
    };

    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    if (extra->split_dim < 0) {
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto nbytes = ggml_nbytes(tensor);
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(ggml_are_same_shape(tensor, split));
            GGML_ASSERT(ggml_nbytes(split) == nbytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(split->data, data, nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
    }
    else if (extra->split_dim == 0) {
        int n_interleave = 1;
        if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
        auto tt = ggml_internal_get_type_traits(tensor->type);
        std::vector<char> host_buffer;
        GGML_ASSERT(ggml_is_contiguous(tensor));
        int nrows = ggml_nrows(tensor);
        auto bs = tt.blck_size;
        auto ts = tt.type_size;
        void * extra_ptr;
        memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
        if (extra_ptr) {
            auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
            GGML_ASSERT(extra->n_device == int(ranges.size()));
            GGML_ASSERT(tensor->ne[2]*tensor->ne[3] == 1);
            GGML_ASSERT(n_interleave == 1);
            GGML_ASSERT(tt.row_meta_size == 0);
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) {
                    GGML_ASSERT(ranges[i].empty());
                    continue;
                }
                GGML_ASSERT(!ranges[i].empty());
                GGML_ASSERT((int)ggml_nrows(split) == nrows);
                auto split_row_size = ggml_row_size(split->type, split->ne[0]);
                if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
                auto dst = host_buffer.data();
                for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                    for (auto & p : ranges[i]) {
                        GGML_ASSERT(p.first  % bs == 0);
                        GGML_ASSERT(p.second % bs == 0);
                        auto src = (const char *)data + i01*tensor->nb[1] + (p.first/bs)*ts;
                        auto size = (p.second/bs)*ts;
                        memcpy(dst, src, size);
                        dst += size;
                    }
                }
                ggml_cuda_set_device(i);
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
        } else {
            int ne = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                GGML_ASSERT(split->ne[1]%n_interleave == 0);
                ggml_cuda_set_device(i);
                GGML_ASSERT(split->type == tensor->type);
                GGML_ASSERT((int)ggml_nrows(split) == nrows);
                GGML_ASSERT(split->ne[0] % bs == 0);
                auto source_offset = n_interleave*(tt.row_meta_size + (ne / bs) * ts);
                auto split_row_size = ggml_row_size(split->type, split->ne[0]);
                if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                        auto dst = host_buffer.data() + (i02*split->ne[1] + i01)*split_row_size;
                        auto src = (const char *)data + i02*tensor->nb[2] + i01*tensor->nb[1];
                        if (tt.row_meta_size > 0) {
                            memcpy(dst, src, tt.row_meta_size*n_interleave);
                        }
                        memcpy(dst + tt.row_meta_size*n_interleave, src + source_offset, n_interleave*(split_row_size - tt.row_meta_size));
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne += split->ne[0];
            }
        }
    }
    else if (extra->split_dim == 1) {
        void * extra_ptr;
        memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
        if (tensor->ne[2] > 1) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<char> host_buffer;
            int ne1 = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                auto size = ggml_nbytes(split);
                if (host_buffer.size() < size) host_buffer.resize(size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    auto dst = host_buffer.data() + i02*split->ne[1]*row_size;
                    if (extra_ptr) {
                        auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                        for (auto & p : ranges[i]) {
                            auto this_src = (const char *)data + i02*tensor->nb[2] + p.first*tensor->nb[1];
                            auto this_size = p.second*tensor->nb[1];
                            memcpy(dst, this_src, this_size);
                            dst += this_size;
                        }
                    } else {
                        auto src = (const char *)data + i02*tensor->nb[2] + ne1*tensor->nb[1];
                        memcpy(dst, src, split->ne[1]*row_size);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne1 += split->ne[1];
            }
        } else {
            int n_interleave = 1;
            if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
            if (extra_ptr) {
                auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                GGML_ASSERT(extra->n_device == int(ranges.size()));
                GGML_ASSERT(tensor->ne[2]*tensor->ne[3] == 1);
                GGML_ASSERT(n_interleave == 1);
                for (int i = 0; i < extra->n_device; ++i) {
                    auto split = extra->splits[i];
                    if (!split) {
                        GGML_ASSERT(ranges[i].empty());
                        continue;
                    }
                    GGML_ASSERT(!ranges[i].empty());
                    ggml_cuda_set_device(i);
                    auto dst = (char *)split->data;
                    for (auto & p : ranges[i]) {
                        GGML_ASSERT(p.first  >= 0 && p.first < tensor->ne[1]);
                        GGML_ASSERT(p.second >= 0 && p.first + p.second <= tensor->ne[1]);
                        auto src = (const char *)data + p.first*tensor->nb[1];
                        auto size = p.second*tensor->nb[1];
                        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                        dst += size;
                    }
                }
            } else {
                size_t cur_offset = 0;
                for (int i = 0; i < extra->n_device; ++i) {
                    auto split = extra->splits[i];
                    if (!split) continue;
                    GGML_ASSERT(split->ne[1]%n_interleave == 0);
                    ggml_cuda_set_device(i);
                    auto size = ggml_nbytes(split);
                    const char * buf_host = (const char *)data + cur_offset;
                    CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    cur_offset += size;
                }
            }
        }
    }
    else if (extra->split_dim == 2) {
        size_t cur_offset = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            ggml_cuda_set_device(i);
            auto size = ggml_nbytes(split);
            const char * buf_host = (const char *)data + cur_offset;
            CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            cur_offset += size;
        }
    }
    else {
        fprintf(stderr, "%s: not implemented for split dim %d\n", __func__, extra->split_dim == 0);
        GGML_ABORT("fatal error");
    }

    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

}

GGML_CALL static void ggml_backend_cuda_split_buffer_get_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    // split tensors must always be read in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    if (!tensor->extra) return;

    // Inverse of split_buffer_set_tensor; refuses paths with no defined inverse.
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    // Repacked types are block-de-interleaved by set_tensor; no runtime inverse.
    {
        const ggml_type t = tensor->type;
        const bool is_repacked =
            t == GGML_TYPE_Q4_0_R8  || t == GGML_TYPE_Q5_0_R4  || t == GGML_TYPE_Q8_0_R8  ||
            t == GGML_TYPE_Q2_K_R4  || t == GGML_TYPE_Q3_K_R4  || t == GGML_TYPE_Q4_K_R4  ||
            t == GGML_TYPE_Q5_K_R4  || t == GGML_TYPE_Q6_K_R4  || t == GGML_TYPE_IQ4_NL_R4 ||
            t == GGML_TYPE_IQ4_XS_R8 || t == GGML_TYPE_Q6_0_R4;
        if (is_repacked) {
            GGML_ABORT("%s: get_tensor of repacked type %s is not invertible",
                       __func__, ggml_type_name(t));
        }
    }

    // Explicit-ranges form (non-contiguous expert assignments) is not invertible.
    void * extra_ptr = nullptr;
    memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
    if (extra_ptr) {
        GGML_ABORT("%s: get_tensor with explicit ranges is not implemented", __func__);
    }

    if (extra->split_dim < 0) {
        // Replicated: read from first present device.
        GGML_ASSERT(ggml_is_contiguous(tensor));
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(data, split->data, ggml_nbytes(tensor),
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            return;
        }
        GGML_ABORT("%s: no device holds a copy of the replicated tensor", __func__);
    }
    else if (extra->split_dim == 0) {
        // Row-split (concat along ne[0]).
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto tt = ggml_internal_get_type_traits(tensor->type);
        GGML_ASSERT(tt.row_meta_size == 0);
        std::vector<char> host_buffer;
        int64_t ne0_acc = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(split->ne[0] % tt.blck_size == 0);
            const size_t split_row_size = ggml_row_size(split->type, split->ne[0]);
            const size_t dev_bytes      = (size_t)ggml_nrows(split) * split_row_size;
            if (host_buffer.size() < dev_bytes) host_buffer.resize(dev_bytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), split->data, dev_bytes,
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            const size_t source_offset = (ne0_acc / tt.blck_size) * tt.type_size;
            for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                for (int64_t i01 = 0; i01 < split->ne[1]; ++i01) {
                    const char * src = host_buffer.data() + (i02*split->ne[1] + i01) * split_row_size;
                    char       * dst = (char *)data + i02*tensor->nb[2] + i01*tensor->nb[1] + source_offset;
                    memcpy(dst, src, split_row_size);
                }
            }
            ne0_acc += split->ne[0];
        }
    }
    else if (extra->split_dim == 1) {
        // Column/ne[1] split.
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        if (tensor->ne[2] > 1) {
            std::vector<char> host_buffer;
            int64_t ne1_acc = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                const size_t dev_bytes = ggml_nbytes(split);
                if (host_buffer.size() < dev_bytes) host_buffer.resize(dev_bytes);
                ggml_cuda_set_device(i);
                CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), split->data, dev_bytes,
                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    const char * src = host_buffer.data() + i02 * split->ne[1] * row_size;
                    char       * dst = (char *)data + i02*tensor->nb[2] + ne1_acc*tensor->nb[1];
                    memcpy(dst, src, split->ne[1] * row_size);
                }
                ne1_acc += split->ne[1];
            }
        } else {
            size_t cur_offset = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                const size_t dev_bytes = ggml_nbytes(split);
                CUDA_CHECK(cudaMemcpyAsync((char *)data + cur_offset, split->data, dev_bytes,
                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
                cur_offset += dev_bytes;
            }
            for (int i = 0; i < extra->n_device; ++i) {
                if (!extra->splits[i]) continue;
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    else {
        GGML_ABORT("%s: not implemented for split_dim %d", __func__, extra->split_dim);
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

GGML_CALL static const char * ggml_backend_cuda_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size([[maybe_unused]] ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (!tensor->extra) return 0;
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    size_t total_size = 0;
    for (int i = 0; i < extra->n_device; ++i) {
        auto split = extra->splits[i];
        if (!split) continue;
        total_size += ggml_nbytes(split);
        auto ne0 = split->ne[0];
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            auto nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto row_size = ggml_row_size(split->type, ne0);
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            total_size += padded_row_size - row_size;
        }
    }
    return total_size;

}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * /*tensor_split*/) {
    static ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .context = */ new ggml_backend_cuda_split_buffer_type_context{}, //{tensor_split_arr},
    };
    return &buft;
}

// ============================================================================
// PXA-SHARD (M1, 2026-07-08): expert-shard buffer type.
//
//   A distinct CUDA buffer type that shards a 3D expert tensor
//   [d_model, d_ff, n_expert] on ne[2] (expert-id) across a MATCHED,
//   P2P-connected device group (e.g. the V100 pair, or the P100 quad).
//
//   It is modeled on ggml_backend_cuda_split_buffer_* above. This fork's split
//   buffer already knows how to split on split_dim==2 (see the split_dim==2
//   branches in set_tensor/get_tensor and init_tensor), and the per-device
//   slicing is produced by prepare_split_tensors(2, ...) in the loader. So the
//   expert-shard type does NOT re-implement any allocation/upload/download
//   numerics — it reuses the split buffer's proven vtable verbatim. The ONLY
//   thing it changes is IDENTITY: a distinct type/buffer name so that
//     * ggml_cuda_moe_up_gate_unary (M2) can detect an expert-sharded parent
//       and take the disjoint-write, NO-all-reduce path, and
//     * the scheduler (supports_op / supports_buft) keeps such ops on CUDA,
//   while the stock CUDA_Split type keeps its own row-split/all-reduce
//   semantics untouched.
//
//   FLAG-GATED: nothing in the codebase instantiates this type unless
//   PXA_EXPERT_SHARD is set (wired in M3's loader). With the flag unset this
//   type is never created, no op ever has an expert-shard parent, and every
//   predicate below returns false => zero behavioral change => the binary is
//   bit-identical to the stock (flag-off) build. This is the "one-line
//   rollback is real" guarantee (unset the env var).
//
//   The type context carries the group device list + n_shard for M2/M3; those
//   fields are unused in M1 (the buffer type alone).
// ============================================================================

struct pxa_expert_shard_buffer_type_context {
    std::vector<int> group;   // group device ids (matched arch, P2P-reachable)
    int              n_shard; // == group.size()
    pxa_expert_shard_buffer_type_context() : n_shard(0) {}
};

GGML_CALL static const char * pxa_expert_shard_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_ExpertShard";
    GGML_UNUSED(buffer);
}

// Per-buffer predicate (parallels ggml_backend_buffer_is_cuda_split).
static bool pxa_buffer_is_expert_shard(ggml_backend_buffer_t buffer) {
    return buffer && buffer->iface.get_name == pxa_expert_shard_buffer_get_name;
    GGML_UNUSED(pxa_buffer_is_expert_shard); // referenced by M2 op path; silence unused in M1
}

// PER-BUFFER context: the group + per-tensor per-device slice base pointers.
// Unlike the stock split buffer, the expert-shard buffer sets NO ->extra on the
// tensor (extra stays NULL) so that llm_build_moe_ffn keeps the FUSED single-op
// MoE path (its dispatcher takes the fused branch iff none of the exps carry a
// ggml_split_tensor_t extra). The per-device bases live HERE and the M2 op path
// reads them via pxa_expert_shard_tensor_info().
struct pxa_expert_shard_buffer_context {
    std::vector<int> group;   // group device ids
    int              n_shard = 0;
    size_t           total_size = 0;                 // full mass for this buffer (alloc_buffer size)
    std::map<const ggml_tensor *, std::vector<void *>> bases; // tensor -> per-device slice ptr per group idx
};

// PROCESS-WIDE malloc-once registry: weight tensor* -> per-group device slice ptrs.
// ggml can materialize the CUDA_ExpertShard buffer more than once (a fresh ctx on a
// scheduler reserve pass); without this each instance re-mallocs all slices -> OOM.
// Keyed by the stable model->tensors object pointer, so a 2nd buffer reuses slice 1.
static std::map<std::string, std::vector<void *>> s_pxa_shard_slices; // KEY = tensor->name (stable across reserve-pass struct copies)
static std::vector<int> s_pxa_shard_group; // group device ids (one M2 group); captured at load for view-resolved tensor_info

GGML_CALL static void pxa_expert_shard_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    // erase-guarded free: free each tensor's slices exactly once (the first buffer
    // instance to free wins + drops it from the registry) so a 2nd instance's free
    // can never double-free the shared device pointers.
    for (auto & kv : ctx->bases) {
        auto git = s_pxa_shard_slices.find(kv.first->name);
        if (git == s_pxa_shard_slices.end()) continue; // already freed by another instance
        for (size_t k = 0; k < kv.second.size() && k < ctx->group.size(); ++k) {
            if (kv.second[k]) { ggml_cuda_set_device(ctx->group[k]); (void)cudaFree(kv.second[k]); }
        }
        s_pxa_shard_slices.erase(git);
    }
    delete ctx;
}
GGML_CALL static void * pxa_expert_shard_buffer_get_base(ggml_backend_buffer_t buffer) {
    return (void *)0x1000; // dummy; real per-device bases are in the context
    GGML_UNUSED(buffer);
}
// allocate expert slice [k*EP..(k+1)*EP) on group device k for a 3D expert tensor.
GGML_CALL static void pxa_expert_shard_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    const int n = ctx->n_shard;
    GGML_ASSERT(n >= 2);
    GGML_ASSERT(tensor->ne[2] % n == 0); // uniform expert shards (M2)
    // VIEW tensors alias their view_src's data (ggml sets tensor->data = view_src->data
    // + view_offs BEFORE init_tensor). Views must NEVER allocate: the decode-graph
    // reserve pass re-inits expert-weight VIEWS, and mallocing per view was the OOM.
    // Match the stock cuda buffer, which simply returns for a view.
    if (tensor->view_src != nullptr) { tensor->extra = nullptr; return; }
    // MALLOC-ONCE keyed by tensor->name: the reserve pass copies weight structs to
    // fresh addresses but keeps the GGUF name, so a repeat init for the same logical
    // weight (any buffer instance / pass) reuses the slices instead of re-mallocing.
    {
        auto g0 = s_pxa_shard_slices.find(tensor->name);
        if (g0 != s_pxa_shard_slices.end()) {
            ctx->bases[tensor] = g0->second;
            tensor->data = g0->second[0]; tensor->extra = nullptr; return;
        }
    }
    const int64_t EP = tensor->ne[2] / n;
    const size_t  es = tensor->nb[2];    // bytes per expert (row-aligned already for d_model%128==0)
    const size_t  slice = (size_t)EP*es; // per-device slice = EP experts of this tensor
    // Per-slice cudaMalloc on each group device (exactly parallels the stock
    // split buffer's init_tensor). No pre-sized arena / capacity bookkeeping ->
    // can never over/under-count -> the capacity assert is gone. One malloc per
    // device per tensor, once at load; net-neutral VRAM (~half-mass/device).
    std::vector<void *> bs(n, nullptr);
    for (int k = 0; k < n; ++k) {
        const int dev = ctx->group[k];
        ggml_cuda_set_device(dev);
        char * p = nullptr;
        CUDA_CHECK(ggml_cuda_device_malloc((void **)&p, slice, dev));
        bs[k] = p;
    }
    s_pxa_shard_slices[tensor->name] = bs; // record globally (name key) so a repeat init reuses these
    ctx->bases[tensor] = bs;
    tensor->data  = bs[0];   // home = group[0]; the op branches before any cuda-ctx device deref
    tensor->extra = nullptr; // MUST stay NULL -> fused MoE path kept
}
GGML_CALL static void pxa_expert_shard_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    const int n = ctx->n_shard;
    const int64_t EP = tensor->ne[2] / n;
    const size_t  es = tensor->nb[2];
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor)); // MoE exps upload whole
    auto it = ctx->bases.find(tensor); GGML_ASSERT(it != ctx->bases.end());
    for (int k = 0; k < n; ++k) {
        ggml_cuda_set_device(ctx->group[k]);
        const char * src = (const char *)data + (size_t)k*EP*es;
        CUDA_CHECK(cudaMemcpy(it->second[k], src, (size_t)EP*es, cudaMemcpyHostToDevice));
    }
}
GGML_CALL static void pxa_expert_shard_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    auto * ctx = (pxa_expert_shard_buffer_context *)buffer->context;
    const int n = ctx->n_shard;
    const int64_t EP = tensor->ne[2] / n;
    const size_t  es = tensor->nb[2];
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor));
    auto it = ctx->bases.find(tensor); GGML_ASSERT(it != ctx->bases.end());
    for (int k = 0; k < n; ++k) {
        ggml_cuda_set_device(ctx->group[k]);
        char * dst = (char *)data + (size_t)k*EP*es;
        CUDA_CHECK(cudaMemcpy(dst, it->second[k], (size_t)EP*es, cudaMemcpyDeviceToHost));
    }
}
GGML_CALL static void pxa_expert_shard_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer); GGML_UNUSED(value); // weights buffer: never cleared
}

static struct ggml_backend_buffer_i pxa_expert_shard_buffer_interface = {
    /* .get_name        = */ pxa_expert_shard_buffer_get_name,
    /* .free_buffer     = */ pxa_expert_shard_buffer_free_buffer,
    /* .get_base        = */ pxa_expert_shard_buffer_get_base,
    /* .init_tensor     = */ pxa_expert_shard_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ pxa_expert_shard_buffer_set_tensor,
    /* .get_tensor      = */ pxa_expert_shard_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ pxa_expert_shard_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * pxa_expert_shard_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_ExpertShard";
    GGML_UNUSED(buft);
}

// Type-level predicate used by the MoE op path (M2), supports_op and
// supports_buft. Returns false for every non-expert-shard buft (incl. the
// stock CUDA_Split type), so it is behavior-preserving when the flag is off.
GGML_CALL bool pxa_buft_is_expert_shard(ggml_backend_buffer_type_t buft) {
    return buft && buft->iface.get_name == pxa_expert_shard_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t pxa_expert_shard_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // Per-device slices are allocated per tensor in init_tensor; the buffer only
    // carries the group + the per-tensor base table (see init_tensor/set_tensor).
    auto * tctx = (pxa_expert_shard_buffer_type_context *)buft->context;
    auto * ctx  = new pxa_expert_shard_buffer_context();
    ctx->group      = tctx->group;
    ctx->n_shard    = tctx->n_shard;
    if (s_pxa_shard_group.empty()) s_pxa_shard_group = tctx->group; // global group for view-resolved tensor_info
    ctx->total_size = size;   // full mass of this buffer's shard tensors
    ggml_backend_buffer_t buf = ggml_backend_buffer_init(buft, pxa_expert_shard_buffer_interface, ctx, size);
    // Pin as a WEIGHTS buffer so the scheduler never treats it as reallocatable on a
    // reserve pass (the root cause of the 2nd materialization). Belt-and-suspenders
    // with the malloc-once registry above.
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    return buf;
}

// get_alloc_size for the extra-less shard buffer: return the FULL tensor size
// (the group splits it across devices in init_tensor). MUST be non-zero so the
// generic allocator creates the (dummy) buffer and calls init_tensor. The split
// get_alloc_size returns 0 when ->extra is NULL (the shard tensor has none),
// which is why allocation previously failed.
GGML_CALL static size_t pxa_expert_shard_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    size_t total = ggml_nbytes(tensor);
    const int64_t ne0 = tensor->ne[0];
    if (ne0 % MATRIX_ROW_PADDING != 0) {
        const int64_t nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
        total += ggml_row_size(tensor->type, nblock*MATRIX_ROW_PADDING) - ggml_row_size(tensor->type, ne0);
    }
    return total;
}

static ggml_backend_buffer_type_i pxa_expert_shard_buffer_type_interface = {
    /* .get_name         = */ pxa_expert_shard_buffer_type_name,
    /* .alloc_buffer     = */ pxa_expert_shard_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ pxa_expert_shard_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

// Returns a distinct expert-shard buffer type for the given matched device
// group. One cached instance per distinct group signature (M4 may create a
// V100 pair AND a P100 quad). Called ONLY by the M3 loader when
// PXA_EXPERT_SHARD is set — never on the flag-off path.
GGML_CALL ggml_backend_buffer_type_t pxa_expert_shard_buffer_type(const int * group, int n_shard) {
    static std::vector<ggml_backend_buffer_type *>               s_bufts;
    static std::vector<pxa_expert_shard_buffer_type_context *>   s_ctxs;
    for (size_t k = 0; k < s_ctxs.size(); ++k) {
        auto * c = s_ctxs[k];
        if ((int) c->group.size() == n_shard) {
            bool same = true;
            for (int i = 0; i < n_shard; ++i) {
                if (c->group[i] != group[i]) { same = false; break; }
            }
            if (same) return s_bufts[k];
        }
    }
    auto * ctx = new pxa_expert_shard_buffer_type_context();
    ctx->n_shard = n_shard;
    ctx->group.assign(group, group + n_shard);
    auto * buft = new ggml_backend_buffer_type();
    buft->iface   = pxa_expert_shard_buffer_type_interface;
    buft->context = ctx;
    s_ctxs.push_back(ctx);
    s_bufts.push_back(buft);
    return buft;
}

// PXA-SHARD (M2) op accessor: fetch the per-device slice bases + group device
// ids for an expert-sharded tensor. Returns false for any non-shard tensor, so
// the op path is a no-op when the flag is off.
GGML_CALL bool pxa_expert_shard_tensor_info(const ggml_tensor * t, void ** bases, int * group, int * n_shard) {
    // The MoE op is handed a VIEW of the expert weight: its ->buffer is the compute
    // buffer (not the shard buft) and its pointer differs from the recorded weight.
    // Resolve the view chain to the root, then look slices up by NAME (immune to the
    // graph's tensor-struct copies). Group comes from the global captured at load.
    while (t && t->view_src) t = t->view_src;
    if (!t) return false;
    auto g = s_pxa_shard_slices.find(t->name);
    if (g == s_pxa_shard_slices.end() || s_pxa_shard_group.empty()) return false;
    const int ns = (int)s_pxa_shard_group.size();
    if ((int)g->second.size() < ns) return false;
    for (int k = 0; k < ns; ++k) { bases[k] = g->second[k]; group[k] = s_pxa_shard_group[k]; }
    *n_shard = ns;
    return true;
}

// host buffer type

GGML_CALL static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cuda_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    constexpr double k_warn_limit = 8.0;

    void * ptr = nullptr;
    double size_GiB = size/(1024.*1024.*1024.);
    auto tim1 = ggml_time_us();
    if (size_GiB > k_warn_limit) {
        GGML_CUDA_LOG_INFO("\n\nAllocating %.2f GiB of pinned host memory, this may take a while.\n", size_GiB);
        GGML_CUDA_LOG_INFO("Using pinned host memory improves PP performance by a significant margin.\n");
        GGML_CUDA_LOG_INFO("But if it takes too long for your model and amount of patience, kill the process and run using\n\n");
        GGML_CUDA_LOG_INFO("GGML_CUDA_NO_PINNED=1 your_command_goes_here\n");
    }
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (size_GiB > k_warn_limit) {
        auto tim2 = ggml_time_us();
        GGML_CUDA_LOG_INFO("    done allocating %.2f GiB in %.1f ms\n\n", size_GiB, 1e-3*(tim2-tim1));
    }
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cuda_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x     = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel   = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi = __half2float(x[ix]);

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t rs = ggml_row_size(type, ne0);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == rs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        // TODO: this only works if the row does not contain meta data
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*rs);
            // pretend the row is a matrix with cols=1
            // TODO: this only works if the row does not contain meta data
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {

        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
        return;
    }

#ifdef GGML_CUDA_IQK_FORCE_BF16
    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src0->type);
        to_bf16_cuda_t to_bf16_cuda_1 = src1->type != GGML_TYPE_BF16 ? ggml_get_to_bf16_cuda(src1->type) : nullptr;
        if (to_bf16_cuda && (src1->type == GGML_TYPE_BF16 || to_bf16_cuda_1)) {
            size_t ne = row_diff*ne00;
            ggml_cuda_pool_alloc<nv_bfloat16> src0_as_bf16(ctx.pool(id), ne);
            to_bf16_cuda(src0_dd_i, src0_as_bf16.get(), row_diff, ne00, stream);

            ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
            if (src1->type != GGML_TYPE_BF16) {
                size_t ne = src1_ncols*ne10;
                src1_as_bf16.alloc(ne);
                to_bf16_cuda_1(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
            }
            const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
            const nv_bfloat16 * src0_ptr = src0_as_bf16.get();

            ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

            const float alpha_f32 = 1.0f;
            const float beta_f32  = 0.0f;

            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
            CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                        src1_ptr,       CUDA_R_16BF, ne10,
                        &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
            to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
            return;
        }
    }
#endif

    // PXA_P100_FP16_GEMM_v1 (2026-07-15): GP100 (sm_60) has NATIVE double-rate fp16 (HFMA2/half2,
    // ~19 TF vs 9.5 TF fp32) — it was the flagship fp16 card before Volta. The old CC_VOLTA gate
    // (aimed at GP102/GP104, whose fp16 is 1/64 rate) forced P100 onto dequant->fp32 + SGEMM for
    // every quantized prompt matmul: measured 80%% of gpt-oss prefill wall (dequantize_block_mxfp4<float>
    // 30%% + maxwell_sgemm/sgemm_*_vec 49%%). fast_fp16_available() (cc>=600 && cc!=610) already knows
    // the truth — use it here: fp16 dequant (half the write traffic) + GemmEx 16F (2x FLOP rate).
    // sm_61 (1080Ti) stays excluded. Env PXA_P100_FP16_GEMM=0 rolls back to the old fp32 path.
    // PXA 0a hygiene (2026-07-22): level-aware resolver (pxa-enhance.cuh) — REFERENCE -> false so
    // PXA_REFERENCE=1 baselines on sm_60 really run the pure fp32 SGEMM reference path. Env wins.
    const bool pxa_fp16_gemm_ok = compute_capability >= CC_VOLTA ||
        (pxa_p100_fp16_gemm() && compute_capability < CC_VOLTA && fast_fp16_available(compute_capability));
    if (pxa_fp16_gemm_ok && (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff, ne00, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), src1_ncols, ne10, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                                src1_ptr,       CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff, src1_ncols, stream);
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff, ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols, ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static bool ggml_cuda_set_peer_access(int main_device) {
    ggml_cuda_set_device(main_device);

    bool all_enabled = true;
    for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
        if (main_device == id_other) {
            continue;
        }

        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, main_device, id_other));
        if (can_access_peer) {
//~ #ifdef NDEBUG
            GGML_CUDA_LOG_INFO(" =========================== %s: Enabling Peer Access between Devices %d->%d\n", __func__, main_device, id_other);
//~ #endif //NDEBUG
            cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
            if (err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            } else {
                // reset the error
                (void)cudaGetLastError();
            }
        } else {
            all_enabled = false;
        }
    }
    return all_enabled;
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_rs = ggml_row_size(src0->type, ne00);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

    }

    bool quantization_done = false;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer, it may then be necessary to clear padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            const int64_t nbytes_data    = ggml_row_size(src0->type, (dev[id].row_high - dev[id].row_low)*ne00);
            const int64_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data , 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && (src1_is_contiguous || (src1->ne[1] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float)))) {
                if (src1_is_contiguous) {
                    quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12*ne13, src1_padded_col_size, src0->type, stream);
                } else {
                    //printf("Calling quantize_tensor_q8_1_cuda for %s\n", src0->name);
                    quantize_tensor_q8_1_cuda(src1, dev[id].src1_ddq, src0->type, stream);
                }
                CUDA_CHECK(cudaGetLastError());
                quantization_done = true;
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    const int64_t src1_col_stride = ne11;
    // split-buffer src0 has data == NULL; per-device dispatch happens in the slow path below.
    const bool src0_is_split = src0->buffer && ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    if (quantization_done && ne11 == 1 && ne12 > 1 && ne13 == 1 && ne02 == ne12 && ne02 == dst->ne[2] && !src0_is_split) {
        int id = ctx.device;
        char  *  src0_dd_i =  dev[id].src0_dd;
        float * src1_ddf_i = dev[id].src1_ddf;
        char  * src1_ddq_i = dev[id].src1_ddq;
        float *   dst_dd_i =   dev[id].dst_dd;
        cudaStream_t stream = ctx.stream(id, 0);
        ggml_cuda_op_mul_mat_vec_q_3D(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                dev[id].row_low, dev[id].row_high, ne11, src1_padded_col_size, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[id].src0_dd + (i0/i02_divisor) * ne01*src0_rs;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    if (!quantization_done) {
                        //printf("Copying %s\n", src1->name);
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                    src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                    }
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous && !quantization_done) {
                    //printf("Quantizing %s\n", src1->name);
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                    GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                    dhf_dst_i += src1_col_0*ne0;
                    CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                }

            }
        }
    }
}

static void ggml_cuda_mul_mat_vec_p021(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        //printf("Using cublasGemmStridedBatchedEx for %s\n", dst->name);
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        //const int64_t smb = ne12 == 1 ? s13       : s12;
        const int64_t smb = ne12 == 1 ? nb13/nb10 : nb12/nb10;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        //printf("Using cublasGemmBatchedEx for %s\n", dst->name);
        //printf("    src0: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n",src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        //printf("    src1: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n",src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, 1, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

// ---------------------------------------------------------------------------------------------
// G2-F3 NORMFUSE (PXA_G2_NORMFUSE=1, default OFF): a FUSED_RMS_NORM whose output feeds q8_1-
// quantized GEMV consumers emits the q8_1 of its own output as a SIDECAR in the same kernel
// (bit-identical to norm-then-quantize_q8_1, see norm.cu). Consumers (the mmvq fast-TG chain,
// FUSED_UP_GATE) look the sidecar up by producer-tensor identity and skip their quantize launch.
// The sidecar is only valid within the same graph eval (serial-checked); a miss falls back to
// the normal quantize path, so behavior is always correct.
// ---------------------------------------------------------------------------------------------
static inline bool pxa_g2_normfuse() {
    static const bool on = [](){
        const char * e = getenv("PXA_G2_NORMFUSE");
        bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_G2_NORMFUSE: ON (G2-F3 fused rmsnorm + q8_1 sidecar producer fusion, bit-exact)\n");
        return v;
    }();
    return on;
}

struct pxa_g2_q8sc_t {
    const ggml_tensor * t = nullptr;   // producer node (sidecar = q8_1 of its output)
    const void * data     = nullptr;   // producer dst->data at emit time
    char * buf            = nullptr;   // persistent device buffer
    size_t sz             = 0;
    int64_t padded        = 0;         // ne10_padded the sidecar was laid out for
    uint64_t eval         = 0;         // graph-eval serial the sidecar was emitted in
};
static pxa_g2_q8sc_t pxa_g2_q8sc[GGML_CUDA_MAX_DEVICES];
static uint64_t pxa_g2_eval_serial = 1;

// G2-F2 QUANTFOLD (PXA_G2_QUANTFOLD=1, default OFF): same sidecar mechanism, but the emitter is
// the deltanet out-gate fused kernel (pxa_dn_rms_silu_gate_f32) -> kills the quantize_q8_1 that
// feeds linear_attn_out. Chosen over the spec's fold-into-mmvq-prologue form because each mmvq
// block traverses the FULL x vector: a per-block redundant quantize multiplies work by gridDim,
// which the busy-wall physics (g1) says is a loss; the producer-side emit removes the same
// launch + HBM round-trip with O(1) redundant compute.
static inline bool pxa_g2_quantfold() {
    static const bool on = [](){
        const char * e = getenv("PXA_G2_QUANTFOLD");
        bool v = e && atoi(e) != 0;
        if (v) fprintf(stderr, "PXA_G2_QUANTFOLD: ON (G2-F2 producer-side q8_1 sidecar from the deltanet out-gate kernel, bit-exact)\n");
        return v;
    }();
    return on;
}

static char * pxa_g2_q8_buf(int device, cudaStream_t stream, size_t need) {
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    auto & sc = pxa_g2_q8sc[device];
    if (sc.sz >= need) return sc.buf;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &st);
    if (st != cudaStreamCaptureStatusNone) return nullptr;   // can't grow mid-capture -> decline
    if (sc.buf) cudaFree(sc.buf);
    sc.buf = nullptr; sc.sz = 0;
    if (cudaMalloc(&sc.buf, need) != cudaSuccess) { sc.buf = nullptr; cudaGetLastError(); return nullptr; }
    sc.sz = need;
    return sc.buf;
}

static const char * pxa_g2_q8_lookup(int device, const ggml_tensor * src1, int64_t ne10_padded) {
    if (!pxa_g2_normfuse()) return nullptr;
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    const auto & sc = pxa_g2_q8sc[device];
    if (!sc.t || !sc.buf || sc.eval != pxa_g2_eval_serial) return nullptr;
    const ggml_tensor * base = src1->view_src ? src1->view_src : src1;
    if (base != sc.t || src1->data != sc.data) return nullptr;
    if (ggml_nrows(src1) != 1 || !ggml_is_contiguous(src1)) return nullptr;
    if (ggml_nelements(src1) != ggml_nelements(sc.t)) return nullptr;   // flat identity (reshape-safe)
    if (ne10_padded != sc.padded) return nullptr;
    return sc.buf;
}

static int ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n, bool is_gemv) {

    auto stream = ctx.stream();

    auto fusion = ctx.fusion && src1->ne[1] == 1;

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*ggml_nrows(src1);
    if (!is_gemv) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    const char * q8v = nullptr;   // G2-F3: sidecar from a fused norm+quantize producer, if valid
    if (is_gemv) {
        q8v = pxa_g2_q8_lookup(ctx.device, src1, ne10_padded);
        if (!q8v) {
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], src1->ne[2], ne10_padded,
                    src0->type, stream);
            CUDA_CHECK(cudaGetLastError());
            q8v = src1_quantized.get();
        }

        // The code below handles the case when Q, K, V have a bias applied after the resepctive matrix multiplication.
        // In that case the graph contains mul_mat(Q) -> mul_mat(K) -> mul_mat(V) -> add(Q) -> add(K) -> add(V)
        if (fusion && cgraph && node_n + 5 < cgraph->n_nodes &&
            cgraph->nodes[node_n+1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[node_n+2]->op == GGML_OP_MUL_MAT &&
            ggml_is_quantized(cgraph->nodes[node_n+1]->src[0]->type) &&
            ggml_is_quantized(cgraph->nodes[node_n+2]->src[0]->type) &&
            cgraph->nodes[node_n+3]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+4]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+5]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+0] == cgraph->nodes[node_n+3]->src[0] &&
            cgraph->nodes[node_n+1] == cgraph->nodes[node_n+4]->src[0] &&
            cgraph->nodes[node_n+2] == cgraph->nodes[node_n+5]->src[0]) {
            for (int i = 0; i < 3; ++i) {
                auto src0_i = cgraph->nodes[node_n+i]->src[0];
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, src0_i, src1, cgraph->nodes[node_n+i], cgraph->nodes[node_n+i+3]->src[1],
                        (const char *)src0_i->data, nullptr, q8v, (float *)cgraph->nodes[node_n+i]->data,
                        0, src0_i->ne[1], src1->ne[1], ne10_padded, stream);
                CUDA_CHECK(cudaGetLastError());
            }
            node_n += 5;
        } else if (fusion && cgraph && node_n + 1 < cgraph->n_nodes &&
                   cgraph->nodes[node_n+1]->op == GGML_OP_ADD &&
                   dst == cgraph->nodes[node_n+1]->src[0] &&
                   dst->ne[0] == cgraph->nodes[node_n+1]->src[1]->ne[0] &&
                   cgraph->nodes[node_n+1]->src[1]->type == GGML_TYPE_F32 &&
                   ggml_nrows(cgraph->nodes[node_n+1]->src[1]) == 1) {
            // We have a bias applied after the matrix multiplication and we can fuse it
            ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+1], cgraph->nodes[node_n+1]->src[1],
                 (const char *)dst->src[0]->data, nullptr, q8v, (float *)cgraph->nodes[node_n+1]->data,
                 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            ++node_n;
        } else {
            ggml_cuda_op_mul_mat_vec_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, q8v, (float *)dst->data,
                    0, src0->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded, src0->type, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!cgraph) return node_n;

    while (node_n + 1 < cgraph->n_nodes) {
        dst = cgraph->nodes[node_n+1];
        if (ggml_is_empty(dst) || dst->op == GGML_OP_RESHAPE || dst->op == GGML_OP_TRANSPOSE || dst->op == GGML_OP_VIEW
                               || dst->op == GGML_OP_PERMUTE || dst->op == GGML_OP_NONE) {
            ++node_n; continue;
        }
        if (dst->op != GGML_OP_MUL_MAT || dst->src[1] != src1 || !ggml_is_quantized(dst->src[0]->type)) break;
        if (!is_gemv && mmq_get_q8_1_ds_layout(src0->type) != mmq_get_q8_1_ds_layout(dst->src[0]->type)) break;
        if (is_gemv) {
            if (fusion && node_n + 2 < cgraph->n_nodes &&
                cgraph->nodes[node_n+2]->op == GGML_OP_ADD &&
                dst == cgraph->nodes[node_n+2]->src[0] &&
                dst->ne[0] == cgraph->nodes[node_n+2]->src[1]->ne[0] &&
                cgraph->nodes[node_n+2]->src[1]->type == GGML_TYPE_F32 &&
                ggml_nrows(cgraph->nodes[node_n+2]->src[1]) == 1) {
                // We have a bias applied after the matrix multiplication and we can fuse it
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+2], cgraph->nodes[node_n+2]->src[1],
                        (const char *)dst->src[0]->data, nullptr, q8v, (float *)cgraph->nodes[node_n+2]->data,
                        0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
                ++node_n;
            } else {
                ggml_cuda_op_mul_mat_vec_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, q8v,
                        (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            }
        } else {
            ggml_cuda_op_mul_mat_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, src1_quantized.get(),
                    (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        ++node_n;
    }

    return node_n;

}

// PXA_SPEC_1ROW (2026-07-23): one block per src1 token column (Ny<=8). The ne01==1 F32
// shared-expert gate at MTP spec-verify batch sizes (Ny=2..4) misses EVERY fast dispatch path —
// dmmv/mmvq/mmq need a quantized src0, batched-cublas needs src1->ne[2]*ne[3] > 1, and the
// ne11==1-only branches below don't fire — so it fell through to a bare `cublasSgemm`
// (per-call overhead every decode token, and cuBLAS's lazy workspace alloc can fail
// mid-inference on a near-full card as CUBLAS_STATUS_INVALID_VALUE; reported on sm_86).
// Ny==1 launches grid=1 with the identical arithmetic DAG as before (bit-identical).
template <typename src_t, int block_size = 256>
static __global__ void mul_mat_row(int n, const src_t * x, const float * y, float * z,
        const size_t nb11, const size_t nb1) {
    const float * ycol = (const float *)((const char *) y + (size_t) blockIdx.x * nb11);
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += block_size) {
        float xi = ggml_cuda_cast<float, src_t>(x[i]);
        sum += xi * ycol[i];
    }
    sum = warp_reduce_sum(sum);
    if constexpr (block_size > WARP_SIZE) {
        __shared__ float tmp[block_size/WARP_SIZE];
        if (threadIdx.x % WARP_SIZE == 0) {
            tmp[threadIdx.x / WARP_SIZE] = sum;
        }
        __syncthreads();
        sum = threadIdx.x < block_size / WARP_SIZE ? tmp[threadIdx.x] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    if (threadIdx.x == 0) {
        *(float *)((char *) z + (size_t) blockIdx.x * nb1) = sum;
    }
}

static void mul_mat_1row(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_backend_cuda_context & ctx) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const int ncols = (int) src1->ne[1];
    if (src0->type == GGML_TYPE_F16) {
        mul_mat_row<<<ncols, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const half *)src0->data, (const float *)src1->data, (float *)dst->data, src1->nb[1], dst->nb[1]);
    }
    else if (src0->type == GGML_TYPE_BF16) {
        mul_mat_row<<<ncols, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const nv_bfloat16 *)src0->data, (const float *)src1->data, (float *)dst->data, src1->nb[1], dst->nb[1]);
    }
    else if (src0->type == GGML_TYPE_F32) {
        mul_mat_row<<<ncols, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const float *)src0->data, (const float *)src1->data, (float *)dst->data, src1->nb[1], dst->nb[1]);
    }
    else {
        GGML_ABORT("Fatal error");
    }
}

// PXA_ROUTER_FUSE B3 phase-1 (2026-07-22): the MoE router-logits GEMV (`ffn_gate_inp`, kept F32
// for precision — n_expert x n_embd against a single decode token) misses EVERY fast dispatch
// path above: dmmv/mmvq/mmq all require a quantized src0, and the F32 batched-cublas branch
// requires src1->ne[2]*ne[3] > 1 (a real batch dim). A plain F32 x F32, ne11==1 GEMV therefore
// falls all the way through to the generic `ggml_cuda_op_mul_mat_cublas`, whose fp16-GEMM branch
// also requires src0 to be F16/BF16/quantized — so an F32 src0 lands on a bare `cublasSgemm`
// call. Measured (PXA_PROFILE, this session): ~450-480us/call on P100/V100 for what is a
// ~1M-FLOP GEMV (256 rows x 2048 K) that should cost low single-digit us — the wall is cuBLAS's
// per-call launch/workspace overhead for a GEMM shape it is not built for (N=1), paid on EVERY
// decode token, per MoE layer. Fix: a dedicated warp-per-output-row GEMV kernel — sequential
// per-thread partial sums + one warp-shuffle reduction, same "row dot x" summation shape a naive
// reference implementation would use, so top-1/top-k expert *selection* is unaffected (logits
// differ from cuBLAS's blocked/tiled reduction only at float ULP — gated on expert-id-stream
// equality, not bit-for-bit logits, matching the spec's BX-for-fp32-math-variant call).
// Restricted to the router's shape family (F32/F32/F32, ne11==1, no batch, small-ish output row
// count) so it can never intercept a real dense/attention GEMM. Default OFF pending fair-battle.
static __global__ void k_pxa_router_gemv_f32(
        const float * __restrict__ w, const float * __restrict__ x, float * __restrict__ y,
        const int ne00, const size_t nb01) {
    const int row = blockIdx.x;
    const float * wrow = (const float *)((const char *) w + (size_t) row * nb01);
    float sum = 0.0f;
    for (int k = threadIdx.x; k < ne00; k += WARP_SIZE) {
        sum += wrow[k] * x[k];
    }
#pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (threadIdx.x == 0) {
        y[row] = sum;
    }
}

// Returns true if it fully handled the mul_mat (caller should return immediately).
static bool ggml_cuda_router_gemv_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    // Per-ARCH gate (resolver in pxa-enhance.cuh, INT8_PREFILL-pattern): ENHANCE auto-enables
    // on cc==700 ONLY (+5.1..7.0% decode measured sm_70; +1.6% KILL sm_60, Pascal stays off).
    // Env always wins: PXA_ROUTER_FUSE=0 forces OFF at any level, =1 the sm_70 ship gate,
    // =2 TEST all-arch. REFERENCE/DEFAULT: off unless env-forced.
    if (!pxa_router_fuse_on(ggml_cuda_info().devices[ctx.device].cc)) {
        return false;
    }
    // belt: never intercept a row-split weight (-sm row) — the raw data-pointer GEMV below
    // reads src0 as one dense local matrix. Production layouts are -sm layer (non-split).
    if (src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer)) {
        return false;
    }
    if (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
        return false;
    }
    if (ggml_is_transposed(src0) || ggml_is_transposed(src1)) {
        return false;
    }
    if (src0->ne[2] != 1 || src0->ne[3] != 1) {
        return false;
    }
    if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // router shape family: many small output rows (experts), guard against ever matching a real
    // dense/output-head projection (those have ne01 in the tens-of-thousands / vocab-sized range,
    // or ne01==1 which mul_mat_1row above already owns).
    if (src0->ne[1] < 2 || src0->ne[1] > 4096) {
        return false;
    }
    const int ne00 = (int) src0->ne[0];
    const int ne01 = (int) src0->ne[1];
    k_pxa_router_gemv_f32<<<ne01, WARP_SIZE, 0, ctx.stream()>>>(
        (const float *) src0->data, (const float *) src1->data, (float *) dst->data,
        ne00, src0->nb[1]);
    return true;
}

static int ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n) {

    if (ggml_cuda_router_gemv_f32(ctx, src0, src1, dst)) {
        return node_n;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;

    bool use_dequantize_mul_mat_vec = ggml_cuda_dmmv_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % (GGML_CUDA_DMMV_X*2) == 0 && src1->ne[1] == 1;
    bool          use_mul_mat_vec_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && ggml_cuda_mmvq_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
    bool              use_mul_mat_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // PXA_ADAPTIVE_DMMV (2026-06-15): per-CARD kernel selection, replacing the blunt compile-time
    // GGML_CUDA_FORCE_DMMV. Pascal (P100 sm_60 / 1080Ti sm_61) keeps the DMMV F16 fast path; Volta+
    // (V100 sm_70, tensor cores) prefers MMVQ. Decided per DEVICE by cc, so a mixed P100+V100 -sm-layer
    // rig automatically uses the best matmul kernel on EACH card in ONE binary (no -DGGML_CUDA_FORCE_DMMV).
    const int cc            = ggml_cuda_info().devices[ctx.device].cc;
    if (cc >= CC_VOLTA) {
        // mmvq (DP4A / tensor-core path) beats dmmv on Volta+; on Pascal we keep dmmv (no fast int8).
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
    }

    bool any_gpus_with_slow_fp16 = false;
    use_mul_mat_q           = use_mul_mat_q           && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
    any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_available(cc);

    // PXA_PASCAL_DMMV (2026-07-19, env-gated): the PXA_ADAPTIVE_DMMV intent (Pascal keeps the
    // DMMV F16 fast path) was dead code — this early return routed EVERY mmvq-capable quantized
    // GEMV (incl. the whole dense backbone at decode) to int8 MMVQ on all cards, and sm_60 has no
    // DP4A. With PXA_PASCAL_DMMV=1, a cc<CC_VOLTA device with a dmmv-supported type falls through
    // to the dequantize_mul_mat_vec branch below for ne11==1 GEMVs (loses the mmvq bias/TG fusion
    // for those nodes; A/B decides).
    // PXA_SPEC_SMALLN (B4 SPEC_VERIFY_ENGINE, 2026-07-22): on pre-Volta cards route the dense
    // quantized backbone at spec-verify batch sizes (ne11 2..8) to the multi-column dequant-FMA
    // GEMV instead of emulated-dp4a MMVQ. See pxa-smalln.cuh. Default OFF (PXA_SPEC_SMALLN=1).
    static const bool pxa_spec_smalln = getenv("PXA_SPEC_SMALLN") && atoi(getenv("PXA_SPEC_SMALLN")) != 0;
    if (pxa_spec_smalln && cc < CC_VOLTA && ggml_cuda_pxa_smalln_supported(src0->type)
        && src1->ne[1] >= 2 && src1->ne[1] <= 8
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[2]*src1->ne[3] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1
        && src0->ne[0] % 32 == 0 && !bad_padding_clear
        && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_pxa_smalln, nullptr);
        return node_n;
    }

    static const bool pxa_pascal_dmmv = getenv("PXA_PASCAL_DMMV") && atoi(getenv("PXA_PASCAL_DMMV")) != 0;
    const bool pxa_dmmv_take = pxa_pascal_dmmv && cc < CC_VOLTA && use_dequantize_mul_mat_vec;
    if ((use_mul_mat_vec_q || use_mul_mat_q) && src1->ne[2]*src1->ne[3] == 1 && !pxa_dmmv_take) {
        return ggml_cuda_mul_mat_q(ctx, src0, src1, dst, cgraph, node_n, use_mul_mat_vec_q);
    }

    // PXA_SPEC_1ROW (default ON, =0 rollback to the ne11==1-only dispatch): single-output-row
    // GEMV, now also at spec-verify batch sizes (Ny<=8, the MMVQ_MAX_BATCH_SIZE convention).
    // Layout/type belts mirror the router-fuse checks above; shapes that fail them fall through
    // to the generic paths exactly as before. Note this also retires the old GGML_ABORT for a
    // 1-row quantized src0 that missed mmvq/mmq — those now fall through to the cuBLAS dequant
    // path instead of aborting.
    static const bool pxa_spec_1row = !(getenv("PXA_SPEC_1ROW") && atoi(getenv("PXA_SPEC_1ROW")) == 0);
    if (ggml_nrows(src0) == 1 && (src1->ne[1] == 1 || (pxa_spec_1row && src1->ne[1] <= 8)) && src1->ne[2]*src1->ne[3] == 1
        && (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && ggml_is_contiguous(src0) && src1->nb[0] == sizeof(float)
        && !(src0->buffer && ggml_backend_buffer_is_cuda_split(src0->buffer))) {
        mul_mat_1row(src0, src1, dst, ctx);
        return node_n;
    }
    bool debug = false; //src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32;

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        if (debug) printf("%s(%s): using ggml_cuda_mul_mat_vec_p021\n", __func__, dst->name);
        // FP32 precision KQ single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        if (debug) printf("%s(%s): using ggml_cuda_mul_mat_vec_nc\n", __func__, dst->name);
        // FP32 precision KQV single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if ((src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32) && (src1->type == src0->type || !any_gpus_with_slow_fp16)
               && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        if (debug) printf("%s(%s): ggml_cuda_mul_mat_batched_cublas\n", __func__, dst->name);
        // KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_dequantize_mul_mat_vec)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_vec_q)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_q)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        if (debug) printf("%s(%s, %s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_cublas)\n", __func__, dst->name, ggml_type_name(src0->type));
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
    return node_n;
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

template <typename data_t = float>
static __global__ void k_copy_src_to_contiguous(const char * __restrict__ src_original, char * __restrict__ src_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne10, int64_t ne11, size_t nb11, size_t nb12) {
    int32_t i = blockIdx.x;

    const int32_t i11 = row_mapping[i].i1 % ne11;
    const int32_t i12 = row_mapping[i].i2;

    data_t * src_row_contiguous = (data_t *)(src_contiguous + i*nb11);
    const data_t * src_row_original = (const data_t *)(src_original + i11*nb11 + i12*nb12);

    for (int j = threadIdx.x; j < ne10; j += blockDim.x) {
        src_row_contiguous[j] = src_row_original[j];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

//static __global__ void k_quick_add(uint32_t n, uint32_t n_per_row, const float * src1, const float * src2, float * dst) {
//
//    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
//        dst[j] = src1[j] + src2[j % n_per_row];
//    }
//}

static __global__ void k_quick_add(uint32_t n_per_row, const float * src1, const float * src2, float * dst) {

    uint32_t row = blockIdx.x;
    const float * src1_row = src1 + row*n_per_row;
    float * dst_row = dst + row*n_per_row;

    for (uint32_t j = threadIdx.x; j < n_per_row; j += blockDim.x) {
        dst_row[j] = src1_row[j] + src2[j];
    }
}

static inline bool prepare_row_mappigs(ggml_backend_cuda_context& ctx, int64_t n_as, int64_t n_ids,
        const ggml_tensor * ids, std::vector<int>& moe_counts, std::vector<int>& cum_moe_counts,
        ggml_cuda_pool_alloc<mmid_row_mapping>& dev_row_mapping) {

    GGML_ASSERT(moe_counts.empty() && cum_moe_counts.empty());

    auto stream = ctx.stream();

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<mmid_row_mapping> rmapping(ids->ne[1]*n_ids);
    moe_counts.resize(n_as, 0);
    cum_moe_counts.resize(n_as + 1);

    bool is_ser = false;
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) ++moe_counts[row_id_i];
            else is_ser = true;
        }
    }
    cum_moe_counts[0] = 0;
    for (int i = 0; i < (int)n_as; ++i) {
        cum_moe_counts[i+1] = cum_moe_counts[i] + moe_counts[i];
    }

    dev_row_mapping.alloc(cum_moe_counts[n_as]);

    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) {
                rmapping[cum_moe_counts[row_id_i]++] = {(int)id, (int)iid1};
            }
        }
    }

    for (int i = 0; i < (int)n_as; ++i) cum_moe_counts[i] -= moe_counts[i];

    CUDA_CHECK(cudaMemcpyAsync(dev_row_mapping.get(), rmapping.data(),
                cum_moe_counts[n_as]*sizeof(mmid_row_mapping), cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaStreamSynchronize(stream));

    return is_ser;
}

static bool ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * next) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];


    CUDA_CHECK(cudaMemsetAsync((char *)dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    if (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_quantized(src0->type) &&
        ggml_cuda_mmvq_type_supported(src0->type) &&   // no mmvq kernel (e.g. PXQ slabs) -> per-expert dequant/cublas loop below
        ggml_backend_buffer_is_cuda(src0->buffer) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_ctx = (ggml_backend_cuda_buffer_context *) src0->buffer->context;
        ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_ctx->device == device_id &&
            src1_ctx->device == device_id &&
            dst_ctx->device  == device_id) {
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[2] = local_dst.nb[1];

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto local_src1 = *src1;
            local_src1.ne[1] = 1;
            local_src1.nb[1] = src_1_ddq_size;
            local_src1.nb[2] = src1->ne[1] > 1 ? src_1_ddq_size : 0;
            local_src1.nb[3] = local_src1.nb[2];

            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            local_src1.data = src1_quantized.alloc(src_1_ddq_size*src1->ne[1]);
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    src1_padded_col_size, src0->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_vec_q_id(ctx, src0, &local_src1, ids, &local_dst, nullptr,
                (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], 1, src1_padded_col_size, stream);
            CUDA_CHECK(cudaGetLastError());

            if (next && next->op == GGML_OP_MUL_MAT_ID && next->src[0]->type == src0->type && src1 == next->src[1] &&
                ggml_are_same_shape(src0, next->src[0]) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ggml_backend_buffer_is_cuda(next->buffer)) {
                ggml_backend_cuda_buffer_context * next_src0_ctx = (ggml_backend_cuda_buffer_context *) next->src[0]->buffer->context;
                ggml_backend_cuda_buffer_context * next_dst_ctx  = (ggml_backend_cuda_buffer_context *) next->buffer->context;
                if (next_src0_ctx->device == device_id &&
                    next_dst_ctx->device  == device_id) {
                    local_dst.data = next->data;
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, next->src[0], &local_src1, ids, &local_dst, nullptr,
                        (const char *)next->src[0]->data, nullptr, src1_quantized.get(), (float *)next->data,
                        0, src0->ne[1], 1, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                    return true;
                }
            }

            return false;
        }
    }

    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0->type) && ggml_cuda_can_use_mmq_id(src0->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
        ggml_cuda_mul_mat_q_id(ctx, src0, src1, ids, dst, nullptr, nullptr);
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;


    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }

    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
    ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

    src1_row.data = src1_contiguous.get();
    dst_row.data  =  dst_contiguous.get();

    for (int64_t i02 = 0; i02 < n_as; i02++) {

        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) {
            continue;
        }

        size_t mapping_offset = cum_moe_counts[i02];

        {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
        }

        src0_row.data = src0_original + i02*nb02;

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = nb11;
        src1_row.nb[2] = num_src1_rows*nb11;
        src1_row.nb[3] = num_src1_rows*nb11;

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1;
        dst_row.nb[2] = num_src1_rows*nb1;
        dst_row.nb[3] = num_src1_rows*nb1;

        if (ggml_is_quantized(src0->type) &&
            ggml_cuda_should_use_mmq(src0->type, ggml_cuda_info().devices[ctx.device].cc, num_src1_rows)) {
            auto src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            auto src1_quantized_size  = src1_padded_row_size*num_src1_rows;
            if (true || num_src1_rows > MMVQ_MAX_BATCH_SIZE) {
                src1_quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_mmq_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_mul_mat_q_id(ctx, &src0_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());

                CUDA_CHECK(cudaGetLastError());
            } else {
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_row_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_op_mul_mat_vec_q(ctx, &src0_row, &src1_row, &dst_row, (const char *)src0_row.data, nullptr,
                        src1_quantized.get(), (float *)dst_row.data,
                        0, src0_row.ne[1], num_src1_rows, src1_padded_num_cols, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row, nullptr, 0);
        }

        {
            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return false;
}

#include "ggml-cuda/grouped_moe_verify.cuh"
#include "ggml-cuda/pxq4.cuh"
#include "ggml-cuda/pxq6.cuh"
#include "ggml-cuda/pxq6i8.cuh"

// ============================== PXQ4 (PXA-native quant) drivers ==============================
// PXQ4 = MXFP4 numerics in a GEMM-tile-ordered slab layout (see pxq4.cuh). These drivers wire the
// fused kernels into the MoE dispatch: prefill = ONE grouped launch over all routed experts
// (replaces the per-expert dequant+cublas loop that starved the P100), decode = fused
// up+gate+swiglu / down mmv straight from f32 activations (no q8_1 stage, no host syncs).
// Both return -1 to decline (shape/type/arch not handled) -> the stock fallback paths run,
// which for PXQ4 means dequant(convert.cu)->cublas: functionally correct on every arch.

static bool pxa_pxq4_bufs_on_device(ggml_backend_cuda_context & ctx, std::initializer_list<const ggml_tensor *> ts) {
    for (const ggml_tensor * t : ts) {
        if (!t || !t->buffer || !ggml_backend_buffer_is_cuda(t->buffer)) return false;
        if (((ggml_backend_cuda_buffer_context *) t->buffer->context)->device != ctx.device) return false;
    }
    return true;
}

// decode / fast-TG: ny <= 8 tokens, one fused mmv per (token, routed expert slot).
// All PXA slab formats (PXQ4/PXQ4-HQ/PXQ2/PXQ3/PXQ6) dispatch here on the policy kernel
// family. (The legacy id-250/id-251 old-family kernels were removed 2026-07-21.) Fastpath gates:
// K1 PXA_PXQ6_KSPLIT (bit-exact) / PXA_PXQ6_KSPLIT_GEN=S (G3-gated), K2 PXA_PXQ6_PAIRLUT /
// PXA_PXQ6_VECX (bit-exact) — all format-independent via the pxq6 policy family.
static int pxa_pxq4_moe_fast_tg(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;
    const ggml_tensor * src0_1 = dst->src[0];   // up   (bias = dst->src[4])
    const ggml_tensor * src0_2 = dst->src[1];   // gate (bias = dst->src[5])
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];

    const int fmt   = pxa_pxq_fmt(src0_1->type);                       // up
    const int fmt_g = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;  // gate
    if (!src0_2 || fmt == PXA_PXQ_FMT_NONE || fmt_g == PXA_PXQ_FMT_NONE) return -1;
    if (fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2 || fmt_g >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SWIGLU_OAI && uop != GGML_UNARY_OP_SILU) return -1;
    if (src1->type != GGML_TYPE_F32) return -1;
    const int64_t R = src0_1->ne[1], K = src0_1->ne[0];
    if (R % PXQ4_BM || K % PXQ4_QK || src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;
    const size_t smem_gu = (size_t)K*sizeof(float) + 2*PXQ4_MMV_KSEG*64*sizeof(float);
    if (smem_gu > 46*1024) return -1;

    cudaStream_t stream = ctx.stream();
    const int64_t n_as  = src0_1->ne[2];
    const int64_t n_ids = ids->ne[0];
    const int64_t Ny    = src1->ne[2];
    const int   unary = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 1 : 0;
    const float limit = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 7.0f : *(const float *)(dst->op_params + 1);

    const ggml_tensor * bu = dst->src[4], * bg = dst->src[5];

    const int  pair = pxa_pxq6_decode_mode();   // sourcing MODE (tab/pairlut/prmt x cs) — name kept for call-site diff-min
    const bool vecx = pxa_pxq6_vecx();
    const int  sgen = pxa_pxq6_ksplit_gen();
    const bool newfam = fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6 || pair || vecx || sgen || pxa_pxq6_ksplit();

    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)n_ids, (unsigned)Ny);

    // G2-F1 REDFUSE eligibility (PXA_G2_REDFUSE=1, bit-exact KSPLIT form only): the down
    // MUL_MAT_ID immediately follows consuming EXACTLY this gateup dst (same ids), the gateup
    // dst has NO other consumer anywhere in the graph, and a redfuse kernel exists for the down
    // format. When it fires, the standalone k_pxq6_gateup_reduce is skipped and the down mmv
    // reconstructs its x from the KSPLIT partials (bit-identical by construction).
    bool redfuse_ok = false;
    int  rf_fmt_d   = PXA_PXQ_FMT_NONE;
    if (pxa_g2_redfuse() && !sgen && pxa_pxq6_ksplit() && graph &&
        next && next->op == GGML_OP_MUL_MAT_ID &&
        next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 &&
        pxa_pxq_fmt(next->src[0]->type) != PXA_PXQ_FMT_NONE &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next})) {
        rf_fmt_d = pxa_pxq_fmt(next->src[0]->type);
        const int64_t Kd_rf = next->src[0]->ne[0];
        const size_t smem_d_rf = (size_t)Kd_rf*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
        if (smem_d_rf <= 46*1024 && pxq6_pick_mmv_redfuse(rf_fmt_d, pair, vecx)) {
            bool sole = true;   // sole-consumer guard: nothing besides `next` may reference dst
            for (int jn = i + 2; jn < graph->n_nodes && sole; ++jn) {
                const ggml_tensor * nn = graph->nodes[jn];
                if (nn->view_src == dst) { sole = false; break; }
                for (int s = 0; s < GGML_MAX_SRC; ++s) {
                    if (nn->src[s] == dst) { sole = false; break; }
                }
            }
            redfuse_ok = sole;
        }
    }

    bool gu_done = false;
    if (newfam && (pxa_pxq6_ksplit() || sgen)) {
        // K1: split gateup launch + fixed-order reducer. Workspace grows only outside graph
        // capture; if unavailable we fall through to the unsplit kernel (bit-identical for the
        // bit-exact form, so replayed graphs stay correct whatever capture saw).
        const size_t need = (size_t)Ny*n_ids*2*(sgen ? 8 : PXQ4_MMV_KSEG)*R;
        float * ws = pxq6_ksplit_workspace(ctx.device, stream, need);
        if (ws) {
            dim3 gridr((unsigned)((R + 255)/256), (unsigned)n_ids, (unsigned)Ny);
            if (sgen) {
                const int kslabs = (int)(K/PXQ4_QK);
                const int kcmax  = ((kslabs + sgen - 1)/sgen)*PXQ4_QK;
                const size_t smem_s = (size_t)kcmax*sizeof(float) + 2*PXQ4_MMV_KSEG*64*sizeof(float);
                dim3 grids((unsigned)(R/PXQ4_BM*sgen), (unsigned)n_ids, (unsigned)Ny);
                auto * ks = pxq6_pick_gateup_ksplit_gen(fmt, fmt_g, pair, vecx);
                if (ks) {
                ks<<<grids, 256, smem_s, stream>>>(
                    (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                    (const char *)src1->data, src1->nb[2], ws,
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    (int)R, (int)K, (int)n_as, (int)n_ids, sgen);
                CUDA_CHECK(cudaGetLastError());
                k_pxq6_gateup_reduce_gen<<<gridr, 256, 0, stream>>>(ws,
                    (char *)dst->data, dst->nb[2], dst->nb[1],
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                    (int)R, (int)n_as, (int)n_ids, unary, 1.702f, limit, sgen);
                CUDA_CHECK(cudaGetLastError());
                gu_done = true;
                }
            } else {
                dim3 grids((unsigned)(R/PXQ4_BM*PXQ4_MMV_KSEG), (unsigned)n_ids, (unsigned)Ny);
                auto * ks = pxq6_pick_gateup_ksplit(fmt, fmt_g, pair, vecx);
                if (ks) {
                ks<<<grids, 64, (size_t)K*sizeof(float), stream>>>(
                    (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                    (const char *)src1->data, src1->nb[2], ws,
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    (int)R, (int)K, (int)n_as, (int)n_ids);
                CUDA_CHECK(cudaGetLastError());
                if (redfuse_ok) {
                    // G2-F1: skip the standalone reduce; the down mmv reconstructs x from ws.
                    const int64_t Rd = next->src[0]->ne[1], Kd = next->src[0]->ne[0];
                    const size_t smem_d = (size_t)Kd*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
                    dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)n_ids, (unsigned)Ny);
                    auto * kern_rf = pxq6_pick_mmv_redfuse(rf_fmt_d, pair, vecx);
                    kern_rf<<<gridd, 256, smem_d, stream>>>(
                        (const uint8_t *)next->src[0]->data, ws,
                        (char *)next->data, next->nb[2], next->nb[1],
                        (const char *)ids->data, ids->nb[0], ids->nb[1],
                        bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                        bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                        (int)Rd, (int)Kd, (int)n_as, (int)n_ids, unary, 1.702f, limit);
                    CUDA_CHECK(cudaGetLastError());
                    return i + 1;
                }
                k_pxq6_gateup_reduce<<<gridr, 256, 0, stream>>>(ws,
                    (char *)dst->data, dst->nb[2], dst->nb[1],
                    (const char *)ids->data, ids->nb[0], ids->nb[1],
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                    (int)R, (int)n_as, (int)n_ids, unary, 1.702f, limit);
                CUDA_CHECK(cudaGetLastError());
                gu_done = true;
                }
            }
        }
    }
    if (!gu_done) {
        // every remaining PXA slab format is new-family (the id-250/251 legacy kernels were
        // removed 2026-07-21); newfam is true by construction here.
        auto * kern_gu = pxq6_pick_gateup(fmt, fmt_g, pair, vecx);
        if (!kern_gu) return -1;
        kern_gu<<<grid, 256, smem_gu, stream>>>(
            (const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
            (const char *)src1->data, src1->nb[2],
            (char *)dst->data, dst->nb[2], dst->nb[1],
            (const char *)ids->data, ids->nb[0], ids->nb[1],
            bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
            bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
            (int)R, (int)K, (int)n_as, unary, 1.702f, limit);
        CUDA_CHECK(cudaGetLastError());
    }

    // fuse the PXQ4 down-proj when it is the next node (mirrors the stock fast-TG fusion;
    // a following ADD_ID runs as its own node)
    if (next && next->op == GGML_OP_MUL_MAT_ID &&
        next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 &&
        pxa_pxq_fmt(next->src[0]->type) != PXA_PXQ_FMT_NONE &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next})) {
        const int fmt_d = pxa_pxq_fmt(next->src[0]->type);
        const int64_t Rd = next->src[0]->ne[1], Kd = next->src[0]->ne[0];
        const size_t smem_d = (size_t)Kd*sizeof(float) + PXQ4_MMV_KSEG*64*sizeof(float);
        if (smem_d <= 46*1024) {
            dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)n_ids, (unsigned)Ny);
            auto * kern_d = pxq6_pick_mmv(fmt_d, pair, vecx);
            if (!kern_d) return i;   // no kernel for this format combination — skip the fusion
            kern_d<<<gridd, 256, smem_d, stream>>>(
                (const uint8_t *)next->src[0]->data,
                (const char *)dst->data, dst->nb[2], dst->nb[1],
                (char *)next->data, next->nb[2], next->nb[1],
                (const char *)ids->data, ids->nb[0], ids->nb[1],
                (int)Rd, (int)Kd, (int)n_as);
            CUDA_CHECK(cudaGetLastError());
            return i + 1;
        }
    }
    return i;
}

// prefill: grouped fused GEMMs over ALL routed experts in one launch per projection
static int pxa_pxq4_moe_prefill(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;
    const ggml_tensor * src0_1 = dst->src[0];   // up   (bias = dst->src[4])
    const ggml_tensor * src0_2 = dst->src[1];   // gate (bias = dst->src[5])
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];

    const int fmt   = pxa_pxq_fmt(src0_1->type);                       // up
    const int fmt_g = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;  // gate
    if (!src0_2 || fmt == PXA_PXQ_FMT_NONE || fmt_g == PXA_PXQ_FMT_NONE) return -1;
    if (fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6) pxq6_maybe_upload_tables(ctx.device);
    if (fmt >= PXA_PXQ_FMT_P2 || fmt_g >= PXA_PXQ_FMT_P2) pxq23_maybe_upload_books(ctx.device);
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SWIGLU_OAI && uop != GGML_UNARY_OP_SILU) return -1;
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[3] != 1) return -1;
    if (!ggml_is_contiguous(dst)) return -1;
    const int64_t R = src0_1->ne[1], K = src0_1->ne[0];
    if (R % PXQ4_BM || K % PXQ4_QK || src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!(fast_fp16_available(cc) && cc < CC_TURING) && !pxa_pxq6_force_prefill()) return -1;   // half2 kernel: sm_60 / sm_70

    cudaStream_t stream = ctx.stream();
    const int64_t n_as  = src0_1->ne[2];
    const int64_t n_ids = ids->ne[0];
    const int   unary = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 1 : 0;
    const float glu_limit = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 7.0f : *(const float *)(dst->op_params + 1);

    const bool fuse_down = next && next->op == GGML_OP_MUL_MAT_ID &&
        pxa_pxq_fmt(next->src[0]->type) != PXA_PXQ_FMT_NONE && next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 && ggml_is_contiguous(next) &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next});

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        ggml_tensor * t = fuse_down ? next : dst;
        CUDA_CHECK(cudaMemsetAsync(t->data, 0, ggml_nbytes(t), stream));
    }
    const int64_t total = cum_moe_counts[n_as];
    if (total == 0) return fuse_down ? i + 1 : i;

    // host tile map: (expert, flat row0, nrows<=64) per 64-token tile, expert-grouped order
    std::vector<pxq4_tile_info> tiles;
    tiles.reserve((size_t)(total/PXQ4_BN + n_as + 1));
    for (int e = 0; e < (int)n_as; ++e) {
        for (int t0 = 0; t0 < moe_counts[e]; t0 += PXQ4_BN) {
            tiles.push_back({e, cum_moe_counts[e] + t0, std::min((int)PXQ4_BN, moe_counts[e] - t0), 0});
        }
    }
    if (tiles.empty() || tiles.size() > 65535) return -1;   // grid.y limit; fallback handles it
    ggml_cuda_pool_alloc<pxq4_tile_info> dev_tiles(ctx.pool(), tiles.size());
    CUDA_CHECK(cudaMemcpyAsync(dev_tiles.get(), tiles.data(), tiles.size()*sizeof(pxq4_tile_info),
                               cudaMemcpyHostToDevice, stream));

    // gather activations once (f32 -> f16, same convert the cublas fp16 path does)
    ggml_cuda_pool_alloc<half> A_f16(ctx.pool(), total*K);
    k_pxq4_gather_a_f16<<<(unsigned)total, 256, 0, stream>>>((const char *)src1->data, A_f16.get(),
            (const pxq4_rowmap *)dev_row_mapping.get(), K, src1->ne[1], src1->nb[1], src1->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    // up + gate grouped GEMMs (biases folded into the epilogue).
    // Kernel-family selection (PXQ6 spec K3/K4/K5/K6): with NO PXA_PXQ6_* prefill gate set,
    // (The legacy id-250/251 old-family prefill kernels were removed 2026-07-21.)
    const ggml_tensor * bu = dst->src[4], * bg = dst->src[5];
    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)tiles.size());

    const int  wmma_env  = cc == 700 ? pxa_pxq6_wmma() : 0;   // K6: exactly Volta (sm_70)
    // v1 (modes 1/2) excludes 2/3-bit + mixed pairs; v2 (mode 3) covers P2/P3/P6/P6HQ + mixed
    // pairs but excludes P6R and requires GUFUSE (its fused kernel IS the up+gate path).
    const int  wmma_mode = wmma_env == 3
        ? ((fmt != PXA_PXQ_FMT_P6R && fmt_g != PXA_PXQ_FMT_P6R && pxa_pxq6_gufuse()) ? 3 : 0)
        : ((fmt < PXA_PXQ_FMT_P2 && fmt == fmt_g) ? wmma_env : 0);
    const bool wmma2     = wmma_mode == 3;   // K6 v2: double-buffered, GUFUSE-fused, WMMA scat down
    const bool rag    = pxa_pxq6_ragtail();
    const bool pipe   = pxa_pxq6_pipe();
    const bool newfam = fmt >= PXA_PXQ_FMT_P6 || fmt_g >= PXA_PXQ_FMT_P6 || wmma_mode || rag || pipe ||
                        pxa_pxq6_gufuse() || pxa_pxq6_scatfuse() || pxa_pxq6_force_prefill();
    if (!newfam) return -1;   // belt: every remaining PXA slab format is new-family
    const bool gufuse = newfam && pxa_pxq6_gufuse() && (!wmma_mode || wmma2) && (fmt == fmt_g || wmma2);   // K6 v1 keeps split GEMMs; v2 has its own fused (mixed-pair-capable) kernel
    const bool scat   = newfam && pxa_pxq6_scatfuse();
    // K6 launch fix (2026-07-19): k_pxq6_gemm_grouped_wmma is a 256-thread (8-warp) block
    // (__launch_bounds__(256); srow = tid>>2 stages 64 rows x 4 thr; wm/wn = 4x2 warp grid).
    // It was being launched with the plain grouped gemm's 64 threads -> 3/4 of the smem tile
    // unstaged + 6/8 warps missing = garbage output (no CUDA error: launch_bounds is a max).
    const unsigned nthr_gu = wmma_mode ? 256u : 64u;

    pxq6_gemm_fn kern_up   = (wmma_mode && !wmma2) ? pxq6_pick_gemm_wmma(fmt,   wmma_mode != 2) : pxq6_pick_gemm(fmt,   rag, pipe);
    pxq6_gemm_fn kern_gate = (wmma_mode && !wmma2) ? pxq6_pick_gemm_wmma(fmt_g, wmma_mode != 2) : pxq6_pick_gemm(fmt_g, rag, pipe);
    if (!kern_up || !kern_gate) return -1;

    if (fuse_down) {
        const int64_t Rd = next->src[0]->ne[1], Kd = next->src[0]->ne[0];   // Kd == R
        const int fmt_d = pxa_pxq_fmt(next->src[0]->type);
            ggml_cuda_pool_alloc<half> H_f16(ctx.pool(), total*R);
        if (gufuse) {
            // K3 GUFUSE: one kernel = up GEMM + gate GEMM + GLU -> f16 (bit-exact vs the
            // 3-kernel pipeline; identical accumulation chains + identical GLU/convert)
            auto * kg = wmma2 ? pxq6_pick_gufuse_wmma_h(fmt, fmt_g) : pxq6_pick_gufuse_h(fmt, rag, pipe);
            kg<<<grid, wmma2 ? 256u : 64u, 0, stream>>>((const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                    A_f16.get(), H_f16.get(),
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                    dev_tiles.get(), (int)R, (int)K, unary, 1.702f, glu_limit);
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_pool_alloc<float> C_up(ctx.pool(), total*R);
            ggml_cuda_pool_alloc<float> C_gate(ctx.pool(), total*R);
            kern_up<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_1->data, A_f16.get(), C_up.get(),
                    bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
            kern_gate<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_2->data, A_f16.get(), C_gate.get(),
                    bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
            CUDA_CHECK(cudaGetLastError());
            const int64_t k = total*R;
            k_pxq4_glu<half><<<(unsigned)((k + 255)/256), 256, 0, stream>>>(
                    C_gate.get(), C_up.get(), H_f16.get(), k, unary, 1.702f, glu_limit);
        }
        dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)tiles.size());
        if (scat) {
            // K3 SCATFUSE: down GEMM scatters straight to the MoE output rows
            pxq6_scat_fn kd = wmma2 && fmt_d != PXA_PXQ_FMT_P6R ? pxq6_pick_down_scat_wmma(fmt_d) : nullptr;
            const unsigned nthr_d = kd ? 256u : 64u;
            if (!kd) kd = pxq6_pick_down_scat(fmt_d, rag, pipe);
            kd<<<gridd, nthr_d, 0, stream>>>((const uint8_t *)next->src[0]->data, H_f16.get(),
                    (char *)next->data, next->nb[1], next->nb[2],
                    (const pxq4_rowmap *)dev_row_mapping.get(), dev_tiles.get(), (int)Rd, (int)Kd);
            CUDA_CHECK(cudaGetLastError());
        } else {
            ggml_cuda_pool_alloc<float> C_down(ctx.pool(), total*Rd);
            const bool down_wmma = wmma_mode && !wmma2 && fmt_d == fmt;
            pxq6_gemm_fn kern_down = down_wmma ? pxq6_pick_gemm_wmma(fmt_d, wmma_mode != 2) : pxq6_pick_gemm(fmt_d, rag, pipe);
            kern_down<<<gridd, down_wmma ? 256u : 64u, 0, stream>>>((const uint8_t *)next->src[0]->data, H_f16.get(),
                    C_down.get(), nullptr, 0, dev_tiles.get(), (int)Rd, (int)Kd);
            CUDA_CHECK(cudaGetLastError());
            dim3 bd(std::min((unsigned)Rd, 768u));
            k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)next->data,
                    (const char *)C_down.get(), dev_row_mapping.get(), Rd, next->nb[1], next->nb[2]);
            CUDA_CHECK(cudaGetLastError());
        }
        return i + 1;
    }

    // no down fusion: GLU to f32, scatter to dst
    ggml_cuda_pool_alloc<float> Out(ctx.pool(), total*R);
    if (gufuse) {
        auto * kg = wmma2 ? pxq6_pick_gufuse_wmma_f(fmt, fmt_g) : pxq6_pick_gufuse_f(fmt, rag, pipe);
        kg<<<grid, wmma2 ? 256u : 64u, 0, stream>>>((const uint8_t *)src0_1->data, (const uint8_t *)src0_2->data,
                A_f16.get(), Out.get(),
                bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0,
                bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0,
                dev_tiles.get(), (int)R, (int)K, unary, 1.702f, glu_limit);
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_cuda_pool_alloc<float> C_up(ctx.pool(), total*R);
        ggml_cuda_pool_alloc<float> C_gate(ctx.pool(), total*R);
        kern_up<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_1->data, A_f16.get(), C_up.get(),
                bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
        kern_gate<<<grid, nthr_gu, 0, stream>>>((const uint8_t *)src0_2->data, A_f16.get(), C_gate.get(),
                bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
        CUDA_CHECK(cudaGetLastError());
        const int64_t k = total*R;
        k_pxq4_glu<float><<<(unsigned)((k + 255)/256), 256, 0, stream>>>(
                C_gate.get(), C_up.get(), Out.get(), k, unary, 1.702f, glu_limit);
    }
    dim3 bd(std::min((unsigned)R, 768u));
    k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)dst->data,
            (const char *)Out.get(), dev_row_mapping.get(), R, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return i;
}

// N13: int8 dp4a MMQ-tile prefill (env PXA_PXQ_INT8_PREFILL, default OFF; see pxq6i8.cuh for
// the design + numeric contract). Mirrors pxa_pxq4_moe_prefill's setup so the proven driver
// stays textually untouched; every decline (-1) falls through to it. NOT bit-exact vs the
// fused fp16 pipeline (G3-gated). Decode never reaches here (fast-TG owns Ny<=8), so decode
// is byte-identical whatever the flag.
static int pxa_pxq_moe_prefill_i8(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    const int i8mode = pxa_pxq_int8_prefill();
    if (i8mode == 0) return -1;
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (i8mode == 1 && cc != 610) return -1;   // ship gate: exactly sm_61; mode 2 = any arch (TEST)

    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;
    const ggml_tensor * src0_1 = dst->src[0];   // up   (bias = dst->src[4])
    const ggml_tensor * src0_2 = dst->src[1];   // gate (bias = dst->src[5])
    const ggml_tensor * src1   = dst->src[2];
    const ggml_tensor * ids    = dst->src[3];

    const int fmt   = src0_1 ? pxa_pxq_fmt(src0_1->type) : PXA_PXQ_FMT_NONE;
    const int fmt_g = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;
    if (!src0_2 || !pxqi8_pick_gemm(fmt) || !pxqi8_pick_gemm(fmt_g)) return -1;
    pxq6_maybe_upload_tables(ctx.device);
    pxq23_maybe_upload_books(ctx.device);
    const ggml_unary_op uop = (ggml_unary_op)dst->op_params[0];
    if (uop != GGML_UNARY_OP_SWIGLU_OAI && uop != GGML_UNARY_OP_SILU) return -1;
    if (src1->type != GGML_TYPE_F32 || src1->ne[1] != 1 || src1->ne[3] != 1) return -1;
    if (!ggml_is_contiguous(dst)) return -1;
    const int64_t R = src0_1->ne[1], K = src0_1->ne[0];
    if (R % PXQ4_BM || K % PXQ4_QK || src0_2->ne[0] != K || src0_2->ne[1] != R) return -1;
    if ((size_t)K*sizeof(float) > 46*1024 || (size_t)R*sizeof(float) > 46*1024) return -1;   // stage smem
    if (!pxa_pxq4_bufs_on_device(ctx, {src0_1, src0_2, src1, dst})) return -1;

    cudaStream_t stream = ctx.stream();
    const int64_t n_as  = src0_1->ne[2];
    const int64_t n_ids = ids->ne[0];
    const int   unary = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 1 : 0;
    const float glu_limit = (uop == GGML_UNARY_OP_SWIGLU_OAI) ? 7.0f : *(const float *)(dst->op_params + 1);

    const int fmt_d = (next && next->op == GGML_OP_MUL_MAT_ID) ? pxa_pxq_fmt(next->src[0]->type) : PXA_PXQ_FMT_NONE;
    const bool fuse_down = next && next->op == GGML_OP_MUL_MAT_ID &&
        pxqi8_pick_gemm(fmt_d) != nullptr && next->src[1] == dst && next->src[2] == ids &&
        next->src[0]->ne[0] == dst->ne[0] && next->src[0]->ne[0] % PXQ4_QK == 0 &&
        next->src[0]->ne[1] % PXQ4_BM == 0 && ggml_is_contiguous(next) &&
        pxa_pxq4_bufs_on_device(ctx, {next->src[0], next});

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        ggml_tensor * t = fuse_down ? next : dst;
        CUDA_CHECK(cudaMemsetAsync(t->data, 0, ggml_nbytes(t), stream));
    }
    const int64_t total = cum_moe_counts[n_as];
    if (total == 0) return fuse_down ? i + 1 : i;

    std::vector<pxq4_tile_info> tiles;
    tiles.reserve((size_t)(total/PXQ4_BN + n_as + 1));
    for (int e = 0; e < (int)n_as; ++e) {
        for (int t0 = 0; t0 < moe_counts[e]; t0 += PXQ4_BN) {
            tiles.push_back({e, cum_moe_counts[e] + t0, std::min((int)PXQ4_BN, moe_counts[e] - t0), 0});
        }
    }
    if (tiles.empty() || tiles.size() > 65535) return -1;   // grid.y limit; fallback handles it
    if (getenv("PXA_PXQI8_DEBUG")) {
        static long long ncalls = 0;
        if (ncalls++ < 8) {
            int full = 0, ragged_rows = 0;
            for (auto & t : tiles) { if (t.nrows == PXQ4_BN) full++; else ragged_rows += (PXQ4_BN - t.nrows); }
            fprintf(stderr, "PXA_PXQI8_DEBUG: n_as=%lld total_tok=%lld tiles=%zu panels(R/64)=%lld "
                    "full_tiles=%d ragged_tiles=%zu wasted_rows=%d (112-block ceiling on 28-SM 1080Ti)\n",
                    (long long)n_as, (long long)total, tiles.size(), (long long)(R/PXQ4_BM),
                    full, tiles.size()-full, ragged_rows);
        }
    }
    ggml_cuda_pool_alloc<pxq4_tile_info> dev_tiles(ctx.pool(), tiles.size());
    CUDA_CHECK(cudaMemcpyAsync(dev_tiles.get(), tiles.data(), tiles.size()*sizeof(pxq4_tile_info),
                               cudaMemcpyHostToDevice, stream));

    // gather + q8 quantize activations once (per-32 absmax/127 — the stock-MMQ activation class)
    const int kgroups = (int)(K/PXQ4_QK);
    ggml_cuda_pool_alloc<uint8_t> A_q8(ctx.pool(), (size_t)total*K);
    ggml_cuda_pool_alloc<float>   A_d (ctx.pool(), (size_t)total*kgroups);
    k_pxqi8_gather_quant<<<(unsigned)total, 256, (size_t)K*sizeof(float), stream>>>(
            (const char *)src1->data, A_q8.get(), A_d.get(),
            (const pxq4_rowmap *)dev_row_mapping.get(), K, src1->ne[1], src1->nb[1], src1->nb[2]);
    CUDA_CHECK(cudaGetLastError());

    const ggml_tensor * bu = dst->src[4], * bg = dst->src[5];
    dim3 grid((unsigned)(R/PXQ4_BM), (unsigned)tiles.size());
    pxqi8_gemm_fn kern_up = pxqi8_pick_gemm(fmt), kern_gate = pxqi8_pick_gemm(fmt_g);

    ggml_cuda_pool_alloc<float> C_up(ctx.pool(), (size_t)total*R);
    ggml_cuda_pool_alloc<float> C_gate(ctx.pool(), (size_t)total*R);
    kern_up<<<grid, 64, 0, stream>>>((const uint8_t *)src0_1->data, A_q8.get(), A_d.get(), C_up.get(),
            bu ? (const float *)bu->data : nullptr, bu ? bu->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
    kern_gate<<<grid, 64, 0, stream>>>((const uint8_t *)src0_2->data, A_q8.get(), A_d.get(), C_gate.get(),
            bg ? (const float *)bg->data : nullptr, bg ? bg->nb[1] : 0, dev_tiles.get(), (int)R, (int)K);
    CUDA_CHECK(cudaGetLastError());

    if (fuse_down) {
        const int64_t Rd = next->src[0]->ne[1];   // Kd == R
        ggml_cuda_pool_alloc<uint8_t> H_q8(ctx.pool(), (size_t)total*R);
        ggml_cuda_pool_alloc<float>   H_d (ctx.pool(), (size_t)total*(R/PXQ4_QK));
        k_pxqi8_glu_quant<<<(unsigned)total, 256, (size_t)R*sizeof(float), stream>>>(
                C_gate.get(), C_up.get(), H_q8.get(), H_d.get(), (int)R, unary, 1.702f, glu_limit);
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_pool_alloc<float> C_down(ctx.pool(), (size_t)total*Rd);
        dim3 gridd((unsigned)(Rd/PXQ4_BM), (unsigned)tiles.size());
        pxqi8_gemm_fn kern_down = pxqi8_pick_gemm(fmt_d);
        kern_down<<<gridd, 64, 0, stream>>>((const uint8_t *)next->src[0]->data, H_q8.get(), H_d.get(),
                C_down.get(), nullptr, 0, dev_tiles.get(), (int)Rd, (int)R);
        CUDA_CHECK(cudaGetLastError());
        dim3 bd(std::min((unsigned)Rd, 768u));
        k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)next->data,
                (const char *)C_down.get(), dev_row_mapping.get(), Rd, next->nb[1], next->nb[2]);
        CUDA_CHECK(cudaGetLastError());
        return i + 1;
    }

    // no down fusion: GLU to f32, scatter to dst (existing kernels)
    ggml_cuda_pool_alloc<float> Out(ctx.pool(), (size_t)total*R);
    const int64_t kel = total*R;
    k_pxq4_glu<float><<<(unsigned)((kel + 255)/256), 256, 0, stream>>>(
            C_gate.get(), C_up.get(), Out.get(), kel, unary, 1.702f, glu_limit);
    dim3 bd(std::min((unsigned)R, 768u));
    k_copy_dst_from_contiguous<<<(unsigned)total, bd, 0, stream>>>((char *)dst->data,
            (const char *)Out.get(), dev_row_mapping.get(), R, dst->nb[1], dst->nb[2]);
    CUDA_CHECK(cudaGetLastError());
    return i;
}

#include "ggml-cuda/pxa_expert_shard.cuh"

static int ggml_cuda_moe_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;

    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src0 = src0_1;
    const ggml_tensor * src1 = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];

    // PXA-SHARD (M2): if the expert weights live in an expert-shard buffer type
    // (M3 loader places the group's *_exps there when PXA_EXPERT_SHARD is set),
    // run the disjoint-row sharded MoE op (no all-reduce) across the matched
    // device group. Flag-off / non-shard parent => the predicate is false and
    // this whole block is skipped (bit-identical to the stock build).
    // PXA-SHARD detection + ntok>1 reconstruct. Holders at FUNCTION scope so the
    // reconstructed full temps + ->data swaps outlive the fall-through stock MMQ enqueue
    // and auto-restore ->data at any function return (RAII).
    ggml_cuda_pool_alloc<char> pxa_ru, pxa_rg;
    pxa_data_swap pxa_sw_u, pxa_sw_g;
    {
        const ggml_tensor * up_root = src0_1;
        while (up_root && up_root->view_src) up_root = up_root->view_src;
        if (pxa_expert_shard_enabled() && up_root && s_pxa_shard_slices.count(up_root->name)) {
            int r = pxa_moe_shard_up_gate_down(ctx, dst, graph, i);
            if (r >= 0) return r;   // ntok==1 decode -> custom parallel kernel: DONE
            // ntok>1 (prefill/reserve): the split weight is a landmine for the stock MMQ
            // below. Reconstruct the full weight on home + repoint src0_1/src0_2->data at it.
            const int home = ctx.device;
            const ggml_tensor * gate_root = src0_2;
            while (gate_root && gate_root->view_src) gate_root = gate_root->view_src;
            cudaStream_t st = ctx.stream(ctx.device, 0);
            if (pxa_shard_reconstruct(ctx, up_root, pxa_ru, home, st) &&
                gate_root && pxa_shard_reconstruct(ctx, gate_root, pxa_rg, home, st)) {
                pxa_sw_u.set(src0_1, pxa_ru.get());
                pxa_sw_g.set(src0_2, pxa_rg.get());
            }
        }
    }

    // PXA_A1 (2026-07-01): fast-TG per-token iy-loop re-reads each routed expert once PER TOKEN of a
    // K-token MTP verify batch. Env-tunable gate: PXA_MOE_FASTTG_MAX_NY=1 routes Ny>1 verify batches to
    // the expert-grouped batched path below (weights read once per traversal). Default 8 = unchanged.
    static const int pxa_fast_tg_max_ny = getenv("PXA_MOE_FASTTG_MAX_NY") ? atoi(getenv("PXA_MOE_FASTTG_MAX_NY")) : 8;
    static const bool pxa_moe_dbg = getenv("PXA_MOE_DEBUG") != nullptr;

    // PXQ4 decode: fused up+gate+swiglu (+down) mmv, f32-in / f32-out, ids on device.
    // Declines (-1) fall through; the q8_1 fast-TG below is gated off for PXQ4 (no vec_dot).
    if (pxa_pxq_fmt(src0_1->type) != PXA_PXQ_FMT_NONE &&
        src1->ne[1] == 1 && src1->ne[2] <= pxa_fast_tg_max_ny && src1->ne[3] == 1) {
        int r = pxa_pxq4_moe_fast_tg(ctx, dst, graph, i);
        if (r >= 0) return r;
    }
    if (pxa_moe_dbg && src1->ne[2] > 1 && src1->ne[2] <= 8) { static int pxa_cnt = 0; if (pxa_cnt < 48) { ++pxa_cnt;
        fprintf(stderr, "PXA_MOE ny=%d fast=%d dev=%d\n", (int)src1->ne[2], src1->ne[2] <= pxa_fast_tg_max_ny ? 1 : 0, ctx.device); } }
    if (src1->ne[1] == 1 && src1->ne[2] <= pxa_fast_tg_max_ny && src1->ne[3] == 1 &&
        ggml_is_quantized(src0_1->type) &&
        src0_1->type != GGML_TYPE_PXQ4 && src0_1->type != GGML_TYPE_PXQ4HQ &&
        src0_1->type != GGML_TYPE_PXQ2 && src0_1->type != GGML_TYPE_PXQ3 &&      // ADD
        src0_1->type != GGML_TYPE_PXQ1 &&
        src0_1->type != GGML_TYPE_PXQ6 &&                                       // ADD (no vec_dot)
        (!src0_2 || ggml_is_quantized(src0_2->type)) &&
        ggml_backend_buffer_is_cuda(src0_1->buffer) &&
        (!src0_2 || ggml_backend_buffer_is_cuda(src0_2->buffer)) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_1_ctx = (ggml_backend_cuda_buffer_context *) src0_1->buffer->context;
        ggml_backend_cuda_buffer_context * src0_2_ctx = src0_2 ? (ggml_backend_cuda_buffer_context *) src0_2->buffer->context : nullptr;
        ggml_backend_cuda_buffer_context * src1_ctx   = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx    = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_1_ctx->device == device_id &&
            (!src0_2_ctx || src0_2_ctx->device == device_id) &&
            src1_ctx->device   == device_id &&
            dst_ctx->device    == device_id) {
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[1] = local_dst.nb[2] = local_dst.nb[3] = local_dst.ne[0]*sizeof(float);

            auto local_src1 = *src1;
            local_src1.nb[2] = local_src1.nb[3] = 0;
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;

            int Ny = src1->ne[2];

            // PXA_A1 E2 (2026-07-06): read-only expert-union saturation probe. On a Ny>1 MTP verify
            // batch, copy routed ids to host and count DISTINCT experts over running prefixes k=1..Ny vs
            // k*n_ids total selections. Decides whether a batched (read-experts-once) verify can pay.
            if (pxa_moe_dbg && Ny > 1 && Ny <= 8 && n_ids > 0 && n_ids <= 32) {
                static int    pxa_e2_calls = 0;
                static double pxa_e2_union[9] = {0};
                static long   pxa_e2_cnt[9]   = {0};
                const int nids = (int)n_ids;
                int32_t hids[32*8];
                for (int t = 0; t < Ny; ++t) {
                    cudaMemcpyAsync(hids + (size_t)t*nids,
                                    (const char *)ids->data + (size_t)t*ids->nb[1],
                                    nids*sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                }
                cudaStreamSynchronize(stream);
                bool seen[1024]; for (int z = 0; z < 1024; ++z) seen[z] = false;
                int uni = 0;
                for (int k = 1; k <= Ny; ++k) {
                    for (int j = 0; j < nids; ++j) {
                        int e = hids[(size_t)(k-1)*nids + j];
                        if (e >= 0 && e < 1024 && !seen[e]) { seen[e] = true; ++uni; }
                    }
                    pxa_e2_union[k] += uni; pxa_e2_cnt[k] += 1;
                }
                if (++pxa_e2_calls % 200 == 0) {
                    fprintf(stderr, "PXA_E2 calls=%d nids=%d dev=%d union/k:", pxa_e2_calls, nids, ctx.device);
                    for (int k = 1; k <= 8; ++k) if (pxa_e2_cnt[k]) fprintf(stderr, " k%d=%.2f/%d", k, pxa_e2_union[k]/pxa_e2_cnt[k], k*nids);
                    fprintf(stderr, "\n");
                }
            }

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            local_src1.data = src1_quantized.alloc(src_1_ddq_size * Ny);
            // Note: no use is currently made of the quantization type passed into quantize_row_q8_1_cuda.
            //       If that were to change, we would need to adjust the code to handle src0_1->type != src0_2->type
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], Ny, 1, src1_padded_col_size,
                    src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_src1.nb[1] = src_1_ddq_size;

            bool fuse_next = next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->src[0]->buffer->context)->device == device_id &&
                ggml_backend_buffer_is_cuda(next->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->buffer->context)->device == device_id;

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            float limit = *(const float *)(dst->op_params + 1);

            auto local_ids = *ids;
            local_ids.ne[1] = 1;

            // PXA A1 grouped up+gate (env PXA_MOE_BATCHED_VERIFY / legacy PXA_MOE_GROUPED,
            // bit0): read each routed expert ONCE across the Ny verify tokens. v2
            // (2026-07-07): half2 kernel, no host sync, per-device shadow-verify mode,
            // and the v1 gate/up operand-order FIX (src0_1=UP, src0_2=GATE — see
            // grouped_moe_verify.cuh header). Default OFF; one-line env rollback.
            bool pxa_grouped_done = false;
            if (src0_2 && Ny >= 2 && pxa_moe_grouped_flags() != 0 &&
                src0_1->type == GGML_TYPE_MXFP4 && src0_2->type == GGML_TYPE_MXFP4 &&
                unary_op == GGML_UNARY_OP_SILU) {
                pxa_grouped_done = pxa_moe_grouped_gateup(ctx, src0_1, src0_2, src1, ids, dst,
                                                          Ny, (int)n_ids, limit,
                                                          (const char *)src1_quantized.get(), src_1_ddq_size, stream);
            }

            pxa_shard_time_begin(device_id, stream);  // meter UNSHARDED up+gate
            for (int iy = 0; iy < Ny && !pxa_grouped_done; ++iy) {
                local_src1.data = src1_quantized.get() + iy*src_1_ddq_size;
                local_ids.data  = (char *)ids->data + iy*ids->nb[1];
                local_dst.data  = (char *)dst->data + iy*dst->nb[2];
                if (src0_2) {
                    ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, &local_src1, &local_ids, &local_dst,
                            dst->src[4], dst->src[5],
                            (const char *)src0_1->data, (const char *)src0_2->data,
                            (const float *)src1->data, (const char *)local_src1.data,
                            (float *)local_dst.data, 0, src0_1->ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                } else {
                    auto local_src0_1 = *src0_1;
                    local_src0_1.ne[1] /= 2;
                    auto local_src0_2 = local_src0_1;
                    local_src0_2.data = (char *)local_src0_1.data + local_src0_1.ne[1]*local_src0_1.nb[1];
                    if (!dst->src[4]) {
                        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, &local_ids, &local_dst,
                                nullptr, nullptr,
                                (const char *)local_src0_2.data, (const char *)local_src0_1.data,
                                (const float *)src1->data, (const char *)local_src1.data,
                                (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                    } else {
                        GGML_ASSERT(!dst->src[5]);
                        auto local_bias_1 = *dst->src[4];
                        local_bias_1.ne[0] /= 2;
                        auto local_bias_2 = local_bias_1;
                        local_bias_2.data = (char *)local_bias_1.data + local_bias_1.ne[0]*local_bias_1.nb[0];
                        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, &local_ids, &local_dst,
                                &local_bias_2, &local_bias_1,
                                (const char *)local_src0_2.data, (const char *)local_src0_1.data,
                                (const float *)src1->data, (const char *)local_src1.data,
                                (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                    }
                }
                CUDA_CHECK(cudaGetLastError());
            }

            pxa_shard_time_end(device_id, stream, "UNSHARDED");  // meter UNSHARDED up+gate
            pxa_moe_grouped_shadow_check(ctx, dst, stream);  // no-op unless PXA_MOE_GROUPED_VERIFY

            if (!fuse_next) return i;

            const int64_t dst_padded_col_size = GGML_PAD(dst->ne[0], MATRIX_ROW_PADDING);
            GGML_ASSERT(dst->ne[0] % QK8_1 == 0);
            auto dst_row_size = dst_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto dst_ddq_size = n_ids*dst_row_size;
            ggml_cuda_pool_alloc<char> dst_quantized(ctx.pool(), dst_ddq_size*Ny);
            quantize_row_q8_1_cuda((const float *)dst->data, (void *)dst_quantized.get(), dst->ne[0], n_ids*Ny, 1,
                    dst_padded_col_size, next->src[0]->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_dst.ne[2] = 1;

            auto local_next = *next;
            local_next.ne[2] = local_next.ne[1];
            local_next.ne[1] = local_next.ne[3] = 1;
            local_next.nb[2] = local_next.nb[1];

            local_src1 = *next->src[1];
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;
            local_src1.nb[1] = local_src1.nb[2] = local_src1.nb[3] = dst_row_size;

            auto local_src0 = *next->src[0];
            local_src0.ne[2] = local_src0.ne[3] = 1;

            int result = i + 1;

            //printf("next: %ld x %ld x %ld x %ld,    %zu x %zu x %zu x %zu\n", next->ne[0], next->ne[1], next->ne[2], next->ne[3], next->nb[0], next->nb[1], next->nb[2], next->nb[3]);

            for (int iy = 0; iy < Ny; ++iy) {
                local_ids.data  = (char *)ids->data + iy*ids->nb[1];
                auto this_dst_quantized = dst_quantized.get() + iy*dst_ddq_size;
                if (i+2 < graph->n_nodes &&
                    graph->nodes[i+2]->op == GGML_OP_ADD_ID &&
                    graph->nodes[i+2]->src[0] == next &&
                    graph->nodes[i+2]->src[2] == ids) {
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, &local_ids, &local_next, graph->nodes[i+2]->src[1],
                            (const char *)next->src[0]->data, nullptr, this_dst_quantized, (float *)graph->nodes[i+2]->data + iy*next->ne[0]*n_ids,
                            0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                    if (iy == 0) {
                        ++result;
                    }
                } else {
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, &local_ids, &local_next, nullptr,
                            (const char *)next->src[0]->data, nullptr, this_dst_quantized, (float *)next->data + iy*next->ne[0]*n_ids,
                            0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            return result;
        }
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor dst_row = *dst;
    // PXA_P100_FP16_GEMM_v1: on a MOE_FUSED_UP_GATE node op_params[0] is the UNARY OP id (silu/
    // swiglu_oai), NOT a ggml_prec — but the per-expert ggml_cuda_mul_mat fallback below hands this
    // local copy to ggml_cuda_op_mul_mat_cublas, which reads op_params[0] as precision and spuriously
    // vetoed the fp16 GemmEx path for every up/gate GEMM (down-proj, a real MUL_MAT_ID node with
    // op_params[0]=0, got fp16 — measured 2/3 of expert GEMMs stuck on fp32 SGEMM). The unary op is
    // read from the ORIGINAL dst above; clearing it on the local copy is safe for every consumer here.
    dst_row.op_params[0] = GGML_PREC_DEFAULT;

    // PXQ4 prefill: ONE grouped fused GEMM launch per projection over ALL routed experts
    // (replaces the per-expert dequant+cublas loop below — the PoC-measured 2.53x P100 path).
    if (pxa_pxq_fmt(src0_1->type) != PXA_PXQ_FMT_NONE) {
        int r = pxa_pxq_moe_prefill_i8(ctx, dst, graph, i);   // N13 (env-gated, default OFF)
        if (r < 0) r = pxa_pxq4_moe_prefill(ctx, dst, graph, i);
        if (r >= 0) return r;
    }

    // The heuristics src1->ne[2] <= 32*src0->ne[2] to use the mul_mat_id implementation instead of the original version
    // is derived from
    //    * DeepSeek-Lite:  64 total, 6 active experts
    //    * GPT-OSS-20B  :  32 total, 4 active experts
    //    * Qwen3-30B-A3B: 128 total, 8 active experts
    // My original hypothesis was that it is dependent on the total/active experts ratio, but from these 3 it
    // looks like it really depends just on the total number of experts.
    // TODO: verify with more models, or perhaps make the magic constant '32' to be defined via a compile time define.
    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1 &&
        ggml_cuda_can_use_mmq_id(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {

        const int64_t ne_get_rows = ne12 * n_ids;
        ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool(), ne_get_rows + ne_get_rows + n_as + 1);
        auto ids_src1 = ids_device.get();
        auto ids_dst  = ids_src1 + ne_get_rows;
        auto expert_bounds = ids_dst + ne_get_rows;

        compute_row_ids((const int32_t *)ids->data, ids_src1, ids_dst, expert_bounds,
                ne02, ne12, n_ids, ne11, nb11, nb12, ids->nb[1], stream);

        const int64_t ne11_flat = ne12*n_ids;
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        size_t nbytes_src1_q8_1 = ne11_flat*ne10_padded * sizeof(block_q8_1)/QK8_1 +
                get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), nbytes_src1_q8_1);

        size_t ts_src1 = ggml_type_size(src1->type);
        quantize_mmq_q8_1_cuda_id((const float *)src1->data, ids_src1, src1_quantized.get(),
                src0_1->type, ne10, src1->nb[1] / ts_src1, src1->nb[2] / ts_src1, src1->nb[2] / ts_src1,
                ne10_padded, ne11_flat, 1, 1, stream);

        if (src0_2) {
        ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
        ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        dst_row.data = dst_up_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[4]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        dst_row.data = dst_gate_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_2, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[5]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[5]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
            ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst->data, ggml_nelements(dst), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
        } else {
            float limit = *((const float *)(dst->op_params + 1));
            //printf("%s: using limit = %g\n", __func__, limit);
            ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                    (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                    (float *)dst->data, limit);
        }
        } else {

            ggml_cuda_pool_alloc<char> dst_up_gate_contiguous(ctx.pool(), 2*sizeof(float)*ggml_nelements(dst));
            ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
            dst_row.ne[0] *= 2;
            dst_row.nb[1] *= 2;
            dst_row.nb[2] *= 2;
            dst_row.nb[3] *= 2;
            dst_row.data = dst_up_gate_contiguous.get();
            ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
            if (dst->src[4]) {
                GGML_ASSERT(!dst->src[5]);
                ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                        (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                        dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
                CUDA_CHECK(cudaGetLastError());
            }

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_gate_contiguous.get(), (const float *)dst_up_gate_contiguous.get() + dst->ne[0],
                        (float *)dst->data, ggml_nelements(dst), dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                float limit = *((const float *)(dst->op_params + 1));
                //printf("%s: using limit = %g\n", __func__, limit);
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst), dst->ne[0],
                        (const float *)dst_up_gate_contiguous.get(), (float *)dst->data, limit);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        if (next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
            ggml_cuda_should_use_mmq(next->src[0]->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            //ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, (char *)ids_device.get(), nullptr);
            ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, nullptr, nullptr);
            return i+1;
        }

        return i;
    }

    ggml_tensor src0_1_row = *src0_1;
    ggml_tensor src0_2_row; if (src0_2) src0_2_row = *src0_2;
    ggml_tensor src1_row   = *src1;
    ggml_tensor final_dst;
    ggml_tensor final_src;

    char * src0_1_original = (char *) src0_1->data;
    char * src0_2_original = src0_2 ? (char *) src0_2->data : nullptr;
    char * src1_original   = (char *) src1->data;
    char * dst_original    = (char *)  dst->data;

    src0_1_row.ne[2] = 1;
    src0_1_row.ne[3] = 1;
    src0_1_row.nb[3] = nb02;
    if (src0_2) {
        src0_2_row.ne[2] = 1;
        src0_2_row.ne[3] = 1;
        src0_2_row.nb[3] = nb02;
    }

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    bool fuse_down = false;
    if (next && next->op == GGML_OP_MUL_MAT_ID) {
        fuse_down = true;
        final_dst = *next;
        final_dst.ne[1] = final_dst.ne[2] = final_dst.ne[3] = 1;
        final_dst.nb[2] = final_dst.nb[3] = final_dst.nb[1];
        final_src = *next->src[0];
        final_src.ne[2] = final_src.ne[3] = 1;
        final_src.nb[3] = final_src.nb[2];
    }

    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
    bool use_quantized_src1 = false;
    int64_t src1_padded_num_cols = 0, src1_padded_row_size = 0, src1_quantized_size = 0;
    if (ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            src1_quantized_size  = src1_padded_row_size*src1->ne[2] + get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
            src1_quantized.alloc(src1_quantized_size);
            use_quantized_src1 = true;
        }
    }
    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool());
    if (!use_quantized_src1) {
        src1_contiguous.alloc(sizeof(float)*ggml_nelements(src1));
    }
    ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool()), dst_gate_contiguous(ctx.pool());
    if (src0_2) {
        dst_up_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    } else {
        dst_up_contiguous.alloc(2*sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    }
    ggml_cuda_pool_alloc<char> final_dst_contiguous(ctx.pool());
    if (fuse_down) {
        final_dst.data = final_dst_contiguous.alloc(sizeof(float)*ggml_nelements(next));
        final_dst.src[1] = &dst_row;
    }

    src1_row.data = src1_contiguous.get();

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;

    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        if (fuse_down) {
            CUDA_CHECK(cudaMemsetAsync(next->data, 0, ggml_nbytes(next), stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        }
    }

    for (int64_t i02 = 0; i02 < n_as; i02++) {
        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) continue;
        size_t mapping_offset = cum_moe_counts[i02];

        if (use_quantized_src1) {
            quantize_mmq_q8_1_id_cuda((const float *)src1->data, src1_quantized.get(), (const char *)(dev_row_mapping.get() + mapping_offset),
                    src1->ne[0], num_src1_rows, src1_padded_num_cols, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_quantized.get();
        }
        else {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_contiguous.get();
        }

        src0_1_row.data = src0_1_original + i02*nb02;
        if (src0_2_original) src0_2_row.data = src0_2_original + i02*src0_2->nb[2];   // gate stride, NOT up nb02 (mixed PXQ pairs differ)

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        auto nb1l = nb1;
        if (!src0_2) {
            nb1l = nb1*2;
            dst_row.ne[0] = dst->ne[0] * 2;
        }

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = use_quantized_src1 ? src1_padded_row_size : nb11;
        src1_row.nb[2] = num_src1_rows*src1_row.nb[1];
        src1_row.nb[3] = num_src1_rows*src1_row.nb[1];

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1l;
        dst_row.nb[2] = num_src1_rows*nb1l;
        dst_row.nb[3] = num_src1_rows*nb1l;

        dst_row.data  =  dst_up_contiguous.get();
        if (use_quantized_src1) {
            ggml_cuda_mul_mat_q_id(ctx, &src0_1_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
        } else {
            ggml_cuda_mul_mat(ctx, &src0_1_row, &src1_row, &dst_row, nullptr, 0);
        }
        CUDA_CHECK(cudaGetLastError());

        if (dst->src[4]) {
            GGML_ASSERT(dst_row.ne[0] == dst->src[4]->ne[0]);
            dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
            dim3 grid_dims(num_src1_rows);
            k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                    (const float *)((const char *)dst->src[4]->data + i02*dst->src[4]->nb[1]), (float *)dst_row.data);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        float limit = *(const float *)(dst->op_params + 1);
        //printf("%s: using limit = %g\n", __func__, limit);
        if (src0_2) {
            dst_row.data  = dst_gate_contiguous.get();
            if (use_quantized_src1) {
                ggml_cuda_mul_mat_q_id(ctx, &src0_2_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
            } else {
                ggml_cuda_mul_mat(ctx, &src0_2_row, &src1_row, &dst_row, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            if (dst->src[5]) {
                dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
                dim3 grid_dims(num_src1_rows);
                k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                        (const float *)((const char *)dst->src[5]->data + i02*dst->src[5]->nb[1]), (float *)dst_row.data);
                CUDA_CHECK(cudaGetLastError());
            }
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                        (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), limit);
            }
        } else {
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_contiguous.get(), (const float *)dst_up_contiguous.get() + dst->ne[0],
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row)/2, dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row)/2, dst->ne[0],
                        (const float *)dst_up_contiguous.get(), (float *)dst_gate_contiguous.get(), limit);
            }
            dst_row.data = dst_gate_contiguous.get();
            dst_row.ne[0] /= 2;
            dst_row.nb[1] /= 2;
            dst_row.nb[2] /= 2;
            dst_row.nb[3] /= 2;
        }
        CUDA_CHECK(cudaGetLastError());

        if (fuse_down) {

            final_dst.ne[1] = num_src1_rows;
            final_dst.nb[1] = final_dst.ne[0]*sizeof(float);
            final_dst.nb[2] = final_dst.nb[3] = num_src1_rows*final_dst.nb[1];
            final_src.data = (char *)next->src[0]->data + i02*next->src[0]->nb[2];
            if (ggml_is_quantized(next->src[0]->type) &&
                ggml_cuda_should_use_mmq(final_src.type, ggml_cuda_info().devices[ctx.device].cc, dst_row.ne[1])) {
                ggml_cuda_mul_mat_q_id(ctx, &final_src, &dst_row, nullptr, &final_dst, nullptr, nullptr);
            } else {
                ggml_cuda_mul_mat(ctx, &final_src, &dst_row, &final_dst, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            dim3 block_dims(std::min((unsigned int)next->ne[0], 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    (char *)next->data, final_dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    next->ne[0],
                    next->nb[1], next->nb[2]);
            CUDA_CHECK(cudaGetLastError());

        }
        else {

            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_gate_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return fuse_down ? i+1 : i;
}

static inline bool pxa_is_pxq_type(ggml_type t) {
    return t == GGML_TYPE_PXQ4 ||
           t == GGML_TYPE_PXQ4HQ || t == GGML_TYPE_PXQ2 || t == GGML_TYPE_PXQ3 ||
           t == GGML_TYPE_PXQ1 ||
           t == GGML_TYPE_PXQ6;
}

static void ggml_cuda_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src1 = dst->src[2];

    // Crash guard: dense FUSED_UP_GATE with PXQ-slab-typed up/gate tensors (a file that puts a
    // PXQ type on a DENSE tensor, e.g. via external surgery -- llama-quantize itself demotes
    // those). The stock q8_1 mmvq/mmq calls below have NO PXQ kernels and fault; divert to the
    // generic dequant->cublas pair + fused GLU, correct at any ne11. No file produced by this
    // repo's quantizer reaches this branch.
    if (pxa_is_pxq_type(src0_1->type) || pxa_is_pxq_type(src0_2->type)) {
        const float limit_px = *(const float *)(dst->op_params + 1);
        ggml_cuda_pool_alloc<float> dst_up_px(ctx.pool(), ggml_nelements(dst));
        auto local_dst = *dst;
        local_dst.data = dst_up_px.get();
        ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
        ggml_cuda_mul_mat(ctx, src0_2, src1, dst, nullptr, 0);
        ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst),
                        (const float *)dst->data, dst_up_px.get(), (float *)dst->data, limit_px);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    GGML_ASSERT(ggml_is_quantized(src0_1->type));
    GGML_ASSERT(src0_1->type == src0_2->type);
    GGML_ASSERT(src1->ne[2] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    auto stream = ctx.stream();

    float limit = *(const float *)(dst->op_params + 1);

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*src1->ne[1];
    if (src1->ne[1] > 8) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<float> dst_up(ctx.pool(), ggml_nelements(dst));
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    if (src1->ne[1] <= 8) {
        // G2-F3: reuse a fused-norm q8_1 sidecar of src1 if one is valid (skips the quantize)
        const char * q8u = pxa_g2_q8_lookup(ctx.device, src1, ne10_padded);
        if (!q8u) {
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded,
                    src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());
            q8u = src1_quantized.get();
        }

        if (src1->ne[1] == 1 && src0_1->type == src0_2->type) {
            ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, src1, nullptr, dst,
                    dst->src[4], dst->src[5],
                    (const char *)src0_1->data, (const char *)src0_2->data, (const float *)src1->data, q8u,
                    (float *)dst->data, 0, src0_1->ne[1], 1, ne10_padded,
                    (ggml_unary_op)dst->op_params[0], limit, stream);
            return;
        }

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, q8u, dst_up.get(),
                0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, q8u, (float *)dst->data,
                0, src0_2->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {

        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[1])) {
            quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    ne10_padded, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, src1_quantized.get(), dst_up.get(),
                    0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, src1_quantized.get(), (float *)dst->data,
                    0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto local_dst = *dst;
            local_dst.data = dst_up.get();
            ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
            ggml_cuda_mul_mat(ctx, src0_2, src1, dst, nullptr, 0);
        }
    }

    //printf("%s: using limit = %g\n", __func__, limit);
    ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst),
                    (const float *)dst->data, dst_up.get(), (float *)dst->data, limit);
    CUDA_CHECK(cudaGetLastError());

}

static inline bool ops_are_same_device(const ggml_cgraph * cgraph, int first, int last) {
    if (last <= first) return true;
    int device = ((const ggml_backend_cuda_buffer_context *)cgraph->nodes[first]->buffer->context)->device;
    for (int i = first; i <= last; ++i) {
        auto node = cgraph->nodes[i];
        if (((const ggml_backend_cuda_buffer_context *)node->buffer->context)->device != device) return false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || !node->src[j]->buffer) continue;
            if (((const ggml_backend_cuda_buffer_context *)node->src[j]->buffer->context)->device != device) return false;
        }
    }
    return true;
}

// G2-F3: does the FUSED_RMS_NORM output `dst` feed at least one q8_1-quantized GEMV consumer
// (a quantized mmvq MUL_MAT, or a FUSED_UP_GATE, on this device) within the lookahead window?
// If so, report the consumer's ne10_padded so the sidecar is laid out to match.
static bool pxa_g2_normfuse_wanted(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i,
        const ggml_tensor * dst, int64_t & padded_out) {
    if (!ggml_is_contiguous(dst)) return false;
    if (ggml_nelements(dst) > 65536) return false;               // decode-shape only
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 20; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        const ggml_tensor * s1 = nullptr;
        if (n->op == GGML_OP_MUL_MAT)            s1 = n->src[1];
        else if (n->op == GGML_OP_FUSED_UP_GATE) s1 = n->src[2];
        else continue;
        if (!s1) continue;
        const ggml_tensor * base = s1->view_src ? s1->view_src : s1;
        if (base != dst || s1->data != dst->data) continue;
        if (ggml_nrows(s1) != 1 || !ggml_is_contiguous(s1)) continue;
        if (ggml_nelements(s1) != ggml_nelements(dst)) continue;
        const ggml_tensor * w = n->src[0];
        if (!w || !ggml_is_quantized(w->type) || !ggml_cuda_mmvq_type_supported(w->type)) continue;
        if (!n->buffer || !ggml_backend_buffer_is_cuda(n->buffer)) continue;
        if (((ggml_backend_cuda_buffer_context *)n->buffer->context)->device != ctx.device) continue;
        padded_out = GGML_PAD(s1->ne[0], MATRIX_ROW_PADDING);
        if (ggml_nrows(dst) > 1 && padded_out != ggml_nelements(dst)) continue;  // flat emit needs pad-free layout
        return true;
    }
    return false;
}

// G2-F4: does tensor t have any consumer in nodes (i+1..end) besides `except`?
static bool pxa_g2_sole_consumer(const ggml_cgraph * cgraph, int i, const ggml_tensor * t, const ggml_tensor * except) {
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == except) continue;
        if (n->view_src == t) return false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == t) return false;
        }
    }
    return true;
}


#include "ggml-cuda/pxa-deltanet-fuse.cuh"

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst, const ggml_cgraph * cgraph, int & i) {

#if IK_PRINT_TIMING
    int64_t tim1 = ggml_time_us();
#endif

    if (ggml_is_noop(dst)) {
        return true;
    }

    // In case we forget to do that in some kernel.
    ggml_cuda_set_device(ctx.device);

    auto next = i < cgraph->n_nodes - 1 ? cgraph->nodes[i+1] : nullptr;

    auto fusion = ctx.fusion;

    //printf("%4d %s(%s) on device %d. time = %ld\n", i, ggml_op_name(dst->op), dst->name, ctx.device, ggml_time_us());
    switch (dst->op) {
        case GGML_OP_REDUCE:
            ggml_cuda_op_reduce(ctx, dst);
            break;
        case GGML_OP_FAKE_CPY:
            break;
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_HADAMARD:
            ggml_cuda_op_hadamard(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_cpy_2(ctx, dst->src[0], cgraph->nodes[i+2]->src[0], dst->src[1], cgraph->nodes[i+2]->src[1])) {
                i += 2;
            } else {
                ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            }
            break;
        case GGML_OP_CONT:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i+2]->op == GGML_OP_TRANSPOSE &&
                dst->src[0]->op == GGML_OP_TRANSPOSE) {
                ggml_cuda_op_sum_rows_nc(ctx, cgraph->nodes[i+1]);
                i += 2;
            } else {
                ggml_cuda_dup(ctx, dst);
            }
            break;
        case GGML_OP_ADD:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ADD &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                dst->src[0]->type == GGML_TYPE_F32 &&               // with split mode "attn" we can end up having f16
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                dst == cgraph->nodes[i+1]->src[0] &&
                ggml_is_contiguous(cgraph->nodes[i+1]->src[1]) &&
                ggml_are_same_shape(dst, cgraph->nodes[i+1]->src[1]) &&
                cgraph->nodes[i+1] == cgraph->nodes[i+2]->src[0] &&
                ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_fused_add_add_rms_norm(ctx, dst, cgraph->nodes[i+1], cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_SOFTPLUS &&
                cgraph->nodes[i+2]->op == GGML_OP_MUL &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+1]->src[0] == cgraph->nodes[i] &&
                ggml_nrows(cgraph->nodes[i+0]->src[1]) == 1 &&
                ggml_nrows(cgraph->nodes[i+2]->src[1]) == 1) {
                ggml_cuda_fused_softplus(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            // G2-F4 ADDFUSE (2026-07-19): re-enable the upstream ADD+FUSED_RMS_NORM pair fusion,
            // env-gated. ne0 >= 256 keeps the fused kernel's block-size selection (256/1024)
            // identical to the standalone fused_rms_norm launcher's, so the reduction order and
            // therefore the output are bit-identical to k_add_same + fused_rms_norm_f32.
            else if (pxa_g2_addfuse() && fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_FUSED_RMS_NORM &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                dst->src[0]->type == GGML_TYPE_F32 &&
                dst->src[1]->type == GGML_TYPE_F32 &&
                dst->ne[0] >= 256 &&
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                cgraph->nodes[i+1]->src[1]->type == GGML_TYPE_F32 &&
                ggml_nrows(cgraph->nodes[i+1]->src[1]) == 1 &&
                dst == cgraph->nodes[i+1]->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_fused_add_rms_norm(ctx, dst, cgraph->nodes[i+1]);
                ++i;
            } else {
                ggml_cuda_op_add(ctx, dst);
            }
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_MULTI_ADD:
            ggml_cuda_op_multi_add(ctx, dst);
            break;
        case GGML_OP_MUL_MULTI_ADD:
            // G2-F4 ADDFUSE: fold the following residual ADD into the mul_multi_add epilogue
            // (experts summed first, residual added last => bit-identical in either operand
            // order). Requires the multi-add output to have NO consumer besides that ADD.
            if (pxa_g2_addfuse() && fusion && next && next->op == GGML_OP_ADD &&
                (next->src[0] == dst || next->src[1] == dst) &&
                next->type == GGML_TYPE_F32 &&
                (next->src[0] == dst ? next->src[1] : next->src[0])->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(next->src[0], next->src[1]) &&
                ggml_are_same_shape(next, dst) &&
                ggml_is_contiguous(next->src[0]) && ggml_is_contiguous(next->src[1]) &&
                ops_are_same_device(cgraph, i, i+1) &&
                pxa_g2_sole_consumer(cgraph, i, dst, next)) {
                ggml_cuda_op_mul_multi_add_fused(ctx, dst, next);
                ++i;
            } else {
                ggml_cuda_op_mul_multi_add(ctx, dst);
            }
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                cgraph->nodes[i+2]->op == GGML_OP_MUL &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_EXP &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[1] == dst->src[1]) {
                ggml_cuda_fused_mul_exp_mul(ctx, cgraph->nodes[i+2]);
                i += 2;
                //printf("mul(%s) -> exp(%s) -> mul(%s), %d, %d, %zu, %zu; %ld x %ld x %ld x %ld - %ld x %ld x %ld x %ld\n", dst->name, cgraph->nodes[i+1]->name, cgraph->nodes[i+2]->name,
                //        ggml_is_contiguous(dst->src[0]), ggml_is_contiguous(dst->src[1]), ggml_nelements(dst->src[0]), ggml_nelements(dst->src[1]),
                //        dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3],
                //        dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3]);
            } else {
                //printf("mul(%s): %d, %d, %d, %ld x %ld x %ld x %ld * %ld x %ld x %ld x %ld\n", dst->name, ggml_is_contiguous(dst->src[0]), ggml_is_contiguous(dst->src[1]), ggml_is_contiguous(dst),
                //        dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3],
                //        dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3]);
                ggml_cuda_op_mul(ctx, dst);
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY:
            ggml_cuda_op_fused_mul_unary(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_UNARY:
            //printf("unary(%s, %s)\n", dst->name, ggml_unary_op_name((ggml_unary_op)dst->op_params[0]));
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    if (fusion) {
                        // PXA_FUSE_DELTANET bit0: the DeltaNet decode qk-norm/state-writeback cluster
                        int pxa_n = pxa_try_deltanet_cluster(ctx, cgraph, i);
                        if (pxa_n > 0) { i += pxa_n; break; }
                    }
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (fusion && i + 5 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                        cgraph->nodes[i+4]->op == GGML_OP_VIEW &&
                        cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+5)) {
                        cuda_glm45moe_experts(ctx, cgraph->nodes[i+5], cgraph->nodes[i+4]);
                        i += 5;
                    }
                    else if (fusion && i + 4 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_GROUPED_TOPK &&
                        cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+4)) {
                        cuda_bailingmoev2_experts(ctx, cgraph->nodes[i+4], cgraph->nodes[i+3]);
                        i += 4;
                    } else if (fusion && i + 2 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD && ops_are_same_device(cgraph, i, i+2)) {
                        ggml_cuda_op_biased_sigmoid(ctx, cgraph->nodes[i+2]);
                        i += 2;
                    } else {
                        ggml_cuda_op_sigmoid(ctx, dst);
                    }
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return -1;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_concat_cpy(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            } else {
                ggml_cuda_op_concat(ctx, dst);
            }
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_FUSED_RMS_NORM:
            if (fusion && pxa_try_deltanet_outgate(ctx, cgraph, i)) {
                // PXA_FUSE_DELTANET bit1: DeltaNet out-gate rms+silu fusion
                i += 1;
            }
            else if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+3], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+2]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+3]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+1], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                dst->ne[2] == 1 && cgraph->nodes[i+2]->ne[2] == 1) {
                ggml_cuda_op_fused_rms_rms_norm(ctx, dst, cgraph->nodes[i+2]);
                i += 2;
            } else {
                // G2-F3 NORMFUSE: emit the q8_1 sidecar alongside the norm when a quantized
                // GEMV consumer is coming (f32 dst stays bit-identical; miss -> plain path).
                bool g2_done = false;
                int64_t g2_padded = 0;
                if (pxa_g2_normfuse() && fusion && dst->src[1] && ggml_nrows(dst) == 1 &&
                    pxa_g2_normfuse_wanted(ctx, cgraph, i, dst, g2_padded)) {
                    const size_t need = (size_t)(g2_padded/QK8_1)*sizeof(block_q8_1);
                    char * q8 = pxa_g2_q8_buf(ctx.device, ctx.stream(), need);
                    if (q8 && ggml_cuda_op_fused_rms_norm_q8(ctx, dst, q8, g2_padded)) {
                        auto & sc = pxa_g2_q8sc[ctx.device];
                        sc.t = dst; sc.data = dst->data; sc.padded = g2_padded; sc.eval = pxa_g2_eval_serial;
                        g2_done = true;
                    }
                }
                if (!g2_done) {
                    ggml_cuda_op_fused_rms_norm(ctx, dst);
                }
            }
            break;
        case GGML_OP_FUSED_RMS_RMS_ADD:
            ggml_cuda_op_fused_rms_rms_add(ctx, dst);
            break;
        case GGML_OP_FUSED_NORM:
            ggml_cuda_op_fused_rms_norm(ctx, dst, true);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                GGML_CUDA_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return -1;
            } else {
                i = ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst, cgraph, i);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (ggml_cuda_mul_mat_id(ctx, dst, next)) ++i;
            break;
        case GGML_OP_MOE_FUSED_UP_GATE:
            i = ggml_cuda_moe_up_gate_unary(ctx, dst, cgraph, i);
            break;
        case GGML_OP_FUSED_UP_GATE:
            ggml_cuda_up_gate_unary(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SOFTCAP:
            ggml_cuda_op_softcap(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            if (fusion && i + 8 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                cgraph->nodes[i+4]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+6]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+7]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i+8]->op == GGML_OP_DIV) {
                ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+8], cgraph->nodes[i+4], cgraph->nodes[i+2]->src[1]);
                i += 8;
            }
            else if (fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ARGSORT  &&
                cgraph->nodes[i+3]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS &&
                ggml_cuda_should_use_topk_moe(cgraph->nodes[i], cgraph->nodes[i+4]) &&
                ops_are_same_device(cgraph, i, i+4)) {
                if (i + 7 < cgraph->n_nodes &&
                    cgraph->nodes[i+5]->op == GGML_OP_RESHAPE  &&
                    cgraph->nodes[i+6]->op == GGML_OP_SUM_ROWS &&
                    cgraph->nodes[i+7]->op == GGML_OP_DIV) {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+7], cgraph->nodes[i+3]);
                    i += 7;
                } else {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+4], cgraph->nodes[i+3]);
                    i += 4;
                }
            } else {
                ggml_cuda_op_soft_max(ctx, dst);
            }
            break;
        case GGML_OP_SOFT_CAP_MAX:
            ggml_cuda_op_soft_cap_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            } else {
                ggml_cuda_op_rope(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROPE_FAST:
            if (fusion && i + 3 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
               (cgraph->nodes[i+2]->op == GGML_OP_RESHAPE || cgraph->nodes[i+2]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+3])) {
                i += 3;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST   &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            }
            else {
                ggml_cuda_op_rope_fast(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_CACHE:
            ggml_cuda_op_rope_cache(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SCALE &&
                cgraph->nodes[i+2]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[1] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[1] == dst &&
                cgraph->nodes[i+1]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+1]);
                ++i;
            } else {
                ggml_cuda_op_sum_rows(ctx, dst);
            }
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            if (fusion && i + 5 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+3]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+4]->op == GGML_OP_SOFT_MAX &&
                cgraph->nodes[i+5]->op == GGML_OP_RESHAPE && ops_are_same_device(cgraph, i, i+4)) {
                cuda_openai_experts(ctx, dst, cgraph->nodes[i+4]);
                i += 5;
            } else {
                ggml_cuda_op_argsort(ctx, dst);
            }
            break;
        case GGML_OP_ARGSORT_THRESH:
            ggml_cuda_op_argsort_thresh(ctx, dst);
            break;
        case GGML_OP_GROUPED_TOPK:
            ggml_cuda_op_grouped_topk(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_DELTA_NET:
            ggml_cuda_op_delta_net(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        default:
            return false;
    }

#if 0
    if (auto err = cudaStreamSynchronize(ctx.stream()); err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

#if IK_PRINT_TIMING
    if (auto err = cudaStreamSynchronize(ctx.stream()); err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }
    int64_t tim2 = ggml_time_us();
    printf("%s(%s): %d us\n", ggml_op_name(dst->op), dst->name, (int)(tim2 - tim1));
#endif

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

GGML_CALL static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cuda_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return ggml_backend_cuda_buffer_type(cuda_ctx->device);
}

GGML_CALL static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_CUDA_LOG_WARN("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        ggml_cuda_pool_alloc<half> tmp_src(cuda_ctx_src->pool());
        ggml_cuda_pool_alloc<half> tmp_dst(cuda_ctx_dst->pool());
        bool needs_f16_f32_copy = false;
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (false && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && dst->ne[1] >= 32) {
                //
                // The goal here is to reduce traffic between GPU's, which is entirely non-negligible
                // for prompt processing.
                // We cast the tensor to be copied to f16, copy the f16 data peer-to-peer
                // and then cast back to f32 on the destination side.
                // The cost for converting to/from f16 is much lower than the cost of copying
                // two times more data over PCI-E (well, at least the 30 GB/s PCI-E I have).
                // But for some reason the following is slower.
                // Can somebody tell me why?
                //

                ggml_cuda_set_device(cuda_ctx_dst->device);
                tmp_dst.alloc(ggml_nelements(dst));

                ggml_cuda_set_device(cuda_ctx_src->device);
                tmp_src.alloc(ggml_nelements(src));

                auto src_f16 = *src;
                src_f16.type = GGML_TYPE_F16;
                for (int i = 0; i < 4; ++i) src_f16.nb[i] /= 2;
                src_f16.data = tmp_src.get();

                ggml_cuda_cpy(*cuda_ctx_src, src, &src_f16, true);

                CUDA_CHECK(cudaMemcpyPeerAsync(tmp_dst.ptr, cuda_ctx_dst->device, src_f16.data, cuda_ctx_src->device, ggml_nbytes(&src_f16), cuda_ctx_src->stream()));

                needs_f16_f32_copy = true;

            } else {
#ifdef GGML_USE_NCCL__
                auto & info = ggml_cuda_info();
                auto nbytes = ggml_nbytes(src);
                ncclGroupStart();
                ggml_cuda_set_device(cuda_ctx_src->device);
                auto status1 = ncclSend(src->data, nbytes, ncclUint8, cuda_ctx_dst->device, info.nccl_coms[cuda_ctx_src->device],
                        info.all_ctx[cuda_ctx_src->device]->stream());
                ggml_cuda_set_device(cuda_ctx_dst->device);
                auto status2 = ncclRecv(dst->data, nbytes, ncclUint8, cuda_ctx_src->device, info.nccl_coms[cuda_ctx_dst->device],
                        info.all_ctx[cuda_ctx_dst->device]->stream());
                ncclGroupEnd();
                GGML_ASSERT(status1 == ncclSuccess && status2 == ncclSuccess);
                return true;
#else
                ggml_cuda_set_device(cuda_ctx_src->device);
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif
            }
#endif
        }

        // record event on src stream after the copy
        ggml_cuda_set_device(cuda_ctx_src->device);
        if (!cuda_ctx_src->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        ggml_cuda_set_device(cuda_ctx_dst->device);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
        if (needs_f16_f32_copy) {
            auto dst_f16 = *dst;
            dst_f16.type = GGML_TYPE_F16;
            for (int i = 0; i < 4; ++i) dst_f16.nb[i] /= 2;
            dst_f16.data = tmp_dst.get();
            ggml_cuda_cpy(*cuda_ctx_dst, &dst_f16, dst, true);
        }
    } else {
        // src and dst are on the same backend
        // printf("Why is this being invoked?\n");
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

GGML_CALL static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH

// PXA_CUDA_GRAPH_BATCH (S2, 2026-07-07): opt-in CUDA-graph capture for small multi-token
// decode batches (spec-decode verify shapes, np>1 coalesced multi-slot decode). Default OFF; the
// n_tokens==1 behavior is byte-identical when the env is unset. See FORK-OPTIMIZATION-SURVEY F1/F2.
static bool pxa_cuda_graph_batch_enabled() {
    static const bool on = [] {
        const char * e = getenv("PXA_CUDA_GRAPH_BATCH");
        if (!e || atoi(e) == 0) return false;
        // The E2 union probe (PXA_MOE_DEBUG) does a cudaMemcpy D2H + stream sync INSIDE the MoE op,
        // and the A1 shadow-verify (PXA_MOE_GROUPED_VERIFY) relies on device-printf flushing at
        // stream syncs -- both are incompatible with (or defeated by) graph capture/replay.
        if (getenv("PXA_MOE_DEBUG") || getenv("PXA_MOE_GROUPED_VERIFY")) return false;
        return true;
    }();
    return on;
}
static int pxa_cuda_graph_batch_max_ny() {
    static const int v = [] {
        int m = 8; // the fast-TG dispatch ceiling in ggml_cuda_moe_up_gate_unary
        if (const char * e = getenv("PXA_MOE_FASTTG_MAX_NY"))       { int t = atoi(e); if (t < m) m = t; }
        if (const char * e = getenv("PXA_CUDA_GRAPH_BATCH_MAX_NY")) { int t = atoi(e); if (t >= 1 && t < m) m = t; }
        return m;
    }();
    return v;
}

static bool pxa_cuda_graph_log_enabled() {
    static const bool on = getenv("PXA_CUDA_GRAPH_LOG") != nullptr;
    return on;
}
static int pxa_cuda_graph_log_level() {
    static const int v = [] { const char * e = getenv("PXA_CUDA_GRAPH_LOG"); return e ? atoi(e) : 0; }();
    return v;
}

// PXA_CUDA_GRAPH_V2 (G1, 2026-07-19): fixed graph-cache semantics, opt-in, default OFF.
// =0/unset is byte-identical to the pre-G1 behavior. When on:
//   1. the shape-FNV cache key is ALWAYS used (not only under PXA_CUDA_GRAPH_BATCH) and folds the
//      (op, ne1, ne2) of EVERY node -- each shape class (prefill chunk, decode variant, MTP
//      warmup/verify, np2 ny=2) owns a stable slot instead of all colliding on nodes[0];
//   2. disable_due_to_too_many_updates is replaced by a cooldown (PXA_CUDA_GRAPH_REARM evals,
//      default 256; 0 = keep the permanent disable) so early churn can't kill replay forever;
//   3. the capturability check runs BEFORE the property store, so an uncapturable graph (prefill
//      MoE ny>8) no longer clobbers the stored decode properties at a shared/collided key;
//   4. the per-context graph cache is LRU-capped (PXA_CUDA_GRAPH_LRU slots, default 8).
static bool pxa_cuda_graph_v2_enabled() {
    // NOTE (2026-07-21): whole-token graph replay is measured NEGATIVE on BOTH box arches with
    // captures verified firing (PXA_CUDA_GRAPH_LOG): V100 −2..−4%, P100 −3.9% (65.0→62.5 t/s,
    // public PXQ2, replays=396/400 tokens, byte-identical). An earlier "+3.5% P100" reading was
    // NOISE from a config where the cc<CC_AMPERE arch gate silently kept captures at 0 — always
    // verify captures>0 before believing a graph number. Decode is GPU-busy; replay bookkeeping
    // is pure tax on these cards. Stays env-only opt-in for instrumentation/diagnostics.
    static const bool on = [] { const char * e = getenv("PXA_CUDA_GRAPH_V2"); return e && atoi(e) != 0; }();
    return on;
}
static int pxa_cuda_graph_rearm_evals() {
    static const int v = [] { const char * e = getenv("PXA_CUDA_GRAPH_REARM"); return e ? atoi(e) : 256; }();
    return v;
}
static size_t pxa_cuda_graph_lru_cap() {
    static const size_t v = [] { const char * e = getenv("PXA_CUDA_GRAPH_LRU");
        long t = e ? atol(e) : 8; return (size_t)(t < 2 ? 2 : t); }();
    return v;
}

// P0 instrumentation: honest global counters + atexit summary (active only under PXA_CUDA_GRAPH_LOG).
struct pxa_cgraph_stats_t {
    std::atomic<long> captures{0};
    std::atomic<long> replays{0};
    std::atomic<long> eager_disabled{0};   // one of the disable flags / arch gate
    std::atomic<long> eager_cooldown{0};   // v2 cooldown active
    std::atomic<long> eager_uncapturable{0}; // compat check said no
    std::atomic<long> disables_arch{0};
    std::atomic<long> disables_too_many{0};
    std::atomic<long> disables_capture_fail{0};
    std::atomic<long> cooldowns_armed{0};
    std::atomic<long> lru_evictions{0};
};
static pxa_cgraph_stats_t & pxa_cgraph_stats() {
    static pxa_cgraph_stats_t s;
    return s;
}
static void pxa_cgraph_atexit_summary() {
    auto & s = pxa_cgraph_stats();
    fprintf(stderr, "PXA_CGRAPH SUMMARY captures=%ld replays=%ld eager{disabled=%ld cooldown=%ld uncapturable=%ld} "
            "disables{arch=%ld too_many=%ld capture_fail=%ld} cooldowns_armed=%ld lru_evictions=%ld\n",
            s.captures.load(), s.replays.load(), s.eager_disabled.load(), s.eager_cooldown.load(),
            s.eager_uncapturable.load(), s.disables_arch.load(), s.disables_too_many.load(),
            s.disables_capture_fail.load(), s.cooldowns_armed.load(), s.lru_evictions.load());
    fflush(stderr);
}
static void pxa_cgraph_stats_arm_atexit() {
    static std::once_flag once;
    std::call_once(once, [] { atexit(pxa_cgraph_atexit_summary); });
}

static inline const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    if (!pxa_cuda_graph_batch_enabled() && !pxa_cuda_graph_v2_enabled()) {
        return cgraph->nodes[0];
    }
    // PXA_CUDA_GRAPH_BATCH shape-keyed cache: with multi-token capture on, DIFFERENT batch shapes
    // alternate step-to-step (verify depth K varies with draft acceptance; server slots join/leave).
    // The host cgraph arena is rebuilt at the same base address after every scheduler reset, so
    // keying on nodes[0] alone maps ALL shapes onto ONE entry -> re-capture on every shape flip ->
    // "too many consecutive updates" permanently disables graphs for the shared key (losing even the
    // banked single-token win). Fold a cheap shape/topology signature into the key so each distinct
    // shape owns a stable entry and replays its own captured graph.
    // Correctness does NOT depend on this key: is_cuda_graph_update_required() compares every node's
    // op/ne/nb/addresses before any replay, so a key collision can only cost a re-capture -- a stale
    // graph can never be replayed at the wrong shape.
    uint64_t h = 0xcbf29ce484222325ULL; // FNV-1a
    auto fold = [&h](uint64_t x) { h ^= x; h *= 0x100000001b3ULL; };
    fold((uint64_t)(uintptr_t)cgraph->nodes[0]);
    fold((uint64_t)cgraph->n_nodes);
    const int n = cgraph->n_nodes;
    // V2: fold (op, ne1, ne2) over ALL nodes (~us-scale for 2500 nodes) so distinct topologies that
    // share a 16-node prefix (e.g. the 2498 vs 2468 decode variants) get distinct keys. Legacy
    // BATCH-only mode keeps the 16-node prefix fold (byte-identical to the shipped behavior).
    const int fold_n = pxa_cuda_graph_v2_enabled() ? n : (n < 16 ? n : 16);
    for (int i = 0; i < fold_n; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        fold((uint64_t)t->op); fold((uint64_t)t->ne[1]); fold((uint64_t)t->ne[2]);
    }
    if (n > 0) {
        const ggml_tensor * t = cgraph->nodes[n-1];
        fold((uint64_t)t->op); fold((uint64_t)t->ne[0]); fold((uint64_t)t->ne[1]); fold((uint64_t)t->ne[2]);
    }
    return (const void *)(uintptr_t)h;
}

static inline ggml_cuda_graph * ggml_cuda_get_graph(ggml_backend_cuda_context & ctx, const void * key) {
    auto & graph = ctx.cuda_graphs[key];
    if (!graph) {
        graph = std::make_unique<ggml_cuda_graph>();
        // PXA_CUDA_GRAPH_V2 P1.4: bound the keyed cache. Each captured 2,500-node exec holds device
        // memory; evict the least-recently-used entry (never the one we just created) beyond the cap.
        if (pxa_cuda_graph_v2_enabled() && ctx.cuda_graphs.size() > pxa_cuda_graph_lru_cap()) {
            const void * lru_key = nullptr;
            uint64_t lru_use = UINT64_MAX;
            for (auto & kv : ctx.cuda_graphs) {
                if (kv.first == key || !kv.second) continue;
                if (kv.second->last_use < lru_use) { lru_use = kv.second->last_use; lru_key = kv.first; }
            }
            if (lru_key) {
                if (pxa_cuda_graph_log_enabled()) {
                    fprintf(stderr, "PXA_CGRAPH lru-evict dev=%d key=%p (cache=%zu cap=%zu)\n",
                            ctx.device, lru_key, ctx.cuda_graphs.size(), pxa_cuda_graph_lru_cap());
                }
                pxa_cgraph_stats().lru_evictions++;
                ctx.cuda_graphs.erase(lru_key);
            }
        }
    }
    auto * g = ctx.cuda_graphs[key].get();
    g->last_use = ++ctx.cuda_graph_eval_no;
    return g;
}

static bool check_node_graph_compatibility_and_refresh_copy_ops(ggml_backend_cuda_context * cuda_ctx,
    ggml_cuda_graph * graph, ggml_cgraph * cgraph, bool use_cuda_graph, cudaStream_t stream) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph
    graph->cpy_dest_ptrs.clear();

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_noop(node)) continue;

        if (node->op == GGML_OP_REDUCE) {
            // PXA_REDUCE_CAPTURE: allow capturing the cross-device ring/direct reduce once
            // every participating device's copy_event AND reduce_kickoff_event already exist
            // (the first reduce of a session runs eager and creates them; capture forbids alloc).
            static const bool _pxa_rc = getenv("PXA_REDUCE_CAPTURE") != nullptr;
            if (!_pxa_rc) { use_cuda_graph = false; break; }
            auto & _ri = ggml_cuda_info();
            int _rn = node->op_params[1];
            bool _ready = true;
            for (int _rj = 0; _rj < _rn && _ready; ++_rj) {
                if (node->src[_rj]) {
                    auto _c = _ri.all_ctx[_rj];
                    if (!_c || !_c->copy_event || !_c->reduce_kickoff_event) _ready = false;
                }
            }
            if (!_ready) { use_cuda_graph = false; break; }
            continue; // reduce is capturable this token
        }

        if (node->op == GGML_OP_MUL_MAT_ID) {
            // PXA_CUDA_GRAPH_MOE: the original guard disabled capture whenever n_expert_used>1
            // (src[2]->ne[0]!=1) -> fired on EVERY MoE token (qwen35moe routes 8 experts) so graph
            // capture NEVER engaged. But the expert ids are read ON-DEVICE by the Fast-TG kernel
            // (capturable); only the host-grouping path (large batch / non-quant) does a real
            // cudaStreamSynchronize and is truly uncapturable. Relax (opt-in PXA_CUDA_GRAPH_MOE) to
            // the predicate the Fast-TG dispatch itself uses, so MoE decode can be captured.
            const bool pxa_uncapturable = node->ne[2] > 8 || node->src[1]->ne[2] != 1 ||
                                          !ggml_is_quantized(node->src[0]->type);
            const bool mmid_disable = getenv("PXA_CUDA_GRAPH_MOE")
                ? pxa_uncapturable
                : (node->ne[2] != 1 || node->src[2]->ne[0] != 1);
            if (mmid_disable) {
                use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
                GGML_CUDA_LOG_DEBUG("%s(%s): disabling CUDA graphs due to unsupported node type %ld %ld\n",
                        __func__, node->src[0]->name, node->ne[2], node->src[2]->ne[0]);
#endif
            }
        }
        if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
            auto src0_1 = node->src[0];
            auto src0_2 = node->src[1];
            auto src1   = node->src[2];
            // PXA_CUDA_GRAPH_BATCH (S2): the original gate restricted capture to single-token MoE
            // (src1->ne[2]==1), so every multi-token decode step (MTP/ngram verify batches, np>1
            // coalesced multi-slot decode) ran permanently eager and forfeited the banked graph win
            // (survey F1). The fast-TG iy-loop and the A1 grouped up+gate path are both
            // host-sync-free for ne[2] in 2..8 (pool allocs are legal under relaxed capture and the
            // VMM pool is pointer-stable), so those shapes are capturable. The allowed Ny range
            // follows the execution dispatch (PXA_MOE_FASTTG_MAX_NY), and the same-device buffer
            // predicate is mirrored from ggml_cuda_moe_up_gate_unary, so we never capture a shape
            // that would fall through to the host-syncing general/mmq-id path.
            bool moe_capturable =
                src1->ne[1] == 1 && src1->ne[3] == 1 && src1->type == GGML_TYPE_F32 &&
                ggml_is_quantized(src0_1->type) && (!src0_2 || ggml_is_quantized(src0_2->type));
            if (moe_capturable && src1->ne[2] != 1) {
                moe_capturable = pxa_cuda_graph_batch_enabled() &&
                    src1->ne[2] >= 2 && src1->ne[2] <= pxa_cuda_graph_batch_max_ny() &&
                    src0_1->buffer && src1->buffer && node->buffer && (!src0_2 || src0_2->buffer) &&
                    ggml_backend_buffer_is_cuda(src0_1->buffer) &&
                    (!src0_2 || ggml_backend_buffer_is_cuda(src0_2->buffer)) &&
                    ggml_backend_buffer_is_cuda(src1->buffer) &&
                    ggml_backend_buffer_is_cuda(node->buffer) &&
                    ((ggml_backend_cuda_buffer_context *) src0_1->buffer->context)->device == cuda_ctx->device &&
                    (!src0_2 || ((ggml_backend_cuda_buffer_context *) src0_2->buffer->context)->device == cuda_ctx->device) &&
                    ((ggml_backend_cuda_buffer_context *) src1->buffer->context)->device == cuda_ctx->device &&
                    ((ggml_backend_cuda_buffer_context *) node->buffer->context)->device == cuda_ctx->device;
            }
            // PXQ4: only the fused decode path is device-only/capturable. If the type is PXQ4
            // but that path would decline (env off, wrong unary, unaligned dims, smem), decode
            // falls to the host-syncing per-expert loop -> must NOT capture.
            if (moe_capturable && (src0_1->type == GGML_TYPE_PXQ4 || src0_1->type == GGML_TYPE_PXQ4HQ ||
                                   src0_1->type == GGML_TYPE_PXQ2 || src0_1->type == GGML_TYPE_PXQ3 ||
                                   src0_1->type == GGML_TYPE_PXQ1 ||
                                   src0_1->type == GGML_TYPE_PXQ6)) {
                const ggml_unary_op puop = (ggml_unary_op)node->op_params[0];
                const int cfu = pxa_pxq_fmt(src0_1->type);
                const int cfg = src0_2 ? pxa_pxq_fmt(src0_2->type) : PXA_PXQ_FMT_NONE;
                const bool pair_ok = cfu != PXA_PXQ_FMT_NONE && cfg != PXA_PXQ_FMT_NONE &&
                    (cfu == cfg || ((cfu == PXA_PXQ_FMT_P2 || cfu == PXA_PXQ_FMT_P3 || cfu == PXA_PXQ_FMT_P6) &&
                                    (cfg == PXA_PXQ_FMT_P2 || cfg == PXA_PXQ_FMT_P3 || cfg == PXA_PXQ_FMT_P6)));
                moe_capturable = pair_ok &&
                    src0_2 &&
                    (puop == GGML_UNARY_OP_SWIGLU_OAI || puop == GGML_UNARY_OP_SILU) &&
                    src0_1->ne[1] % PXQ4_BM == 0 && src0_1->ne[0] % PXQ4_QK == 0 &&
                    (size_t)src0_1->ne[0]*sizeof(float) + 2*PXQ4_MMV_KSEG*64*sizeof(float) <= 46*1024;
            }
            if (!moe_capturable) {
                use_cuda_graph = false;
            } else {
                if (i < cgraph->n_nodes-1) {
                    auto next = cgraph->nodes[i+1];
                    if (next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type)) {
                        ++i;
                    }
                }
            }
        }

        // Why was this needed? Leaving it in place but disabled in case it is actually needed.
        if (false && node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&
            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
            strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(), ffn_moe_gate_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_up_bias_prefix.c_str(), ffn_moe_up_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_down_bias_prefix.c_str(), ffn_moe_down_bias_prefix.size()) != 0) {
            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
            // by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (node->op == GGML_OP_CPY) {

            // Store the pointers which are updated for each token, such that these can be sent
            // to the device and accessed using indirection from CUDA graph
            graph->cpy_dest_ptrs.push_back((char *) node->src[1]->data);

            // store a pointer to each copy op CUDA kernel to identify it later
            void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
            if (!ptr) {
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
            }
        }
        if (!use_cuda_graph) {
            break;
        }
    }

    if (use_cuda_graph) {
        graph->use_cpy_indirection = true;
        // copy pointers to GPU so they can be accessed via indirection within CUDA graph
        ggml_cuda_cpy_dest_ptrs_copy(graph, graph->cpy_dest_ptrs.data(), graph->cpy_dest_ptrs.size(), stream);
    }

    return use_cuda_graph;
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool ggml_graph_node_has_matching_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    if (node->data != graph_node_properties->node_address &&
          node->op != GGML_OP_CPY &&
          node->op != GGML_OP_VIEW) {
        return false;
    }

    if (node->op != graph_node_properties->node_op) {
        return false;
    }

    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->ne[i] != graph_node_properties->ne[i]) {
            return false;
        }
        if (node->nb[i] != graph_node_properties->nb[i]) {
            return false;
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (node->src[i] &&
            node->src[i]->data != graph_node_properties->src_address[i] &&
            node->op != GGML_OP_VIEW &&
            // PXQ port of upstream ik_llama.cpp PR #2136 (merged 2026-07-17): only a CPY node's
            // DESTINATION (src[1], the KV-cache view whose address legitimately moves between
            // captures) is exempt from the address check. The READ source (src[0]) must force a
            // re-capture when it moves, or the captured graph keeps reading the old address
            // (stale-read bug, e.g. SWA mask/KV re-pointing).
            !(node->op == GGML_OP_CPY && i == 1)
        ) {
            return false;
        }
    }

    if (node->op == GGML_OP_SCALE &&
        memcmp(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
        return false;
    }

    return true;
}

// P0 instrumentation: name WHICH field of WHICH node forced the update (first mismatch only).
static void pxa_cgraph_log_first_mismatch(const void * key, int i, ggml_tensor * node, ggml_graph_node_properties * p) {
    const char * field = "?";
    char detail[192] = {0};
    if (node->op != p->node_op) {
        field = "op";
        snprintf(detail, sizeof(detail), "%s->%s", ggml_op_name(p->node_op), ggml_op_name(node->op));
    } else if (node->data != p->node_address && node->op != GGML_OP_CPY && node->op != GGML_OP_VIEW) {
        field = "data";
        snprintf(detail, sizeof(detail), "%p->%p", p->node_address, node->data);
    } else {
        for (int d = 0; d < GGML_MAX_DIMS; d++) {
            if (node->ne[d] != p->ne[d]) {
                field = "ne";
                snprintf(detail, sizeof(detail), "dim%d %lld->%lld", d, (long long)p->ne[d], (long long)node->ne[d]);
                break;
            }
            if (node->nb[d] != p->nb[d]) {
                field = "nb";
                snprintf(detail, sizeof(detail), "dim%d %zu->%zu", d, p->nb[d], node->nb[d]);
                break;
            }
        }
        if (detail[0] == 0) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (node->src[s] && node->src[s]->data != p->src_address[s] &&
                    node->op != GGML_OP_VIEW &&
                    !(node->op == GGML_OP_CPY && s == 1)) { // PR #2136: keep logger in lockstep with the matcher
                    field = "src";
                    snprintf(detail, sizeof(detail), "src%d(%s) %p->%p", s, node->src[s]->name,
                             p->src_address[s], node->src[s]->data);
                    break;
                }
            }
        }
        if (detail[0] == 0 && node->op == GGML_OP_SCALE &&
            memcmp(p->op_params, node->op_params, GGML_MAX_OP_PARAMS) != 0) {
            field = "op_params";
        }
    }
    fprintf(stderr, "PXA_CGRAPH mismatch key=%p node[%d] name=%s op=%s field=%s %s\n",
            key, i, node->name, ggml_op_name(node->op), field, detail);
}

static bool is_cuda_graph_update_required(ggml_cuda_graph * graph, ggml_cgraph * cgraph, const void * key = nullptr) {

    bool cuda_graph_update_required = false;
    const bool log_on = pxa_cuda_graph_log_enabled();

    if (graph->instance == nullptr) {
        cuda_graph_update_required = true;
        if (log_on) fprintf(stderr, "PXA_CGRAPH update-required key=%p reason=no-instance\n", key);
    }

    // Check if the graph size has changed
    if (graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        if (log_on && !cuda_graph_update_required) {
            fprintf(stderr, "PXA_CGRAPH update-required key=%p reason=n_nodes %zu->%d\n",
                    key, graph->ggml_graph_properties.size(), cgraph->n_nodes);
        }
        cuda_graph_update_required = true;
        graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        bool has_matching_properties = true;
        if (!cuda_graph_update_required) {
            has_matching_properties = ggml_graph_node_has_matching_properties(cgraph->nodes[i], &graph->ggml_graph_properties[i]);
        }
        if (!has_matching_properties) {
            if (log_on) pxa_cgraph_log_first_mismatch(key, i, cgraph->nodes[i], &graph->ggml_graph_properties[i]);
            cuda_graph_update_required = true;
        }
        set_ggml_graph_node_properties(cgraph->nodes[i], &graph->ggml_graph_properties[i]);
    }

    return cuda_graph_update_required;
}

static void update_cuda_graph_executable(ggml_cuda_graph * graph) {

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    // PXA_CUDA_GRAPH_V2: treat ANYTHING but clean success as "cannot update in place" and
    // destroy+reinstantiate (CUDA 12.x sm70 partial-update edge cases); legacy behavior asserts
    // on unexpected errors.
    if (stat == cudaErrorGraphExecUpdateFailure || (pxa_cuda_graph_v2_enabled() && stat != cudaSuccess)) {
#ifndef NDEBUG
        GGML_CUDA_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif
        if (pxa_cuda_graph_log_enabled()) {
            fprintf(stderr, "PXA_CGRAPH exec-update-failed (%s) -> reinstantiate\n", cudaGetErrorString(stat));
        }

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif

static void evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {
    // flag used to determine whether it is an integrated_gpu
    // TODO
    [[maybe_unused]] const bool integrated = false; //ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ++pxa_g2_eval_serial;   // G2-F3: q8 sidecars are only valid within one graph eval

#ifdef USE_CUDA_GRAPH
    auto graph = use_cuda_graph ? ggml_cuda_get_graph(*cuda_ctx, ggml_cuda_graph_get_key(cgraph)) : nullptr;
#endif

#if IK_PRINT_TIMING
    printf("======================== %s: graph with %d nodes on device %d. time = %ld\n", __func__, cgraph->n_nodes, cuda_ctx->device, ggml_time_us());
#endif
    // PXA_WAVE4_DIAG (PXA_OP_SYNC_CHECK=1): report a sticky CUDA error set BEFORE this graph eval
    // (means the fault came from an external/API path between graphs, not a graph op).
    if (g_pxa_op_check_on < 0) g_pxa_op_check_on = getenv("PXA_OP_SYNC_CHECK") ? 1 : 0;
    if (g_pxa_op_check_on > 0) {
        cudaError_t _perr = cudaGetLastError();
        if (_perr != cudaSuccess) {
            fprintf(stderr, "PXA_WAVE4_DIAG ENTRY-DIRTY dev=%d err=%s (fault set before graph eval)\n",
                    cuda_ctx->device, cudaGetErrorString(_perr));
            fflush(stderr);
            GGML_ABORT("PXA_WAVE4_DIAG entry-dirty");
        }
    }
    // PXA_NODE_PROF per-graph wall summary
    int64_t _pxa_gt0 = 0;
    if (g_pxa_prof_on > 0) { cudaStreamSynchronize(cuda_ctx->stream()); _pxa_gt0 = ggml_time_us(); }
    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_noop(node)) continue;

                if (g_pxa_prof_on < 0) { g_pxa_prof_on = getenv("PXA_PROFILE") ? 1 : 0; if (getenv("PXA_PROFILE_EVERY")) g_pxa_prof_every = atol(getenv("PXA_PROFILE_EVERY")); if (g_pxa_prof_every < 1) g_pxa_prof_every = 20000; }
                int64_t _pxt0 = 0;
                if (g_pxa_prof_on > 0) { cudaStreamSynchronize(cuda_ctx->stream()); _pxt0 = ggml_time_us(); }
                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node, cgraph, i);
                if (!ok) {
                    GGML_CUDA_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                if (g_pxa_prof_on > 0) { cudaStreamSynchronize(cuda_ctx->stream()); double _dus=(double)(ggml_time_us()-_pxt0); int _o=(int)node->op; if(_o>=0&&_o<128){ g_pxa_op_us[_o]+=_dus; g_pxa_op_cnt[_o]++; } { auto & pr = g_pxa_name_us[pxa_name_bucket(node->name)]; pr.first += _dus; pr.second++; } if ((++g_pxa_total_ops % g_pxa_prof_every)==0) pxa_prof_dump(); }
                // PXA_WAVE4_DIAG: sync+check after EVERY op -> pin the exact faulting node.
                if (g_pxa_op_check_on > 0) {
                    cudaError_t _perr = cudaStreamSynchronize(cuda_ctx->stream());
                    if (_perr == cudaSuccess) _perr = cudaGetLastError();
                    if (_perr != cudaSuccess) {
                        fprintf(stderr, "PXA_WAVE4_DIAG FAULT dev=%d node[%d/%d] name=%s op=%s err=%s\n",
                                cuda_ctx->device, i, cgraph->n_nodes, node->name, ggml_op_name(node->op), cudaGetErrorString(_perr));
                        ggml_tensor * _tt[1 + GGML_MAX_SRC]; _tt[0] = node;
                        for (int _s = 0; _s < GGML_MAX_SRC; ++_s) _tt[1+_s] = node->src[_s];
                        for (int _s = 0; _s < 1 + GGML_MAX_SRC; ++_s) {
                            ggml_tensor * _t = _tt[_s]; if (!_t) continue;
                            fprintf(stderr, "  %s name=%s type=%s ne=[%ld,%ld,%ld,%ld] nb=[%zu,%zu,%zu,%zu] data=%p bufbase=%p\n",
                                    _s == 0 ? "dst" : "src", _t->name, ggml_type_name(_t->type),
                                    (long)_t->ne[0], (long)_t->ne[1], (long)_t->ne[2], (long)_t->ne[3],
                                    _t->nb[0], _t->nb[1], _t->nb[2], _t->nb[3], _t->data,
                                    _t->buffer ? ggml_backend_buffer_get_base(_t->buffer) : (void *)0);
                        }
                        for (int _k = i >= 4 ? i - 4 : 0; _k < i; ++_k) {
                            fprintf(stderr, "  prior node[%d] name=%s op=%s\n", _k, cgraph->nodes[_k]->name, ggml_op_name(cgraph->nodes[_k]->op));
                        }
                        fflush(stderr);
                        GGML_ABORT("PXA_WAVE4_DIAG: pinned at node %s", node->name);
                    }
                }
            }
        }
#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            // PXA_CUDA_GRAPH_BATCH fail-soft (S2): a capture that hit an uncapturable operation
            // must not abort the process. Disable graphs for THIS key only and re-run the ggml graph
            // eagerly -- work recorded during capture was never executed, so the eager re-run is the
            // first (and only) execution. One-way: the key stays eager for the rest of the session.
            cudaError_t pxa_capture_err = cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph);
            {
                std::lock_guard<std::mutex> lock(ggml_cuda_lock);
                if (--ggml_cuda_lock_counter == 0) {
                    ggml_cuda_lock_cv.notify_all();
                }
            }
            if (pxa_capture_err != cudaSuccess || graph->graph == nullptr) {
                (void)cudaGetLastError(); // clear the sticky capture error
                GGML_CUDA_LOG_ERROR("%s: CUDA graph capture failed (%s) - falling back to eager for this graph\n",
                        __func__, cudaGetErrorString(pxa_capture_err));
                if (graph->graph != nullptr) {
                    (void)cudaGraphDestroy(graph->graph);
                    graph->graph = nullptr;
                }
                graph->disable_due_to_failed_graph_capture = true;
                use_cuda_graph = false;
                cuda_graph_update_required = false;
                pxa_cgraph_stats().disables_capture_fail++;
                if (pxa_cuda_graph_log_enabled()) {
                    fprintf(stderr, "PXA_CGRAPH DISABLE capture-fail dev=%d key=%p\n",
                            cuda_ctx->device, ggml_cuda_graph_get_key(cgraph));
                }
                continue; // re-run this ggml graph eagerly
            }
            graph->n_captures++;
            graph->last_n_nodes = cgraph->n_nodes;
            pxa_cgraph_stats().captures++;
            if (pxa_cuda_graph_log_enabled()) {
                int pxa_ny = -1; // Ny of the first MoE node in this (sub)graph, if any
                for (int j = 0; j < cgraph->n_nodes; ++j) {
                    if (cgraph->nodes[j]->op == GGML_OP_MOE_FUSED_UP_GATE) { pxa_ny = (int)cgraph->nodes[j]->src[2]->ne[2]; break; }
                }
                fprintf(stderr, "PXA_CGRAPH capture dev=%d nodes=%d moe_ny=%d key=%p captures=%ld replays=%ld consec=%d\n",
                        cuda_ctx->device, cgraph->n_nodes, pxa_ny, ggml_cuda_graph_get_key(cgraph),
                        graph->n_captures, graph->n_replays, graph->number_consecutive_updates);
                // LOG=2: full node list per capture -> diff two captures to NAME a topology delta.
                if (pxa_cuda_graph_log_level() >= 2) {
                    for (int j = 0; j < cgraph->n_nodes; ++j) {
                        const ggml_tensor * t = cgraph->nodes[j];
                        fprintf(stderr, "PXA_CGRAPH_NODE key=%p [%d] op=%s name=%s ne=%lld,%lld,%lld,%lld\n",
                                ggml_cuda_graph_get_key(cgraph), j, ggml_op_name(t->op), t->name,
                                (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3]);
                    }
                }
            }
            graph_evaluated_or_captured = true; // CUDA graph has been captured
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }
    if (g_pxa_prof_on > 0 && _pxa_gt0 && cgraph->n_nodes >= 64) {
        cudaStreamSynchronize(cuda_ctx->stream());
        fprintf(stderr, "PXA_GRAPH dev=%d nodes=%d us=%lld\n", cuda_ctx->device, cgraph->n_nodes, (long long)(ggml_time_us() - _pxa_gt0));
    }

    if (use_cuda_graph) {
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            update_cuda_graph_executable(graph);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
        if (!cuda_graph_update_required) {
            // Honest replay accounting: a launch right after capture is NOT a replay.
            graph->n_replays++;
            long r = ++pxa_cgraph_stats().replays;
            if (pxa_cuda_graph_log_enabled()) {
                if (pxa_cuda_graph_log_level() >= 2) {
                    fprintf(stderr, "PXA_CGRAPH replay dev=%d nodes=%d key=%p key_replays=%ld total=%ld\n",
                            cuda_ctx->device, cgraph->n_nodes, ggml_cuda_graph_get_key(cgraph), graph->n_replays, r);
                } else if (r == 1 || r % 500 == 0) {
                    fprintf(stderr, "PXA_CGRAPH replays=%ld (captures=%ld)\n", r, pxa_cgraph_stats().captures.load());
                }
            }
        }
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

GGML_CALL static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    // PXA_MTP_REDUCE_CTX_FIX: ggml_cuda_op_reduce synchronizes its cross-device peer copies using
    // the GLOBAL ggml_cuda_info().all_ctx[i] streams/events. all_ctx[i] is set ONLY by the FIRST
    // constructed context for device i (the ctor refuses to overwrite and logs a-context-already
    // -exists), so once a SECOND llama_context exists (the MTP speculative/draft context), its
    // reduces gather the other-device partials on the WRONG first/main context streams. The
    // producer kernel ran on THIS context device-i stream while the gather waits on the other
    // context stream -> unordered -> intermittent cross-device illegal memory access in the MTP
    // head decode under -sm attn (crash device varies). Re-register THIS context for its device on
    // every graph compute: the scheduler runs each device splits host-sequentially before any later
    // cross-device reduce, so by the reduce all participating all_ctx[] point to the CURRENT context.
    // Single-context (no-MTP) decode is unaffected (idempotent no-op).
    {
        auto _pxa_info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
        if (cuda_ctx->device >= 0 && cuda_ctx->device < GGML_CUDA_MAX_DEVICES) {
            _pxa_info->all_ctx[cuda_ctx->device] = cuda_ctx;
        }
    }

#ifdef USE_CUDA_GRAPH
    cuda_ctx->cur_graph = nullptr;

    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    bool use_cuda_graph = !disable_cuda_graphs_due_to_env && cuda_ctx->use_cuda_graph;

    ggml_cuda_graph * graph = nullptr;
    if (use_cuda_graph) {
        auto graph_key = ggml_cuda_graph_get_key(cgraph);
        graph = ggml_cuda_get_graph(*cuda_ctx, graph_key);
    }
    cuda_ctx->cur_graph = graph;

    bool cuda_graph_update_required = false;

    const bool pxa_v2 = pxa_cuda_graph_v2_enabled();
    const bool pxa_glog = pxa_cuda_graph_log_enabled();
    if (pxa_glog) pxa_cgraph_stats_arm_atexit();
    const void * pxa_gkey = (pxa_glog && graph) ? ggml_cuda_graph_get_key(cgraph) : nullptr;

    if (use_cuda_graph && graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < CC_AMPERE && !getenv("PXA_CUDA_GRAPHS_PASCAL")) {
            if (!graph->disable_due_to_gpu_arch) {
                pxa_cgraph_stats().disables_arch++;
                if (pxa_glog) fprintf(stderr, "PXA_CGRAPH DISABLE gpu-arch dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
            }
            graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && (
        graph->disable_due_to_gpu_arch ||
        graph->disable_due_to_too_many_updates ||
        graph->disable_due_to_failed_graph_capture)) {
        use_cuda_graph = false;
        graph->n_eager++;
        pxa_cgraph_stats().eager_disabled++;
    }

    // PXA_CUDA_GRAPH_V2 P1.2: cooldown in place of the permanent too-many-updates disable.
    if (use_cuda_graph && pxa_v2 && graph->cooldown_remaining > 0) {
        graph->cooldown_remaining--;
        if (graph->cooldown_remaining == 0) {
            graph->number_consecutive_updates = 0; // re-arm
            if (pxa_glog) fprintf(stderr, "PXA_CGRAPH cooldown-rearm dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
        }
        use_cuda_graph = false;
        graph->n_eager++;
        pxa_cgraph_stats().eager_cooldown++;
    }

    if (use_cuda_graph && pxa_v2) {
        // PXA_CUDA_GRAPH_V2 P1.3: capturability check BEFORE the property store, so an uncapturable
        // graph (prefill MoE chunk, unsupported CPY) can no longer clobber the stored properties of
        // a capturable one at the same key and force a spurious recapture of the next decode token.
        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, graph, cgraph, use_cuda_graph, cuda_ctx->stream());
        if (!use_cuda_graph) {
            graph->n_eager++;
            pxa_cgraph_stats().eager_uncapturable++;
            if (pxa_cuda_graph_log_level() >= 2) {
                fprintf(stderr, "PXA_CGRAPH eager-uncapturable dev=%d nodes=%d key=%p\n", cuda_ctx->device, cgraph->n_nodes, pxa_gkey);
            }
        } else {
            cuda_graph_update_required = is_cuda_graph_update_required(graph, cgraph, pxa_gkey);

            // P3 pool-generation guard: if pool memory was cudaFree'd since the last capture, the
            // captured kernels may hold dangling scratch pointers -> force recapture, never replay.
            const uint64_t pxa_pool_gen = cuda_ctx->pool().generation();
            if (!cuda_graph_update_required && pxa_pool_gen != graph->pool_generation) {
                cuda_graph_update_required = true;
                if (pxa_glog) fprintf(stderr, "PXA_CGRAPH update-required key=%p reason=pool-generation %llu->%llu\n",
                        pxa_gkey, (unsigned long long)graph->pool_generation, (unsigned long long)pxa_pool_gen);
            }
            graph->pool_generation = pxa_pool_gen;

            if (cuda_graph_update_required) {
                graph->number_consecutive_updates++;
            } else {
                graph->number_consecutive_updates = 0;
            }

            if (graph->number_consecutive_updates >= 4) {
                const int rearm = pxa_cuda_graph_rearm_evals();
                if (rearm > 0) {
                    graph->cooldown_remaining = rearm;
                    pxa_cgraph_stats().cooldowns_armed++;
                    if (pxa_glog) fprintf(stderr, "PXA_CGRAPH cooldown-armed dev=%d key=%p evals=%d\n", cuda_ctx->device, pxa_gkey, rearm);
                } else {
                    graph->disable_due_to_too_many_updates = true; // PXA_CUDA_GRAPH_REARM=0 keeps the permanent disable
                    pxa_cgraph_stats().disables_too_many++;
                    if (pxa_glog) fprintf(stderr, "PXA_CGRAPH DISABLE too-many-updates dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
                }
                use_cuda_graph = false;
                cuda_ctx->cur_graph = nullptr;
                graph->n_eager++;
            }
        }
    } else if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(graph, cgraph, pxa_gkey);

        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(cuda_ctx, graph, cgraph, use_cuda_graph, cuda_ctx->stream());
        if (!use_cuda_graph) {
            graph->n_eager++;
            pxa_cgraph_stats().eager_uncapturable++;
        }

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph) {
            if (cuda_graph_update_required) {
                graph->number_consecutive_updates++;
            } else {
                graph->number_consecutive_updates = 0;
            }
        }

        if (graph->number_consecutive_updates >= 4) {
            if (!graph->disable_due_to_too_many_updates) {
                pxa_cgraph_stats().disables_too_many++;
                if (pxa_glog) fprintf(stderr, "PXA_CGRAPH DISABLE too-many-updates dev=%d key=%p\n", cuda_ctx->device, pxa_gkey);
            }
            graph->disable_due_to_too_many_updates = true;
            use_cuda_graph = false;
            cuda_ctx->cur_graph = nullptr;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        // Why are we protecting an atomic_int with a mutex?
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ++ggml_cuda_lock_counter;
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    if (graph && !use_cuda_graph) {
        graph->use_cpy_indirection = false;
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required);

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    // Non-mul_mat ops can't read a split-buffer parent (no data ptr); let the scheduler fall back to CPU.
    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        // PXA-SHARD: the fused MoE up+gate op HAS a CUDA handler that reads the
        // expert-shard per-device slices (ggml_cuda_moe_up_gate_unary's shard branch
        // resolves root->name->registry), so it must STAY on CUDA — do NOT fall it
        // back to CPU for an expert-shard src. (Split buffers still always fall back.)
        const bool pxa_shard_fused_ok = (op->op == GGML_OP_MOE_FUSED_UP_GATE || op->op == GGML_OP_FUSED_UP_GATE);
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            // no single data ptr for split/shard bufts, so non-mul_mat ops fall back
            // to CPU. Both terms are false when PXA_EXPERT_SHARD is off.
            if (op->src[i] && op->src[i]->buffer &&
                ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
            if (op->src[i] && op->src[i]->buffer &&
                pxa_buft_is_expert_shard(op->src[i]->buffer->buft) && !pxa_shard_fused_ok) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SWIGLU:
                case GGML_UNARY_OP_SWIGLU_OAI:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_NEG:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY: return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MOE_FUSED_UP_GATE:
        case GGML_OP_FUSED_UP_GATE:
            {
                bool is_fused_up_gate = op->op == GGML_OP_MOE_FUSED_UP_GATE || op->op == GGML_OP_FUSED_UP_GATE;
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = is_fused_up_gate ? op->src[2] : op->src[1];
                if (is_fused_up_gate && op->src[1] && !ggml_moe_up_gate_can_fuse(a->type, op->src[1]->type)) {
                    fprintf(stderr, "%s: returning false for GGML_OP_MOE_FUSED_UP_GATE because src0->type != src1->type\n", __func__);
                    return false;
                }
                //==================================================================
                //if (ggml_is_quantized(a->type) && ggml_is_quantized(b->type)) {
                //    return false;
                //}
                //==================================================================
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_is_quantized(a->type)) {
                    printf("%s: returning false for op %d because (case 1)\n", __func__, (int)op->op);
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT && a->ne[3] != b->ne[3]) {
                    return false;
                }
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q6_0:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_PXQ4:
                    case GGML_TYPE_PXQ4HQ:
                    case GGML_TYPE_PXQ2:      // ADD
                    case GGML_TYPE_PXQ3:      // ADD
                    case GGML_TYPE_PXQ1:
                    case GGML_TYPE_PXQ6:     // ADD
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ2_KL:
                    case GGML_TYPE_IQ3_KS:
                    case GGML_TYPE_IQ4_KS:
                    case GGML_TYPE_IQ4_KSS:
                    case GGML_TYPE_IQ5_KS:
                    case GGML_TYPE_IQ2_K:
                    case GGML_TYPE_IQ2_KS:
                    case GGML_TYPE_IQ1_KT:
                    case GGML_TYPE_IQ2_KT:
                    case GGML_TYPE_IQ3_KT:
                    case GGML_TYPE_IQ4_KT:
                    case GGML_TYPE_IQ3_K:
                    case GGML_TYPE_IQ4_K:
                    case GGML_TYPE_IQ5_K:
                    case GGML_TYPE_IQ6_K:
                    case GGML_TYPE_IQ1_BN:
                    case GGML_TYPE_IQ2_BN:
                    case GGML_TYPE_IQ2_K_R4:
                    case GGML_TYPE_IQ3_K_R4:
                    case GGML_TYPE_IQ4_K_R4:
                    case GGML_TYPE_IQ4_KS_R4:
                    case GGML_TYPE_IQ5_K_R4:
                    case GGML_TYPE_IQ5_KS_R4:
                    case GGML_TYPE_IQ1_S_R4:
                    case GGML_TYPE_IQ1_M_R4:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q6_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (ggml_is_quantized(src0_type) && (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_F32)) {
                    return true;
                }
                if (ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op->src[1])) {
                    if (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                if (ggml_are_same_shape(op->src[0], op->src[1]) && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_Q8_0) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_REDUCE:
        case GGML_OP_FAKE_CPY:
        case GGML_OP_ARGMAX:
            return true;
        case GGML_OP_HADAMARD: {
            if (!(op->op_params[0] == 64 || op->op_params[0] == 128 || op->op_params[0] == 256 || op->op_params[0] == 512)) return false;
            if (op->ne[0] % op->op_params[0] != 0) return false;
            if (op->type != GGML_TYPE_F32) return false;
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q6_0:
                case GGML_TYPE_IQ4_NL:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_L2_NORM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_MULTI_ADD:
        case GGML_OP_MUL_MULTI_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
        case GGML_OP_FUSED_RMS_NORM:
        case GGML_OP_FUSED_RMS_RMS_ADD:
        case GGML_OP_SCALE:
        case GGML_OP_SOFTCAP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_CAP_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ROPE_FAST:
        case GGML_OP_ROPE_CACHE:
            return true;
        case GGML_OP_FUSED_NORM:
            return ggml_is_contiguous(op->src[0]);
        //case GGML_OP_ROPE:
        //    return ggml_is_contiguous(op->src[0]);
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_ARGSORT:
        case GGML_OP_ARGSORT_THRESH:
        case GGML_OP_GROUPED_TOPK:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        case GGML_OP_CUMSUM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_TRI:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[0]->type == op->type;
        case GGML_OP_FILL:
            return ggml_is_contiguous(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_SOLVE_TRI:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op) &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] == op->src[0]->ne[1] &&
                   op->src[0]->ne[1] == op->src[1]->ne[1] &&
                   op->src[0]->ne[2] == op->src[1]->ne[2] &&
                   op->src[0]->ne[3] == op->src[1]->ne[3];
        case GGML_OP_SSM_CONV:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 &&
                   op->src[3]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) &&
                   op->src[1]->nb[0] == sizeof(float) &&
                   op->src[2]->nb[0] == sizeof(float) &&
                   op->src[3]->nb[0] == sizeof(int32_t) &&
                   op->src[2]->ne[0] == op->src[0]->ne[0] + 1 &&
                   op->src[2]->ne[1] == op->src[0]->ne[1] &&
                   op->src[1]->ne[0] == op->src[0]->ne[1] &&
                   op->src[3]->ne[0] == op->src[0]->ne[2];
        case GGML_OP_DELTA_NET:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
            return (op->src[0]->ne[0] == 64 && op->src[1]->type == GGML_TYPE_F16) || op->src[0]->ne[0] == 128;
#else
            return ggml_cuda_fattn_is_supported(*cuda_ctx, op);
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cuda_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    //printf("%s(%s, %s): %d, %d\n", __func__, ggml_backend_name(backend), ggml_backend_buft_name(buft), ggml_backend_buft_is_cuda_split(buft), ggml_backend_buft_is_cuda(buft));
    if (ggml_backend_buft_is_cuda_split(buft)) {
        return true;
    }

    // PXA-SHARD: keep expert-sharded weights on the CUDA backend (same as split).
    // Always false when PXA_EXPERT_SHARD is off, so this is a no-op flag-off.
    if (pxa_buft_is_expert_shard(buft)) {
        return true;
    }

    if (ggml_backend_buft_is_cuda(buft)) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
        ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
        return buft_ctx->device == cuda_ctx->device;
    }

    return false;
}

GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    auto ctx = (ggml_backend_cuda_context *)backend->context;
    int min_batch_size = ctx->offload_batch_size; // originally: GGML_CUDA_MIN_BATCH_OFFLOAD;

    // Why do we want to do this? The heuristics that the batch must have more than min_batch_size tokens to be worth it
    // offloading the required model weights comes from dense models. For MoE models, the average number of tokens
    // each expert deals with in a batch is (active_experts / total_experts) * batch_size. Hence, according to the
    // learned heuristics, we need (active_experts / total_experts) * batch_size >= min_batch_size.
    // Rearranging we get
    //
    //           batch_size * active_experts >= min_batch_size * total_experts
    //
    // as the condition for offloading model weights resinding in RAM to the GPU.
    // In this case, the number of tokens is not as usual in op->ne[1] but rather in op->ne[2].
    if (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE) {
        if (ctx->offload_batch_size_per_byte >= 0) {
            auto src0 = op->src[0];
            auto row_size = ggml_row_size(src0->type, src0->ne[0]);
            min_batch_size = int(1.*ctx->offload_batch_size_per_byte*row_size/src0->ne[0]);
        }
        auto ids = op->op == GGML_OP_MUL_MAT_ID ? op->src[2] : op->src[3];
        int64_t batch_size = op->ne[2];
        if (batch_size < min_batch_size) return false;
        int64_t n_experts_tot    = op->src[0]->ne[2];
        int64_t n_experts_active = ids->ne[0];
        bool should_offload = batch_size*n_experts_active >= min_batch_size*n_experts_tot;
        return should_offload;
    }

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;

    // Original:
    //return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
    //       (op->ne[2] >= min_batch_size && (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE));

    GGML_UNUSED(backend);
}

static ggml_backend_event_t ggml_backend_cuda_event_new(ggml_backend_t backend) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .backend = */ backend,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_event_free(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));

    delete event;
}

static void ggml_backend_cuda_event_record(ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)event->backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(event->backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_event_synchronize(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .get_default_buffer_type = */ ggml_backend_cuda_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .supports_op             = */ ggml_backend_cuda_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_offload_op,
    /* .event_new               = */ ggml_backend_cuda_event_new,
    /* .event_free              = */ ggml_backend_cuda_event_free,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .event_synchronize       = */ ggml_backend_cuda_event_synchronize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

struct cuda_params {
    int  fusion = GGML_CUDA_FUSION;
    int  offload_batch_size = GGML_CUDA_MIN_BATCH_OFFLOAD;
    int  offload_batch_size_per_byte = -1;
    int  mmq_id_thresh = 32;
    float fa_offset = 0.6931f;
#ifdef USE_CUDA_GRAPH
    bool use_cuda_graph = true;
#else
    bool use_cuda_graph = false;
#endif
    bool enable_p2p = true;
};

static std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

template <typename T> bool read_value(const std::string& val, T& result) {
    std::istringstream str(val);
    T tmp; str >> tmp;
    if (!str.fail()) {
        result = tmp;
        return true;
    }
    return false;
}

static cuda_params ggml_cuda_parse_params(const char * params_string) {
    cuda_params params{};
    if (!params_string) return params;
    auto values = string_split(std::string{params_string}, ",");
    if (values.empty()) return params;
    for (auto& value : values) {
        auto parsed = string_split(value, "=");
        bool is_good = false;
        if (parsed.size() == 2) {
            if (parsed[0] == "fusion") {
                is_good = read_value(parsed[1], params.fusion);
            }
            else if (parsed[0] == "offload-batch-size") {
                int tmp = 0;
                is_good = read_value(parsed[1], tmp);
                if (is_good) {
                    params.offload_batch_size = tmp;
                }
            }
            else if (parsed[0] == "offload-batch-size-per-byte") {
                is_good = read_value(parsed[1], params.offload_batch_size_per_byte);
            }
            else if (parsed[0] == "mmq-id-size") {
                is_good = read_value(parsed[1], params.mmq_id_thresh);
            }
            else if (parsed[0] == "enable-p2p") {
                is_good = read_value(parsed[1], params.enable_p2p);
            }
            else if (parsed[0] == "fa-offset") {
                float tmp;
                is_good = read_value(parsed[1], tmp);
                if (is_good) {
                    if (tmp < 0.0f || tmp > 3.0f) {
                        GGML_CUDA_LOG_WARN("%s: bad value for %s. It is %g, but must be in [0...3]\n", __func__, parsed[0].c_str(), tmp);
                    } else {
                        params.fa_offset = tmp;
                    }
                }
            }
#ifdef USE_CUDA_GRAPH
            else if (parsed[0] == "graphs") {
                is_good = read_value(parsed[1], params.use_cuda_graph);
            }
#endif
        }
        if (!is_good) {
            GGML_CUDA_LOG_WARN("%s: invalid parameter %s (%d) -> ignored\n", __func__, value.c_str(), (int)parsed.size());
        }
    }
    return params;
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, [[maybe_unused]] const void * param_string) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_CUDA_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_CUDA_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .context   = */ ctx
    };

    bool enable_p2p = true;
    if (param_string) {
        [[maybe_unused]] auto params = ggml_cuda_parse_params((const char *)param_string);
        if (params.fusion != ctx->fusion) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fusion to %d\n", __func__, params.fusion);
            ctx->fusion             = params.fusion;
        }
        if (params.offload_batch_size != ctx->offload_batch_size) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size to %d\n", __func__, params.offload_batch_size);
            ctx->offload_batch_size = params.offload_batch_size;
        }
        if (params.offload_batch_size_per_byte != ctx->offload_batch_size_per_byte) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size_per_byte to %d\n", __func__, params.offload_batch_size_per_byte);
            ctx->offload_batch_size_per_byte = params.offload_batch_size_per_byte;
        }
        if (params.mmq_id_thresh != ctx->mmq_id_thresh) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting mmq_id_thresh to %d\n", __func__, params.mmq_id_thresh);
            ctx->mmq_id_thresh      = params.mmq_id_thresh;
        }
        if (params.fa_offset != ctx->fa_offset) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fa_offset to %g\n", __func__, params.fa_offset);
            ctx->fa_offset = params.fa_offset;
        }
        enable_p2p = params.enable_p2p;
#ifdef USE_CUDA_GRAPH
        if (params.use_cuda_graph != ctx->use_cuda_graph) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting use_cuda_graph to %d\n", __func__, params.use_cuda_graph);
            ctx->use_cuda_graph = params.use_cuda_graph;
        }
#endif
    }

#ifdef GGML_USE_NCCL
    if (!enable_p2p) {
        printf("================== P2P disabled, but needed for NCCL\n");
        enable_p2p = true;
    }
#endif

#if !defined(GGML_CUDA_NO_PEER_COPY)
    if (enable_p2p) {
        ctx->p2p_enabled = ggml_cuda_set_peer_access(device);
    }
#endif

    return cuda_backend;
}

GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

GGML_CALL int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

GGML_CALL int ggml_backend_cuda_get_device_cc(int device) {
    if (device < 0 || device >= ggml_cuda_info().device_count) {
        return -1;
    }
    return ggml_cuda_info().devices[device].cc;
}

// Read-only P2P topology probe: true iff every ordered device pair can peer-access.
// Uses cudaDeviceCanAccessPeer ONLY (no cudaDeviceEnablePeerAccess) so it is side-effect free
// and safe to call at model-load time. Result is cached in a tri-state static.
GGML_CALL bool ggml_backend_cuda_all_pairs_can_peer(void) {
    static int cached = -1; // -1 unknown, 0 false, 1 true
    if (cached != -1) {
        return cached != 0;
    }
    const int device_count = ggml_backend_cuda_get_device_count();
    if (device_count <= 1) {
        cached = 1;
        return true;
    }
    bool all_peer = true;
    for (int id = 0; id < device_count && all_peer; ++id) {
        ggml_cuda_set_device(id);
        for (int id_other = 0; id_other < device_count; ++id_other) {
            if (id == id_other) {
                continue;
            }
            int can_access_peer = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
            if (!can_access_peer) {
                all_peer = false;
                break;
            }
        }
    }
    cached = all_peer ? 1 : 0;
    return all_peer;
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11100 || defined(GGML_USE_MUSA)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();

        GGML_CUDA_LOG_WARN("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init((int) (intptr_t) user_data, nullptr);
    return cuda_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cuda_reg_devices();

GGML_CALL int ggml_backend_cuda_reg_devices() {
    int device_count = ggml_backend_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
