// pxq4_kernel_launch.h — the CUDA-free launcher declarations consumed by pxq4_kernel_torch.cpp.
// Keeping these in a plain header lets the torch binding TU be compiled by the host compiler
// only; nothing here mentions a __global__ or a device type.
#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// dequantize the whole tensor: out[N, K] fp16 (row-major) from the PXQ4 two-tensor split.
// N = panels*64, K = kslabs*32. `anchor` points at fp16 data ([panels, 64]).
void pxq4_launch_dequant_f16(const uint8_t * slabs, const void * anchor, void * out,
                             int panels, int kslabs, cudaStream_t stream);

// out[M, N] fp16 = x[M, K] fp16 * W[N, K]^T. Small M only (see PXQ4_MMV_MAX_M).
void pxq4_launch_mmv_f16(const uint8_t * slabs, const void * anchor, const void * x, void * out,
                         int M, int panels, int kslabs, bool vecx, cudaStream_t stream);

// dynamic shared-memory bytes the mmv needs for this K, and whether that fits the device.
int  pxq4_mmv_smem_bytes(int kslabs);
bool pxq4_mmv_supported(int kslabs);

// overwrite the device-resident book / sublevel tables (16 floats each) on the CURRENT device.
// Must be called in eager mode, before any cuda-graph capture: it is a cudaMemcpyToSymbol.
void pxq4_upload_tables(const float * book16, const float * sub16);

// read the tables back for validation (G6).
void pxq4_download_tables(float * book16, float * sub16);
