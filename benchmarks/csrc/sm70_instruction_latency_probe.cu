// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

#ifndef SM70_LATENCY_PROBE_ITERATIONS
  #define SM70_LATENCY_PROBE_ITERATIONS 128
#endif

constexpr int kThreads = 32;
constexpr int kProbeIterations = SM70_LATENCY_PROBE_ITERATIONS;
constexpr unsigned kWarpMask = 0xffffffffU;

void check_cuda(cudaError_t status, const char* expression, const char* file,
                int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::cerr << "CUDA failure at " << file << ':' << line << " for "
            << expression << ": " << cudaGetErrorString(status) << '\n';
  std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression) \
  check_cuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ uint32_t pack_half2(float low, float high) {
  const uint32_t low_bits = __half_as_ushort(__float2half_rn(low));
  const uint32_t high_bits = __half_as_ushort(__float2half_rn(high));
  return low_bits | (high_bits << 16);
}

__device__ __forceinline__ void mma_m8n8k4_row_col(float (&d)[8], uint32_t a0,
                                                   uint32_t a1, uint32_t b0,
                                                   uint32_t b1,
                                                   const float (&c)[8]) {
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, "
      "{%8, %9}, {%10, %11}, {%12, %13, %14, %15, %16, %17, %18, %19};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
        "=f"(d[6]), "=f"(d[7])
      : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]),
        "f"(c[3]), "f"(c[4]), "f"(c[5]), "f"(c[6]), "f"(c[7]));
}

__device__ __forceinline__ uint64_t read_clock() {
  __syncwarp(kWarpMask);
  asm volatile("" ::: "memory");
  const uint64_t value = clock64();
  asm volatile("" ::: "memory");
  return value;
}

__global__ void empty_loop_probe(uint64_t* cycles, float* sink) {
  uint32_t value = threadIdx.x + 1;
  const uint64_t start = read_clock();
#pragma unroll
  for (int iteration = 0; iteration < kProbeIterations; ++iteration) {
    asm volatile("add.u32 %0, %0, 1;" : "+r"(value));
  }
  const uint64_t end = read_clock();
  if (threadIdx.x == 0) {
    cycles[0] = end - start;
  }
  sink[threadIdx.x] = static_cast<float>(value);
}

__global__ void dependent_hmma_probe(uint64_t* cycles, float* sink) {
  const uint32_t a0 = pack_half2(0.015625f, -0.0078125f);
  const uint32_t a1 = pack_half2(0.00390625f, 0.01171875f);
  const uint32_t b0 = pack_half2(-0.0078125f, 0.015625f);
  const uint32_t b1 = pack_half2(0.01171875f, 0.00390625f);
  float accum[8];
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    accum[index] = static_cast<float>(threadIdx.x + index) * 0.0001f;
  }

  const uint64_t start = read_clock();
#pragma unroll
  for (int iteration = 0; iteration < kProbeIterations; ++iteration) {
    float next[8];
    mma_m8n8k4_row_col(next, a0, a1, b0, b1, accum);
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accum[index] = next[index];
    }
  }
  const uint64_t end = read_clock();
  if (threadIdx.x == 0) {
    cycles[0] = end - start;
  }
  sink[threadIdx.x] = accum[threadIdx.x & 7];
}

__global__ void independent_hmma_probe(uint64_t* cycles, float* sink) {
  const uint32_t a0 = pack_half2(0.015625f, -0.0078125f);
  const uint32_t a1 = pack_half2(0.00390625f, 0.01171875f);
  const uint32_t b0 = pack_half2(-0.0078125f, 0.015625f);
  const uint32_t b1 = pack_half2(0.01171875f, 0.00390625f);
  float accum0[8];
  float accum1[8];
  float accum2[8];
  float accum3[8];
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    accum0[index] = static_cast<float>(threadIdx.x + index) * 0.0001f;
    accum1[index] = static_cast<float>(threadIdx.x + index + 1) * 0.0001f;
    accum2[index] = static_cast<float>(threadIdx.x + index + 2) * 0.0001f;
    accum3[index] = static_cast<float>(threadIdx.x + index + 3) * 0.0001f;
  }

  const uint64_t start = read_clock();
#pragma unroll
  for (int iteration = 0; iteration < kProbeIterations; ++iteration) {
    float next0[8];
    float next1[8];
    float next2[8];
    float next3[8];
    mma_m8n8k4_row_col(next0, a0, a1, b0, b1, accum0);
    mma_m8n8k4_row_col(next1, a0, a1, b0, b1, accum1);
    mma_m8n8k4_row_col(next2, a0, a1, b0, b1, accum2);
    mma_m8n8k4_row_col(next3, a0, a1, b0, b1, accum3);
#pragma unroll
    for (int index = 0; index < 8; ++index) {
      accum0[index] = next0[index];
      accum1[index] = next1[index];
      accum2[index] = next2[index];
      accum3[index] = next3[index];
    }
  }
  const uint64_t end = read_clock();
  if (threadIdx.x == 0) {
    cycles[0] = end - start;
  }
  float total = 0.0f;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    total += accum0[index] + accum1[index] + accum2[index] + accum3[index];
  }
  sink[threadIdx.x] = total;
}

__global__ void dependent_shared_load_probe(uint64_t* cycles, float* sink) {
  __shared__ volatile uint32_t links[kThreads];
  const int lane = threadIdx.x;
  links[lane] = (lane + 1) & (kThreads - 1);
  __syncwarp(kWarpMask);
  uint32_t index = lane;
  const uint64_t start = read_clock();
#pragma unroll
  for (int iteration = 0; iteration < kProbeIterations; ++iteration) {
    index = links[index];
  }
  const uint64_t end = read_clock();
  if (lane == 0) {
    cycles[0] = end - start;
  }
  sink[lane] = static_cast<float>(index);
}

__global__ void dependent_global_load_probe(const uint32_t* __restrict__ links,
                                            uint64_t* cycles, float* sink) {
  uint32_t index = threadIdx.x;
  const uint64_t start = read_clock();
#pragma unroll
  for (int iteration = 0; iteration < kProbeIterations; ++iteration) {
    index = __ldcg(links + index);
  }
  const uint64_t end = read_clock();
  if (threadIdx.x == 0) {
    cycles[0] = end - start;
  }
  sink[threadIdx.x] = static_cast<float>(index);
}

__global__ void shared_roundtrip_probe(uint64_t* cycles, float* sink) {
  __shared__ volatile uint32_t values[kThreads];
  const int lane = threadIdx.x;
  uint32_t value = lane + 1;
  const uint64_t start = read_clock();
#pragma unroll
  for (int iteration = 0; iteration < kProbeIterations; ++iteration) {
    values[lane] = value;
    __syncwarp(kWarpMask);
    value = values[(lane + 1) & (kThreads - 1)] + 1;
    __syncwarp(kWarpMask);
  }
  const uint64_t end = read_clock();
  if (lane == 0) {
    cycles[0] = end - start;
  }
  sink[lane] = static_cast<float>(value);
}

struct Arguments {
  int device = 0;
  int iterations = 1024;
  int warmup = 8;
  int samples = 51;
};

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    if (index + 1 >= argc) {
      std::cerr << "missing value for " << argv[index] << '\n';
      std::exit(EXIT_FAILURE);
    }
    const std::string option = argv[index];
    const int value = std::atoi(argv[++index]);
    if (option == "--device") {
      arguments.device = value;
    } else if (option == "--iterations") {
      arguments.iterations = value;
    } else if (option == "--warmup") {
      arguments.warmup = value;
    } else if (option == "--samples") {
      arguments.samples = value;
    } else {
      std::cerr << "unknown option: " << option << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  if (arguments.iterations <= 0 || arguments.warmup < 0 ||
      arguments.samples <= 0) {
    std::cerr << "invalid benchmark argument\n";
    std::exit(EXIT_FAILURE);
  }
  return arguments;
}

template <typename Launch>
std::vector<uint64_t> measure(Launch launch, uint64_t* device_cycles,
                              int warmup, int samples) {
  for (int index = 0; index < warmup; ++index) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint64_t> values;
  values.reserve(samples);
  for (int index = 0; index < samples; ++index) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    uint64_t cycles = 0;
    CUDA_CHECK(cudaMemcpy(&cycles, device_cycles, sizeof(cycles),
                          cudaMemcpyDeviceToHost));
    values.push_back(cycles);
  }
  std::sort(values.begin(), values.end());
  return values;
}

void print_cycles(const std::vector<uint64_t>& values, int iterations,
                  int operations_per_iteration) {
  const double mean = static_cast<double>(std::accumulate(
                          values.begin(), values.end(), uint64_t{0})) /
                      values.size();
  const uint64_t median = values[values.size() / 2];
  const double operations =
      static_cast<double>(iterations) * operations_per_iteration;
  std::cout << std::fixed << std::setprecision(6)
            << "{\"min_cycles\":" << values.front()
            << ",\"median_cycles\":" << median << ",\"mean_cycles\":" << mean
            << ",\"max_cycles\":" << values.back()
            << ",\"median_cycles_per_operation\":" << median / operations
            << '}';
}

}  // namespace

int main(int argc, char** argv) {
  const Arguments arguments = parse_arguments(argc, argv);
  CUDA_CHECK(cudaSetDevice(arguments.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, arguments.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "SM70 GPU required, got " << properties.major << '.'
              << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  uint64_t* device_cycles = nullptr;
  float* device_sink = nullptr;
  uint32_t* device_links = nullptr;
  CUDA_CHECK(cudaMalloc(&device_cycles, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&device_sink, kThreads * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_links, kThreads * sizeof(uint32_t)));
  std::vector<uint32_t> links(kThreads);
  for (int index = 0; index < kThreads; ++index) {
    links[index] = (index + 1) & (kThreads - 1);
  }
  CUDA_CHECK(cudaMemcpy(device_links, links.data(),
                        links.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));

  if (arguments.iterations != kProbeIterations) {
    std::cerr << "binary was built for " << kProbeIterations
              << " iterations, got " << arguments.iterations << '\n';
    return EXIT_FAILURE;
  }
  const int iterations = kProbeIterations;
  const auto empty = measure(
      [&] { empty_loop_probe<<<1, kThreads>>>(device_cycles, device_sink); },
      device_cycles, arguments.warmup, arguments.samples);
  const auto dependent_hmma = measure(
      [&] {
        dependent_hmma_probe<<<1, kThreads>>>(device_cycles, device_sink);
      },
      device_cycles, arguments.warmup, arguments.samples);
  const auto independent_hmma = measure(
      [&] {
        independent_hmma_probe<<<1, kThreads>>>(device_cycles, device_sink);
      },
      device_cycles, arguments.warmup, arguments.samples);
  const auto shared_load = measure(
      [&] {
        dependent_shared_load_probe<<<1, kThreads>>>(device_cycles,
                                                     device_sink);
      },
      device_cycles, arguments.warmup, arguments.samples);
  const auto global_load = measure(
      [&] {
        dependent_global_load_probe<<<1, kThreads>>>(
            device_links, device_cycles, device_sink);
      },
      device_cycles, arguments.warmup, arguments.samples);
  const auto shared_roundtrip = measure(
      [&] {
        shared_roundtrip_probe<<<1, kThreads>>>(device_cycles, device_sink);
      },
      device_cycles, arguments.warmup, arguments.samples);

  std::cout << "{\n  \"device\":\"" << properties.name << "\",\n"
            << "  \"clock_rate_khz\":" << properties.clockRate << ",\n"
            << "  \"iterations\":" << iterations << ",\n"
            << "  \"probes\":{\n    \"empty_loop\":";
  print_cycles(empty, iterations, 1);
  std::cout << ",\n    \"dependent_hmma_m8n8k4\":";
  print_cycles(dependent_hmma, iterations, 1);
  std::cout << ",\n    \"four_chain_hmma_m8n8k4\":";
  print_cycles(independent_hmma, iterations, 4);
  std::cout << ",\n    \"dependent_shared_load\":";
  print_cycles(shared_load, iterations, 1);
  std::cout << ",\n    \"dependent_global_cg_load_l2_hit\":";
  print_cycles(global_load, iterations, 1);
  std::cout << ",\n    \"shared_store_load_warp_roundtrip\":";
  print_cycles(shared_roundtrip, iterations, 1);
  std::cout << "\n  }\n}\n";

  cudaFree(device_links);
  cudaFree(device_sink);
  cudaFree(device_cycles);
  return EXIT_SUCCESS;
}
