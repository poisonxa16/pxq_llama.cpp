// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kSourceValues = 8;
constexpr int kTargetWords = 16;

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

__host__ __device__ constexpr int accumulator_row(int lane, int reg) {
  return (lane & 1) + (reg & 2) + ((lane >> 4) * 4);
}

__host__ __device__ constexpr int accumulator_col(int lane, int reg) {
  return (lane & 12) * 2 + (lane & 2) + (reg & 4) + (reg & 1);
}

__host__ __device__ constexpr int matrix_a_row(int lane) {
  return (lane & 3) + ((lane >> 4) * 4);
}

__device__ __forceinline__ uint32_t pack_fp32_to_half2(float low, float high) {
  const uint32_t low_bits = __half_as_ushort(__float2half_rn(low));
  const uint32_t high_bits = __half_as_ushort(__float2half_rn(high));
  return low_bits | (high_bits << 16);
}

__device__ __forceinline__ void convert_reference(
    const float (&source)[kSourceValues], uint32_t (&target)[kTargetWords],
    int lane) {
  const int target_row_hi = (lane >> 1) & 1;
#pragma unroll
  for (int word = 0; word < kTargetWords; ++word) {
    uint32_t result = 0;
#pragma unroll
    for (int key_in_word = 0; key_in_word < 2; ++key_in_word) {
      const int key = 2 * word + key_in_word;
      const int source_lane = (lane & 1) | (((key >> 1) & 1) << 1) |
                              (((key >> 3) & 3) << 2) | (lane & 16);
      const int source_reg = key_in_word + 4 * ((key >> 2) & 1);
      const uint32_t payload =
          pack_fp32_to_half2(source[source_reg], source[source_reg + 2]);
      const uint32_t received = __shfl_sync(0xffffffffU, payload, source_lane);
      const uint32_t selected =
          target_row_hi ? (received >> 16) : (received & 0xffffU);
      result |= selected << (16 * key_in_word);
    }
    target[word] = result;
  }
}

template <int kRound>
__device__ __forceinline__ uint32_t
minimum_communication_round(const float (&source)[kSourceValues], int lane) {
  constexpr int kSourceWordBase = 4 * ((kRound >> 1) & 1);
  const bool source_row_hi = ((kRound & 1) ^ ((lane >> 1) & 1)) != 0;
  const float low =
      source_row_hi ? source[kSourceWordBase + 2] : source[kSourceWordBase];
  const float high =
      source_row_hi ? source[kSourceWordBase + 3] : source[kSourceWordBase + 1];
  const uint32_t payload = pack_fp32_to_half2(low, high);
  const int target_row_hi = (lane >> 1) & 1;
  const int source_lane = (lane & 1) |
                          ((((kRound & 1) ^ target_row_hi) & 1) << 1) |
                          (kRound & 12) | (lane & 16);
  return __shfl_sync(0xffffffffU, payload, source_lane);
}

template <int kPhase>
__device__ __forceinline__ void minimum_communication_phase(
    const float (&source)[kSourceValues], uint32_t (&target)[kTargetWords],
    int lane) {
  const uint32_t even = minimum_communication_round<2 * kPhase>(source, lane);
  const uint32_t odd =
      minimum_communication_round<2 * kPhase + 1>(source, lane);
  const bool target_row_hi = ((lane >> 1) & 1) != 0;
  target[2 * kPhase] = target_row_hi ? odd : even;
  target[2 * kPhase + 1] = target_row_hi ? even : odd;
}

// Edge-color the source-lane-to-query-row broadcast graph in 16 rounds.
// Two rounds complete one K4 phase. Static phase destinations keep all words
// in registers and avoid a lane-dependent local-memory index.
__device__ __forceinline__ void convert_minimum_communication(
    const float (&source)[kSourceValues], uint32_t (&target)[kTargetWords],
    int lane) {
  minimum_communication_phase<0>(source, target, lane);
  minimum_communication_phase<1>(source, target, lane);
  minimum_communication_phase<2>(source, target, lane);
  minimum_communication_phase<3>(source, target, lane);
  minimum_communication_phase<4>(source, target, lane);
  minimum_communication_phase<5>(source, target, lane);
  minimum_communication_phase<6>(source, target, lane);
  minimum_communication_phase<7>(source, target, lane);
}

template <bool kMinimumCommunication>
__device__ __forceinline__ void convert_and_store(
    const float* __restrict__ source, uint32_t* __restrict__ target) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int lane = threadIdx.x & 31;
  float source_values[kSourceValues];
#pragma unroll
  for (int index = 0; index < kSourceValues; ++index) {
    source_values[index] = source[global_thread * kSourceValues + index];
  }

  uint32_t target_words[kTargetWords];
  if constexpr (kMinimumCommunication) {
    convert_minimum_communication(source_values, target_words, lane);
  } else {
    convert_reference(source_values, target_words, lane);
  }
#pragma unroll
  for (int index = 0; index < kTargetWords; ++index) {
    target[global_thread * kTargetWords + index] = target_words[index];
  }
}

}  // namespace

extern "C" __global__ __launch_bounds__(
    kThreads, 4) void sm70_hmma_c2a_reference(const float* __restrict__ source,
                                              uint32_t* __restrict__ target) {
  convert_and_store<false>(source, target);
}

extern "C" __global__ __launch_bounds__(kThreads, 4) void sm70_hmma_c2a_mincomm(
    const float* __restrict__ source, uint32_t* __restrict__ target) {
  convert_and_store<true>(source, target);
}

namespace {

struct Arguments {
  int device = 0;
  int blocks = 144;
  int warmup = 20;
  int rounds = 100;
  int launches = 8;
};

struct Timing {
  double mean_us = 0.0;
  double median_us = 0.0;
  double minimum_us = 0.0;
  double maximum_us = 0.0;
};

uint32_t host_pack(float low, float high) {
  const uint32_t low_bits = __half_as_ushort(__float2half_rn(low));
  const uint32_t high_bits = __half_as_ushort(__float2half_rn(high));
  return low_bits | (high_bits << 16);
}

float logical_value(int row, int col) {
  return static_cast<float>(row * 32 + col) / 16.0f;
}

Timing summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  Timing result;
  result.mean_us =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  result.median_us = samples[samples.size() / 2];
  result.minimum_us = samples.front();
  result.maximum_us = samples.back();
  return result;
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments args;
  for (int index = 1; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) {
      std::cerr << "missing value for " << option << '\n';
      std::exit(EXIT_FAILURE);
    }
    const int value = std::atoi(argv[++index]);
    if (option == "--device") {
      args.device = value;
    } else if (option == "--blocks") {
      args.blocks = value;
    } else if (option == "--warmup") {
      args.warmup = value;
    } else if (option == "--rounds") {
      args.rounds = value;
    } else if (option == "--launches") {
      args.launches = value;
    } else {
      std::cerr << "unknown option: " << option << '\n';
      std::exit(EXIT_FAILURE);
    }
  }
  if (args.blocks <= 0 || args.warmup < 0 || args.rounds <= 0 ||
      args.launches <= 0) {
    std::cerr << "invalid benchmark argument\n";
    std::exit(EXIT_FAILURE);
  }
  return args;
}

void print_timing(const Timing& timing) {
  std::cout << std::fixed << std::setprecision(6)
            << "{\"mean_us\":" << timing.mean_us
            << ",\"median_us\":" << timing.median_us
            << ",\"min_us\":" << timing.minimum_us
            << ",\"max_us\":" << timing.maximum_us << '}';
}

}  // namespace

int main(int argc, char** argv) {
  const Arguments args = parse_arguments(argc, argv);
  CUDA_CHECK(cudaSetDevice(args.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, args.device));
  if (properties.major != 7 || properties.minor != 0) {
    std::cerr << "SM70 GPU required, got " << properties.major << '.'
              << properties.minor << '\n';
    return EXIT_FAILURE;
  }

  const size_t threads = static_cast<size_t>(args.blocks) * kThreads;
  std::vector<float> host_source(threads * kSourceValues);
  std::vector<uint32_t> expected(threads * kTargetWords);
  for (size_t thread = 0; thread < threads; ++thread) {
    const int lane = static_cast<int>(thread % kThreads) & 31;
    for (int reg = 0; reg < kSourceValues; ++reg) {
      const int row = accumulator_row(lane, reg);
      const int col = accumulator_col(lane, reg);
      host_source[thread * kSourceValues + reg] = logical_value(row, col);
    }
    const int row = matrix_a_row(lane);
    for (int word = 0; word < kTargetWords; ++word) {
      expected[thread * kTargetWords + word] = host_pack(
          logical_value(row, 2 * word), logical_value(row, 2 * word + 1));
    }
  }

  float* device_source = nullptr;
  uint32_t* device_reference = nullptr;
  uint32_t* device_candidate = nullptr;
  CUDA_CHECK(cudaMalloc(&device_source, host_source.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_reference, expected.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&device_candidate, expected.size() * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(device_source, host_source.data(),
                        host_source.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  auto launch_reference = [&] {
    sm70_hmma_c2a_reference<<<args.blocks, kThreads>>>(device_source,
                                                       device_reference);
  };
  auto launch_candidate = [&] {
    sm70_hmma_c2a_mincomm<<<args.blocks, kThreads>>>(device_source,
                                                     device_candidate);
  };
  for (int index = 0; index < args.warmup; ++index) {
    launch_reference();
    launch_candidate();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint32_t> reference(expected.size());
  std::vector<uint32_t> candidate(expected.size());
  launch_reference();
  launch_candidate();
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(reference.data(), device_reference,
                        reference.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(candidate.data(), device_candidate,
                        candidate.size() * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost));
  size_t reference_mismatches = 0;
  size_t candidate_mismatches = 0;
  size_t pair_mismatches = 0;
  for (size_t index = 0; index < expected.size(); ++index) {
    reference_mismatches += reference[index] != expected[index];
    candidate_mismatches += candidate[index] != expected[index];
    pair_mismatches += candidate[index] != reference[index];
  }

  cudaEvent_t start;
  cudaEvent_t end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));
  auto measure = [&](bool minimum_communication) {
    CUDA_CHECK(cudaEventRecord(start));
    for (int launch = 0; launch < args.launches; ++launch) {
      if (minimum_communication) {
        launch_candidate();
      } else {
        launch_reference();
      }
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    return 1000.0 * milliseconds / args.launches;
  };

  std::vector<double> reference_samples;
  std::vector<double> candidate_samples;
  reference_samples.reserve(args.rounds);
  candidate_samples.reserve(args.rounds);
  for (int round = 0; round < args.rounds; ++round) {
    if ((round & 1) == 0) {
      reference_samples.push_back(measure(false));
      candidate_samples.push_back(measure(true));
    } else {
      candidate_samples.push_back(measure(true));
      reference_samples.push_back(measure(false));
    }
  }
  const Timing reference_timing = summarize(reference_samples);
  const Timing candidate_timing = summarize(candidate_samples);

  cudaFuncAttributes reference_attributes{};
  cudaFuncAttributes candidate_attributes{};
  CUDA_CHECK(
      cudaFuncGetAttributes(&reference_attributes, sm70_hmma_c2a_reference));
  CUDA_CHECK(
      cudaFuncGetAttributes(&candidate_attributes, sm70_hmma_c2a_mincomm));

  const double speedup =
      100.0 * (reference_timing.median_us - candidate_timing.median_us) /
      reference_timing.median_us;
  std::cout << "{\n"
            << "  \"device\":\"" << properties.name << "\",\n"
            << "  \"blocks\":" << args.blocks << ",\n"
            << "  \"quality\":{\"reference_expected_mismatches\":"
            << reference_mismatches
            << ",\"candidate_expected_mismatches\":" << candidate_mismatches
            << ",\"pair_mismatches\":" << pair_mismatches << "},\n"
            << "  \"resources\":{\"reference\":{\"registers\":"
            << reference_attributes.numRegs
            << ",\"local_bytes\":" << reference_attributes.localSizeBytes
            << "},\"candidate\":{\"registers\":" << candidate_attributes.numRegs
            << ",\"local_bytes\":" << candidate_attributes.localSizeBytes
            << "}},\n"
            << "  \"timing_us_per_launch\":{\"reference\":";
  print_timing(reference_timing);
  std::cout << ",\"candidate\":";
  print_timing(candidate_timing);
  std::cout << ",\"candidate_speedup_pct\":" << speedup << "},\n"
            << "  \"gate_passed\":"
            << ((reference_mismatches == 0 && candidate_mismatches == 0 &&
                 pair_mismatches == 0 &&
                 candidate_attributes.localSizeBytes == 0 && speedup >= 5.0)
                    ? "true"
                    : "false")
            << "\n}\n";

  cudaEventDestroy(start);
  cudaEventDestroy(end);
  cudaFree(device_candidate);
  cudaFree(device_reference);
  cudaFree(device_source);
  return pair_mismatches == 0 && candidate_mismatches == 0 ? EXIT_SUCCESS
                                                           : EXIT_FAILURE;
}
