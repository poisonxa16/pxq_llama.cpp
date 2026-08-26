#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen a last-CTA DeepSeek V4 GEMV + sqrt-softplus top-k fusion."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

import vllm._custom_ops as ops
from vllm.models.deepseek_v4.sm70.gemv import (
    _sm70_dsv4_fp16_gemv_kernel,
)
from vllm.triton_utils import tl, triton

K = 4096
NUM_EXPERTS = 256
TOP_K = 6
BLOCK_K = 1024


@triton.jit
def _sm70_dsv4_fused_router_kernel(
    x_ptr,
    weight_ptr,
    correction_bias_ptr,
    logits_scratch_ptr,
    counter_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    source_rows_ptr,
    K: tl.constexpr,
    NUM_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
    NUM_PROGRAMS: tl.constexpr,
    ROUND_TO_FP16: tl.constexpr,
):
    program = tl.program_id(0)
    rows = program * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
    row_mask = rows < NUM_EXPERTS
    k_offsets = tl.arange(0, BLOCK_K)
    accumulator = tl.zeros((ROWS_PER_PROGRAM,), dtype=tl.float32)
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(x_ptr + block_start + k_offsets)
        weight_offsets = rows[:, None] * K + block_start + k_offsets[None, :]
        weight = tl.load(weight_ptr + weight_offsets, mask=row_mask[:, None])
        accumulator += tl.sum(weight.to(tl.float32) * x[None, :], axis=1)

    if ROUND_TO_FP16:
        accumulator = accumulator.to(tl.float16).to(tl.float32)
    tl.store(logits_scratch_ptr + rows, accumulator, mask=row_mask)

    previous = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    if previous == NUM_PROGRAMS - 1:
        expert_offsets = tl.arange(0, NUM_EXPERTS)
        logits = tl.load(logits_scratch_ptr + expert_offsets)
        softplus = tl.where(
            logits > 20.0,
            logits,
            tl.log(1.0 + tl.exp(logits)),
        )
        unbiased_scores = tl.sqrt(softplus)
        choice_scores = unbiased_scores + tl.load(correction_bias_ptr + expert_offsets)

        for topk_index in tl.static_range(0, TOP_K):
            expert = tl.argmax(choice_scores, axis=0, tie_break_left=True)
            selected = tl.sum(
                tl.where(expert_offsets == expert, unbiased_scores, 0.0), axis=0
            )
            tl.store(topk_weights_ptr + topk_index, selected)
            tl.store(topk_ids_ptr + topk_index, expert)
            tl.store(source_rows_ptr + topk_index, topk_index)
            choice_scores = tl.where(
                expert_offsets == expert, -float("inf"), choice_scores
            )

        topk_offsets = tl.arange(0, 8)
        topk_mask = topk_offsets < TOP_K
        selected_scores = tl.load(
            topk_weights_ptr + topk_offsets, mask=topk_mask, other=0.0
        )
        scale = 1.5 / tl.sum(selected_scores, axis=0)
        tl.store(
            topk_weights_ptr + topk_offsets,
            selected_scores * scale,
            mask=topk_mask,
        )
        tl.store(counter_ptr, 0)


def _load_tensor(
    model: Path,
    weight_map: dict[str, str],
    key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    with safe_open(model / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device="cuda", dtype=dtype)


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    graph.replay()
    stream.synchronize()
    return graph


def _time_graph(
    graph: torch.cuda.CUDAGraph,
    replays: int,
    repeats: int,
) -> list[float]:
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / replays)
    return samples


def _make_outputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((1, TOP_K), device="cuda", dtype=torch.float32),
        torch.empty((1, TOP_K), device="cuda", dtype=torch.int32),
        torch.empty((1, TOP_K), device="cuda", dtype=torch.int32),
    )


def _compare(
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, object]:
    candidate_weights, candidate_ids, candidate_rows = candidate
    reference_weights, reference_ids, reference_rows = reference
    weight_diff = (candidate_weights - reference_weights).abs()
    return {
        "ids_equal": torch.equal(candidate_ids, reference_ids),
        "source_rows_equal": torch.equal(candidate_rows, reference_rows),
        "weights_equal": torch.equal(candidate_weights, reference_weights),
        "weights_max_abs": float(weight_diff.max().item()),
        "weights_mean_abs": float(weight_diff.mean().item()),
        "candidate_ids": candidate_ids.cpu().tolist(),
        "reference_ids": reference_ids.cpu().tolist(),
    }


def _compare_layers(
    candidates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    references: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    layer_indices: list[int],
) -> dict[str, object]:
    comparisons = [
        _compare(candidate, reference)
        for candidate, reference in zip(candidates, references, strict=True)
    ]
    weight_differences = torch.cat(
        [
            (candidate[0] - reference[0]).abs().flatten()
            for candidate, reference in zip(candidates, references, strict=True)
        ]
    )
    return {
        "ids_equal": all(bool(result["ids_equal"]) for result in comparisons),
        "source_rows_equal": all(
            bool(result["source_rows_equal"]) for result in comparisons
        ),
        "weights_equal": all(bool(result["weights_equal"]) for result in comparisons),
        "weights_max_abs": float(weight_differences.max().item()),
        "weights_mean_abs": float(weight_differences.mean().item()),
        "id_mismatch_layers": [
            layer
            for layer, result in zip(layer_indices, comparisons, strict=True)
            if not result["ids_equal"]
        ],
        "source_row_mismatch_layers": [
            layer
            for layer, result in zip(layer_indices, comparisons, strict=True)
            if not result["source_rows_equal"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--programs", type=int, nargs="+", default=[32, 64, 80, 128])
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--all-regular-layers", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70.")

    weight_map = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    layer_indices = list(range(3, 43)) if args.all_regular_layers else [3]
    weights = [
        _load_tensor(
            args.model,
            weight_map,
            f"layers.{layer}.ffn.gate.weight",
            torch.float16,
        )
        for layer in layer_indices
    ]
    biases = [
        _load_tensor(
            args.model,
            weight_map,
            f"layers.{layer}.ffn.gate.bias",
            torch.float32,
        )
        for layer in layer_indices
    ]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    inputs = [
        torch.randn((1, K), device="cuda", dtype=torch.float16) for _ in layer_indices
    ]

    separate_logits = [
        torch.empty((1, NUM_EXPERTS), device="cuda", dtype=torch.float32)
        for _ in layer_indices
    ]
    separate_outputs = [_make_outputs() for _ in layer_indices]

    def separate_call() -> None:
        for x, weight, bias, logits, outputs in zip(
            inputs,
            weights,
            biases,
            separate_logits,
            separate_outputs,
            strict=True,
        ):
            _sm70_dsv4_fp16_gemv_kernel[(NUM_EXPERTS,)](
                x,
                weight,
                logits,
                K=K,
                BLOCK_K=BLOCK_K,
                num_warps=4,
            )
            ops.topk_hash_softplus_sqrt(
                *outputs,
                logits,
                True,
                1.5,
                bias,
                None,
                None,
            )

    separate_graph = _capture(separate_call)
    separate_samples = _time_graph(separate_graph, args.replays, args.repeats)
    separate_graph.replay()
    torch.cuda.synchronize()
    separate_reference = [
        tuple(output.clone() for output in outputs) for outputs in separate_outputs
    ]

    cublas_logits_half = [
        torch.empty((1, NUM_EXPERTS), device="cuda", dtype=torch.float16)
        for _ in layer_indices
    ]
    cublas_logits = [torch.empty_like(logits) for logits in separate_logits]
    cublas_outputs = [_make_outputs() for _ in layer_indices]

    def cublas_call() -> None:
        for x, weight, bias, logits_half, logits, outputs in zip(
            inputs,
            weights,
            biases,
            cublas_logits_half,
            cublas_logits,
            cublas_outputs,
            strict=True,
        ):
            torch.mm(x, weight.T, out=logits_half)
            logits.copy_(logits_half)
            ops.topk_hash_softplus_sqrt(
                *outputs,
                logits,
                True,
                1.5,
                bias,
                None,
                None,
            )

    cublas_graph = _capture(cublas_call)
    cublas_samples = _time_graph(cublas_graph, args.replays, args.repeats)
    cublas_graph.replay()
    torch.cuda.synchronize()
    cublas_reference = [
        tuple(output.clone() for output in outputs) for outputs in cublas_outputs
    ]

    variants = []
    for programs in args.programs:
        rows_per_program = math.ceil(NUM_EXPERTS / programs)
        for round_to_fp16 in (False, True):
            scratches = [
                torch.empty((NUM_EXPERTS,), device="cuda", dtype=torch.float32)
                for _ in layer_indices
            ]
            counters = [
                torch.zeros((1,), device="cuda", dtype=torch.int32)
                for _ in layer_indices
            ]
            outputs_list = [_make_outputs() for _ in layer_indices]

            def fused_call(
                programs: int = programs,
                rows_per_program: int = rows_per_program,
                round_to_fp16: bool = round_to_fp16,
                scratches: list[torch.Tensor] = scratches,
                counters: list[torch.Tensor] = counters,
                outputs_list: list[
                    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                ] = outputs_list,
            ) -> None:
                for x, weight, bias, scratch, counter, outputs in zip(
                    inputs,
                    weights,
                    biases,
                    scratches,
                    counters,
                    outputs_list,
                    strict=True,
                ):
                    _sm70_dsv4_fused_router_kernel[(programs,)](
                        x,
                        weight,
                        bias,
                        scratch,
                        counter,
                        *outputs,
                        K=K,
                        NUM_EXPERTS=NUM_EXPERTS,
                        TOP_K=TOP_K,
                        BLOCK_K=BLOCK_K,
                        ROWS_PER_PROGRAM=rows_per_program,
                        NUM_PROGRAMS=programs,
                        ROUND_TO_FP16=round_to_fp16,
                        num_warps=4,
                    )

            graph = _capture(fused_call)
            samples = _time_graph(graph, args.replays, args.repeats)
            graph.replay()
            torch.cuda.synchronize()
            candidate = [
                tuple(output.clone() for output in outputs) for outputs in outputs_list
            ]
            variants.append(
                {
                    "programs": programs,
                    "rows_per_program": rows_per_program,
                    "round_to_fp16": round_to_fp16,
                    "samples_ms": samples,
                    "median_ms": statistics.median(samples),
                    "vs_separate": _compare_layers(
                        candidate, separate_reference, layer_indices
                    ),
                    "vs_cublas": _compare_layers(
                        candidate, cublas_reference, layer_indices
                    ),
                }
            )

    separate_median = statistics.median(separate_samples)
    cublas_median = statistics.median(cublas_samples)
    for variant in variants:
        variant_median = float(variant["median_ms"])
        variant["speedup_vs_separate"] = separate_median / variant_median
        measured_saving = separate_median - variant_median
        variant["measured_saving_ms_per_graph"] = measured_saving
        variant["projected_saving_ms_per_token"] = measured_saving * (
            1 if args.all_regular_layers else 40
        )

    payload = {
        "contract": {
            "model": str(args.model),
            "shape_nk": [NUM_EXPERTS, K],
            "layers": layer_indices,
            "top_k": TOP_K,
            "renormalize": True,
            "routed_scaling_factor": 1.5,
            "cuda_graph": True,
            "seed": args.seed,
            "replays": args.replays,
            "repeats": args.repeats,
        },
        "separate_candidate": {
            "samples_ms": separate_samples,
            "median_ms": separate_median,
            "vs_cublas": _compare_layers(
                separate_reference, cublas_reference, layer_indices
            ),
        },
        "cublas_reference": {
            "samples_ms": cublas_samples,
            "median_ms": cublas_median,
        },
        "variants": variants,
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
