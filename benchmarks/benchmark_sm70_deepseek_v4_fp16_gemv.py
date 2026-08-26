# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen fixed-shape SM70 FP16 GEMV kernels with real DeepSeek V4 weights."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.triton_utils import tl, triton


@triton.jit
def _sm70_fp16_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(x_ptr + block_start + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + row * K + block_start + offsets).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(out_ptr + row, acc)


@dataclass(frozen=True)
class Case:
    name: str
    keys: tuple[str, ...]
    output_dtype: torch.dtype
    cast_to_fp32: bool
    layer_count: int


CASES = (
    Case("router", ("layers.3.ffn.gate.weight",), torch.float16, True, 43),
    Case(
        "indexer_weights",
        ("layers.2.attn.indexer.weights_proj.weight",),
        torch.float16,
        False,
        21,
    ),
    Case(
        "c4_indexer_compressor",
        (
            "layers.2.attn.indexer.compressor.wkv.weight",
            "layers.2.attn.indexer.compressor.wgate.weight",
        ),
        torch.float32,
        False,
        21,
    ),
    Case(
        "c4_main_compressor",
        (
            "layers.2.attn.compressor.wkv.weight",
            "layers.2.attn.compressor.wgate.weight",
        ),
        torch.float32,
        False,
        21,
    ),
    Case(
        "c128_main_compressor",
        (
            "layers.3.attn.compressor.wkv.weight",
            "layers.3.attn.compressor.wgate.weight",
        ),
        torch.float32,
        False,
        20,
    ),
)


def _load_weight(
    model: Path, weight_map: dict[str, str], keys: tuple[str, ...]
) -> torch.Tensor:
    tensors = []
    for key in keys:
        with safe_open(model / weight_map[key], framework="pt", device="cpu") as handle:
            tensors.append(handle.get_tensor(key))
    return torch.cat(tensors, dim=0).to(device="cuda", dtype=torch.float16)


def _load_tensor(
    model: Path, weight_map: dict[str, str], key: str, dtype: torch.dtype
) -> torch.Tensor:
    with safe_open(model / weight_map[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).to(device="cuda", dtype=dtype)


def _capture(call: Callable[[], None]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def _parallel_call(
    calls: list[Callable[[], None]],
    streams: list[torch.cuda.Stream],
    start_event: torch.cuda.Event,
    done_events: list[torch.cuda.Event],
) -> None:
    origin = torch.cuda.current_stream()
    start_event.record(origin)
    for call, stream, done_event in zip(calls, streams, done_events):
        stream.wait_event(start_event)
        with torch.cuda.stream(stream):
            call()
            done_event.record(stream)
    for done_event in done_events:
        origin.wait_event(done_event)


def _capture_parallel(calls: list[Callable[[], None]]) -> torch.cuda.CUDAGraph:
    streams = [torch.cuda.Stream() for _ in calls]
    start_event = torch.cuda.Event()
    done_events = [torch.cuda.Event() for _ in calls]
    return _capture(lambda: _parallel_call(calls, streams, start_event, done_events))


def _time_graph(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / replays


def _baseline_call(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    raw_out: torch.Tensor,
    final_out: torch.Tensor,
    output_dtype: torch.dtype,
    cast_to_fp32: bool,
) -> None:
    if output_dtype == torch.float16:
        torch.mm(x, weight_t, out=raw_out)
        if cast_to_fp32:
            final_out.copy_(raw_out)
    else:
        torch.mm(x, weight_t, out=final_out, out_dtype=torch.float32)


def _candidate_call(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    block_k: int,
    num_warps: int,
) -> None:
    _sm70_fp16_gemv_kernel[(weight.shape[0],)](
        x,
        weight,
        out,
        K=weight.shape[1],
        BLOCK_K=block_k,
        num_warps=num_warps,
    )


def _top6_match(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    return torch.equal(
        torch.topk(reference, 6, dim=-1).indices,
        torch.topk(candidate, 6, dim=-1).indices,
    )


def _benchmark_case(
    case: Case,
    weight: torch.Tensor,
    block_k_values: list[int],
    num_warps_values: list[int],
    replays: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn((1, weight.shape[1]), device="cuda", dtype=torch.float16)
    weight_t = weight.T
    raw_out = torch.empty((1, weight.shape[0]), device="cuda", dtype=torch.float16)
    final_dtype = torch.float32 if case.cast_to_fp32 else case.output_dtype
    reference = torch.empty((1, weight.shape[0]), device="cuda", dtype=final_dtype)
    if not case.cast_to_fp32:
        raw_out = reference

    baseline_graph = _capture(
        lambda: _baseline_call(
            x,
            weight_t,
            raw_out,
            reference,
            case.output_dtype,
            case.cast_to_fp32,
        )
    )
    baseline_samples = [_time_graph(baseline_graph, replays) for _ in range(repeats)]
    baseline_median = statistics.median(baseline_samples)
    baseline_graph.replay()
    torch.cuda.synchronize()
    reference_value = reference.clone()

    variants = []
    for block_k in block_k_values:
        if weight.shape[1] % block_k != 0:
            continue
        for num_warps in num_warps_values:
            candidate = torch.empty_like(reference)
            candidate_graph = _capture(
                lambda candidate=candidate,
                block_k=block_k,
                num_warps=num_warps: _candidate_call(
                    x, weight, candidate, block_k, num_warps
                )
            )
            samples = [_time_graph(candidate_graph, replays) for _ in range(repeats)]
            candidate_graph.replay()
            torch.cuda.synchronize()
            diff = (candidate - reference_value).abs()
            variants.append(
                {
                    "block_k": block_k,
                    "num_warps": num_warps,
                    "samples_ms": samples,
                    "median_ms": statistics.median(samples),
                    "speedup": baseline_median / statistics.median(samples),
                    "max_abs": float(diff.max().item()),
                    "mean_abs": float(diff.mean().item()),
                    "top6_match": (
                        _top6_match(reference_value, candidate)
                        if case.name == "router"
                        else None
                    ),
                }
            )

    best = min(variants, key=lambda item: float(item["median_ms"]))
    return {
        "name": case.name,
        "shape_nk": [weight.shape[0], weight.shape[1]],
        "baseline_output_dtype": str(case.output_dtype),
        "final_output_dtype": str(final_dtype),
        "cast_to_fp32": case.cast_to_fp32,
        "layer_count": case.layer_count,
        "baseline_samples_ms": baseline_samples,
        "baseline_median_ms": baseline_median,
        "variants": variants,
        "best": best,
        "projected_service_saving_ms": case.layer_count
        * (baseline_median - float(best["median_ms"])),
    }


def _benchmark_c4_parallel(
    model: Path,
    weight_map: dict[str, str],
    case_results: list[dict[str, object]],
    replays: int,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    c4_names = (
        "indexer_weights",
        "c4_indexer_compressor",
        "c4_main_compressor",
    )
    case_by_name = {case.name: case for case in CASES}
    result_by_name = {str(result["name"]): result for result in case_results}
    cases = [case_by_name[name] for name in c4_names]
    weights = [_load_weight(model, weight_map, case.keys) for case in cases]

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn((1, 4096), device="cuda", dtype=torch.float16)
    baseline_calls = []
    candidate_calls = []
    for case, weight in zip(cases, weights):
        final_dtype = torch.float32 if case.cast_to_fp32 else case.output_dtype
        raw_out = torch.empty((1, weight.shape[0]), device="cuda", dtype=torch.float16)
        final_out = torch.empty((1, weight.shape[0]), device="cuda", dtype=final_dtype)
        if not case.cast_to_fp32:
            raw_out = final_out
        baseline_calls.append(
            lambda case=case,
            weight=weight,
            raw_out=raw_out,
            final_out=final_out: _baseline_call(
                x,
                weight.T,
                raw_out,
                final_out,
                case.output_dtype,
                case.cast_to_fp32,
            )
        )

        best = result_by_name[case.name]["best"]
        assert isinstance(best, dict)
        candidate_out = torch.empty_like(final_out)
        candidate_calls.append(
            lambda weight=weight,
            candidate_out=candidate_out,
            block_k=int(best["block_k"]),
            num_warps=int(best["num_warps"]): _candidate_call(
                x, weight, candidate_out, block_k, num_warps
            )
        )

    baseline_graph = _capture_parallel(baseline_calls)
    candidate_graph = _capture_parallel(candidate_calls)
    baseline_samples = [_time_graph(baseline_graph, replays) for _ in range(repeats)]
    candidate_samples = [_time_graph(candidate_graph, replays) for _ in range(repeats)]
    baseline_median = statistics.median(baseline_samples)
    candidate_median = statistics.median(candidate_samples)
    return {
        "cases": list(c4_names),
        "baseline_samples_ms": baseline_samples,
        "candidate_samples_ms": candidate_samples,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "speedup": baseline_median / candidate_median,
        "projected_21_layer_saving_ms": 21 * (baseline_median - candidate_median),
    }


def _router_quality(
    model: Path,
    weight_map: dict[str, str],
    router_result: dict[str, object],
    seeds: list[int],
) -> dict[str, object]:
    case = next(case for case in CASES if case.name == "router")
    weight = _load_weight(model, weight_map, case.keys)
    bias = _load_tensor(model, weight_map, "layers.3.ffn.gate.bias", torch.float32)
    best = router_result["best"]
    assert isinstance(best, dict)
    block_k = int(best["block_k"])
    num_warps = int(best["num_warps"])

    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x = torch.randn((1, weight.shape[1]), device="cuda", dtype=torch.float16)
        raw_reference = torch.empty(
            (1, weight.shape[0]), device="cuda", dtype=torch.float16
        )
        reference = torch.empty(
            (1, weight.shape[0]), device="cuda", dtype=torch.float32
        )
        candidate = torch.empty_like(reference)
        _baseline_call(
            x,
            weight.T,
            raw_reference,
            reference,
            torch.float16,
            True,
        )
        _candidate_call(x, weight, candidate, block_k, num_warps)
        torch.cuda.synchronize()

        reference_scores = torch.nn.functional.softplus(reference).sqrt()
        candidate_scores = torch.nn.functional.softplus(candidate).sqrt()
        reference_ids = torch.topk(reference_scores + bias, 6, dim=-1).indices
        candidate_ids = torch.topk(candidate_scores + bias, 6, dim=-1).indices
        reference_weights = reference_scores.gather(1, reference_ids)
        candidate_weights = candidate_scores.gather(1, reference_ids)
        reference_weights /= reference_weights.sum(dim=-1, keepdim=True)
        candidate_weights /= candidate_weights.sum(dim=-1, keepdim=True)
        choice_values = torch.topk(reference_scores + bias, 7, dim=-1).values
        rows.append(
            {
                "seed": seed,
                "top6_match": torch.equal(reference_ids, candidate_ids),
                "logit_max_abs": float((reference - candidate).abs().max().item()),
                "score_max_abs": float(
                    (reference_scores - candidate_scores).abs().max().item()
                ),
                "selected_weight_max_abs": float(
                    (reference_weights - candidate_weights).abs().max().item()
                ),
                "reference_choice_margin_6v7": float(
                    (choice_values[:, 5] - choice_values[:, 6]).item()
                ),
            }
        )

    return {
        "seeds": seeds,
        "all_top6_match": all(bool(row["top6_match"]) for row in rows),
        "max_logit_abs": max(float(row["logit_max_abs"]) for row in rows),
        "max_score_abs": max(float(row["score_max_abs"]) for row in rows),
        "max_selected_weight_abs": max(
            float(row["selected_weight_max_abs"]) for row in rows
        ),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--block-k", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--replays", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--quality-seeds",
        type=int,
        nargs="+",
        default=list(range(20260802, 20260818)),
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires an NVIDIA V100/SM70 GPU.")

    weight_map = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    results = []
    for case in CASES:
        weight = _load_weight(args.model, weight_map, case.keys)
        results.append(
            _benchmark_case(
                case,
                weight,
                args.block_k,
                args.num_warps,
                args.replays,
                args.repeats,
                args.seed,
            )
        )
        del weight

    c4_parallel = _benchmark_c4_parallel(
        args.model,
        weight_map,
        results,
        args.replays,
        args.repeats,
        args.seed,
    )
    router_result = next(result for result in results if result["name"] == "router")
    router_quality = _router_quality(
        args.model, weight_map, router_result, args.quality_seeds
    )

    payload = {
        "contract": {
            "model": str(args.model),
            "m": 1,
            "k": 4096,
            "input_dtype": "torch.float16",
            "candidate_output_dtype": "torch.float32",
            "cuda_graph": True,
        },
        "results": results,
        "c4_parallel": c4_parallel,
        "router_quality": router_quality,
        "projected_service_saving_ms": sum(
            float(result["projected_service_saving_ms"]) for result in results
        ),
    }
    encoded = json.dumps(payload, indent=2)
    print(encoded)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
