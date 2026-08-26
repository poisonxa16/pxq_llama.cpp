#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Profile the exact DeepSeek V4 M=1 mHC decode path on SM70."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors import safe_open

from vllm.model_executor.kernels.mhc.tilelang import (
    _tilelang_hc_prenorm_gemm,
    mhc_fused_post_pre_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_fused_tilelang,
    mhc_post_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    sm70_mhc_dot_from_fp32_stage_tilelang as _mhc_dot_from_fp32_stage_tilelang,
)
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    sm70_mhc_post_fp32_stage_tilelang as _mhc_post_fp32_stage_tilelang,
)

HIDDEN_SIZE = 4096
HC_MULT = 4
HC_OUT = 2 * HC_MULT + HC_MULT * HC_MULT
CALLS_PER_TOKEN = 85


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


def _tensor_result(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    difference = (actual.float() - expected.float()).abs()
    raw = actual.detach().cpu().contiguous().numpy().tobytes()
    return {
        "equal": torch.equal(actual, expected),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument(
        "--path", choices=("fused", "separate", "fp32_stage"), default="fused"
    )
    parser.add_argument("--tile-n", type=int, default=2)
    parser.add_argument("--n-splits", type=int, default=8)
    parser.add_argument(
        "--component",
        choices=("full", "main", "pre", "post", "dot"),
        default="full",
    )
    parser.add_argument("--replays", type=int, default=3000)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--profile-replay", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70.")
    if HC_OUT % args.tile_n != 0:
        raise ValueError(f"tile-n must divide {HC_OUT}")
    if HIDDEN_SIZE % args.n_splits != 0:
        raise ValueError(f"n-splits must divide {HIDDEN_SIZE}")
    if args.path == "fused" and args.component in ("post", "dot"):
        raise ValueError("post and dot components require a staged path")

    weight_map = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    prefix = f"layers.{args.layer}"
    fn = _load_tensor(args.model, weight_map, f"{prefix}.hc_attn_fn", torch.float32)
    hc_scale = _load_tensor(
        args.model, weight_map, f"{prefix}.hc_attn_scale", torch.float32
    )
    hc_base = _load_tensor(
        args.model, weight_map, f"{prefix}.hc_attn_base", torch.float32
    )
    norm_weight = _load_tensor(
        args.model, weight_map, f"{prefix}.attn_norm.weight", torch.float16
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    x = torch.randn((1, HIDDEN_SIZE), device="cuda", dtype=torch.float16)
    residual = torch.randn(
        (1, HC_MULT, HIDDEN_SIZE), device="cuda", dtype=torch.float16
    )
    post_mix = torch.sigmoid(
        torch.randn((1, HC_MULT), device="cuda", dtype=torch.float32)
    )
    comb_mix = torch.softmax(
        torch.randn((1, HC_MULT, HC_MULT), device="cuda", dtype=torch.float32),
        dim=1,
    )

    oracle = mhc_fused_post_pre_tilelang(
        x,
        residual,
        post_mix,
        comb_mix,
        fn,
        hc_scale,
        hc_base,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        norm_weight=norm_weight,
        norm_eps=1e-6,
    )
    torch.cuda.synchronize()

    active_splits = 1 if args.path == "separate" else args.n_splits
    gemm_out_mul = torch.empty(
        (active_splits, 1, HC_OUT), device="cuda", dtype=torch.float32
    )
    gemm_out_sqrsum = torch.empty(
        (active_splits, 1), device="cuda", dtype=torch.float32
    )
    residual_out = torch.empty_like(residual)
    residual_fp32 = torch.empty(
        (1, HC_MULT, HIDDEN_SIZE), device="cuda", dtype=torch.float32
    )
    post_out = torch.empty((1, HC_MULT), device="cuda", dtype=torch.float32)
    comb_out = torch.empty((1, HC_MULT * HC_MULT), device="cuda", dtype=torch.float32)
    layer_input = torch.empty((1, HIDDEN_SIZE), device="cuda", dtype=torch.float16)

    def fused_call() -> None:
        mhc_fused_tilelang(
            comb_mix,
            residual,
            post_mix,
            x,
            fn.view(HC_OUT, HC_MULT, HIDDEN_SIZE),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_out,
            HC_MULT,
            HIDDEN_SIZE,
            HC_OUT,
            tile_n=args.tile_n,
            split_k=args.n_splits,
            use_fp16=True,
        )

    def post_call() -> None:
        mhc_post_tilelang(
            comb_mix,
            residual,
            post_mix,
            x,
            residual_out,
            HC_MULT,
            HIDDEN_SIZE,
            use_fp16=True,
        )

    def dot_call() -> None:
        _tilelang_hc_prenorm_gemm(
            residual_out.view(1, HC_MULT * HIDDEN_SIZE),
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            HIDDEN_SIZE,
            HC_MULT,
        )

    def fp32_stage_call() -> None:
        _mhc_post_fp32_stage_tilelang(
            comb_mix,
            residual,
            post_mix,
            x,
            residual_fp32,
            residual_out,
            gemm_out_sqrsum,
            HIDDEN_SIZE,
            HC_MULT,
            n_splits=args.n_splits,
        )

    def fp32_dot_call() -> None:
        _mhc_dot_from_fp32_stage_tilelang(
            residual_fp32,
            fn.view(HC_OUT, HC_MULT, HIDDEN_SIZE),
            gemm_out_mul,
            HIDDEN_SIZE,
            HC_MULT,
            HC_OUT,
            tile_n=args.tile_n,
            n_splits=args.n_splits,
        )

    def pre_call() -> None:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_out,
            post_out,
            comb_out,
            layer_input,
            norm_weight,
            HIDDEN_SIZE,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            1e-6,
            active_splits,
            HC_MULT,
            use_fp16=True,
        )

    if args.path == "fused":
        main_call = fused_call
        post_component_call = None
        dot_component_call = None
    elif args.path == "separate":
        main_call = lambda: (post_call(), dot_call())
        post_component_call = post_call
        dot_component_call = dot_call
    else:
        main_call = lambda: (fp32_stage_call(), fp32_dot_call())
        post_component_call = fp32_stage_call
        dot_component_call = fp32_dot_call

    main_call()
    pre_call()
    torch.cuda.synchronize()
    eager_outputs = (
        residual_out.clone(),
        post_out.view(1, HC_MULT, 1).clone(),
        comb_out.view(1, HC_MULT, HC_MULT).clone(),
        layer_input.clone(),
    )

    if args.component == "full":
        call = lambda: (main_call(), pre_call())
    elif args.component == "main":
        call = main_call
    elif args.component == "pre":
        call = pre_call
    elif args.component == "post":
        assert post_component_call is not None
        call = post_component_call
    else:
        assert dot_component_call is not None
        call = dot_component_call

    graph = _capture(call)
    if args.profile_replay:
        torch.cuda.cudart().cudaProfilerStart()
        graph.replay()
        torch.cuda.cudart().cudaProfilerStop()
        torch.cuda.synchronize()

    samples = _time_graph(graph, args.replays, args.repeats)
    graph_outputs = (
        residual_out,
        post_out.view(1, HC_MULT, 1),
        comb_out.view(1, HC_MULT, HC_MULT),
        layer_input,
    )
    names = ("residual", "post_mix", "comb_mix", "layer_input")
    exactness = {
        name: _tensor_result(actual, expected)
        for name, actual, expected in zip(names, graph_outputs, oracle, strict=True)
    }
    graph_exactness = {
        name: _tensor_result(actual, expected)
        for name, actual, expected in zip(
            names, graph_outputs, eager_outputs, strict=True
        )
    }

    median_ms = statistics.median(samples)
    result = {
        "contract": {
            "layer": args.layer,
            "num_tokens": 1,
            "hidden_size": HIDDEN_SIZE,
            "hc_mult": HC_MULT,
            "hc_out": HC_OUT,
            "path": args.path,
            "tile_n": args.tile_n,
            "n_splits": active_splits,
            "component": args.component,
            "calls_per_token": CALLS_PER_TOKEN,
            "cuda_graph": True,
        },
        "samples_ms": samples,
        "median_ms": median_ms,
        "projected_ms_per_token": median_ms * CALLS_PER_TOKEN,
        "exactness_vs_runtime_oracle": exactness,
        "exactness_graph_vs_eager": graph_exactness,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
