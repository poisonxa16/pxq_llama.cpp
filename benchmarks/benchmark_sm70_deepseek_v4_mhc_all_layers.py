#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B all 85 exact DeepSeek V4 decode mHC calls on SM70."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch
from benchmark_sm70_deepseek_v4_mhc import (
    HC_MULT,
    HC_OUT,
    HIDDEN_SIZE,
    _capture,
    _load_tensor,
    _mhc_dot_from_fp32_stage_tilelang,
    _mhc_post_fp32_stage_tilelang,
    _time_graph,
)

from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    mhc_fused_tilelang,
    mhc_pre_big_fuse_with_norm_tilelang,
)


@dataclass
class Outputs:
    gemm: torch.Tensor
    sqrsum: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    layer_input: torch.Tensor
    residual_fp32: torch.Tensor | None = None


@dataclass
class Call:
    name: str
    fn: torch.Tensor
    scale: torch.Tensor
    base: torch.Tensor
    norm: torch.Tensor
    x: torch.Tensor
    residual: torch.Tensor
    post: torch.Tensor
    comb: torch.Tensor
    baseline: Outputs
    candidate: Outputs


def _make_outputs(with_fp32_stage: bool) -> Outputs:
    return Outputs(
        gemm=torch.empty((8, 1, HC_OUT), device="cuda", dtype=torch.float32),
        sqrsum=torch.empty((8, 1), device="cuda", dtype=torch.float32),
        residual=torch.empty(
            (1, HC_MULT, HIDDEN_SIZE), device="cuda", dtype=torch.float16
        ),
        post=torch.empty((1, HC_MULT), device="cuda", dtype=torch.float32),
        comb=torch.empty((1, HC_MULT * HC_MULT), device="cuda", dtype=torch.float32),
        layer_input=torch.empty((1, HIDDEN_SIZE), device="cuda", dtype=torch.float16),
        residual_fp32=(
            torch.empty((1, HC_MULT, HIDDEN_SIZE), device="cuda", dtype=torch.float32)
            if with_fp32_stage
            else None
        ),
    )


def _pre(call: Call, outputs: Outputs) -> None:
    mhc_pre_big_fuse_with_norm_tilelang(
        outputs.gemm,
        outputs.sqrsum,
        call.scale,
        call.base,
        outputs.residual,
        outputs.post,
        outputs.comb,
        outputs.layer_input,
        call.norm,
        HIDDEN_SIZE,
        1e-6,
        1e-6,
        1e-6,
        2.0,
        20,
        1e-6,
        8,
        HC_MULT,
        use_fp16=True,
    )


def _baseline(call: Call) -> None:
    outputs = call.baseline
    mhc_fused_tilelang(
        call.comb,
        call.residual,
        call.post,
        call.x,
        call.fn.view(HC_OUT, HC_MULT, HIDDEN_SIZE),
        outputs.gemm,
        outputs.sqrsum,
        outputs.residual,
        HC_MULT,
        HIDDEN_SIZE,
        HC_OUT,
        tile_n=2,
        split_k=8,
        use_fp16=True,
    )
    _pre(call, outputs)


def _candidate(call: Call) -> None:
    outputs = call.candidate
    assert outputs.residual_fp32 is not None
    _mhc_post_fp32_stage_tilelang(
        call.comb,
        call.residual,
        call.post,
        call.x,
        outputs.residual_fp32,
        outputs.residual,
        outputs.sqrsum,
        HIDDEN_SIZE,
        HC_MULT,
        n_splits=8,
    )
    _mhc_dot_from_fp32_stage_tilelang(
        outputs.residual_fp32,
        call.fn.view(HC_OUT, HC_MULT, HIDDEN_SIZE),
        outputs.gemm,
        HIDDEN_SIZE,
        HC_MULT,
        HC_OUT,
        tile_n=2,
        n_splits=8,
    )
    _pre(call, outputs)


def _compare(calls: list[Call]) -> dict[str, object]:
    fields = ("residual", "post", "comb", "layer_input")
    summary: dict[str, object] = {}
    for field in fields:
        mismatch_calls = []
        max_abs = 0.0
        total_abs = 0.0
        total_values = 0
        for call in calls:
            expected = getattr(call.baseline, field)
            actual = getattr(call.candidate, field)
            difference = (actual.float() - expected.float()).abs()
            if not torch.equal(actual, expected):
                mismatch_calls.append(call.name)
            max_abs = max(max_abs, float(difference.max().item()))
            total_abs += float(difference.sum().item())
            total_values += difference.numel()
        summary[field] = {
            "equal": not mismatch_calls,
            "max_abs": max_abs,
            "mean_abs": total_abs / total_values,
            "mismatch_calls": mismatch_calls,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--clock-mhz", type=int)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("This benchmark requires NVIDIA V100/SM70.")

    weight_map = json.loads(
        (args.model / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    specs = [(0, "ffn")]
    specs.extend((layer, kind) for layer in range(1, 43) for kind in ("attn", "ffn"))
    calls = []
    for layer, kind in specs:
        prefix = f"layers.{layer}"
        calls.append(
            Call(
                name=f"layer{layer}.{kind}",
                fn=_load_tensor(
                    args.model,
                    weight_map,
                    f"{prefix}.hc_{kind}_fn",
                    torch.float32,
                ),
                scale=_load_tensor(
                    args.model,
                    weight_map,
                    f"{prefix}.hc_{kind}_scale",
                    torch.float32,
                ),
                base=_load_tensor(
                    args.model,
                    weight_map,
                    f"{prefix}.hc_{kind}_base",
                    torch.float32,
                ),
                norm=_load_tensor(
                    args.model,
                    weight_map,
                    f"{prefix}.{kind}_norm.weight",
                    torch.float16,
                ),
                x=torch.randn((1, HIDDEN_SIZE), device="cuda", dtype=torch.float16),
                residual=torch.randn(
                    (1, HC_MULT, HIDDEN_SIZE),
                    device="cuda",
                    dtype=torch.float16,
                ),
                post=torch.sigmoid(
                    torch.randn((1, HC_MULT), device="cuda", dtype=torch.float32)
                ),
                comb=torch.softmax(
                    torch.randn(
                        (1, HC_MULT, HC_MULT),
                        device="cuda",
                        dtype=torch.float32,
                    ),
                    dim=1,
                ),
                baseline=_make_outputs(with_fp32_stage=False),
                candidate=_make_outputs(with_fp32_stage=True),
            )
        )

    def baseline_token() -> None:
        for call in calls:
            _baseline(call)

    def candidate_token() -> None:
        for call in calls:
            _candidate(call)

    baseline_graph = _capture(baseline_token)
    candidate_graph = _capture(candidate_token)
    baseline_graph.replay()
    candidate_graph.replay()
    torch.cuda.synchronize()
    exactness = _compare(calls)

    samples = {"baseline_ms": [], "candidate_ms": []}
    graphs = (
        ("baseline_ms", baseline_graph),
        ("candidate_ms", candidate_graph),
    )
    for repeat in range(args.repeats):
        order = graphs if repeat % 2 == 0 else tuple(reversed(graphs))
        for name, graph in order:
            samples[name].append(_time_graph(graph, args.replays, 1)[0])

    baseline_median = statistics.median(samples["baseline_ms"])
    candidate_median = statistics.median(samples["candidate_ms"])
    result = {
        "contract": {
            "calls": len(calls),
            "hidden_size": HIDDEN_SIZE,
            "hc_mult": HC_MULT,
            "tile_n": 2,
            "n_splits": 8,
            "cuda_graph": True,
            "clock_mhz": args.clock_mhz,
        },
        "samples": samples,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "saving_ms_per_token": baseline_median - candidate_median,
        "speedup_percent": (baseline_median / candidate_median - 1.0) * 100.0,
        "exactness": exactness,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
