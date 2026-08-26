# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark bounded-memory SM70 AWQ long-prefill weight expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from benchmarks import benchmark_sm70_awq_verifier_micro as awq_bench
from vllm.model_executor.layers.quantization.awq import _awq_exact_f16_weight


def _time_cuda_call(
    fn: Any,
    *,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> float:
    for _ in range(warmup):
        fn()
    torch.accelerator.synchronize(device)
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return float(torch.tensor(samples).median().item())


def _run_case(
    *,
    model: Path,
    case: awq_bench.BenchCase,
    device: torch.device,
    tp_size: int,
    tp_rank: int,
    group_size: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    qweight, scales, qzeros = awq_bench._load_case_awq(
        model,
        case,
        tp_size,
        tp_rank,
        group_size,
        device,
    )
    tm_weight, tm_scales, meta = torch.ops._C.awq_sm70_prepare(
        qweight,
        scales,
        qzeros,
        group_size,
        False,
    )
    exact_weight = _awq_exact_f16_weight(qweight, scales, qzeros, group_size)
    workspace = torch.empty_like(exact_weight)
    x = awq_bench._make_input(case.m, qweight.shape[0], device)
    tm_output = torch.empty(
        (case.m, qweight.shape[1] * 8),
        dtype=torch.float16,
        device=device,
    )
    dense_output = torch.empty_like(tm_output)
    workspace_output = torch.empty_like(tm_output)

    def run_tm() -> None:
        torch.ops._C.awq_gemm_sm70_out(
            tm_output,
            x,
            tm_weight,
            tm_scales,
            group_size,
            int(meta[0]),
            int(meta[1]),
            False,
        )

    def run_dense() -> None:
        torch.mm(x, exact_weight, out=dense_output)

    def run_dequant() -> None:
        torch.ops._C.awq_sm70_dequantize_out(
            workspace,
            tm_weight,
            tm_scales,
            group_size,
        )

    def run_workspace() -> None:
        run_dequant()
        torch.mm(x, workspace, out=workspace_output)

    run_tm()
    run_dense()
    run_workspace()
    torch.accelerator.synchronize(device)

    dequant_equal = torch.equal(workspace, exact_weight)
    output_equal = torch.equal(workspace_output, dense_output)
    tm_equal = torch.equal(tm_output, dense_output)
    result = {
        "label": case.label,
        "count": case.count,
        "m": case.m,
        "k": int(qweight.shape[0]),
        "n": int(qweight.shape[1] * 8),
        "workspace_mib": workspace.numel() * workspace.element_size() / 1024**2,
        "dequant_equal": dequant_equal,
        "dequant_mismatches": int(torch.count_nonzero(workspace != exact_weight)),
        "dequant_max_abs_error": float(
            (workspace.float() - exact_weight.float()).abs().max().item()
        ),
        "output_equal": output_equal,
        "output_mismatches": int(torch.count_nonzero(workspace_output != dense_output)),
        "output_max_abs_error": float(
            (workspace_output.float() - dense_output.float()).abs().max().item()
        ),
        "turbomind_equal": tm_equal,
        "turbomind_mismatches": int(torch.count_nonzero(tm_output != dense_output)),
        "turbomind_max_abs_error": float(
            (tm_output.float() - dense_output.float()).abs().max().item()
        ),
        "turbomind_ms": _time_cuda_call(
            run_tm,
            device=device,
            warmup=warmup,
            iterations=iterations,
        ),
        "dense_ms": _time_cuda_call(
            run_dense,
            device=device,
            warmup=warmup,
            iterations=iterations,
        ),
        "dequant_ms": _time_cuda_call(
            run_dequant,
            device=device,
            warmup=warmup,
            iterations=iterations,
        ),
        "workspace_ms": _time_cuda_call(
            run_workspace,
            device=device,
            warmup=warmup,
            iterations=iterations,
        ),
    }
    result["workspace_vs_turbomind_speedup"] = (
        result["turbomind_ms"] / result["workspace_ms"]
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    device = torch.device(args.device)
    awq_bench._require_sm70(device)
    rows = []
    for case in awq_bench._default_cases(args.m):
        row = _run_case(
            model=args.model,
            case=case,
            device=device,
            tp_size=args.tp_size,
            tp_rank=args.tp_rank,
            group_size=args.group_size,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        rows.append(row)
        torch.accelerator.empty_cache()

    weighted = {
        name: sum(row["count"] * row[name] for row in rows)
        for name in ("turbomind_ms", "dense_ms", "dequant_ms", "workspace_ms")
    }
    result = {
        "all_dequant_equal": all(row["dequant_equal"] for row in rows),
        "all_output_equal": all(row["output_equal"] for row in rows),
        "max_workspace_mib": max(row["workspace_mib"] for row in rows),
        "weighted": weighted,
        "weighted_workspace_vs_turbomind_speedup": (
            weighted["turbomind_ms"] / weighted["workspace_ms"]
        ),
        "weighted_workspace_over_dense_ms": (
            weighted["workspace_ms"] - weighted["dense_ms"]
        ),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    return 0 if result["all_dequant_equal"] and result["all_output_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
