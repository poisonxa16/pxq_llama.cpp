# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.warmup import awq_sm70_warmup as warmup


def _grouped_fp8_layer() -> nn.Module:
    layer = nn.Module()
    layer.sm70_fp8_turbomind = True
    layer.sm70_fp8_bmm = True
    layer.sm70_fp8_bmm_output_size = 64
    layer.sm70_fp8_k_ld = 128
    layer.sm70_fp8_q_ld = 64
    layer.output_size_per_partition = 192
    layer.weight = nn.Parameter(
        torch.empty((3, 128, 64), dtype=torch.uint8), requires_grad=False
    )
    layer.weight_scale_inv = nn.Parameter(
        torch.empty((3, 1, 64), dtype=torch.float32), requires_grad=False
    )
    return layer


def test_fp8_warmup_discovers_grouped_bmm_by_per_group_shape():
    layer = _grouped_fp8_layer()
    model = nn.Sequential(layer)

    discovered = list(warmup._iter_unique_fp8_dense_layers(model))

    assert discovered == [(layer, False)]


def test_fp8_warmup_matches_grouped_bmm_runtime_slice(monkeypatch):
    layer = _grouped_fp8_layer()
    calls = []
    monkeypatch.setattr(torch.ops._C, "fp8_gemm_sm70_out_meta", object(), raising=False)

    def record_call(out, x, weight, scales, group_size, k_ld, q_ld, gated_silu):
        calls.append(
            SimpleNamespace(
                out_shape=tuple(out.shape),
                x_shape=tuple(x.shape),
                weight_shape=tuple(weight.shape),
                scale_shape=tuple(scales.shape),
                group_size=group_size,
                k_ld=k_ld,
                q_ld=q_ld,
                gated_silu=gated_silu,
            )
        )

    monkeypatch.setattr(warmup.sm70_ops, "fp8_gemm_sm70_out", record_call)

    count = warmup._warmup_fp8_dense_layers([(layer, False)], [1, 4])

    assert count == 2
    assert [call.out_shape for call in calls] == [(1, 64), (4, 64)]
    assert [call.x_shape for call in calls] == [(1, 128), (4, 128)]
    assert all(call.weight_shape == (128, 64) for call in calls)
    assert all(call.scale_shape == (1, 64) for call in calls)
    assert all(call.group_size == 128 for call in calls)
    assert all(call.k_ld == 128 and call.q_ld == 64 for call in calls)
    assert all(not call.gated_silu for call in calls)
