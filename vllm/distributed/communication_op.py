# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_reduce_sum2(
    input_a: torch.Tensor, input_b: torch.Tensor
) -> torch.Tensor:
    """All-reduce the elementwise sum of two tensors across model parallel."""
    return get_tp_group().all_reduce_sum2(input_a, input_b)


def tensor_model_parallel_sm70_awq_mlp_down_tile_all_reduce(
    input_: torch.Tensor,
) -> torch.Tensor:
    """Try the SM70 AWQ MLP down-proj tile-runtime all-reduce lane."""
    return get_tp_group().sm70_awq_mlp_down_tile_all_reduce(input_)


def tensor_model_parallel_sm70_awq_mlp_down_tile_gemm_reduce(
    input_: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    group_size: int,
    k_ld: int,
    q_ld: int,
) -> torch.Tensor:
    """Run the SM70 AWQ down-proj GEMM with tile-ready TP2 reduction."""
    return get_tp_group().sm70_awq_mlp_down_tile_gemm_reduce(
        input_, qweight, scales, group_size, k_ld, q_ld
    )


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, dim: int = -1
) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> torch.Tensor | None:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: dict[Any, torch.Tensor | Any] | None = None, src: int = 0
):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
