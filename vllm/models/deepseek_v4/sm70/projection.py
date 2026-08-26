# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""SM70 inverse-RoPE and grouped output projection helpers."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _sm70_inverse_rope_kernel(
    x_ptr,
    out_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    nope_dim: tl.constexpr,
    half_rope: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    base = token_idx * num_heads * head_dim + head_idx * head_dim

    offsets = tl.arange(0, head_dim)
    tl.store(
        out_ptr + base + offsets,
        tl.load(x_ptr + base + offsets, mask=offsets < nope_dim),
        mask=offsets < nope_dim,
    )

    pos = tl.load(positions_ptr + token_idx).to(tl.int64)
    pair_idx = tl.arange(0, half_rope)
    cos = tl.load(cos_sin_cache_ptr + pos * rope_dim + pair_idx).to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * rope_dim + half_rope + pair_idx).to(
        tl.float32
    )
    even_offsets = nope_dim + pair_idx * 2
    odd_offsets = even_offsets + 1
    even = tl.load(x_ptr + base + even_offsets).to(tl.float32)
    odd = tl.load(x_ptr + base + odd_offsets).to(tl.float32)

    # Inverse of GPT-J interleaved forward rotation.
    tl.store(
        out_ptr + base + even_offsets,
        (even * cos + odd * sin).to(out_ptr.type.element_ty),
    )
    tl.store(
        out_ptr + base + odd_offsets,
        (odd * cos - even * sin).to(out_ptr.type.element_ty),
    )


def sm70_inverse_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    assert x.dtype == torch.float16
    assert x.ndim == 3 and x.is_contiguous()
    head_dim = x.shape[-1]
    nope_dim = head_dim - rope_dim
    assert rope_dim > 0 and rope_dim % 2 == 0 and nope_dim > 0

    out = torch.empty_like(x)
    _sm70_inverse_rope_kernel[(x.shape[0], x.shape[1])](
        x,
        out,
        positions,
        cos_sin_cache,
        num_heads=x.shape[1],
        head_dim=head_dim,
        rope_dim=rope_dim,
        nope_dim=nope_dim,
        half_rope=rope_dim // 2,
        num_warps=1,
    )
    return out


def sm70_grouped_output_projection(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
    num_groups: int,
    wo_a: torch.nn.Module,
    wo_b: torch.nn.Module,
) -> torch.Tensor:
    """Run inverse RoPE followed by the prepared grouped TurboMind WO path."""
    o_unrotated = sm70_inverse_rope(o, positions, cos_sin_cache, rope_dim)
    grouped = o_unrotated.reshape(o.shape[0], num_groups, -1)
    z = wo_a(grouped)
    assert isinstance(z, torch.Tensor)
    projected = wo_b(z.flatten(1))
    assert isinstance(projected, torch.Tensor)
    return projected
