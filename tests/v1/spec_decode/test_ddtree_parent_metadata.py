# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.spec_decode.ddtree_parent_metadata import (
    ROOT_PARENT,
    build_padded_parent_ids,
    full_parent_ids_from_payload,
)
from vllm.v1.spec_decode.ddtree_payload import DDTreeDraftPayload


def _payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 22, 31),
        parent_indices=(-1, 0, 0, 2),
        node_depths=(1, 2, 2, 3),
        node_scores=(0.0, -0.1, -0.2, -0.3),
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=4,
        top_k=2,
        chain_seed=True,
    )


def test_full_parent_ids_from_payload_adds_synthetic_root() -> None:
    assert full_parent_ids_from_payload(_payload()) == (
        ROOT_PARENT,
        ROOT_PARENT,
        1,
        1,
        3,
    )


def test_build_padded_parent_ids_aligns_to_request_ids() -> None:
    metadata = build_padded_parent_ids(
        ["req-a", "req-b"],
        {"req-b": _payload()},
        device="cpu",
        pad_to=6,
    )

    assert metadata is not None
    assert metadata.request_ids == ("req-a", "req-b")
    assert metadata.num_tree_tokens_cpu.tolist() == [0, 4]
    assert metadata.parent_ids.shape == (2, 6)
    assert metadata.parent_ids[0].tolist() == [0, 0, 0, 0, 0, 0]
    assert metadata.parent_ids[1].tolist() == [-1, -1, 1, 1, 3, 0]


def test_build_padded_parent_ids_returns_none_without_payloads() -> None:
    assert build_padded_parent_ids(["req-a"], None, device="cpu") is None
    assert build_padded_parent_ids(["req-a"], {}, device="cpu") is None


def test_build_padded_parent_ids_rejects_too_small_pad_to() -> None:
    try:
        build_padded_parent_ids(
            ["req-a"],
            {"req-a": _payload()},
            device="cpu",
            pad_to=3,
        )
    except ValueError as exc:
        assert "pad_to" in str(exc)
    else:
        raise AssertionError("expected ValueError for too-small pad_to")


def test_build_padded_parent_ids_honors_device_and_dtype() -> None:
    metadata = build_padded_parent_ids(
        ["req-a"],
        {"req-a": _payload()},
        device="cpu",
        dtype=torch.int64,
    )

    assert metadata is not None
    assert metadata.parent_ids.dtype == torch.int64
    assert metadata.num_tree_tokens_cpu.dtype == torch.int32
