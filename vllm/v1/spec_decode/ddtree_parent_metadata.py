# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parent metadata for DDTree verifier state replay.

DDTree payloads store only non-root verifier nodes. GDN/Mamba state replay
needs an explicit root-plus-tree parent row so each tree node can eventually
read the recurrent state of its true parent instead of a linear draft offset.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from vllm.v1.spec_decode.ddtree_payload import DDTreeDraftPayload

ROOT_PARENT = -1
PADDING_PARENT = 0


@dataclass(frozen=True)
class DDTreeParentMetadata:
    # [num_reqs, max_tree_tokens + 1]. Column 0 is the synthetic root.
    parent_ids: torch.Tensor
    # [num_reqs], CPU int32. Counts non-root tree nodes per request.
    num_tree_tokens_cpu: torch.Tensor
    request_ids: tuple[str, ...]


def full_parent_ids_from_payload(payload: DDTreeDraftPayload) -> tuple[int, ...]:
    """Return root-plus-tree parent ids for model-state replay.

    Payload parent indices are verifier-local non-root indices. ``-1`` means a
    node is a root child and should read the pre-tree recurrent state. Any
    non-negative parent ``p`` maps to compact parent slot ``p + 1`` because
    slot 0 is the synthetic root.
    """

    if len(payload.tree_token_ids) != len(payload.parent_indices):
        raise ValueError(
            "DDTree payload has mismatched tree_token_ids/parent_indices: "
            f"{len(payload.tree_token_ids)} != {len(payload.parent_indices)}"
        )

    parents = [ROOT_PARENT]
    for parent in payload.parent_indices:
        parents.append(ROOT_PARENT if parent < 0 else int(parent) + 1)
    return tuple(parents)


def build_padded_parent_ids(
    req_ids: Sequence[str],
    payload_by_req_id: Mapping[str, DDTreeDraftPayload] | None,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.int32,
    pad_to: int | None = None,
) -> DDTreeParentMetadata | None:
    """Build padded DDTree parent rows aligned with the active vLLM batch."""

    if not payload_by_req_id:
        return None

    parents_by_req: list[tuple[int, ...]] = []
    lengths: list[int] = []
    max_len = 0
    found = False
    for req_id in req_ids:
        payload = payload_by_req_id.get(req_id)
        if payload is None or not payload.tree_token_ids:
            parents: tuple[int, ...] = ()
        else:
            parents = full_parent_ids_from_payload(payload)
            found = True
        parents_by_req.append(parents)
        lengths.append(max(0, len(parents) - 1))
        max_len = max(max_len, len(parents))

    if not found:
        return None

    if pad_to is not None:
        pad_to = int(pad_to)
        if pad_to < max_len:
            raise ValueError(
                f"DDTree parent metadata requires pad_to >= {max_len}, got {pad_to}"
            )
        max_len = pad_to
    max_len = max(max_len, 1)

    parent_ids = torch.full(
        (len(req_ids), max_len),
        PADDING_PARENT,
        dtype=dtype,
        device=device,
    )
    for row, parents in enumerate(parents_by_req):
        if not parents:
            continue
        if len(parents) > max_len:
            raise ValueError(
                f"DDTree parent row for {req_ids[row]} has length "
                f"{len(parents)} > pad_to {max_len}"
            )
        parent_ids[row, : len(parents)] = torch.tensor(
            parents,
            dtype=dtype,
            device=device,
        )

    return DDTreeParentMetadata(
        parent_ids=parent_ids,
        num_tree_tokens_cpu=torch.tensor(lengths, dtype=torch.int32, device="cpu"),
        request_ids=tuple(req_ids),
    )
