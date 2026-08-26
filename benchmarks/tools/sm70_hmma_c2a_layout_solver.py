# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Solve SM70 FP32-accumulator QK-C to PV-operand lane ownership.

The ownership formulas are the host form of the CuTe SM70 layouts in
``cute/atom/mma_traits_sm70.hpp``.  The search deliberately permits an
arbitrary fixed permutation of the eight lanes in one Volta quadpair.  This
is a superset of the permutations realizable by TN/NT/NN/TT tile layouts, so
failure to find a zero-transfer mapping is a useful impossibility result.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

QUADPAIR_LANES = (0, 1, 2, 3, 16, 17, 18, 19)
LANE_INDEX = {lane: index for index, lane in enumerate(QUADPAIR_LANES)}

# CuTe names describe the logical A/B transposition.  The corresponding
# operand ownership layouts come directly from MMA_Traits<SM70_...>.
MMA_LAYOUTS = {
    "TN": {"a": "row", "b": "row"},
    "NT": {"a": "col", "b": "col"},
    "NN": {"a": "col", "b": "row"},
    "TT": {"a": "row", "b": "col"},
}


@dataclass(frozen=True)
class Candidate:
    qk_output: str
    pv_variant: str
    p_operand: str
    operand_layout: str
    direct_local_values: int
    best_local_values: int
    total_values: int
    minimum_remote_values: int
    best_local_fraction: float
    best_lane_permutation: tuple[int, ...]
    direct_xor_histogram: dict[str, int]


def _bits(value: int, count: int) -> tuple[int, ...]:
    return tuple((value >> bit) & 1 for bit in range(count))


def _c_coordinates(logical_lane: int) -> tuple[tuple[int, int], ...]:
    """Return the eight C[M,N] coordinates owned by one logical lane."""
    thread_bits = _bits(logical_lane, 3)
    coordinates = []
    for value in range(8):
        value_bits = _bits(value, 3)
        linear = (
            thread_bits[0]
            + 16 * thread_bits[1]
            + 4 * thread_bits[2]
            + 8 * value_bits[0]
            + 2 * value_bits[1]
            + 32 * value_bits[2]
        )
        coordinates.append((linear % 8, linear // 8))
    return tuple(coordinates)


def _operand_coordinates(
    logical_lane: int, layout: str, k_phase: int
) -> tuple[tuple[int, int], ...]:
    """Return four logical P[query,key] values consumed in one K4 phase."""
    if layout == "row":
        return tuple((logical_lane, 4 * k_phase + value) for value in range(4))
    if layout != "col":
        raise ValueError(f"unsupported operand layout: {layout}")
    thread_k = logical_lane % 4
    thread_m_group = logical_lane // 4
    return tuple(
        (4 * thread_m_group + value, 4 * k_phase + thread_k) for value in range(4)
    )


def _source_owners(qk_output: str) -> dict[tuple[int, int], int]:
    owners: dict[tuple[int, int], int] = {}
    for logical_lane, physical_lane in enumerate(QUADPAIR_LANES):
        for m_coord, n_coord in _c_coordinates(logical_lane):
            if qk_output == "P":
                coordinate = (m_coord, n_coord)
            elif qk_output == "PT":
                coordinate = (n_coord, m_coord)
            else:
                raise ValueError(f"unsupported QK output: {qk_output}")
            if coordinate in owners:
                raise AssertionError(f"duplicate C owner for {coordinate}")
            owners[coordinate] = physical_lane
    if len(owners) != 64:
        raise AssertionError("SM70 C layout did not cover one 8x8 tile")
    return owners


def _weight_matrix(
    qk_output: str, operand_layout: str
) -> tuple[list[list[int]], dict[str, int]]:
    owners = _source_owners(qk_output)
    weights = [[0 for _ in QUADPAIR_LANES] for _ in QUADPAIR_LANES]
    xor_histogram: dict[str, int] = {}
    for target_index, target_lane in enumerate(QUADPAIR_LANES):
        for k_phase in range(2):
            for coordinate in _operand_coordinates(
                target_index, operand_layout, k_phase
            ):
                source_lane = owners[coordinate]
                source_index = LANE_INDEX[source_lane]
                weights[source_index][target_index] += 1
                lane_xor = str(source_lane ^ target_lane)
                xor_histogram[lane_xor] = xor_histogram.get(lane_xor, 0) + 1
    return weights, dict(sorted(xor_histogram.items(), key=lambda item: int(item[0])))


def _best_lane_permutation(
    weights: list[list[int]],
) -> tuple[int, tuple[int, ...]]:
    best_score = -1
    best_permutation: tuple[int, ...] | None = None
    for permutation in itertools.permutations(range(len(QUADPAIR_LANES))):
        score = sum(
            weights[source_index][permutation[source_index]]
            for source_index in range(len(QUADPAIR_LANES))
        )
        if score > best_score:
            best_score = score
            best_permutation = permutation
    assert best_permutation is not None
    physical_permutation = tuple(
        QUADPAIR_LANES[target_index] for target_index in best_permutation
    )
    return best_score, physical_permutation


def _solve_candidate(qk_output: str, pv_variant: str, p_operand: str) -> Candidate:
    operand_layout = MMA_LAYOUTS[pv_variant][p_operand]
    weights, xor_histogram = _weight_matrix(qk_output, operand_layout)
    total_values = sum(sum(row) for row in weights)
    direct_local_values = sum(weights[index][index] for index in range(8))
    best_local_values, best_lane_permutation = _best_lane_permutation(weights)
    return Candidate(
        qk_output=qk_output,
        pv_variant=pv_variant,
        p_operand=p_operand.upper(),
        operand_layout=operand_layout,
        direct_local_values=direct_local_values,
        best_local_values=best_local_values,
        total_values=total_values,
        minimum_remote_values=total_values - best_local_values,
        best_local_fraction=best_local_values / total_values,
        best_lane_permutation=best_lane_permutation,
        direct_xor_histogram=xor_histogram,
    )


def solve() -> dict[str, object]:
    candidates = [
        _solve_candidate(qk_output, pv_variant, p_operand)
        for qk_output in ("P", "PT")
        for pv_variant in MMA_LAYOUTS
        for p_operand in ("a", "b")
    ]
    candidates.sort(
        key=lambda candidate: (
            candidate.minimum_remote_values,
            candidate.qk_output,
            candidate.pv_variant,
            candidate.p_operand,
        )
    )
    best_remote = candidates[0].minimum_remote_values
    zero_transfer = [
        candidate for candidate in candidates if candidate.minimum_remote_values == 0
    ]
    best = [
        asdict(candidate)
        for candidate in candidates
        if candidate.minimum_remote_values == best_remote
    ]
    payload: dict[str, object] = {
        "model": {
            "instruction": "mma.sync.aligned.m8n8k4.f32.f16.f16.f32",
            "source_tile": "one FP32 C 8x8 tile",
            "target_tile": "one FP16 P 8x8 tile over two K4 phases",
            "lane_domain": list(QUADPAIR_LANES),
            "permutation_scope": (
                "all 8! fixed quadpair lane permutations; a strict superset "
                "of realizable TN/NT/NN/TT ownership permutations"
            ),
        },
        "zero_transfer_mapping_exists": bool(zero_transfer),
        "best_local_fraction": candidates[0].best_local_fraction,
        "minimum_remote_values_per_64": best_remote,
        "best_candidates": best,
        "all_candidates": [asdict(candidate) for candidate in candidates],
        "decision": (
            "P0_admit_zero_transfer"
            if zero_transfer
            else "P0_reject_zero_transfer_enter_P1_minimum_communication"
        ),
    }
    if zero_transfer:
        payload["zero_transfer_candidates"] = [
            asdict(candidate) for candidate in zero_transfer
        ]
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = solve()
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if payload["zero_transfer_mapping_exists"]:
        return 0
    if payload["minimum_remote_values_per_64"] != 32:
        raise RuntimeError("unexpected SM70 minimum-communication lower bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
