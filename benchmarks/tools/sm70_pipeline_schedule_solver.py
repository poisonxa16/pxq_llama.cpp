# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Schedule SM70 attention pipelines with latency and register constraints.

This is a source-level screening model, not a replacement for SASS inspection
or an NCU wall-time gate.  It uses measured Volta instruction costs to search
legal issue orders while accounting for dependency latency, pipe occupancy,
and the lifetime of register-resident fragments.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Operation:
    name: str
    pipe: str
    latency: float
    occupancy: float
    dependencies: tuple[str, ...] = ()
    produces_registers: int = 0
    consumes_fragment: str | None = None


@dataclass(frozen=True)
class ScheduledOperation:
    name: str
    pipe: str
    start_cycle: float
    end_cycle: float


@dataclass(frozen=True)
class SearchState:
    scheduled: tuple[ScheduledOperation, ...]
    completed: frozenset[str]
    end_times: tuple[tuple[str, float], ...]
    pipe_ready: tuple[tuple[str, float], ...]
    fragment_release: tuple[tuple[str, float | None], ...]
    peak_live_registers: int


@dataclass(frozen=True)
class ScheduleResult:
    name: str
    register_budget: int
    makespan_cycles: float
    peak_live_registers: int
    issue_order: tuple[str, ...]
    timeline: tuple[ScheduledOperation, ...]


def _mapping(items: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    return dict(items)


def _active_registers(
    operations: dict[str, Operation],
    releases: dict[str, float | None],
    cycle: float,
) -> int:
    return sum(
        operations[name].produces_registers
        for name, release in releases.items()
        if release is None or release > cycle
    )


def _earliest_register_cycle(
    operations: dict[str, Operation],
    releases: dict[str, float | None],
    requested_cycle: float,
    produced_registers: int,
    register_budget: int,
) -> float | None:
    cycle = requested_cycle
    while True:
        live = _active_registers(operations, releases, cycle)
        if live + produced_registers <= register_budget:
            return cycle
        candidates = sorted(
            release
            for release in releases.values()
            if release is not None and release > cycle
        )
        if not candidates:
            return None
        cycle = candidates[0]


def _state_score(state: SearchState) -> tuple[float, int, tuple[str, ...]]:
    end_times = _mapping(state.end_times)
    makespan = max(end_times.values(), default=0.0)
    order = tuple(operation.name for operation in state.scheduled)
    return makespan, state.peak_live_registers, order


def _schedule(
    name: str,
    operation_list: list[Operation],
    register_budget: int,
    beam_width: int,
) -> ScheduleResult:
    operations = {operation.name: operation for operation in operation_list}
    if len(operations) != len(operation_list):
        raise ValueError("operation names must be unique")
    consumers: dict[str, list[str]] = {
        operation.name: [] for operation in operation_list
    }
    for operation in operation_list:
        for dependency in operation.dependencies:
            if dependency not in operations:
                raise ValueError(f"unknown dependency {dependency!r}")
        if operation.consumes_fragment is not None:
            producer = operation.consumes_fragment
            if producer not in operations:
                raise ValueError(f"unknown fragment producer {producer!r}")
            consumers[producer].append(operation.name)
    for producer, fragment_consumers in consumers.items():
        if operations[producer].produces_registers and len(fragment_consumers) != 1:
            raise ValueError(
                f"register fragment {producer!r} must have exactly one consumer"
            )

    initial = SearchState(
        scheduled=(),
        completed=frozenset(),
        end_times=(),
        pipe_ready=(),
        fragment_release=(),
        peak_live_registers=0,
    )
    beam = [initial]
    while beam and len(beam[0].completed) != len(operation_list):
        next_states: list[SearchState] = []
        for state in beam:
            end_times = _mapping(state.end_times)
            pipe_ready = _mapping(state.pipe_ready)
            releases = _mapping(state.fragment_release)
            issue_ready = pipe_ready.get("issue", 0.0)
            ready = [
                operation
                for operation in operation_list
                if operation.name not in state.completed
                and all(dep in state.completed for dep in operation.dependencies)
            ]
            for operation in ready:
                dependency_ready = max(
                    (end_times[dep] for dep in operation.dependencies),
                    default=0.0,
                )
                start = max(
                    issue_ready,
                    pipe_ready.get(operation.pipe, 0.0),
                    dependency_ready,
                )
                start = _earliest_register_cycle(
                    operations,
                    releases,
                    start,
                    operation.produces_registers,
                    register_budget,
                )
                if start is None:
                    continue
                end = start + operation.latency
                next_end_times = dict(end_times)
                next_end_times[operation.name] = end
                next_pipe_ready = dict(pipe_ready)
                next_pipe_ready["issue"] = start + 1.0
                next_pipe_ready[operation.pipe] = start + operation.occupancy
                next_releases = dict(releases)
                if operation.produces_registers:
                    next_releases[operation.name] = None
                if operation.consumes_fragment is not None:
                    next_releases[operation.consumes_fragment] = end
                live_at_start = _active_registers(operations, releases, start)
                peak = max(
                    state.peak_live_registers,
                    live_at_start + operation.produces_registers,
                )
                scheduled = ScheduledOperation(
                    name=operation.name,
                    pipe=operation.pipe,
                    start_cycle=start,
                    end_cycle=end,
                )
                next_states.append(
                    SearchState(
                        scheduled=state.scheduled + (scheduled,),
                        completed=state.completed | {operation.name},
                        end_times=tuple(sorted(next_end_times.items())),
                        pipe_ready=tuple(sorted(next_pipe_ready.items())),
                        fragment_release=tuple(sorted(next_releases.items())),
                        peak_live_registers=peak,
                    )
                )
        if not next_states:
            raise RuntimeError(
                f"no legal {name} schedule under {register_budget} registers"
            )
        deduplicated: dict[
            tuple[
                frozenset[str],
                tuple[tuple[str, float], ...],
                tuple[tuple[str, float], ...],
                tuple[tuple[str, float | None], ...],
            ],
            SearchState,
        ] = {}
        for state in next_states:
            key = (
                state.completed,
                state.end_times,
                state.pipe_ready,
                state.fragment_release,
            )
            previous = deduplicated.get(key)
            if previous is None or _state_score(state) < _state_score(previous):
                deduplicated[key] = state
        beam = sorted(deduplicated.values(), key=_state_score)[:beam_width]

    best = min(beam, key=_state_score)
    end_times = _mapping(best.end_times)
    return ScheduleResult(
        name=name,
        register_budget=register_budget,
        makespan_cycles=max(end_times.values()),
        peak_live_registers=best.peak_live_registers,
        issue_order=tuple(operation.name for operation in best.scheduled),
        timeline=tuple(
            sorted(best.scheduled, key=lambda operation: operation.start_cycle)
        ),
    )


def _pv_operations(costs: dict[str, float], phases: int) -> list[Operation]:
    # The split-D N32 SASS has 256 PV HMMA instructions over eight K4 phases.
    hmma_step_issue = costs["four_chain_hmma_m8n8k4_issue_cycles"] / 4.0
    hmma_phase_cycles = 32 * hmma_step_issue
    operations: list[Operation] = []
    for phase in range(phases):
        lds = f"pv_lds_{phase}"
        previous_hmma = f"pv_hmma_{phase - 1}" if phase else None
        operations.append(
            Operation(
                name=lds,
                pipe="mio",
                latency=costs["dependent_shared_load_cycles"],
                occupancy=1.0,
                produces_registers=8,
            )
        )
        dependencies = (lds,) if previous_hmma is None else (lds, previous_hmma)
        operations.append(
            Operation(
                name=f"pv_hmma_{phase}",
                pipe="tensor",
                latency=hmma_phase_cycles,
                occupancy=hmma_phase_cycles,
                dependencies=dependencies,
                consumes_fragment=lds,
            )
        )
    return operations


def _k_operations(costs: dict[str, float], phases: int) -> list[Operation]:
    hmma_step_issue = costs["four_chain_hmma_m8n8k4_issue_cycles"] / 4.0
    qk_phase_cycles = 64 * hmma_step_issue
    operations = [
        Operation(
            name="qk_hmma_0",
            pipe="tensor",
            latency=qk_phase_cycles,
            occupancy=qk_phase_cycles,
        )
    ]
    for phase in range(1, phases):
        ldg = f"k_ldg_{phase}"
        sts = f"k_sts_{phase}"
        previous_hmma = f"qk_hmma_{phase - 1}"
        operations.extend(
            [
                Operation(
                    name=ldg,
                    pipe="lg",
                    latency=costs["dependent_global_cg_load_l2_hit_cycles"],
                    occupancy=1.0,
                    produces_registers=8,
                ),
                Operation(
                    name=sts,
                    pipe="mio",
                    latency=1.0,
                    occupancy=1.0,
                    dependencies=(ldg, previous_hmma),
                    consumes_fragment=ldg,
                ),
                Operation(
                    name=f"qk_hmma_{phase}",
                    pipe="tensor",
                    latency=qk_phase_cycles,
                    occupancy=qk_phase_cycles,
                    dependencies=(sts, previous_hmma),
                ),
            ]
        )
    return operations


def solve(latency_payload: dict[str, Any], beam_width: int) -> dict[str, Any]:
    costs = latency_payload["measured_costs"]
    pv_operations = _pv_operations(costs, phases=8)
    k_operations = _k_operations(costs, phases=4)
    schedules = [
        _schedule("pv_single_buffer", pv_operations, 8, beam_width),
        _schedule("pv_double_buffer", pv_operations, 16, beam_width),
        _schedule("k_single_lookahead", k_operations, 8, beam_width),
        _schedule("k_double_lookahead", k_operations, 16, beam_width),
    ]
    by_name = {schedule.name: schedule for schedule in schedules}
    pv_single = by_name["pv_single_buffer"]
    pv_double = by_name["pv_double_buffer"]
    k_single = by_name["k_single_lookahead"]
    k_double = by_name["k_double_lookahead"]
    first_publish = "k_sts_1"
    first_publish_single = next(
        operation.start_cycle
        for operation in k_single.timeline
        if operation.name == first_publish
    )
    first_publish_double = next(
        operation.start_cycle
        for operation in k_double.timeline
        if operation.name == first_publish
    )
    return {
        "device": latency_payload.get("device"),
        "clock_rate_khz": latency_payload.get("clock_rate_khz"),
        "measured_costs": costs,
        "model": {
            "pv_hmma_sass_per_phase": 32,
            "qk_hmma_sass_per_d64_phase": 64,
            "hmma_step_issue_cycles": (
                costs["four_chain_hmma_m8n8k4_issue_cycles"] / 4.0
            ),
            "fragment_registers": 8,
            "caveat": (
                "source-level lower-bound model; accept candidates only after "
                "same-build wall-time, SASS, register, and NCU gates"
            ),
        },
        "schedules": [asdict(schedule) for schedule in schedules],
        "decisions": {
            "pv_double_buffer": {
                "modeled_cycle_reduction_pct": (
                    (pv_single.makespan_cycles - pv_double.makespan_cycles)
                    / pv_single.makespan_cycles
                    * 100.0
                ),
                "decision": "admit_for_wall_time_gate",
            },
            "k_double_lookahead": {
                "modeled_cycle_reduction_pct": (
                    (k_single.makespan_cycles - k_double.makespan_cycles)
                    / k_single.makespan_cycles
                    * 100.0
                ),
                "first_critical_publish_cycle_single": first_publish_single,
                "first_critical_publish_cycle_double": first_publish_double,
                "first_critical_publish_improved": (
                    first_publish_double < first_publish_single
                ),
                "decision": (
                    "admit_for_wall_time_gate"
                    if first_publish_double < first_publish_single
                    else "reject_no_first_critical_edge_improvement"
                ),
            },
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-json", required=True, type=Path)
    parser.add_argument("--beam-width", type=int, default=4096)
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    latency_payload = json.loads(args.latency_json.read_text(encoding="utf-8"))
    payload = solve(latency_payload, args.beam_width)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
