from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass

import numpy as np

from backends.csim.bindings.types import SimInstance

from .fixed_time import solve_fixed_time


@dataclass(frozen=True)
class TimeProbe:
    index: int
    total_time_s: float


@dataclass(frozen=True)
class TimeProbeResult:
    index: int
    total_time_s: float
    wall_s: float
    caught: bool
    feasible: bool
    failure_reason: str
    replay_min_distance_m: float
    replay_final_distance_m: float
    replay_wall_s: float
    replay_steps: int
    error: str = ""


@dataclass(frozen=True)
class TimeSearchResult:
    seed: int
    mode: str
    wall_s: float
    workers: int
    probes_requested: int
    probes_executed: int
    early_exit_reason: str
    fastest_caught_time_s: float
    fastest_caught_probe_index: int
    caught: bool
    replay_min_distance_m: float
    replay_final_distance_m: float
    probe_results: tuple[TimeProbeResult, ...]


def find_fastest_intercept(
    instance: SimInstance,
    reference_controls_rpm: np.ndarray,
    reference_time_s: float,
    *,
    time_multipliers: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1),
    dynamics_substeps: int = 1,
    control_layout: str = "auto",
    mode: str = "serial",
    workers: int = 1,
) -> TimeSearchResult:
    probes = _time_probes(float(reference_time_s), time_multipliers)
    started = time.perf_counter()
    if mode == "serial":
        results = _run_serial(
            instance,
            reference_controls_rpm,
            probes,
            dynamics_substeps=dynamics_substeps,
            control_layout=control_layout,
        )
        early_exit_reason = "first_caught" if any(result.caught for result in results) else "exhausted"
        resolved_workers = 1
    elif mode == "parallel":
        resolved_workers = max(1, min(int(workers), len(probes)))
        results = _run_parallel(
            instance,
            reference_controls_rpm,
            probes,
            dynamics_substeps=dynamics_substeps,
            control_layout=control_layout,
            workers=resolved_workers,
        )
        early_exit_reason = "parallel_all_probes"
    else:
        raise ValueError("mode must be 'serial' or 'parallel'")
    wall_s = time.perf_counter() - started
    fastest = _fastest_caught(results)
    return TimeSearchResult(
        seed=int(instance.seed),
        mode=str(mode),
        wall_s=float(wall_s),
        workers=int(resolved_workers),
        probes_requested=int(len(probes)),
        probes_executed=int(len(results)),
        early_exit_reason=early_exit_reason,
        fastest_caught_time_s=float(fastest.total_time_s) if fastest is not None else float("nan"),
        fastest_caught_probe_index=int(fastest.index) if fastest is not None else -1,
        caught=fastest is not None,
        replay_min_distance_m=float(fastest.replay_min_distance_m) if fastest is not None else float("inf"),
        replay_final_distance_m=float(fastest.replay_final_distance_m) if fastest is not None else float("inf"),
        probe_results=tuple(sorted(results, key=lambda result: result.index)),
    )


def _run_serial(
    instance: SimInstance,
    reference_controls_rpm: np.ndarray,
    probes: tuple[TimeProbe, ...],
    *,
    dynamics_substeps: int,
    control_layout: str,
) -> list[TimeProbeResult]:
    results: list[TimeProbeResult] = []
    for probe in probes:
        result = _run_probe(
            instance,
            reference_controls_rpm,
            probe,
            dynamics_substeps=dynamics_substeps,
            control_layout=control_layout,
        )
        results.append(result)
        if result.caught:
            break
    return results


def _run_parallel(
    instance: SimInstance,
    reference_controls_rpm: np.ndarray,
    probes: tuple[TimeProbe, ...],
    *,
    dynamics_substeps: int,
    control_layout: str,
    workers: int,
) -> list[TimeProbeResult]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(workers)) as executor:
        futures = [
            executor.submit(
                _run_probe,
                instance,
                reference_controls_rpm,
                probe,
                dynamics_substeps=dynamics_substeps,
                control_layout=control_layout,
            )
            for probe in probes
        ]
        return [future.result() for future in concurrent.futures.as_completed(futures)]


def _run_probe(
    instance: SimInstance,
    reference_controls_rpm: np.ndarray,
    probe: TimeProbe,
    *,
    dynamics_substeps: int,
    control_layout: str,
) -> TimeProbeResult:
    started = time.perf_counter()
    try:
        result = solve_fixed_time(
            instance,
            float(probe.total_time_s),
            reference_controls_rpm,
            dynamics_substeps=dynamics_substeps,
            control_layout=control_layout,
        )
        return TimeProbeResult(
            index=int(probe.index),
            total_time_s=float(probe.total_time_s),
            wall_s=time.perf_counter() - started,
            caught=bool(result.caught),
            feasible=bool(result.feasible),
            failure_reason=str(result.failure_reason),
            replay_min_distance_m=float(result.replay_min_distance_m),
            replay_final_distance_m=float(result.replay_final_distance_m),
            replay_wall_s=float(result.replay_wall_s),
            replay_steps=int(result.replay_steps),
            error="",
        )
    except Exception as exc:  # noqa: BLE001
        return TimeProbeResult(
            index=int(probe.index),
            total_time_s=float(probe.total_time_s),
            wall_s=time.perf_counter() - started,
            caught=False,
            feasible=False,
            failure_reason="error",
            replay_min_distance_m=float("inf"),
            replay_final_distance_m=float("inf"),
            replay_wall_s=float("nan"),
            replay_steps=0,
            error=repr(exc),
        )


def _time_probes(reference_time_s: float, multipliers: tuple[float, ...]) -> tuple[TimeProbe, ...]:
    reference = float(reference_time_s)
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_time_s must be finite and positive")
    values = sorted({float(multiplier) for multiplier in multipliers})
    if not values:
        raise ValueError("time_multipliers must not be empty")
    probes = []
    for index, multiplier in enumerate(values):
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            raise ValueError("time_multipliers must be finite and positive")
        probes.append(TimeProbe(index=index, total_time_s=reference * multiplier))
    return tuple(probes)


def _fastest_caught(results: list[TimeProbeResult]) -> TimeProbeResult | None:
    caught = [result for result in results if result.caught and not result.error]
    if not caught:
        return None
    return min(caught, key=lambda result: result.total_time_s)
