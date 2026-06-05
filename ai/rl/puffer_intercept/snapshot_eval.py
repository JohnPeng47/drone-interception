from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from backends.csim.generator.instance_store import read_sim_instances_by_index
from backends.csim.bindings import BatchPufferSimEngineBackend
from backends.csim.bindings.types import SimInstance, SimSnapshots

from .checkpointing import CHECKPOINT_TYPE, SCHEMA_VERSION
from .observations import OBS_SIZE, observation_from_batch_snapshot, observation_from_batch_arrays
from .puffer_ppo import PufferMLPPolicy, sample_logits
from .scenario_table import ScenarioLabel, ScenarioTable


@dataclass(frozen=True)
class SnapshotEvalConfig:
    scenario_table: Path
    checkpoint: Path
    out_dir: Path
    manifest: Path | None = None
    max_scenarios: int | None = None
    max_episodes: int = 36
    samples_per_cell: int | None = None
    num_envs: int = 32
    seed: int = 1
    device: str = "cpu"
    stochastic: bool = False
    snapshot_stride: int = 10
    max_episode_steps: int | None = None
    scenario_indices: tuple[int, ...] | None = None


@dataclass
class _EpisodeTrace:
    scenario_index: int
    label: ScenarioLabel
    records: list[dict[str, Any]]
    episode_return: float = 0.0
    terminal_reason: str = ""
    terminal_tick: int = 0


def run_snapshot_eval(config: SnapshotEvalConfig) -> Path:
    if config.max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if config.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if config.snapshot_stride <= 0:
        raise ValueError("snapshot_stride must be positive")

    start = time.perf_counter()
    scenario_indices = (
        list(config.scenario_indices)
        if config.scenario_indices is not None
        else None
    )
    table = (
        _SelectedScenarioTable(config.scenario_table, config.manifest, scenario_indices)
        if scenario_indices is not None
        else ScenarioTable(config.scenario_table, manifest_path=config.manifest, max_scenarios=config.max_scenarios)
    )
    scenario_indices = (
        scenario_indices
        if scenario_indices is not None
        else select_stratified_indices(
            table,
            max_episodes=config.max_episodes,
            samples_per_cell=config.samples_per_cell,
            seed=config.seed,
        )
    )
    model, checkpoint_info = _load_model(config.checkpoint, config.device)
    run_dir = config.out_dir / f"{checkpoint_info['global_step']:012d}"
    episodes_dir = run_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    timings = {
        "policy_inference_s": 0.0,
        "sim_step_s": 0.0,
        "loop_s": 0.0,
        "policy_batches": 0,
        "policy_items": 0,
        "sim_steps": 0,
        "env_steps": 0,
    }
    for start_index in range(0, len(scenario_indices), config.num_envs):
        chunk = scenario_indices[start_index:start_index + config.num_envs]
        chunk_rows, chunk_timings = _run_chunk(table, model, chunk, episodes_dir, config)
        rows.extend(chunk_rows)
        _accumulate_timings(timings, chunk_timings)

    summary = _summary(rows)
    summary["timing"] = _timing_rates(timings)
    metadata = {
        "checkpoint": str(config.checkpoint),
        "checkpoint_info": checkpoint_info,
        "scenario_table": str(config.scenario_table),
        "manifest": None if config.manifest is None else str(config.manifest),
        "max_scenarios": config.max_scenarios,
        "num_envs": config.num_envs,
        "seed": config.seed,
        "device": config.device,
        "stochastic": config.stochastic,
        "snapshot_stride": config.snapshot_stride,
        "max_episode_steps": config.max_episode_steps,
        "selected_scenario_indices": scenario_indices,
        "episodes": rows,
        "summary": summary,
        "elapsed_wall_s": time.perf_counter() - start,
    }
    summary_path = run_dir / "snapshot_eval.json"
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return summary_path


def select_stratified_indices(
    table: ScenarioTable,
    *,
    max_episodes: int,
    samples_per_cell: int | None = None,
    seed: int = 1,
) -> list[int]:
    rng = np.random.default_rng(int(seed))
    if not table.cells or not table.samples_per_cell:
        count = min(int(max_episodes), table.count)
        return sorted(rng.choice(table.count, size=count, replace=False).astype(int).tolist())

    per_cell = int(samples_per_cell) if samples_per_cell is not None else max(1, math.ceil(int(max_episodes) / len(table.cells)))
    selected_by_cell: list[list[int]] = []
    samples_per_manifest_cell = int(table.samples_per_cell)
    for cell_index in range(len(table.cells)):
        start = cell_index * samples_per_manifest_cell
        stop = min(start + samples_per_manifest_cell, table.count)
        if start >= stop:
            continue
        count = min(per_cell, stop - start)
        selected = rng.choice(np.arange(start, stop), size=count, replace=False).astype(int).tolist()
        selected_by_cell.append(selected)

    ordered: list[int] = []
    for offset in range(per_cell):
        for cell_values in selected_by_cell:
            if offset < len(cell_values):
                ordered.append(cell_values[offset])
            if len(ordered) >= int(max_episodes):
                return ordered
    return ordered[: int(max_episodes)]


class _SelectedScenarioTable:
    def __init__(self, path: Path, manifest_path: Path | None, scenario_indices: list[int]):
        instances, total_count = read_sim_instances_by_index(path, tuple(scenario_indices))
        missing = [index for index in scenario_indices if index not in instances]
        if missing:
            raise IndexError(f"scenario indices out of range for {path}: {missing[:5]}")
        self.path = Path(path)
        self.instances = instances
        self.count = total_count
        self.manifest_path = manifest_path
        self.manifest = _load_manifest(manifest_path)
        self.cells = tuple((self.manifest or {}).get("grid", {}).get("cells", ()))
        self.samples_per_cell = (self.manifest or {}).get("grid", {}).get("samples_per_cell")

    def get(self, index: int) -> SimInstance:
        return self.instances[int(index)]

    def label(self, index: int) -> ScenarioLabel:
        index = int(index)
        if not self.cells or not self.samples_per_cell:
            return ScenarioLabel(index, None, None, None)
        cell_index = min(index // int(self.samples_per_cell), len(self.cells) - 1)
        cell = self.cells[cell_index]
        return ScenarioLabel(
            scenario_index=index,
            cell_index=int(cell["cell_index"]),
            range_m=float(cell["range_m"]),
            closing_speed_mps=float(cell["closing_speed_mps"]),
        )


def _load_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _run_chunk(
    table: ScenarioTable,
    model: PufferMLPPolicy,
    scenario_indices: list[int],
    episodes_dir: Path,
    config: SnapshotEvalConfig,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    instances = tuple(table.get(index) for index in scenario_indices)
    labels = tuple(table.label(index) for index in scenario_indices)
    backend = BatchPufferSimEngineBackend(len(instances))
    snapshot = backend.reset_many(np.arange(len(instances), dtype=np.int64), instances)
    previous_action = np.zeros((len(instances), 4), dtype=np.float32)
    previous_distance = snapshot.arrays.metrics[:, 0].astype(np.float32, copy=True)
    elapsed_s = np.zeros(len(instances), dtype=np.float32)
    episode_length = np.zeros(len(instances), dtype=np.int32)
    episode_return = np.zeros(len(instances), dtype=np.float32)
    active = np.ones(len(instances), dtype=bool)
    bounds_w = _bounds(instances)
    duration_s = _durations(instances)
    traces = [
        _EpisodeTrace(scenario_index=int(index), label=label, records=[])
        for index, label in zip(scenario_indices, labels)
    ]
    _record_snapshot(traces, snapshot, previous_action, np.zeros(len(instances), dtype=np.float32), active, elapsed_s, tick=0)
    obs = observation_from_batch_snapshot(snapshot, previous_action=previous_action)
    timings: dict[str, float | int] = {
        "policy_inference_s": 0.0,
        "sim_step_s": 0.0,
        "loop_s": 0.0,
        "policy_batches": 0,
        "policy_items": 0,
        "sim_steps": 0,
        "env_steps": 0,
    }

    tick = 0
    loop_start = time.perf_counter()
    while np.any(active):
        tick += 1
        action = np.zeros((len(instances), 4), dtype=np.float32)
        active_indices = np.flatnonzero(active)
        policy_start = time.perf_counter()
        with torch.no_grad():
            obs_t = torch.as_tensor(obs[active_indices], dtype=torch.float32, device=next(model.parameters()).device)
            dist, _value, _state = model.forward_eval(obs_t)
            if config.stochastic:
                action_t, _logprob, _entropy = sample_logits(dist)
            else:
                action_t = dist.loc
        action[active_indices] = action_t.detach().cpu().numpy().astype(np.float32, copy=False)
        timings["policy_inference_s"] = float(timings["policy_inference_s"]) + time.perf_counter() - policy_start
        timings["policy_batches"] = int(timings["policy_batches"]) + 1
        timings["policy_items"] = int(timings["policy_items"]) + int(len(active_indices))

        sim_start = time.perf_counter()
        snapshot = backend.step_ctbr_many(action)
        timings["sim_step_s"] = float(timings["sim_step_s"]) + time.perf_counter() - sim_start
        timings["sim_steps"] = int(timings["sim_steps"]) + 1
        timings["env_steps"] = int(timings["env_steps"]) + int(len(active_indices))
        arrays = snapshot.arrays
        elapsed_s[active] += _dt(backend)
        episode_length[active] += 1
        distance = arrays.metrics[:, 0].astype(np.float32, copy=False)
        intercepted = arrays.metrics[:, 2] > 0.5
        failed, fail_reasons = _failed(snapshot, bounds_w)
        timeout = _timeout(duration_s, elapsed_s, episode_length, config.max_episode_steps)
        done = active & (intercepted | failed | timeout)
        reward = _reward(previous_distance, distance, arrays.body_rates_b, intercepted, failed)
        reward = np.where(active, reward, 0.0).astype(np.float32, copy=False)
        episode_return[active] += reward[active]

        should_record = active & ((tick % config.snapshot_stride == 0) | done)
        _record_snapshot(traces, snapshot, action, reward, should_record, elapsed_s, tick=tick, done=done)

        for slot in np.flatnonzero(done):
            reason = "intercepted" if intercepted[slot] else fail_reasons[slot] if failed[slot] else "timeout"
            traces[slot].episode_return = float(episode_return[slot])
            traces[slot].terminal_reason = reason
            traces[slot].terminal_tick = int(tick)
            active[slot] = False

        previous_distance = distance.copy()
        previous_action[:] = action
        obs = observation_from_batch_arrays(arrays, previous_action=previous_action)

    rows = []
    for trace in traces:
        rows.append(_write_episode(episodes_dir, trace))
    timings["loop_s"] = time.perf_counter() - loop_start
    return rows, timings


_COMPATIBLE_CHECKPOINT_TYPES = frozenset((CHECKPOINT_TYPE, "simengine_batch_training"))


def _load_model(checkpoint_path: Path, device_name: str) -> tuple[PufferMLPPolicy, dict[str, Any]]:
    device = torch.device(device_name if torch.cuda.is_available() or device_name == "cpu" else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_type = str(checkpoint.get("checkpoint_type"))
    if checkpoint_type not in _COMPATIBLE_CHECKPOINT_TYPES:
        expected = ", ".join(sorted(_COMPATIBLE_CHECKPOINT_TYPES))
        raise ValueError(f"{checkpoint_path} is not a compatible checkpoint type; expected one of {expected}")
    if int(checkpoint.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema in {checkpoint_path}: {checkpoint.get('schema_version')!r}")
    args = checkpoint.get("args", {}) or {}
    model = PufferMLPPolicy(
        OBS_SIZE,
        4,
        hidden_size=int(args.get("hidden_size", 128)),
        num_layers=int(args.get("num_layers", 4)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {
        "checkpoint_type": checkpoint_type,
        "global_step": int(checkpoint.get("global_step", 0)),
        "ppo_epoch": int(checkpoint.get("ppo_epoch", 0)),
        "hidden_size": int(args.get("hidden_size", 128)),
        "num_layers": int(args.get("num_layers", 4)),
    }


def _record_snapshot(
    traces: list[_EpisodeTrace],
    snapshot: SimSnapshots,
    action: np.ndarray,
    reward: np.ndarray,
    mask: np.ndarray,
    elapsed_s: np.ndarray,
    *,
    tick: int,
    done: np.ndarray | None = None,
) -> None:
    arrays = snapshot.arrays
    done_arr = np.zeros(len(traces), dtype=bool) if done is None else np.asarray(done, dtype=bool)
    for slot in np.flatnonzero(mask):
        traces[int(slot)].records.append({
            "tick": int(tick),
            "elapsed_s": float(elapsed_s[slot]),
            "pursuer": arrays.pursuer[slot].copy(),
            "target": arrays.target[slot].copy(),
            "metrics": arrays.metrics[slot].copy(),
            "camera": arrays.camera[slot].copy(),
            "action": action[slot].copy(),
            "reward": float(reward[slot]),
            "done": bool(done_arr[slot]),
            "body_rates_b": (
                np.zeros(3, dtype=np.float32)
                if arrays.body_rates_b is None
                else arrays.body_rates_b[slot].copy()
            ),
            "thrust_n": (
                np.float32(0.0)
                if arrays.thrust_n is None
                else np.float32(arrays.thrust_n[slot])
            ),
        })


def _write_episode(episodes_dir: Path, trace: _EpisodeTrace) -> dict[str, Any]:
    prefix = f"scenario_{trace.scenario_index:012d}"
    path = episodes_dir / f"{prefix}.npz"
    records = trace.records
    if not records:
        raise RuntimeError(f"no records collected for scenario {trace.scenario_index}")
    np.savez_compressed(
        path,
        ticks=np.asarray([record["tick"] for record in records], dtype=np.int32),
        elapsed_s=np.asarray([record["elapsed_s"] for record in records], dtype=np.float32),
        pursuer=np.asarray([record["pursuer"] for record in records], dtype=np.float32),
        target=np.asarray([record["target"] for record in records], dtype=np.float32),
        metrics=np.asarray([record["metrics"] for record in records], dtype=np.float32),
        camera=np.asarray([record["camera"] for record in records], dtype=np.float32),
        actions=np.asarray([record["action"] for record in records], dtype=np.float32),
        rewards=np.asarray([record["reward"] for record in records], dtype=np.float32),
        dones=np.asarray([record["done"] for record in records], dtype=bool),
        body_rates_b=np.asarray([record["body_rates_b"] for record in records], dtype=np.float32),
        thrust_n=np.asarray([record["thrust_n"] for record in records], dtype=np.float32),
    )
    final_metrics = np.asarray(records[-1]["metrics"], dtype=np.float32)
    return {
        "scenario_index": trace.scenario_index,
        "cell_index": -1 if trace.label.cell_index is None else int(trace.label.cell_index),
        "range_m": trace.label.range_m,
        "closing_speed_mps": trace.label.closing_speed_mps,
        "terminal_reason": trace.terminal_reason,
        "intercepted": bool(final_metrics[2] > 0.5),
        "episode_return": trace.episode_return,
        "episode_length": trace.terminal_tick,
        "min_distance_m": float(final_metrics[1]),
        "final_distance_m": float(final_metrics[0]),
        "snapshot_count": len(records),
        "path": str(path),
    }


def _bounds(instances: tuple[SimInstance, ...]) -> np.ndarray:
    out = np.full((len(instances), 3), np.inf, dtype=np.float32)
    for slot, instance in enumerate(instances):
        bounds = None if instance.config is None else instance.config.bounds_w
        if bounds is not None:
            out[slot] = np.asarray(bounds, dtype=np.float32).reshape(3)
    return out


def _durations(instances: tuple[SimInstance, ...]) -> np.ndarray:
    out = np.zeros(len(instances), dtype=np.float32)
    for slot, instance in enumerate(instances):
        if instance.config is not None:
            out[slot] = float(instance.config.options.duration_s)
    return out


def _failed(snapshot: SimSnapshots, bounds_w: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pos = snapshot.arrays.pursuer[:, 0:3]
    oob = np.any(np.abs(pos) > bounds_w, axis=1)
    nonfinite = ~np.all(np.isfinite(pos), axis=1)
    return oob | nonfinite, ["nonfinite" if nonfinite[i] else "oob" if oob[i] else "" for i in range(len(pos))]


def _timeout(
    duration_s: np.ndarray,
    elapsed_s: np.ndarray,
    episode_length: np.ndarray,
    max_episode_steps: int | None,
) -> np.ndarray:
    timeout = (duration_s > 0.0) & (elapsed_s >= duration_s)
    if max_episode_steps is not None:
        timeout |= episode_length >= int(max_episode_steps)
    return timeout


def _reward(
    previous_distance: np.ndarray,
    distance: np.ndarray,
    body_rates_b: np.ndarray | None,
    intercepted: np.ndarray,
    failed: np.ndarray,
) -> np.ndarray:
    # Training reward is owned by the native C puffer-intercept env. Snapshot
    # eval records trajectories and terminal outcomes without maintaining a
    # second Python reward implementation.
    del previous_distance, body_rates_b, intercepted, failed
    return np.zeros(len(distance), dtype=np.float32)


def _dt(backend: BatchPufferSimEngineBackend) -> float:
    return float(backend._dt if backend._dt is not None else 0.01)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    caught = np.asarray([bool(row["intercepted"]) for row in rows], dtype=float)
    min_distance = np.asarray([float(row["min_distance_m"]) for row in rows], dtype=np.float32)
    terminal_counts: dict[str, int] = {}
    cell_counts: dict[str, int] = {}
    cell_catches: dict[str, list[bool]] = {}
    for row in rows:
        terminal_counts[row["terminal_reason"]] = terminal_counts.get(row["terminal_reason"], 0) + 1
        cell = str(row["cell_index"])
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        cell_catches.setdefault(cell, []).append(bool(row["intercepted"]))
    return {
        "episodes": len(rows),
        "catch_rate": float(np.mean(caught)) if caught.size else math.nan,
        "min_distance_p50_m": float(np.percentile(min_distance, 50)) if min_distance.size else math.nan,
        "terminal_counts": terminal_counts,
        "cell_counts": cell_counts,
        "cell_catch_rates": {cell: float(np.mean(values)) for cell, values in cell_catches.items()},
    }


def _accumulate_timings(total: dict[str, float | int], chunk: dict[str, float | int]) -> None:
    for key, value in chunk.items():
        if isinstance(value, int):
            total[key] = int(total[key]) + value
        else:
            total[key] = float(total[key]) + float(value)


def _timing_rates(timings: dict[str, float | int]) -> dict[str, float | int]:
    policy_s = float(timings["policy_inference_s"])
    sim_s = float(timings["sim_step_s"])
    loop_s = float(timings["loop_s"])
    policy_items = int(timings["policy_items"])
    env_steps = int(timings["env_steps"])
    policy_batches = int(timings["policy_batches"])
    sim_steps = int(timings["sim_steps"])
    return {
        **timings,
        "policy_items_per_s": policy_items / policy_s if policy_s > 0.0 else math.nan,
        "policy_ms_per_batch": 1000.0 * policy_s / policy_batches if policy_batches else math.nan,
        "policy_us_per_item": 1_000_000.0 * policy_s / policy_items if policy_items else math.nan,
        "sim_env_steps_per_s": env_steps / sim_s if sim_s > 0.0 else math.nan,
        "sim_ms_per_batch_step": 1000.0 * sim_s / sim_steps if sim_steps else math.nan,
        "loop_env_steps_per_s": env_steps / loop_s if loop_s > 0.0 else math.nan,
    }
