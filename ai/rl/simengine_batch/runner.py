from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backends.csim.bindings import BatchPufferSimEngineBackend
from backends.csim.bindings.types import SimSnapshots
from ai.rl.simengine_env.rewards import RewardConfig
from ai.rl.simengine_env.scenario_table import ScenarioLabel

from .generator import BatchReset, BatchSimGenerator
from .observations import OBS_SIZE, observation_from_batch_arrays, observation_from_batch_snapshot


@dataclass(frozen=True)
class BatchRunnerConfig:
    max_episode_steps: int | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)


class BatchSimRunner:
    action_size = 4
    observation_size = OBS_SIZE

    def __init__(self, generator: BatchSimGenerator, *, config: BatchRunnerConfig | None = None):
        self.generator = generator
        self.config = config or BatchRunnerConfig()
        self.num_envs = generator.num_envs
        self.backend = BatchPufferSimEngineBackend(self.num_envs)
        self.snapshot: SimSnapshots | None = None
        self.labels: list[ScenarioLabel] = [ScenarioLabel(-1, None, None, None)] * self.num_envs
        self.duration_s = np.zeros(self.num_envs, dtype=np.float32)
        self.bounds_w = np.full((self.num_envs, 3), np.inf, dtype=np.float32)
        self.elapsed_s = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_length = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_return = np.zeros(self.num_envs, dtype=np.float32)
        self.previous_distance_m = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_counter = np.zeros(self.num_envs, dtype=np.int64)

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        reset = self.generator.initial_batch()
        self.snapshot = self._apply_reset(reset)
        obs = observation_from_batch_snapshot(self.snapshot)
        return obs, self._infos(np.zeros(self.num_envs, dtype=bool), [""] * self.num_envs)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if self.snapshot is None:
            self.reset()
        assert self.snapshot is not None
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.action_size)
        self.snapshot = self.backend.step_ctbr_many(actions)
        self.elapsed_s += self._dt()
        self.episode_length += 1

        arrays = self.snapshot.arrays
        metrics = arrays.metrics
        distance = metrics[:, 0].astype(np.float32, copy=False)
        intercepted = metrics[:, 2] > 0.5
        failed, fail_reasons = self._failed(self.snapshot)
        timeout = self._timeout()
        done = intercepted | failed | timeout

        if arrays.body_rates_b is None:
            raise RuntimeError("batch snapshot is missing applied body-rate commands")
        body_rates = arrays.body_rates_b
        rewards = self._reward(distance, body_rates, intercepted, failed)
        self.episode_return += rewards
        reasons = [
            "intercepted" if intercepted[i] else fail_reasons[i] if failed[i] else "timeout" if timeout[i] else ""
            for i in range(self.num_envs)
        ]
        infos = self._infos(done, reasons)
        for i, is_done in enumerate(done):
            if is_done:
                infos[i]["episode"] = self._episode_info(i, reasons[i])

        self.previous_distance_m = distance
        obs = observation_from_batch_arrays(arrays)

        if np.any(done):
            done_indices = np.flatnonzero(done)
            reset = self.generator.refill(done_indices)
            self.snapshot = self._apply_reset(reset)
            reset_obs = observation_from_batch_snapshot(self.snapshot)
            obs[done_indices] = reset_obs[done_indices]
            reset_infos = self._infos(np.zeros(self.num_envs, dtype=bool), [""] * self.num_envs)
            for idx in done_indices:
                infos[int(idx)]["reset"] = reset_infos[int(idx)]

        return obs, rewards.astype(np.float32, copy=False), done.astype(bool, copy=False), infos

    def _apply_reset(self, reset: BatchReset) -> SimSnapshots:
        snapshot = self.backend.reset_many(reset.indices, reset.instances)
        for local, slot in enumerate(reset.indices):
            slot = int(slot)
            self.labels[slot] = reset.labels[local]
            self.duration_s[slot] = reset.duration_s[local]
            instance = reset.instances[local]
            bounds_w = None if instance.config is None else instance.config.bounds_w
            self.bounds_w[slot] = (
                np.full(3, np.inf, dtype=np.float32)
                if bounds_w is None
                else np.asarray(bounds_w, dtype=np.float32).reshape(3)
            )
            self.elapsed_s[slot] = 0.0
            self.episode_length[slot] = 0
            self.episode_return[slot] = 0.0
            self.episode_counter[slot] += 1
            self.previous_distance_m[slot] = snapshot[slot].metrics.distance_m
        return snapshot

    def _reward(
        self,
        distance: np.ndarray,
        body_rates_b: np.ndarray,
        intercepted: np.ndarray,
        failed: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config.reward
        progress = self.previous_distance_m - distance
        return (
            np.where(intercepted, cfg.catch_reward, 0.0)
            + cfg.progress_weight * progress
            - cfg.distance_weight * distance
            - cfg.rate_weight * np.linalg.norm(body_rates_b, axis=1)
            - np.where(failed, cfg.fail_penalty, 0.0)
        ).astype(np.float32)

    def _failed(self, snapshot: SimSnapshots) -> tuple[np.ndarray, list[str]]:
        pos = snapshot.arrays.pursuer[:, 0:3]
        oob = np.any(np.abs(pos) > self.bounds_w, axis=1)
        nonfinite = ~np.all(np.isfinite(pos), axis=1)
        failed = oob | nonfinite
        reasons = ["nonfinite" if nonfinite[i] else "oob" if oob[i] else "" for i in range(self.num_envs)]
        return failed, reasons

    def _timeout(self) -> np.ndarray:
        timeout = np.zeros(self.num_envs, dtype=bool)
        if self.config.max_episode_steps is not None:
            timeout |= self.episode_length >= int(self.config.max_episode_steps)
        timeout |= (self.duration_s > 0.0) & (self.elapsed_s >= self.duration_s)
        return timeout

    def _dt(self) -> float:
        return float(self.backend._dt if self.backend._dt is not None else 0.01)

    def _infos(self, done: np.ndarray, reasons: list[str]) -> list[dict[str, Any]]:
        assert self.snapshot is not None
        metrics = self.snapshot.arrays.metrics
        infos = []
        for i, label in enumerate(self.labels):
            intercepted = bool(metrics[i, 2] > 0.5)
            infos.append({
                "done": bool(done[i]),
                "terminal_reason": reasons[i],
                "scenario_index": int(label.scenario_index),
                "cell_index": -1 if label.cell_index is None else int(label.cell_index),
                "range_m": label.range_m,
                "closing_speed_mps": label.closing_speed_mps,
                "distance_m": float(metrics[i, 0]),
                "min_distance_m": float(metrics[i, 1]),
                "intercepted": intercepted,
                "intercept_time_s": float(metrics[i, 3]) if intercepted else None,
                "elapsed_s": float(self.elapsed_s[i]),
            })
        return infos

    def _episode_info(self, index: int, reason: str) -> dict[str, Any]:
        assert self.snapshot is not None
        metrics = self.snapshot.arrays.metrics
        label = self.labels[index]
        intercepted = bool(metrics[index, 2] > 0.5)
        return {
            "scenario_index": int(label.scenario_index),
            "cell_index": -1 if label.cell_index is None else int(label.cell_index),
            "range_m": label.range_m,
            "closing_speed_mps": label.closing_speed_mps,
            "return": float(self.episode_return[index]),
            "length": int(self.episode_length[index]),
            "min_distance_m": float(metrics[index, 1]),
            "time_to_catch_s": float(metrics[index, 3]) if intercepted else None,
            "intercepted": intercepted,
            "terminal_reason": reason,
        }
