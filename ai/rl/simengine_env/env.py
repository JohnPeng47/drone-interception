from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backends.csim.bindings import PufferSimEngineBackend, SimInstance

from .observations import OBS_SIZE, observation_from_snapshot
from .rewards import RewardConfig, compute_reward
from .scenario_table import ScenarioLabel, ScenarioTable


@dataclass
class EpisodeStats:
    scenario_index: int = -1
    cell_index: int = -1
    range_m: float = float("nan")
    closing_speed_mps: float = float("nan")
    return_: float = 0.0
    length: int = 0
    min_distance_m: float = float("inf")
    time_to_catch_s: float = float("nan")
    terminal_reason: str = ""


@dataclass(frozen=True)
class EnvConfig:
    max_episode_steps: int | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)


class SimEngineInterceptEnv:
    action_size = 4
    observation_size = OBS_SIZE

    def __init__(self, table: ScenarioTable, *, seed: int = 1, config: EnvConfig | None = None):
        self.table = table
        self.rng = np.random.default_rng(int(seed))
        self.config = config or EnvConfig()
        self.backend: PufferSimEngineBackend | None = None
        self.snapshot: dict | None = None
        self.instance: SimInstance | None = None
        self.label = ScenarioLabel(-1, None, None, None)
        self.elapsed_s = 0.0
        self.episode_return = 0.0
        self.episode_length = 0
        self.previous_distance_m = 0.0

    def reset(self, scenario_index: int | None = None) -> tuple[np.ndarray, dict]:
        if scenario_index is None:
            scenario_index = int(self.rng.integers(0, self.table.count))
        scenario_index = int(scenario_index) % self.table.count
        self.instance = self.table.get(scenario_index)
        self.label = self.table.label(scenario_index)
        if self.instance.config is None:
            raise ValueError("SimEngine RL scenarios require SimInstance.config")
        self.backend = PufferSimEngineBackend(self.instance.config)
        self.snapshot = self.backend.reset(self.instance)
        self.elapsed_s = 0.0
        self.episode_return = 0.0
        self.episode_length = 0
        self.previous_distance_m = float(self.snapshot["metrics"]["distance_m"])
        return self._obs(), self._info(done=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        if self.snapshot is None or self.backend is None or self.instance is None:
            self.reset()
        assert self.snapshot is not None
        assert self.backend is not None
        assert self.instance is not None
        action = np.asarray(action, dtype=float).reshape(4)
        command = self._command_from_action(action)
        self.snapshot = self.backend.step_ctbr(self.snapshot, command)
        self.elapsed_s += self._dt()
        self.episode_length += 1

        metrics = self.snapshot["metrics"]
        distance_m = float(metrics["distance_m"])
        intercepted = bool(metrics["intercepted"])
        failed, fail_reason = self._failed()
        timeout = self._timeout()
        done = intercepted or failed or timeout
        reward = compute_reward(
            previous_distance_m=self.previous_distance_m,
            distance_m=distance_m,
            body_rates_b=command["body_rates_b"],
            intercepted=intercepted,
            failed=failed,
            config=self.config.reward,
        )
        self.previous_distance_m = distance_m
        self.episode_return += reward
        reason = "intercepted" if intercepted else fail_reason if failed else "timeout" if timeout else ""
        info = self._info(done=done, terminal_reason=reason)
        obs = self._obs()
        if done:
            info["episode"] = self._episode_stats(reason)
            obs, reset_info = self.reset()
            info["reset"] = reset_info
        return obs, reward, done, info

    def _command_from_action(self, action: np.ndarray) -> dict:
        assert self.instance is not None
        config = self.instance.config
        assert config is not None
        max_thrust = float(config.max_thrust_n or config.pursuer.mass_kg * config.pursuer.gravity_mps2 * 2.0)
        max_rate = float(config.max_rate_rps or config.pursuer.max_omega_rps)
        thrust_n = float(np.clip((action[0] + 1.0) * 0.5, 0.0, 1.0) * max_thrust)
        return {
            "thrust_n": thrust_n,
            "body_rates_b": np.clip(action[1:], -1.0, 1.0) * max_rate,
        }

    def _obs(self) -> np.ndarray:
        assert self.snapshot is not None
        max_rate = 20.0
        if self.instance is not None and self.instance.config is not None:
            max_rate = float(self.instance.config.max_rate_rps or self.instance.config.pursuer.max_omega_rps)
        return observation_from_snapshot(self.snapshot, max_rate_rps=max_rate)

    def _dt(self) -> float:
        assert self.instance is not None
        options = self.instance.config.options if self.instance.config is not None else None
        if options is None:
            return 0.005
        return float(options.backend_dt) * max(1, int(options.action_substeps))

    def _timeout(self) -> bool:
        assert self.instance is not None
        if self.config.max_episode_steps is not None and self.episode_length >= self.config.max_episode_steps:
            return True
        duration_s = self.instance.config.options.duration_s if self.instance.config is not None else 0.0
        return bool(duration_s > 0.0 and self.elapsed_s >= float(duration_s))

    def _failed(self) -> tuple[bool, str]:
        assert self.snapshot is not None
        pos = np.asarray(self.snapshot["vehicle_state"]["x"], dtype=float)
        if self.instance is not None and self.instance.config is not None and self.instance.config.bounds_w is not None:
            bounds = np.asarray(self.instance.config.bounds_w, dtype=float).reshape(3)
            if bool(np.any(np.abs(pos) > bounds)):
                return True, "oob"
        if not np.all(np.isfinite(pos)):
            return True, "nonfinite"
        return False, ""

    def _info(self, *, done: bool, terminal_reason: str = "") -> dict:
        assert self.snapshot is not None
        metrics = self.snapshot["metrics"]
        return {
            "done": bool(done),
            "terminal_reason": terminal_reason,
            "scenario_index": int(self.label.scenario_index),
            "cell_index": -1 if self.label.cell_index is None else int(self.label.cell_index),
            "range_m": self.label.range_m,
            "closing_speed_mps": self.label.closing_speed_mps,
            "distance_m": float(metrics["distance_m"]),
            "min_distance_m": float(metrics["min_distance_m"]),
            "intercepted": bool(metrics["intercepted"]),
            "intercept_time_s": metrics["intercept_time_s"],
            "elapsed_s": self.elapsed_s,
        }

    def _episode_stats(self, reason: str) -> dict:
        assert self.snapshot is not None
        metrics = self.snapshot["metrics"]
        return {
            "scenario_index": int(self.label.scenario_index),
            "cell_index": -1 if self.label.cell_index is None else int(self.label.cell_index),
            "range_m": self.label.range_m,
            "closing_speed_mps": self.label.closing_speed_mps,
            "return": float(self.episode_return),
            "length": int(self.episode_length),
            "min_distance_m": float(metrics["min_distance_m"]),
            "time_to_catch_s": metrics["intercept_time_s"],
            "intercepted": bool(metrics["intercepted"]),
            "terminal_reason": reason,
        }
