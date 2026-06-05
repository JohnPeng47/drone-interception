from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .native_backend import NativeInterceptBackend


@dataclass(frozen=True)
class BenchmarkResult:
    mode: str
    scenario_path: str
    scenario_count: int
    num_envs: int
    steps: int
    env_steps: int
    elapsed_s: float
    sim_sps: float
    policy_latency_us: float
    terminal_count: int
    obs_shape: tuple[int, int]

    def to_dict(self) -> dict:
        out = asdict(self)
        out["obs_shape"] = list(self.obs_shape)
        return out


def run_puffer_native_benchmark(
    scenario_path: str | Path,
    *,
    num_envs: int,
    steps: int,
    seed: int = 1,
    policy_latency_us: float = 0.0,
    max_episode_steps: int | None = None,
) -> BenchmarkResult:
    backend = NativeInterceptBackend(scenario_path, num_envs=num_envs, max_episode_steps=max_episode_steps)
    try:
        obs = backend.reset()
        actions = np.zeros((num_envs, 4), dtype=np.float32)
        terminal_count = 0
        start = time.perf_counter()
        for step in range(int(steps)):
            _fake_policy(obs, actions, step + int(seed), policy_latency_us)
            obs, _rewards, dones = backend.step(actions)
            terminal_count += int(np.count_nonzero(dones))
        elapsed = max(time.perf_counter() - start, 1e-9)
        return BenchmarkResult(
            mode="puffer_native",
            scenario_path=str(scenario_path),
            scenario_count=int(backend.scenario_count),
            num_envs=int(num_envs),
            steps=int(steps),
            env_steps=int(num_envs) * int(steps),
            elapsed_s=elapsed,
            sim_sps=(int(num_envs) * int(steps)) / elapsed,
            policy_latency_us=float(policy_latency_us),
            terminal_count=terminal_count,
            obs_shape=tuple(int(x) for x in obs.shape),
        )
    finally:
        backend.close()


def _fake_policy(obs: np.ndarray, actions: np.ndarray, step: int, policy_latency_us: float) -> None:
    n = actions.shape[0]
    phase = np.float32(step * 0.013)
    rel_pos = obs[:, 19:22] - obs[:, 0:3]
    range_m = np.linalg.norm(rel_pos, axis=1)
    actions[:, 0] = np.float32(0.15) + np.float32(0.05) * np.sin(range_m * np.float32(0.05) + phase)
    actions[:, 1] = np.tanh(rel_pos[:, 1] * np.float32(0.05))
    actions[:, 2] = np.tanh(rel_pos[:, 2] * np.float32(0.05))
    actions[:, 3] = np.float32(0.05) * np.sin(np.arange(n, dtype=np.float32) * np.float32(0.017) + phase)
    if policy_latency_us > 0.0:
        end = time.perf_counter() + float(policy_latency_us) / 1_000_000.0
        while time.perf_counter() < end:
            pass
