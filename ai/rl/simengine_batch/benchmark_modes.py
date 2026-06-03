from __future__ import annotations

import importlib.util
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ai.rl.simengine_env.scenario_table import ScenarioTable

from .generator import BatchSimGenerator
from .runner import BatchRunnerConfig, BatchSimRunner


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


def run_simengine_batch_benchmark(
    scenario_path: str | Path,
    *,
    num_envs: int,
    steps: int,
    seed: int = 1,
    policy_latency_us: float = 0.0,
    max_episode_steps: int | None = None,
) -> BenchmarkResult:
    table = ScenarioTable(scenario_path)
    generator = BatchSimGenerator(table, num_envs=num_envs, seed=seed, strategy="sequential_epoch")
    runner = BatchSimRunner(generator, config=BatchRunnerConfig(max_episode_steps=max_episode_steps))
    obs, _ = runner.reset()
    actions = np.zeros((num_envs, runner.action_size), dtype=np.float32)
    terminal_count = 0
    start = time.perf_counter()
    for step in range(int(steps)):
        _fake_policy(obs, actions, step, policy_latency_us)
        obs, _rewards, dones, _infos = runner.step(actions)
        terminal_count += int(np.count_nonzero(dones))
    elapsed = max(time.perf_counter() - start, 1e-9)
    return BenchmarkResult(
        mode="simengine_batch",
        scenario_path=str(scenario_path),
        scenario_count=table.count,
        num_envs=int(num_envs),
        steps=int(steps),
        env_steps=int(num_envs) * int(steps),
        elapsed_s=elapsed,
        sim_sps=(int(num_envs) * int(steps)) / elapsed,
        policy_latency_us=float(policy_latency_us),
        terminal_count=terminal_count,
        obs_shape=tuple(int(x) for x in obs.shape),
    )


def run_puffer_native_benchmark(
    scenario_path: str | Path,
    *,
    num_envs: int,
    steps: int,
    seed: int = 1,
    policy_latency_us: float = 0.0,
    max_episode_steps: int | None = None,
) -> BenchmarkResult:
    backend_cls = _load_native_backend_class()
    backend = backend_cls(scenario_path, num_envs=num_envs, max_episode_steps=max_episode_steps)
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
    actions[:, 0] = np.float32(0.15) + np.float32(0.05) * np.sin(obs[:, 20] + phase)
    actions[:, 1] = np.tanh(obs[:, 18] * np.float32(1.5))
    actions[:, 2] = np.tanh(obs[:, 19] * np.float32(1.5))
    actions[:, 3] = np.float32(0.05) * np.sin(np.arange(n, dtype=np.float32) * np.float32(0.017) + phase)
    if policy_latency_us > 0.0:
        end = time.perf_counter() + float(policy_latency_us) / 1_000_000.0
        while time.perf_counter() < end:
            pass


def _load_native_backend_class():
    root = Path(__file__).resolve().parents[3]
    path = root / "ai" / "rl" / "puffer-intercept" / "native_backend.py"
    spec = importlib.util.spec_from_file_location("puffer_intercept_native_backend", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load native backend from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativeInterceptBackend
