from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import EnvConfig, SimEngineInterceptEnv
from .scenario_table import ScenarioTable


@dataclass(frozen=True)
class WorkerConfig:
    scenario_table: str
    manifest: str | None
    envs_per_worker: int
    seed: int
    max_scenarios: int | None = None
    max_episode_steps: int | None = None


class ParallelSimEngineVectorEnv:
    def __init__(
        self,
        *,
        scenario_table: str,
        manifest: str | None,
        num_workers: int,
        envs_per_worker: int,
        seed: int = 1,
        max_scenarios: int | None = None,
        max_episode_steps: int | None = None,
    ):
        self.num_workers = int(num_workers)
        self.envs_per_worker = int(envs_per_worker)
        self.num_envs = self.num_workers * self.envs_per_worker
        if self.num_workers <= 0 or self.envs_per_worker <= 0:
            raise ValueError("num_workers and envs_per_worker must be positive")
        ctx = mp.get_context("spawn")
        self._parents = []
        self._procs = []
        for worker_id in range(self.num_workers):
            parent, child = ctx.Pipe()
            cfg = WorkerConfig(
                scenario_table=str(scenario_table),
                manifest=None if manifest is None else str(manifest),
                envs_per_worker=self.envs_per_worker,
                seed=int(seed) + 1009 * worker_id,
                max_scenarios=max_scenarios,
                max_episode_steps=max_episode_steps,
            )
            proc = ctx.Process(target=_worker_main, args=(child, cfg), daemon=True)
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)
        self.observation_size = int(self._broadcast("obs_size")[0])
        self.action_size = int(self._broadcast("action_size")[0])

    def reset(self) -> tuple[np.ndarray, list[dict[str, Any]]]:
        results = self._broadcast("reset")
        obs = np.concatenate([item[0] for item in results], axis=0)
        infos = [info for item in results for info in item[1]]
        return obs, infos

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        actions = np.asarray(actions, dtype=np.float32).reshape(self.num_envs, self.action_size)
        offset = 0
        for parent in self._parents:
            chunk = actions[offset: offset + self.envs_per_worker]
            parent.send(("step", chunk))
            offset += self.envs_per_worker
        results = [parent.recv() for parent in self._parents]
        obs = np.concatenate([item[0] for item in results], axis=0)
        rewards = np.concatenate([item[1] for item in results], axis=0)
        dones = np.concatenate([item[2] for item in results], axis=0)
        infos = [info for item in results for info in item[3]]
        return obs, rewards, dones, infos

    def close(self) -> None:
        for parent in self._parents:
            try:
                parent.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()

    def _broadcast(self, command: str):
        for parent in self._parents:
            parent.send((command, None))
        return [parent.recv() for parent in self._parents]

    def __enter__(self) -> "ParallelSimEngineVectorEnv":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def _worker_main(conn, cfg: WorkerConfig) -> None:
    table = ScenarioTable(cfg.scenario_table, manifest_path=cfg.manifest, max_scenarios=cfg.max_scenarios)
    env_config = EnvConfig(max_episode_steps=cfg.max_episode_steps)
    envs = [
        SimEngineInterceptEnv(table, seed=cfg.seed + idx, config=env_config)
        for idx in range(cfg.envs_per_worker)
    ]
    try:
        while True:
            command, payload = conn.recv()
            if command == "close":
                return
            if command == "obs_size":
                conn.send(SimEngineInterceptEnv.observation_size)
            elif command == "action_size":
                conn.send(SimEngineInterceptEnv.action_size)
            elif command == "reset":
                obs_infos = [env.reset() for env in envs]
                conn.send((
                    np.stack([item[0] for item in obs_infos], axis=0),
                    [item[1] for item in obs_infos],
                ))
            elif command == "step":
                actions = np.asarray(payload, dtype=np.float32)
                steps = [env.step(actions[idx]) for idx, env in enumerate(envs)]
                conn.send((
                    np.stack([item[0] for item in steps], axis=0),
                    np.asarray([item[1] for item in steps], dtype=np.float32),
                    np.asarray([item[2] for item in steps], dtype=bool),
                    [item[3] for item in steps],
                ))
            else:
                raise ValueError(f"unknown worker command {command!r}")
    finally:
        conn.close()
