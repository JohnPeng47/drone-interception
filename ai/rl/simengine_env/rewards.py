from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    progress_weight: float = 28.0
    progress_sigma_m: float = 250.0
    distance_weight: float = 0.001
    distance_scale_m: float = 532.67
    fail_penalty: float = 30.0
    rate_weight: float = 2e-4


def compute_rewards(
    *,
    previous_distance_m: np.ndarray,
    distance_m: np.ndarray,
    body_rates_b: np.ndarray,
    failed: np.ndarray,
    config: RewardConfig,
) -> np.ndarray:
    previous = np.asarray(previous_distance_m, dtype=np.float32)
    distance = np.asarray(distance_m, dtype=np.float32)
    rates = np.asarray(body_rates_b, dtype=np.float32)
    failed_arr = np.asarray(failed, dtype=bool)
    previous_potential = np.exp(-previous / float(config.progress_sigma_m))
    potential = np.exp(-distance / float(config.progress_sigma_m))
    return (
        float(config.progress_weight) * (potential - previous_potential)
        - float(config.distance_weight) * (distance / float(config.distance_scale_m))
        - float(config.rate_weight) * np.linalg.norm(rates, axis=-1)
        - np.where(failed_arr, float(config.fail_penalty), 0.0)
    ).astype(np.float32)


def compute_reward(
    *,
    previous_distance_m: float,
    distance_m: float,
    body_rates_b: np.ndarray,
    intercepted: bool,
    failed: bool,
    config: RewardConfig,
) -> float:
    del intercepted
    return float(
        compute_rewards(
            previous_distance_m=np.asarray([previous_distance_m], dtype=np.float32),
            distance_m=np.asarray([distance_m], dtype=np.float32),
            body_rates_b=np.asarray(body_rates_b, dtype=np.float32).reshape(1, -1),
            failed=np.asarray([failed], dtype=bool),
            config=config,
        )[0]
    )
