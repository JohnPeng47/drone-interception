from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    catch_reward: float = 10.0
    distance_weight: float = 0.001
    progress_weight: float = 0.1
    fail_penalty: float = 30.0
    rate_weight: float = 2e-4


def compute_reward(
    *,
    previous_distance_m: float,
    distance_m: float,
    body_rates_b: np.ndarray,
    intercepted: bool,
    failed: bool,
    config: RewardConfig,
) -> float:
    progress = float(previous_distance_m) - float(distance_m)
    return float(
        (config.catch_reward if intercepted else 0.0)
        + config.progress_weight * progress
        - config.distance_weight * float(distance_m)
        - config.rate_weight * float(np.linalg.norm(body_rates_b))
        - (config.fail_penalty if failed else 0.0)
    )
