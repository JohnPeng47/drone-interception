from __future__ import annotations

import numpy as np

from ai.rl.simengine_env.env import SimEngineInterceptEnv
from ai.rl.simengine_env.scenario_table import ScenarioTable
from backends import (
    PursuerInitialState,
    PursuerParams,
    SimConfig,
    SimInstance,
    SimOptions,
    TargetConfig,
    TargetInitialState,
    write_sim_instances,
)


def test_scalar_simengine_env_uses_generated_scenario_bounds(tmp_path):
    path = tmp_path / "oob.csimin"
    instance = _instance(
        seed=300,
        position_w=np.array([2.0, 0.0, 0.0]),
        target_position_w=np.array([3.0, 0.0, 0.0]),
        bounds_w=(1.0, 1.0, 1.0),
        duration_s=1.0,
    )
    write_sim_instances(path, (instance,))
    env = SimEngineInterceptEnv(ScenarioTable(path), seed=1)
    env.reset(scenario_index=0)

    _, _, done, info = env.step(np.zeros(env.action_size, dtype=np.float32))

    assert done is True
    assert info["terminal_reason"] == "oob"


def _instance(
    *,
    seed: int,
    position_w: np.ndarray,
    target_position_w: np.ndarray,
    bounds_w: tuple[float, float, float],
    duration_s: float,
) -> SimInstance:
    params = PursuerParams(
        mass_kg=0.027,
        ixx=3.85e-6,
        iyy=3.85e-6,
        izz=5.9675e-6,
        arm_len_m=0.0396,
        k_thrust=3.16e-10,
        k_yaw=0.005964552,
        max_rpm=21702.0,
    )
    config = SimConfig(
        pursuer=params,
        options=SimOptions(duration_s=duration_s),
        targets=(TargetConfig(id="target", kind="target", radius_m=0.2),),
        intercept_radius_m=0.1,
        max_thrust_n=0.5,
        max_rate_rps=8.0,
        bounds_w=bounds_w,
    )
    return SimInstance(
        seed=seed,
        pursuer_initial=PursuerInitialState(
            position_w=np.asarray(position_w, dtype=float).copy(),
            velocity_w=np.zeros(3),
            quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            body_rates_b=np.zeros(3),
        ),
        target_initials=(
            TargetInitialState(
                position_w=np.asarray(target_position_w, dtype=float).copy(),
                velocity_w=np.zeros(3),
            ),
        ),
        config=config,
    )
