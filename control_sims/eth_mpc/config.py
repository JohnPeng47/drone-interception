from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EthMpcConfig:
    """Configuration for the ETH MPCC++-style pursuit controller."""

    horizon_steps: int = 8
    horizon_dt_s: float = 0.05
    solve_period_s: float = 0.10
    target_lookahead_s: float = 0.55
    approach_speed_mps: float = 11.0
    max_progress_speed_mps: float = 24.0
    max_pred_speed_mps: float = 35.0
    max_accel_mps2: float = 32.0
    max_lateral_accel_mps2: float = 18.0
    tunnel_radius_m: float = 5.0
    min_tunnel_width_m: float = 0.75
    q_lag: float = 0.20
    q_contour: float = 0.35
    q_terminal: float = 2.5
    q_terminal_set: float = 1.5
    q_velocity: float = 0.025
    q_accel: float = 0.010
    q_accel_rate: float = 0.020
    q_progress_rate: float = 0.015
    q_body_rate: float = 0.015
    q_thrust_rate: float = 0.002
    q_tunnel: float = 1.0
    progress_reward: float = 1.8
    intercept_radius_weight: float = 5.0
    attitude_rate_gain: float = 4.0
    drag_diag: tuple[float, float, float] = (0.06, 0.06, 0.12)
    optimizer_maxiter: int = 4

    def __post_init__(self) -> None:
        if int(self.horizon_steps) <= 0:
            raise ValueError("horizon_steps must be positive")
        if float(self.horizon_dt_s) <= 0.0:
            raise ValueError("horizon_dt_s must be positive")
        if float(self.solve_period_s) <= 0.0:
            raise ValueError("solve_period_s must be positive")
        if float(self.approach_speed_mps) <= 0.0:
            raise ValueError("approach_speed_mps must be positive")
        if float(self.max_accel_mps2) <= 0.0:
            raise ValueError("max_accel_mps2 must be positive")
        if int(self.optimizer_maxiter) <= 0:
            raise ValueError("optimizer_maxiter must be positive")
