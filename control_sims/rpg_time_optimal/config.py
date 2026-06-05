from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RpgTimeOptimalConfig:
    """Configuration for the RPG time-optimal trajectory planner."""

    cpc_tolerance_m: float | None = None
    plan_time_scale: float = 1.0
    motor_command_mode: str = "zoh"
    terminal_nodes: int = 30
    dynamics_substeps: int = 1
    planner_rate_limit_scale: float = 1.0
    command_smoothness_weight: float = 0.0
    body_rate_smoothness_weight: float = 0.0
    terminal_capture_window_nodes: int = 1
    velocity_guess_mps: float = 8.0
    ipopt_max_iter: int = 100
    ipopt_print_level: int = 0
    suppress_solver_stdout: bool = True
    position_gain: float = 4.0
    velocity_gain: float = 2.4
    max_tracking_accel_mps2: float = 35.0

    def __post_init__(self) -> None:
        if int(self.terminal_nodes) <= 0:
            raise ValueError("terminal_nodes must be positive")
        if int(self.dynamics_substeps) <= 0:
            raise ValueError("dynamics_substeps must be positive")
        if not (0.0 < float(self.planner_rate_limit_scale) <= 1.0):
            raise ValueError("planner_rate_limit_scale must be in (0, 1]")
        if float(self.command_smoothness_weight) < 0.0:
            raise ValueError("command_smoothness_weight must be non-negative")
        if float(self.body_rate_smoothness_weight) < 0.0:
            raise ValueError("body_rate_smoothness_weight must be non-negative")
        if int(self.terminal_capture_window_nodes) <= 0:
            raise ValueError("terminal_capture_window_nodes must be positive")
        if self.cpc_tolerance_m is not None and float(self.cpc_tolerance_m) <= 0.0:
            raise ValueError("cpc_tolerance_m must be positive when set")
        if float(self.plan_time_scale) <= 0.0:
            raise ValueError("plan_time_scale must be positive")
        if self.motor_command_mode not in {"zoh", "linear"}:
            raise ValueError("motor_command_mode must be 'zoh' or 'linear'")
        if float(self.velocity_guess_mps) <= 0.0:
            raise ValueError("velocity_guess_mps must be positive")
        if int(self.ipopt_max_iter) <= 0:
            raise ValueError("ipopt_max_iter must be positive")
        if float(self.max_tracking_accel_mps2) <= 0.0:
            raise ValueError("max_tracking_accel_mps2 must be positive")
