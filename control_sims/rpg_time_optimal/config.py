from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RpgTimeOptimalConfig:
    """Configuration for the RPG time-optimal trajectory adapter."""

    solver_root: Path = Path("tools/rpg_time_optimal")
    cpc_tolerance_m: float | None = None
    plan_time_scale: float = 1.0
    terminal_nodes: int = 30
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
        if self.cpc_tolerance_m is not None and float(self.cpc_tolerance_m) <= 0.0:
            raise ValueError("cpc_tolerance_m must be positive when set")
        if float(self.plan_time_scale) <= 0.0:
            raise ValueError("plan_time_scale must be positive")
        if float(self.velocity_guess_mps) <= 0.0:
            raise ValueError("velocity_guess_mps must be positive")
        if int(self.ipopt_max_iter) <= 0:
            raise ValueError("ipopt_max_iter must be positive")
        if float(self.max_tracking_accel_mps2) <= 0.0:
            raise ValueError("max_tracking_accel_mps2 must be positive")
