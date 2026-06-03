from __future__ import annotations

import contextlib
import importlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi as ca
import numpy as np

from backends.csim.bindings.types import PursuerParams, SimInstance

from .config import RpgTimeOptimalConfig


@dataclass(frozen=True)
class RpgTimeOptimalPlan:
    """A solved RPG trajectory in array form suitable for SimRunner policies."""

    seed: int
    solve_wall_s: float
    total_time_s: float
    t_x_s: np.ndarray
    t_u_s: np.ndarray
    position_w: np.ndarray
    velocity_w: np.ndarray
    acceleration_w: np.ndarray
    quat_wxyz: np.ndarray
    body_rates_b: np.ndarray
    motor_thrusts_n: np.ndarray
    motor_speed_commands_rpm: np.ndarray | None = None


class RpgTimeOptimalAdapter:
    """Build and solve RPG time-optimal trajectories for typed SimInstances."""

    def __init__(self, config: RpgTimeOptimalConfig | None = None):
        self.config = config or RpgTimeOptimalConfig()
        self._modules: dict[str, Any] | None = None

    def solve(self, instance: SimInstance) -> RpgTimeOptimalPlan:
        if instance.config is None:
            raise ValueError("RPG time-optimal adapter requires SimInstance.config")

        # 1. Entry Point: use an explicit CPC tolerance when configured, otherwise
        # use the scenario intercept radius as the terminal capture radius.
        tolerance = (
            float(self.config.cpc_tolerance_m)
            if self.config.cpc_tolerance_m is not None
            else float(instance.config.intercept_radius_m)
        )
        tolerance = max(tolerance, 1.0e-6)

        stream = io.StringIO()
        stdout_context = (
            contextlib.redirect_stdout(stream)
            if bool(self.config.suppress_solver_stdout)
            else contextlib.nullcontext()
        )
        start = time.perf_counter()
        with stdout_context:
            plan = self._solve_terminal_ocp(instance, tolerance)
        solve_wall_s = time.perf_counter() - start
        return _plan_with_wall_time(plan, solve_wall_s)

    def _solve_terminal_ocp(
        self,
        instance: SimInstance,
        tolerance_m: float,
    ) -> RpgTimeOptimalPlan:
        """
        x = [
            p_w        3,  world position
            v_w        3,  world velocity
            q_wxyz     4,  attitude quaternion
            omega_b    3,  body rates
            rpm        4,  actual motor speeds
        ]
        """
        assert instance.config is not None
        params = instance.config.pursuer
        n = int(self.config.terminal_nodes)
        state_size = 17
        max_rate = _max_rate_rps(instance)
        rpm_min = _min_rpm(params)
        rpm_max = float(params.max_rpm)
        max_thrust_n = _max_collective_thrust_n(instance)
        thrust_limited_rpm = _thrust_limited_rpm(instance)

        # 2. State Definition: each x is 17D:
        # [p_w(3), v_w(3), q_wxyz(4), omega_b(3), actual_motor_rpm(4)].
        # The control u is commanded motor RPM; motor lag maps u toward the
        # actual_motor_rpm state inside the dynamics.
        variables = []
        guesses = []
        constraints = []
        lower = []
        upper = []

        # 3. Decision Variables: the NLP optimizes total time T, the fixed
        # initial state x_0, and each pair (u_k, x_{k+1}) for k in [0, N).
        total_time = ca.MX.sym("t", 1)
        variables.append(total_time)
        initial_position = np.asarray(instance.pursuer_initial.position_w, dtype=float).reshape(3)
        target_position = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3)
        distance = float(np.linalg.norm(target_position - initial_position))
        guesses.append([max(distance / float(self.config.velocity_guess_mps), 0.1)])
        constraints.append(total_time)
        lower.append([0.05])
        upper.append([150.0])

        state = ca.MX.sym("x0", state_size)
        variables.append(state)
        x0 = _initial_state_wxyz_rpm(instance, rpm_min, rpm_max)
        guesses.append(x0)
        # 4. Initial Condition Constraint: x_0 = x_initial. Bounds are expressed
        # through lbg/ubg, so equal lower and upper values pin the state exactly.
        constraints.append(state)
        lower.append(x0)
        upper.append(x0)

        q_guess = x0[6:10]
        initial_rpm = np.asarray(x0[13:17], dtype=float)
        for index in range(n):
            control = ca.MX.sym(f"u{index}", 4)
            variables.append(control)
            guesses.append([thrust_limited_rpm] * 4)
            # 5. Control Bounds: rpm_min <= u_k[i] <= rpm_max for each motor.
            constraints.append(control)
            lower.append([rpm_min] * 4)
            upper.append([rpm_max] * 4)

            next_state = ca.MX.sym(f"x{index + 1}", state_size)
            variables.append(next_state)
            # 6. Dynamics Constraint: x_{k+1} = RK4(f, x_k, u_k, T / N).
            predicted = _rk4_motor_lag_step(state, control, total_time / n, params)
            constraints.append(next_state - predicted)
            lower.append([0.0] * state_size)
            upper.append([0.0] * state_size)

            alpha = float(index + 1) / float(n)
            position_guess = (1.0 - alpha) * initial_position + alpha * target_position
            velocity_guess = (target_position - initial_position)
            velocity_norm = float(np.linalg.norm(velocity_guess))
            if velocity_norm > 1.0e-9:
                velocity_guess = velocity_guess * (float(self.config.velocity_guess_mps) / velocity_norm)
            rpm_guess = (1.0 - alpha) * initial_rpm + alpha * np.full(4, thrust_limited_rpm, dtype=float)
            guesses.append([*position_guess, *velocity_guess, *q_guess, 0.0, 0.0, 0.0, *rpm_guess])

            # 9. Per-Node Physical Constraints: keep body rates, altitude, actual
            # motor speeds, and collective thrust inside the SimEngine limits.
            constraints.append(next_state[10:13])
            lower.append([-max_rate, -max_rate, -max_rate])
            upper.append([max_rate, max_rate, max_rate])

            constraints.append(next_state[2])
            lower.append([0.5])
            upper.append([100.0])

            constraints.append(next_state[13:17])
            lower.append([rpm_min] * 4)
            upper.append([rpm_max] * 4)

            constraints.append(_collective_thrust_from_rpm(next_state[13:17], params))
            lower.append([0.0])
            upper.append([max_thrust_n])
            state = next_state

        target_velocity = np.asarray(instance.target_initial.velocity_w, dtype=float).reshape(3)
        terminal_target = target_position + target_velocity * total_time
        terminal_error = state[0:3] - terminal_target
        # 10. Terminal Intercept Constraint:
        # ||p_N - (p_target_0 + v_target_0 * T)||^2 <= tolerance_m^2.
        constraints.append(ca.dot(terminal_error, terminal_error))
        lower.append([0.0])
        upper.append([float(tolerance_m) ** 2])

        # 3. Decision Variables, continued: minimize total time T over the flat
        # decision vector [T, x_0, u_0, x_1, ..., u_{N-1}, x_N].
        nlp = {
            "f": total_time,
            "x": ca.vertcat(*variables),
            "g": ca.vertcat(*constraints),
        }
        # 11. IPOPT Solve: CasADi receives the initial guess plus lower/upper
        # bounds for every constraint, including equality constraints.
        solver = ca.nlpsol(
            "solver",
            "ipopt",
            nlp,
            {
                "ipopt": {
                    "max_iter": int(self.config.ipopt_max_iter),
                    "print_level": int(self.config.ipopt_print_level),
                },
                "print_time": 0,
            },
        )
        solution = solver(
            x0=ca.veccat(*guesses),
            lbg=ca.veccat(*lower),
            ubg=ca.veccat(*upper),
        )["x"].full().reshape(-1)
        return _plan_from_terminal_solution(instance, solution, n, params)

    def _load_modules(self) -> dict[str, Any]:
        if self._modules is not None:
            return self._modules

        root = Path(self.config.solver_root)
        if not root.is_absolute():
            root = Path.cwd() / root
        src = root / "src"
        if not src.exists():
            raise FileNotFoundError(f"RPG time-optimal solver source not found: {src}")
        src_text = str(src)
        if src_text not in sys.path:
            sys.path.insert(0, src_text)

        self._modules = {
            "Track": importlib.import_module("track").Track,
            "Quad": importlib.import_module("quad").Quad,
            "RungeKutta4": importlib.import_module("integrator").RungeKutta4,
            "Planner": importlib.import_module("planner").Planner,
            "Trajectory": importlib.import_module("trajectory").Trajectory,
        }
        return self._modules

    @staticmethod
    def _build_track(instance: SimInstance, track_cls: type) -> Any:
        track = track_cls()
        q_xyzw = np.asarray(instance.pursuer_initial.quat_xyzw, dtype=float).reshape(4)
        track.init_pos = np.asarray(instance.pursuer_initial.position_w, dtype=float).reshape(3).tolist()
        track.init_vel = np.asarray(instance.pursuer_initial.velocity_w, dtype=float).reshape(3).tolist()
        track.init_att = [float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]
        track.init_omega = np.asarray(instance.pursuer_initial.body_rates_b, dtype=float).reshape(3).tolist()
        track.end_pos = np.asarray(instance.target_initial.position_w, dtype=float).reshape(3).tolist()
        return track

    @staticmethod
    def _build_quad(instance: SimInstance, quad_cls: type) -> Any:
        assert instance.config is not None
        params = instance.config.pursuer
        quad = quad_cls()
        quad.m = float(params.mass_kg)
        quad.l = float(params.arm_len_m)
        quad.I = ca.DM(
            [
                [float(params.ixx), 0.0, 0.0],
                [0.0, float(params.iyy), 0.0],
                [0.0, 0.0, float(params.izz)],
            ]
        )
        quad.I_inv = ca.inv(quad.I)
        quad.T_max = _max_motor_thrust_n(instance)
        quad.T_min = 0.0
        max_rate = float(instance.config.max_rate_rps)
        if max_rate <= 0.0:
            max_rate = float(params.max_omega_rps)
        quad.omega_max_xy = max_rate
        quad.omega_max_z = max_rate
        quad.ctau = float(params.k_yaw)
        quad.g = float(params.gravity_mps2)
        quad.rampup_dist = 0.0
        quad.T_ramp_start = quad.T_max
        quad.omega_ramp_start = max_rate
        quad.cd = 0.0
        return quad


def _plan_from_trajectory(
    instance: SimInstance,
    trajectory: Any,
    solve_wall_s: float,
) -> RpgTimeOptimalPlan:
    t_x = np.asarray(trajectory.t_x, dtype=float).reshape(-1)
    t_u = np.asarray(trajectory.t_u, dtype=float).reshape(-1)
    position = np.asarray(trajectory.p, dtype=float).reshape(3, -1)
    velocity = np.asarray(trajectory.v, dtype=float).reshape(3, -1)
    quat_wxyz = np.asarray(trajectory.q, dtype=float).reshape(4, -1)
    body_rates = np.asarray(trajectory.w, dtype=float).reshape(3, -1)
    motor_thrusts = np.asarray(trajectory.u, dtype=float).reshape(4, -1)
    acceleration = np.zeros_like(position)
    if t_x.size > 1:
        dt = max(float(t_x[1] - t_x[0]), 1.0e-9)
        acceleration[:, :-1] = np.diff(velocity, axis=1) / dt
        acceleration[:, -1] = acceleration[:, -2]
    return RpgTimeOptimalPlan(
        seed=int(instance.seed),
        solve_wall_s=float(solve_wall_s),
        total_time_s=float(trajectory.t_total),
        t_x_s=t_x,
        t_u_s=t_u,
        position_w=position,
        velocity_w=velocity,
        acceleration_w=acceleration,
        quat_wxyz=quat_wxyz,
        body_rates_b=body_rates,
        motor_thrusts_n=motor_thrusts,
        motor_speed_commands_rpm=None,
    )


def _plan_from_terminal_solution(
    instance: SimInstance,
    solution: np.ndarray,
    nodes: int,
    params: PursuerParams,
) -> RpgTimeOptimalPlan:
    # 12. Plan Extraction: unpack [T, x_0, u_0, x_1, ..., u_{N-1}, x_N]
    # into time-indexed arrays that SimRunner policies can sample.
    values = np.asarray(solution, dtype=float).reshape(-1)
    total_time = float(values[0])
    cursor = 1
    states = [values[cursor:cursor + 17].copy()]
    cursor += 17
    controls = []
    for _ in range(nodes):
        controls.append(values[cursor:cursor + 4].copy())
        cursor += 4
        states.append(values[cursor:cursor + 17].copy())
        cursor += 17

    state_arr = np.asarray(states, dtype=float).T
    control_arr = np.asarray(controls, dtype=float).T
    rpm_arr = np.clip(state_arr[13:17], 0.0, None)
    thrust_arr = float(params.k_thrust) * np.square(rpm_arr[:, :-1])
    t_x = np.linspace(0.0, total_time, nodes + 1)
    t_u = t_x[:-1].copy()
    acceleration = np.zeros((3, nodes + 1), dtype=float)
    if nodes > 0:
        dt = max(total_time / nodes, 1.0e-9)
        acceleration[:, :-1] = np.diff(state_arr[3:6], axis=1) / dt
        acceleration[:, -1] = acceleration[:, -2]
    return RpgTimeOptimalPlan(
        seed=int(instance.seed),
        solve_wall_s=0.0,
        total_time_s=total_time,
        t_x_s=t_x,
        t_u_s=t_u,
        position_w=state_arr[0:3],
        velocity_w=state_arr[3:6],
        acceleration_w=acceleration,
        quat_wxyz=state_arr[6:10],
        body_rates_b=state_arr[10:13],
        motor_thrusts_n=thrust_arr,
        motor_speed_commands_rpm=control_arr,
    )


def _plan_with_wall_time(plan: RpgTimeOptimalPlan, solve_wall_s: float) -> RpgTimeOptimalPlan:
    return RpgTimeOptimalPlan(
        seed=plan.seed,
        solve_wall_s=float(solve_wall_s),
        total_time_s=plan.total_time_s,
        t_x_s=plan.t_x_s,
        t_u_s=plan.t_u_s,
        position_w=plan.position_w,
        velocity_w=plan.velocity_w,
        acceleration_w=plan.acceleration_w,
        quat_wxyz=plan.quat_wxyz,
        body_rates_b=plan.body_rates_b,
        motor_thrusts_n=plan.motor_thrusts_n,
        motor_speed_commands_rpm=plan.motor_speed_commands_rpm,
    )


def _rk4_motor_lag_step(x: ca.MX, u: ca.MX, dt: ca.MX, params: PursuerParams) -> ca.MX:
    # 8. RK4 Integration: classical fourth-order integration. Quaternion
    # components are normalized at intermediate and final states.
    k1 = _motor_lag_dynamics(x, u, params)
    k2 = _motor_lag_dynamics(_normalize_state_quat(x + 0.5 * dt * k1), u, params)
    k3 = _motor_lag_dynamics(_normalize_state_quat(x + 0.5 * dt * k2), u, params)
    k4 = _motor_lag_dynamics(_normalize_state_quat(x + dt * k3), u, params)
    return _normalize_state_quat(x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _motor_lag_dynamics(x: ca.MX, u: ca.MX, params: PursuerParams) -> ca.MX:
    # 7. Continuous Dynamics:
    # thrust_i = k_thrust * rpm_i^2
    # a_w = R(q) * [0, 0, sum(thrust_i)] / m - [0, 0, g] + optional drag
    # q_dot = 0.5 * q (*) [0, omega_b]
    # rpm_dot = (commanded_rpm - actual_rpm) / motor_tau_s
    velocity = x[3:6]
    quat = _normalize_quat_wxyz(x[6:10])
    omega = x[10:13]
    rpm = x[13:17]
    thrusts = float(params.k_thrust) * ca.power(rpm, 2)
    thrust_total = ca.sum1(thrusts)
    force_w = _quat_rotate_wxyz(quat, ca.vertcat(0.0, 0.0, thrust_total))
    accel_w = force_w / float(params.mass_kg)
    if float(params.b_drag) != 0.0:
        accel_w += (-float(params.b_drag) * velocity) / float(params.mass_kg)
    accel_w += ca.vertcat(0.0, 0.0, -float(params.gravity_mps2))

    qdot = 0.5 * _quat_mul_wxyz(quat, ca.vertcat(0.0, omega))
    moment = _sim_moment_from_thrusts(thrusts, params)
    if float(params.k_ang_damp) != 0.0:
        moment += -float(params.k_ang_damp) * omega
    wx, wy, wz = omega[0], omega[1], omega[2]
    ixx, iyy, izz = float(params.ixx), float(params.iyy), float(params.izz)
    inertial = ca.vertcat((iyy - izz) * wy * wz, (izz - ixx) * wz * wx, (ixx - iyy) * wx * wy)
    omega_dot = ca.vertcat(
        (moment[0] + inertial[0]) / ixx,
        (moment[1] + inertial[1]) / iyy,
        (moment[2] + inertial[2]) / izz,
    )
    rpm_dot = (u - rpm) / max(float(params.motor_tau_s), 1.0e-6)
    return ca.vertcat(velocity, accel_w, qdot, omega_dot, rpm_dot)


def _sim_moment_from_thrusts(thrusts: ca.MX, params: PursuerParams) -> ca.MX:
    rotor_positions = params.rotor_positions_b
    rotor_directions = params.rotor_directions
    if rotor_positions is None or rotor_directions is None:
        arm_factor = float(params.arm_len_m) / np.sqrt(2.0)
        return ca.vertcat(
            arm_factor * ((thrusts[2] + thrusts[3]) - (thrusts[0] + thrusts[1])),
            arm_factor * ((thrusts[1] + thrusts[2]) - (thrusts[0] + thrusts[3])),
            float(params.k_yaw) * (-thrusts[0] + thrusts[1] - thrusts[2] + thrusts[3]),
        )
    rotor_positions_arr = np.asarray(rotor_positions, dtype=float).reshape(4, 3)
    rotor_directions_arr = np.asarray(rotor_directions, dtype=float).reshape(4)
    return ca.vertcat(
        ca.dot(ca.DM(rotor_positions_arr[:, 1]), thrusts),
        ca.dot(ca.DM(-rotor_positions_arr[:, 0]), thrusts),
        ca.dot(ca.DM(float(params.k_yaw) * rotor_directions_arr), thrusts),
    )


def _normalize_state_quat(x: ca.MX) -> ca.MX:
    return ca.vertcat(x[0:6], _normalize_quat_wxyz(x[6:10]), x[10:17])


def _normalize_quat_wxyz(q: ca.MX) -> ca.MX:
    return q / ca.sqrt(ca.dot(q, q) + 1.0e-12)


def _quat_rotate_wxyz(q: ca.MX, v: ca.MX) -> ca.MX:
    rotated = _quat_mul_wxyz(_quat_mul_wxyz(q, ca.vertcat(0.0, v)), ca.vertcat(q[0], -q[1], -q[2], -q[3]))
    return rotated[1:4]


def _quat_mul_wxyz(a: ca.MX, b: ca.MX) -> ca.MX:
    return ca.vertcat(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


def _initial_state_wxyz_rpm(instance: SimInstance, rpm_min: float, rpm_max: float) -> list[float]:
    # 4. Initial Condition Constraint, continued: if the SimInstance does not
    # provide initial rotor speeds, initialize the fixed x_0 RPM state at hover.
    base = _initial_state_wxyz(instance)
    if instance.pursuer_initial.rotor_speeds is None:
        assert instance.config is not None
        rpm = np.full(4, _hover_rpm(instance.config.pursuer), dtype=float)
    else:
        rpm = np.asarray(instance.pursuer_initial.rotor_speeds, dtype=float).reshape(4)
    rpm = np.clip(rpm, float(rpm_min), float(rpm_max))
    return [*base, *rpm]


def _initial_state_wxyz(instance: SimInstance) -> list[float]:
    q_xyzw = np.asarray(instance.pursuer_initial.quat_xyzw, dtype=float).reshape(4)
    return [
        *np.asarray(instance.pursuer_initial.position_w, dtype=float).reshape(3),
        *np.asarray(instance.pursuer_initial.velocity_w, dtype=float).reshape(3),
        float(q_xyzw[3]),
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        *np.asarray(instance.pursuer_initial.body_rates_b, dtype=float).reshape(3),
    ]


def _max_rate_rps(instance: SimInstance) -> float:
    assert instance.config is not None
    if float(instance.config.max_rate_rps) > 0.0:
        return float(instance.config.max_rate_rps)
    return float(instance.config.pursuer.max_omega_rps)


def _hover_rpm(params: PursuerParams) -> float:
    return float(np.sqrt((float(params.mass_kg) * float(params.gravity_mps2)) / (4.0 * float(params.k_thrust))))


def _min_rpm(params: PursuerParams) -> float:
    if params.rpm_min is not None:
        return float(np.clip(params.rpm_min, 0.0, params.max_rpm))
    min_rpm = 2.0 * _hover_rpm(params) - float(params.max_rpm)
    return float(np.clip(min_rpm, 0.0, float(params.max_rpm)))


def _collective_thrust_from_rpm(rpm: ca.MX, params: PursuerParams) -> ca.MX:
    return float(params.k_thrust) * ca.sum1(ca.power(rpm, 2))


def _max_collective_thrust_n(instance: SimInstance) -> float:
    assert instance.config is not None
    if float(instance.config.max_thrust_n) > 0.0:
        return float(instance.config.max_thrust_n)
    params = instance.config.pursuer
    return float(4.0 * params.mass_kg * params.gravity_mps2)


def _thrust_limited_rpm(instance: SimInstance) -> float:
    assert instance.config is not None
    params = instance.config.pursuer
    per_motor_thrust = _max_collective_thrust_n(instance) / 4.0
    rpm = np.sqrt(max(per_motor_thrust, 0.0) / max(float(params.k_thrust), 1.0e-12))
    return float(np.clip(rpm, _min_rpm(params), float(params.max_rpm)))


def _max_motor_thrust_n(instance: SimInstance) -> float:
    assert instance.config is not None
    if float(instance.config.max_thrust_n) > 0.0:
        return float(instance.config.max_thrust_n) / 4.0
    params = instance.config.pursuer
    return float(params.mass_kg * params.gravity_mps2)
