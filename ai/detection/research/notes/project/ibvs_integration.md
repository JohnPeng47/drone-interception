# IBVS Controller + Trajectory Evader Integration Plan

Concrete additive integration of Yan/Yang's IBVS+PNG controller into the existing
`data/docs/research/drones/interception` simulator. Single objective: produce
trajectory rollouts of (pursuer_pose, target_pose) tuples for the Phase 1
synthetic-data pipeline ([→ pipeline.md](synthetic/pipeline.md)).

## Scope and non-goals

In scope:
- `IBVSController` class implementing Yan/Yang's analytical IBVS+PNG, conforming
  to the existing controller interface (`params`, `horizon_steps`, `solve()`,
  `command_to_rotor_thrusts()`).
- `CameraParams` struct (intrinsics, FOV).
- `ScriptedTrajectoryEvader` evader policy following a parameterized waypoint
  path with realistic accelerations.
- Driver script + smoke-test validation.
- Trajectory-batch generator that emits `(pursuer_pose, target_pose, range,
  bearing, target_size_px)` tuples for the renderer.

Not in scope (defer to Phase 2 sim work):
- Delayed Kalman Filter observer.
- Wiener-process IMU bias model.
- Delayed-pixel measurement buffer.
- Frame-rate sweep / FOV-loss recovery logic.
- Sim2real residual modeling.

The Phase 1 trajectory dataset only needs the *pose distribution* an interceptor
produces. Sensor-realism layers don't change that distribution materially —
they're load-bearing for Phase 2's RL controller, not for Phase 1.

## Existing architecture (read before modifying)

```diagram
┌─ run_episode() ──────────────────────────────────────────────────────────────┐
│                                                                               │
│   for step in range(num_steps):                                               │
│     target_positions = build_target_trajectory(evader, horizon, dt, p_pursuer)│
│     command          = controller.solve(state, target_positions)              │
│     rotor_command    = controller.command_to_rotor_thrusts(state, command, dt)│
│     state            = integrate_rk4(quad_dynamics_numpy, state, rotor_cmd)   │
│     evader_pos, _    = evader.step(dt, state[SX.position])                    │
│                                                                               │
│   returns EpisodeResult                                                       │
└───────────────────────────────────────────────────────────────────────────────┘

state    : 22-D [p(3), v(3), R(9), ω(3), f(4)]      (dynamics.py:11-25)
command  : 4-D for both SRT and CTBR controllers
evader   : Protocol with reset/state/step/predict_positions   (evader.py:9-24)
```

Files to extend (additive only — no edits to existing code):
- `src/interception/ibvs.py`            (new)
- `src/interception/camera.py`          (new)
- `src/interception/evader.py`          (add `ScriptedTrajectoryEvader`)
- `scripts/run_ibvs_canonical.py`       (new driver)
- `scripts/generate_trajectory_dataset.py`  (new batch generator)
- `tests/test_ibvs.py`                  (new)
- `configs/camera.yaml`                 (new)

## Step 1 — Camera parameters

New file: `src/interception/camera.py`

```python
@dataclass(frozen=True)
class CameraParams:
    image_width:  int       # px
    image_height: int       # px
    focal_length: float     # px (treat fx = fy)
    hfov_rad: float         # derived: 2*arctan(W/(2f))
    vfov_rad: float         # derived: 2*arctan(H/(2f))
    body_to_camera: np.ndarray  # 3x3, rotation from body frame to camera frame
                                # default = identity for forward-strapdown camera

    @classmethod
    def from_yaml(cls, path: Path) -> "CameraParams": ...

    def project(self, p_camera: np.ndarray) -> np.ndarray:
        """Pinhole projection. Returns normalized image coords (px/foc, py/foc)."""
        # ip̄ = [p_x / p_z, p_y / p_z]   with p_camera in CCS
        # caller checks p_z > 0 and FOV gate

    def in_fov(self, p_camera: np.ndarray) -> bool:
        """True if target is in front of camera AND within HFOV/VFOV."""
```

`configs/camera.yaml`: a 1920×1080, 60° HFOV camera. Matches a typical FPV cam.

## Step 2 — IBVS controller

New file: `src/interception/ibvs.py`

The Yan/Yang collinear control law produces `[T_cmd, ω_des]` from a single image
observation plus IMU. We replicate the analytical form, **using ground-truth
target world position projected through the pursuer's known pose** as the image
input. (The point of Phase 1 is to *generate* trajectories, not to test
robustness to perception noise — the synthetic data pipeline is what handles
perception realism downstream.)

```python
class IBVSController:
    control_type = "ctbr"

    def __init__(
        self,
        params: QuadrotorParams,
        camera: CameraParams,
        approach_speed: float = 8.0,    # m/s — desired closing speed cap
        k_omega: float = 4.0,           # rate gain for collinear control
        k_thrust: float = 12.0,         # thrust feedback gain
        ω_max: float = 6.0,             # rad/s body-rate clamp
    ) -> None:
        self.params = params
        self.camera = camera
        self.approach_speed = approach_speed
        self.k_omega = k_omega
        self.k_thrust = k_thrust
        self.ω_max = ω_max
        self.last_predicted_states = None
        self.last_predicted_controls = None
        self.last_realization = None
        self.last_stats = None

    @property
    def horizon_steps(self) -> int:
        return 1   # IBVS is reactive, not predictive

    def solve(
        self,
        state: np.ndarray,
        target_positions: np.ndarray,   # (≥1, 3) world coords; we use [0]
    ) -> np.ndarray:
        # 1. Express target in pursuer body frame, then camera frame.
        p_W = state[SX.position]
        R_WB = state[SX.rotation].reshape(3, 3)
        p_target_W = target_positions[0]
        p_target_B = R_WB.T @ (p_target_W - p_W)
        p_target_C = self.camera.body_to_camera @ p_target_B

        # 2. Out-of-FOV → fall back to "look toward last LOS" coast.
        if not self.camera.in_fov(p_target_C):
            return self._coast(state, p_target_W)

        # 3. Yan/Yang Eq.(6)-style: optical axis n_c = ẑ_camera, LOS n_td = p̂_target_C.
        n_td = p_target_C / np.linalg.norm(p_target_C)
        n_c  = np.array([0.0, 0.0, 1.0])

        # 4. Collinear control: ω_des = k_ω · (n_c × n_td), in camera frame,
        #    rotated back to body frame.
        ω_cam  = self.k_omega * np.cross(n_c, n_td)
        ω_body = self.camera.body_to_camera.T @ ω_cam
        ω_des  = np.clip(ω_body, -self.ω_max, self.ω_max)

        # 5. Thrust: align thrust along n_td direction in body frame, magnitude
        #    sized to maintain approach_speed against current closing rate.
        v_W = state[SX.velocity]
        range_m = float(np.linalg.norm(p_target_W - p_W))
        closing_rate = -float(np.dot(v_W, p_target_W - p_W)) / max(range_m, 1e-3)
        thrust_cmd = (
            self.params.mass * self.params.gravity
            + self.k_thrust * (self.approach_speed - closing_rate) * self.params.mass
        )
        thrust_cmd = float(np.clip(thrust_cmd, 0.0, np.sum(self.params.max_rotor_thrusts)))

        return np.array([thrust_cmd, ω_des[0], ω_des[1], ω_des[2]], dtype=float)

    def _coast(self, state, p_target_W):
        # Out-of-FOV: hold attitude, hover thrust. (Phase-2 problem to do better.)
        return np.array([self.params.mass * self.params.gravity, 0.0, 0.0, 0.0])

    def command_to_rotor_thrusts(self, state, command, dt):
        # Reuse existing CTBR rotor allocation.
        return ctbr_command_to_rotor_thrusts(
            self.params, state[SX.body_rates], command, tau_attitude=0.05,
        )

    def reset_execution_state(self) -> None:
        pass
```

Notes:
- This is **not** the full Yan/Yang controller. It's the *kernel* — collinear
  control on n_c/n_td plus a P-loop on closing speed. Yan adds a PNG outer loop
  that biases n_td toward predicted intercept point. We can add that later;
  for trajectory generation the simpler form gives realistic-shape paths.
- The `last_*` attributes exist because `simulator.py` reads them. They stay
  None — IBVS has no MPC rollout.
- `tau_attitude = 0.05` matches `CTBRController` default (`mpc_ctbr.py:43`).

## Step 3 — Scripted trajectory evader

Add to `src/interception/evader.py`:

```python
@dataclass
class ScriptedTrajectoryEvader:
    """Evader following a piecewise-linear path with constant speed per segment.

    The waypoint sequence and per-segment speeds are sampled at construction by
    the trajectory dataset generator; per-engagement variety comes from sampling
    different (waypoints, speeds) tuples from a parameter distribution.
    """
    waypoints: np.ndarray   # (N, 3) world coords
    speeds: np.ndarray      # (N-1,) m/s for each segment
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    _segment_idx: int = 0
    _arc_param: float = 0.0  # [0, 1] along current segment

    def reset(self, initial_position, initial_velocity=None):
        self.position = np.asarray(initial_position, dtype=float).copy()
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        # initial_position is ignored if not on the path; we override with waypoints[0]
        if np.linalg.norm(self.waypoints[0] - self.position) > 1e-6:
            self.position = self.waypoints[0].copy()
        self._segment_idx = 0
        self._arc_param = 0.0
        self.velocity = self._segment_velocity(0)

    def state(self):
        return self.position.copy(), self.velocity.copy()

    def step(self, dt, pursuer_position):
        # Advance along current segment; spill into next when segment exhausted.
        seg_start = self.waypoints[self._segment_idx]
        seg_end   = self.waypoints[self._segment_idx + 1]
        seg_len   = float(np.linalg.norm(seg_end - seg_start))
        speed     = float(self.speeds[self._segment_idx])
        delta_arc = (speed * dt) / max(seg_len, 1e-6)
        self._arc_param += delta_arc
        while self._arc_param >= 1.0 and self._segment_idx < len(self.waypoints) - 2:
            self._arc_param -= 1.0
            self._segment_idx += 1
            self.velocity = self._segment_velocity(self._segment_idx)
            seg_start = self.waypoints[self._segment_idx]
            seg_end   = self.waypoints[self._segment_idx + 1]
            seg_len   = float(np.linalg.norm(seg_end - seg_start))
        self._arc_param = min(self._arc_param, 1.0)
        self.position = seg_start + self._arc_param * (seg_end - seg_start)
        return self.state()

    def predict_positions(self, times, pursuer_position=None):
        # Return ground-truth future path along the script. Simulator MPC consumers
        # get a real trajectory, not constant-velocity extrapolation.
        out = np.zeros((len(times), 3), dtype=float)
        seg_idx = self._segment_idx
        arc = self._arc_param
        for k, t in enumerate(times):
            ...   # walk the path forward by t seconds; clamp at last waypoint
        return out

    def _segment_velocity(self, idx):
        seg = self.waypoints[idx + 1] - self.waypoints[idx]
        return self.speeds[idx] * seg / max(np.linalg.norm(seg), 1e-6)
```

## Step 4 — Driver and validation

New file: `scripts/run_ibvs_canonical.py`

Mirrors `run_canonical.py` exactly, but instantiates `IBVSController` and a
`ScriptedTrajectoryEvader` with a 3-waypoint path that includes a hard right
turn — should expose any FOV-loss / coast-fallback issues.

New file: `tests/test_ibvs.py`
- **stationary target test**: pursuer 10 m away, target stationary in FOV. Expect
  catch within `catch_radius` in <5 s.
- **constant-velocity crossing test**: same as `run_canonical.py`'s scenario.
  Expect catch.
- **out-of-FOV at start test**: target behind pursuer. Expect coast (no crash).
- **n_c × n_td direction test**: with target above-and-right in image, expect
  `ω_des` to have the sign that rotates the camera toward the target.

## Step 5 — Trajectory dataset generator

New file: `scripts/generate_trajectory_dataset.py`

```python
def main():
    params = load_params(...)
    camera = CameraParams.from_yaml(...)
    controller = IBVSController(params, camera)

    rng = np.random.default_rng(seed=0)
    n_engagements = 10_000
    samples_per_engagement = 30   # at 30 Hz over ~1 s of engagement window

    out_path = Path("results/trajectory_dataset.npz")
    records = []

    for i in range(n_engagements):
        # Sample: pursuer initial pose, target initial position, target waypoints.
        pursuer_init = sample_pursuer_initial(rng)
        target_waypoints, segment_speeds = sample_target_trajectory(rng)
        evader = ScriptedTrajectoryEvader(target_waypoints, segment_speeds)

        result = run_episode(
            initial_state=pursuer_init,
            controller=controller,
            evader_policy=evader,
            evader_initial_position=target_waypoints[0],
            evader_initial_velocity=evader._segment_velocity(0),
            max_time=8.0,
            controller_rate_hz=200.0,    # match Yang paper IMU rate
            catch_radius=0.30,
            include_drag=True,
        )

        # Subsample frames evenly across the engagement.
        n_frames = result.pursuer_states.shape[0]
        idxs = np.linspace(0, n_frames - 1, samples_per_engagement).astype(int)
        for k in idxs:
            p_W   = result.pursuer_states[k, SX.position]
            R_WB  = result.pursuer_states[k, SX.rotation].reshape(3, 3)
            t_W   = result.evader_positions[k]
            range_m = float(np.linalg.norm(t_W - p_W))
            t_C = camera.body_to_camera @ R_WB.T @ (t_W - p_W)
            ip = camera.project(t_C) if camera.in_fov(t_C) else None
            target_size_px = estimate_target_size_px(camera, range_m, target_radius=0.25)
            records.append(dict(
                engagement_id=i,
                frame_idx=k,
                t=result.time[k],
                pursuer_p_W=p_W,
                pursuer_R_WB=R_WB,
                target_p_W=t_W,
                target_v_W=result.evader_velocities[k],
                range_m=range_m,
                normalized_pixel=ip,             # None if out-of-FOV
                target_size_px=target_size_px,
                outcome=result.outcome,
            ))

    np.savez_compressed(out_path, records=records)
```

This is the **bridge to Phase 1** — the rendered dataset's pose distribution
comes from this file, not from random sampling.

## Acceptance criteria

Before declaring this done:

1. `pytest tests/test_ibvs.py` passes.
2. `python scripts/run_ibvs_canonical.py` produces a plot showing the pursuer
   catching a turning evader within 5 s.
3. `python scripts/generate_trajectory_dataset.py --n 100` produces a
   `.npz` with ≥3000 records (100 engagements × ~30 samples). Histogram of
   `range_m` covers 1-30 m. Histogram of `target_size_px` covers 2-200 px.
4. Catch rate ≥80% across 10k engagements with the constant-velocity-crossing
   evader (matching the Yang paper's published rate).
5. No edits to `dynamics.py`, `simulator.py`, `mpc_srt.py`, or `mpc_ctbr.py`.
   The integration is purely additive.

## Open questions to resolve before coding

1. **Camera placement on the body** — strapdown forward (camera optical axis =
   body x-axis or body z-axis?). The Yan paper assumes z_camera aligned with
   the thrust direction (downward-tilted from horizontal in flight).
   Decide: forward-strapdown (camera +x = body +x) is simpler and matches
   typical FPV racing drones. Recommended default. Use `body_to_camera` as the
   single point of configuration for this.

2. **Target physical size for `target_size_px` estimation** — Phase 1 will train
   on different drone models. Encode the target's bounding sphere radius in the
   dataset record so the renderer can reuse it consistently.

3. **Catch radius** — current default 0.15 m is for a small target. The Yang
   paper uses 0.30 m for a tethered balloon. Recommend 0.30 m as a more
   forgiving default that survives FOV loss in the terminal phase.

4. **FOV loss handling** — the placeholder `_coast()` returns hover thrust.
   Better behaviors (use IMU-integrated last-LOS as a dead-reckoning prediction;
   open-loop continue along last n_td) are Phase-2 problems. For now, accept
   the catch-rate hit from naive coast.

## Code locations to consult during implementation

### Existing interception module (mirror these patterns; do not edit)

- `data/docs/research/drones/interception/src/interception/dynamics.py:11-25`
  — `STATE_SIZE = 22`, `StateSlices` definition. The IBVS controller reads
  `state[SX.position]`, `state[SX.rotation]`, `state[SX.velocity]`,
  `state[SX.body_rates]`. **Do not change these slices.**
- `data/docs/research/drones/interception/src/interception/dynamics.py:127-138`
  — `ctbr_command_to_rotor_thrusts(...)`. Reuse verbatim from
  `IBVSController.command_to_rotor_thrusts()`. The `tau_attitude=0.05`
  default lives here.
- `data/docs/research/drones/interception/src/interception/mpc_ctbr.py:35-110`
  — `CTBRController.__init__` and method signatures. Match the public surface
  exactly: `params`, `horizon_steps`, `solve(state, target_positions)`,
  `command_to_rotor_thrusts(state, command, dt)`, `reset_execution_state()`,
  `last_predicted_states`, `last_predicted_controls`, `last_realization`,
  `last_stats`. Look at how the simulator reads these.
- `data/docs/research/drones/interception/src/interception/simulator.py:144-195`
  — the main step loop. Confirms exactly which controller methods are called
  and which `last_*` attributes are read. The `last_realization` dict keys
  (`total_thrust`, `desired_body_rates`, `measured_body_rates`, `rate_error`,
  `torque_command`, `integral_error`, `rotor_thrusts`) are populated in
  `mpc_ctbr.py` — emit a similar dict from `IBVSController` if you want the
  debug log to be populated for IBVS rollouts.
- `data/docs/research/drones/interception/src/interception/evader.py:9-24`
  — `EvaderPolicy` Protocol. `ScriptedTrajectoryEvader` must conform.
- `data/docs/research/drones/interception/src/interception/evader.py:48-74`
  — `ConstantVelocityEvader` is the cleanest existing reference for the
  reset/state/step/predict_positions interface; copy that scaffolding.
- `data/docs/research/drones/interception/scripts/run_canonical.py:1-55`
  — driver template for `run_ibvs_canonical.py`. Shows path setup, param
  loading, initial-state construction (`make_hover_state` + position
  override), and the `run_episode(...)` call.
- `data/docs/research/drones/interception/configs/hummingbird.yaml:1-54`
  — quadrotor params already match the Hummingbird used in omnidrones.
  Reuse for IBVS — the Yang/Yan paper uses a similar-class platform.
- `data/docs/research/drones/interception/src/interception/params.py`
  — `load_params(...)` and `QuadrotorParams` dataclass. Extend
  `CameraParams.from_yaml()` to follow the same loader pattern.

### Beihang reference code (lift control law structure, not scaffolding)

- `data/docs/research/drones/swarm/papers/beihang/code/Drone-vs.-Drone/tello_vs_tello.py`
  — main pursuer process. Skim only:
  - `k_yaw = 0.002`, `k_z = -0.005`, `attack_pitch = 0.6` near `droneSetup()`
    — these are the production gains they actually flew with. Use as a sanity
    check on `IBVSController` gain magnitudes after non-dimensionalization.
  - `saturation()` helper — confirms they clip control output, matches our
    `ω_des` clamp.
- The repo has **no IMU model, no DKF, no FOV gate** in code. Don't waste
  time grepping for them. The DKF lives only on paper.

### CopterSim (only the dynamics noise parameters)

- `data/docs/research/drones/swarm/papers/beihang/code/CopterSim/Init.m`
  — Gaussian noise levels used for the lab's HIL sims. Reference values for
  any later Phase-2 sensor-noise injection: `noisePowerAccel = [0.001, 0.001,
  0.003]`, `noisePowerGyro = [1e-5, 1e-5, 1e-5]`, sample times 1 ms.
  **These are zero-mean only — bias drift is NOT in this repo.**

### Papers (equations to translate to NumPy)

- `data/docs/research/drones/swarm/papers/beihang/extracted/2404.08296.txt:280-360`
  — Section IV.B: full IMU + delayed-image measurement model. Skip for Phase 1
  (deferred), but cite when extending to Phase 2.
- `data/docs/research/drones/swarm/papers/beihang/extracted/2404.08296.txt`
  Section III.A.5 (image Jacobian Eq. 6) — the Lₛ matrix. Useful if you later
  add a feedforward term to `ω_des` based on target velocity.
- `data/docs/research/drones/swarm/papers/beihang/extracted/2404.08296.txt`
  Section IV.A — collinear control law (force vector n_f, target unit vector
  n_td, rotation matrix R_f). The simplified law in our `IBVSController` step 4
  is the kernel of this; the full version augments n_td with a PNG bias.
- `data/docs/research/drones/swarm/control/2409.17497.pdf`
  — Yan 2024 outdoor extension. Read for the gain values that survived 4 m/s
  wind in real flight.

### Synthetic-data consumer (target for the dataset)

- `data/docs/research/drones/swarm/docs/synthetic/pipeline.md`
  Stage 1 — defines exactly which fields the trajectory dataset must emit
  (pursuer pose, target pose, range, lighting_id placeholder, target type).
  `generate_trajectory_dataset.py` should match this schema field-for-field.
- `data/docs/research/drones/swarm/docs/synthetic/symeonidis_2021.txt`
  — bbox annotation method that consumes the trajectory dataset downstream.
  Determines that we need to emit `target_size_px` (used for curriculum
  stratification) and the pursuer R_WB (used to compute the camera pose for
  rendering).

### Lineage

- IBVS controller form: Yang 2024 / Yan 2024 (Beihang). Reference code at
  [`github.com/KennethYangle/Drone-vs.-Drone`](https://github.com/KennethYangle/Drone-vs.-Drone)
  — real-Tello hardware code, not a clean library; lift the gain structure
  and control law, not the multiprocessing scaffolding.
- Existing dynamics + simulator: this repo's `interception/` module.
- Trajectory dataset consumer:
  [synthetic/pipeline.md](synthetic/pipeline.md) Stage 1 (trajectory generator).
