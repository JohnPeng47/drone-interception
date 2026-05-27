# Sim Generator / C Sim Plan

This note tracks the plan for consolidating scenario generation across
`control_sims` and RL/Puffer training using the C sim interfaces now present in
`backends/csim`.

## Current C Sim Surface

`backends/csim` now has three layers:

- `sim_types.h` / `pursuer_sim.c`: low-level quadrotor physics via `PursuerSim`.
- `target_sim.h` / `target_sim.c`: target actor state, behavior, and controller.
- `sim_engine.h` / `sim_engine.c`: multi-actor orchestration via `SimEngine`.

The intended ownership boundary is:

- `PursuerSim` owns only the pursuer vehicle state and `PursuerParams`.
- `TargetSim` owns target state, radius, behavior config, and controller config.
- `SimEngine` owns one pursuer plus zero or more targets and advances both.
- RL owns rewards, observations, resets, episode bookkeeping, and PufferLib glue.
- Python owns scenario distribution authoring and artifact generation.

## Python Binding Input Models

Move the public Python sim data models into:

```text
backends/csim/bindings/types/
```

`backends/input.py` remains a compatibility re-export until all older call sites
have moved to the canonical names.

Use explicit names:

```python
@dataclass(frozen=True)
class PursuerParams:
    mass_kg: float
    ixx: float
    iyy: float
    izz: float
    arm_len_m: float
    k_thrust: float
    k_yaw: float
    k_ang_damp: float = 0.0
    b_drag: float = 0.0
    gravity_mps2: float = 9.81
    max_rpm: float = DEFAULT_MAX_RPM
    max_vel_mps: float = DEFAULT_MAX_VEL_MPS
    max_omega_rps: float = DEFAULT_MAX_OMEGA_RPS
    motor_tau_s: float = 0.15
    rpm_min: float | None = None
    rotor_positions_b: np.ndarray | None = None
    rotor_directions: np.ndarray | None = None
    k_w: float = 1.0


@dataclass(frozen=True)
class PursuerInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray
    quat_xyzw: np.ndarray
    body_rates_b: np.ndarray
    rotor_speeds: np.ndarray | None = None
    wind_w: np.ndarray | None = None


@dataclass(frozen=True)
class SimOptions:
    backend_dt: float = PUFFER_DT
    action_substeps: int = PUFFER_ACTION_SUBSTEPS
    command_mode: str = "ctbr"
    ctbr_rate_gain: float = 0.08
    randomize_params: bool = False
```

Keep the roles separate:

- `PursuerParams`: physical pursuer model.
- `PursuerInitialState`: per-scenario pursuer initial state.
- `SimOptions`: stepping/action-mode settings.

Do not put vehicle params inside the initial state. They vary on a different
axis. If we later need per-episode vehicle randomization, add an optional
vehicle-param override to the generated scenario record.

## Quaternion TODO

Standardize quaternion naming and conversion before broadening the API:

- Python public fields currently use `quat_xyzw`.
- C `Quat` currently stores `w, x, y, z`.

Acceptance criteria:

- No public field named only `quat` unless its convention is local/private.
- Python fields use `_xyzw` suffix.
- C fields or helper docs explicitly say `wxyz`.
- Python/C conversion helpers live in one binding module.
- Tests cover identity, pure pitch, pure yaw, and Python-to-C round trip.

## Target Models

Use the new `target_sim.h` concepts directly in Python:

```python
@dataclass(frozen=True)
class TargetInitialState:
    position_w: np.ndarray
    velocity_w: np.ndarray


@dataclass(frozen=True)
class TargetControllerConfig:
    kind: str  # "none", "linear"
    kp: float = 0.0
    kv: float = 0.0
    max_accel_mps2: float = 0.0


@dataclass(frozen=True)
class TargetBehaviorConfig:
    kind: str  # "hold", "waypoints"
    waypoints_w: tuple[np.ndarray, ...] = ()
    duration_s: float = 0.0
    loop: bool = False


@dataclass(frozen=True)
class TargetConfig:
    id: int
    radius_m: float
    behavior: TargetBehaviorConfig
    controller: TargetControllerConfig
```

The current red-balloon moving-straight-line case can be represented as either:

- `TARGET_BEHAVIOR_HOLD` plus nonzero initial target velocity and
  `TARGET_CONTROLLER_NONE`, if target state should coast in a straight line.
- Or two `TARGET_BEHAVIOR_WAYPOINTS` points with `TARGET_CONTROLLER_LINEAR`, if
  we want target motion to track a reference trajectory.

Before generating RL artifacts, decide which semantics we want for
straight-line targets. The simplest for parity with the existing red-balloon
builder is constant velocity with no controller.

## Sim Instance

`SimGenerator.sample(seed)` should return a fully resolved scenario instance:

```python
@dataclass(frozen=True)
class SimInstance:
    seed: int
    pursuer_initial: PursuerInitialState
    target_initials: tuple[TargetInitialState, ...]
    config: SimConfig | None = None
```

`SimConfig` is the distribution/run-level object:

```python
@dataclass(frozen=True)
class SimConfig:
    pursuer: PursuerParams
    options: SimOptions = field(default_factory=SimOptions)
    targets: tuple[TargetConfig, ...] = ()
    cameras: tuple[CameraConfig, ...] = ()
    intercept_radius_m: float = 0.0
```

So:

- `SimConfig` owns baseline pursuer params, runtime options, target/camera definitions, and intercept settings.
- `SimInstance` owns sampled per-scenario initial state: pursuer initial state and target initial states.

## Control Sim Integration

`backends/generator.py` owns the Python `SimGenerator` boundary. Concrete
generators implement `sample(seed=...) -> SimInstance` and may override `run()`
for deterministic execution.

```python
instance = generator.sample(seed=seed)
```

`ExperimentConfig` is removed. The Drake builder consumes plain resolved config
dicts, and the red-balloon geometry is now local to the generator package.

## RL / Puffer Integration: Option B

Use generated binary scenario tables, not Python callbacks during training.

Artifacts:

```text
rl/generated/intercept.ini
rl/generated/intercept_scenarios.bin
rl/generated/intercept_manifest.json
```

`intercept.ini` contains run-level/static fields:

- vehicle params (`PursuerParams`)
- runtime options (`SimOptions`)
- reward params
- scenario table path
- scenario table count/version/checksum

`intercept_scenarios.bin` contains many fixed `SimInstance` records:

- `PursuerInitialState` mapped to C `State`
- one or more `TargetInitialState` records paired with baseline `SimConfig.targets`
- seed/sample id where useful

The C/Puffer env should load the binary table once in `my_init`, keep it in
memory, and sample scenario records on reset.

## C Runtime Shape

The RL env should eventually store:

```c
typedef struct {
    PursuerParams vehicle_params;
    float backend_dt;
    int action_substeps;
    int command_mode;
    // reward config
    // scenario table pointer/count
} RuntimeSimConfig;
```

Per agent, prefer embedding `SimEngine`:

```c
typedef struct {
    SimEngine engine;
    Vec3 prev_pos;
    float episode_return;
    int episode_length;
    // RL bookkeeping
} Drone;
```

Reset:

```c
ScenarioRecord* scenario = sample_scenario(env);
sim_engine_init(&agent->engine, env->runtime.vehicle_params,
                scenario->pursuer_initial);
sim_engine_set_targets(&agent->engine, scenario->targets,
                       scenario->num_targets);
```

Step:

```c
sim_engine_step_motor_dt(&agent->engine, actions,
                         env->runtime.backend_dt * env->runtime.action_substeps,
                         env->runtime.action_substeps);
```

Rewards/observations read from:

```c
sim_engine_get_pursuer_state(&agent->engine);
sim_engine_get_target_state(&agent->engine, 0);
```

## Binary Scenario Format

Python now has a versioned `SimInstance` table format in
`backends/csim/generator/instance_store.py`. It stores a binary header followed
by packed binary `SimInstance` records:

```c
typedef struct {
    char magic[8];        // "CSIMINST"
    uint32_t version;
    uint32_t num_records;
    uint64_t payload_bytes;
} SimInstanceTableHeader;
```

Records store the resolved simulation fields only: seed, `PursuerInitialState`,
target initial states, and optional `SimConfig` containing baseline targets and cameras. Numeric values are written as
little-endian `float32`/integer fields, and IDs/kinds are length-prefixed UTF-8
strings.

Use `write_sim_instances(path, instances)` to write tables and
`read_sim_instances(path)` or `PregeneratedSimGenerator.sample_many_from_disk`
to load them.

## Implementation Steps

1. Fill `backends/csim/bindings/types/` with the renamed Python dataclasses.
2. Update `backends/csim/bindings/puffer_c.py` and call sites to use the new
   names.
3. Add Python conversion helpers:
   - `PursuerParams -> PursuerParams`
   - `PursuerInitialState -> State`
   - `TargetConfig -> TargetSim`
4. Add `SimGenerator` and red-balloon distribution adapter. DONE for
   `control_sims/beihang_paper_sim`.
5. Add binary writer/reader tests for `intercept_scenarios.bin`. DONE for the
   Python `SimInstance` table.
6. Move RL env sources under `rl/env/intercept`.
7. Update RL env to embed `SimEngine` and load/sample scenario records.
8. Add `rl/scripts/prepare_puffer_env.sh` to copy `rl/env/intercept` and
   `backends/csim` into `puffer/ocean/intercept`.
9. Update `remote_bootstrap.sh` to generate/copy artifacts before building.
10. Delete `intercept_env/` after:
    - `rg "intercept_env|dronelib.h|move_drone"` has no live references.
    - Puffer build succeeds.
    - A short Puffer training smoke run succeeds.

## Open Decisions

- Should straight-line targets be represented as free constant velocity or as a
  waypoint reference with linear controller?
> Waypoint reference with v > 0 and max_accel = 0
- Should `rpm_min` be added to C `PursuerParams`, or continue deriving it from hover
  RPM for normalized motor actions?
> derived
- Should `k_w` remain Python-only until CTBR actions are supported in RL?
- How many generated scenario records are enough for training before repeats
  become a problem?
