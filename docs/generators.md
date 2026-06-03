# Generators Pipeline

This document describes the generator pipeline as system boundaries and data
contracts. It is intended as source material for an ASCII diagram generator, so
the emphasis is on component interactions rather than per-file implementation
details.

## System Boundary Summary

```text
+----------------------------------------------------------------------------+
| PROCEDURAL GENERATORS                                                      |
|                                                                            |
|  scripts/generators/*                                                      |
|  - subclass SimInstanceGenerator                                           |
|  - resolve sampling distributions into typed SimInstance records            |
|  - validate samples before returning them                                  |
|  - write generated samples under scripts/generators/sim_instances          |
|                                                                            |
+------------------------------- typed SimInstance --------------------------+
                                |
                                v
+----------------------------------------------------------------------------+
| SHARED GENERATOR INFRASTRUCTURE                                             |
|                                                                            |
|  backends/csim/generator                                                    |
|  - SimInstanceGenerator: procedural generation contract                     |
|  - SimGenerator: disk-backed reader for generated SimInstance files         |
|  - instance_store: .csimin read/write format                                |
|  - metadata: sidecar JSON for generated sample tables                       |
|                                                                            |
+------------------------------- .csimin table ------------------------------+
                                |
                                v
+----------------------------------------------------------------------------+
| CONSUMERS                                                                   |
|                                                                            |
|  scripts/runners/control_sim                                                |
|  - CLI loads .csimin files for beihang_minimal, beihang_paper, and policies |
|  - SimRunner accepts typed SimInstance workloads                           |
|  - fixed-width slots are reset/refilled from those instances                |
|                                                                            |
|  ai/rl/simengine_*                                                          |
|  - ScenarioTable reads .csimin files                                        |
|  - BatchSimGenerator samples ScenarioTable indices, not procedural code     |
|                                                                            |
+-------------------------- typed SimInstance reset -------------------------+
                                |
                                v
+----------------------------------------------------------------------------+
| PYTHON CSIM BINDINGS                                                        |
|                                                                            |
|  backends/csim/bindings                                                     |
|  - typed dataclasses define the Python boundary                             |
|  - PufferSimEngineBackend and BatchPufferSimEngineBackend translate typed   |
|    Python objects into ctypes structs                                       |
|  - only this layer calls sim_engine_* C API functions                       |
|                                                                            |
+------------------------------- C API calls --------------------------------+
                                |
                                v
+----------------------------------------------------------------------------+
| C SIMENGINE                                                                 |
|                                                                            |
|  backends/csim/sim_engine.h / sim_engine.c                                  |
|  - owns SimEngine state, targets, cameras, metrics, render hooks            |
|  - steps pursuer/target simulation                                          |
|  - returns snapshots, metrics, and camera outputs                           |
+----------------------------------------------------------------------------+
```

## Boundary Contracts

### 1. C SimEngine API

`SimEngine` is the native simulation owner. The public C boundary is
`backends/csim/sim_engine.h`.

Important C-side entities:

- `SimEngine`: native mutable state for one pursuer, targets, cameras, timing,
  intercept metrics, and optional rendering.
- `SimSnapshot`: exported snapshot containing pursuer state, target states,
  target ids/radii, intercept metrics, and camera outputs.
- `sim_engine_init` / `sim_engine_reset`: establish pursuer state and vehicle
  parameters.
- `sim_engine_set_targets` / `sim_engine_clear_targets`: replace target set.
- `sim_engine_set_cameras` / `sim_engine_clear_cameras`: replace camera set.
- `sim_engine_set_intercept_radius`: configure interception metric threshold.
- `sim_engine_step_motor_speeds_dt`: advance one engine with commanded motor
  speeds.
- `sim_engine_batch_step_motor_speeds_dt`: advance an array of independent
  engines in fixed-width batch mode.
- `sim_engine_get_snapshot`: export a full snapshot for Python callers.
- `sim_engine_batch_get_snapshots`: export compact batched snapshot buffers for
  the Python binding layer.

Rule: Python code outside `backends/csim/bindings` must not call `sim_engine_*`
directly. The binding layer is the only place where C structs, ctypes
conversions, and direct C API calls belong.

### 2. Python Binding Layer

`backends/csim/bindings` is the Python boundary to the C engine.

Typed inputs come from `backends/csim/bindings/types`:

- `SimInstance`: seed, pursuer initial state, target initial states, and typed
  `SimConfig`.
- `SimConfig`: pursuer parameters, sim options, targets, cameras, intercept
  radius, command limits, noise, and rendering settings.
- `PursuerInitialState`: world pose/velocity, body rates, optional rotor speeds,
  and optional wind.
- `TargetConfig` and `TargetInitialState`: target identity, size, behavior,
  controller, initial position, and initial velocity.
- `CameraConfig`: camera identity, parent, transform, intrinsics, and capture
  rate.

Binding responsibilities:

- Convert typed Python dataclasses into C structs.
- Normalize state shapes and quaternions before crossing into C.
- Pair `TargetConfig` records with matching `TargetInitialState` records.
- Configure cameras and optional rendering from `SimConfig`.
- Convert C snapshots back into typed `SimSnapshot` / `SimSnapshots` objects.
- Keep vectorized batch data behind typed `SimSnapshotArrays` for high-throughput
  RL/control loops.
- Enforce batch constraints such as homogeneous `backend_dt` and
  `action_substeps`.

Primary binding entry points:

- `PufferSimEngineBackend`: single-engine adapter. It can reset from a
  `SimInstance` and exposes snapshot dictionaries.
- `BatchPufferSimEngineBackend`: fixed-width array of C `SimEngine` slots. It
  resets selected slots from `SimInstance` records and steps all active slots
  with CTBR-derived motor speeds.

## Generator Side

### Procedural Generators

Procedural generators live under `scripts/generators` and subclass
`SimInstanceGenerator`.

The procedural boundary is:

```text
+--------------------+       _sample_once(seed)       +----------------------+
| sampling strategy | -----------------------------> | typed SimInstance    |
| config + seed     |                                | config + initials    |
+--------------------+                                +----------------------+
          |                                                       |
          | validation                                             |
          v                                                       v
   reject invalid samples                              write .csimin records
```

`SimInstanceGenerator.sample()` resolves a seed into one validated
`SimInstance`. `sample_many()` resolves a sequence of seeds into validated
instances. Invalid generated samples are discarded and regenerated up to the
generator attempt limit.

Named sim configs are resolved from `backends/csim/configs` through
`get_config()`. A config module exposes `SIM_CONFIG`, and it must be a typed
`SimConfig`.

### Disk-Backed SimGenerator

`SimGenerator` means a disk-backed reader over already-generated
`SimInstance` records. It is not a procedural sampler.

Expected use:

```text
.csimin file -> read_sim_instances -> SimGenerator -> sample(seed)
```

`SimGenerator.from_disk(path)` reads generated records from disk.
`SimGenerator.sample(seed=...)` returns the instance with that seed.
`SimGenerator.sample_many(...)` returns a deterministic seed range.

There should not be a separate compatibility generator that reintroduces a
parallel implementation of this behavior.

### `.csimin` Tables And Metadata

Generated sample tables are binary `.csimin` files written by
`write_sim_instances()` and read by `read_sim_instances()`.

Table boundary facts:

- Header magic is `CSIMINST`.
- The format has an explicit version.
- Records contain typed `SimInstance` content, including nested `SimConfig`,
  target config, camera config, options, and noise.
- Readers validate magic, version, payload length, offsets, counts, and trailing
  bytes.
- Generated files must be written under
  `scripts/generators/sim_instances`.

Sidecar metadata is written as JSON next to the `.csimin` table. It records the
sample path, file format, generator name, strategy, sampling counts, simulation
settings, parameter specs, labels, record paths, and plot paths when available.

## Consumer Side

### Control Sim CLI Flow

Runnable control-sim CLI entry points live under `scripts/runners/control_sim`.
Each script aliases its concrete `SimControlPolicy` and delegates only shared
artifact mechanics to `control_sims.runner`.

The control-sim CLI boundary is:

```text
scripts/runners/control_sim/*
        |
        | --scenario-table required
        v
read_sim_instances(.csimin)
        |
        v
typed SimInstance tasks
        |
        +--> beihang_minimal SimControlPolicy -> SimRunner -> C SimEngine
        |
        +--> beihang_paper   SimControlPolicy -> SimRunner -> C SimEngine
        |
        +--> neural policy   SimControlPolicy -> SimRunner -> C SimEngine
```

`--scenario-table` is required. Control sims consume generated `.csimin` files;
they do not sample procedural generators directly.

### `backends/csim/runner`

`SimRunner` is the batch-native control runner for typed
`SimInstance` workloads.

Boundary flow:

```text
tuple[SimInstance, ...]
        |
        v
SimRunner.reset()
        |
        | validates SimInstance.config exists
        | validates homogeneous dt/action_substeps
        v
BatchPufferSimEngineBackend.reset_many(indices, instances)
        |
        v
C SimEngine slots
        |
        | CtbrCommandBatch(thrust_n, body_rates_b)
        v
BatchPufferSimEngineBackend.step_ctbr_commands_many()
        |
        v
snapshot arrays: pursuer, target, metrics, camera
        |
        v
completion records + optional snapshot logging callbacks
```

Important runner concepts:

- A workload is a finite sequence of typed `SimInstance` objects.
- Slots are fixed-width C `SimEngine` instances.
- Completed slots are refilled immediately until the workload is exhausted.
- Commands are physical CTBR commands: thrust in newtons and body rates in
  radians per second.
- Completion is based on interception metrics, out-of-bounds/nonfinite failure,
  or `SimConfig.options.duration_s`.
- `CompletedSim` carries the original instance, seed, terminal snapshot, and
  terminal reason. Derived metrics such as visible fraction and control effort
  are computed by consumers from `SimRunnerStep` history when needed.

### RL Consumer Flow

RL consumers also read generated `.csimin` files.

Boundary flow:

```text
.csimin table
        |
        v
ScenarioTable
        |
        | get(index) -> SimInstance
        | label(index) -> ScenarioLabel
        v
BatchSimGenerator
        |
        | random / grid_balanced / sequential_epoch index sampling
        v
BatchSimRunner + BatchPufferSimEngineBackend
```

`BatchSimGenerator` samples scenario-table indices. It does not call procedural
generator classes.

## Logging And Artifacts

Generated sample artifacts:

- `.csimin` binary sample tables live under `scripts/generators/sim_instances`.
- JSON sidecars next to `.csimin` files describe generator strategy, counts,
  sample format, simulation settings, active parameters, labels, and related
  record/plot outputs.
- Analysis artifacts from one-off analysis scripts belong under
  `docs/analysis`, with subfolders matching the source area when relevant.

Control-sim run artifacts:

- `RunsDirLogger` writes date-partitioned run directories under `.runs`.
- Control sim runners write `trials.csv`, `summary.json`, and optional
  `snapshots/<sim_name>.csv`.
- Snapshot logging writes `snapshots/logging_config.json` with the logging rate
  and output location.
- Snapshot rows normalize seed, tick, time, pursuer state, motor state, target
  state, metrics, camera observation, and CTBR commands.

Operational logging rules:

- Use generated `.csimin` files as the source of simulation tasks.
- Record enough metadata to reproduce the generator configuration and sampling
  strategy.
- Keep run logs separate from generated sample tables: generated samples belong
  under `scripts/generators/sim_instances`; run outputs belong under `.runs` or
  explicit run output directories.
- Drone image detection progress is logged in `docs/detection/worklog.md` only
  when the work advances or clarifies drone image detection. Pure generator or
  SimEngine plumbing does not need a detection worklog entry.

## Script And Folder Rules

Project rules that affect this pipeline:

- All interactions with `SimEngine` go through the C API and the Python binding
  layer in `backends/csim/bindings`.
- Python callers pass typed objects from `backends/csim/bindings/types`; do not
  pass raw dictionaries or parallel adapter types across the generator/engine
  boundary.
- Do not call `sim_engine_*` directly outside the binding layer.
- `SimGenerator` is the disk-backed reader for generated `SimInstance` files.
- Do not reintroduce `PregeneratedSimGenerator` or another temporary
  compatibility path.
- Procedural generators subclass `SimInstanceGenerator` and live under
  `scripts/generators`.
- Shared generator infrastructure that affects all generator implementations
  lives under `backends/csim/generator`.
- Generated simulation sample files, including `.csimin`, are written under
  `scripts/generators/sim_instances`.
- Sim consumers, including control sims and RL runners, consume generated
  `.csimin` files rather than sampling procedural generators directly.
- Executable CLI code lives under `scripts/runners`, except for explicitly
  allowed analysis code and existing generator CLIs.
- One-off analysis scripts that generate output live under `docs/analysis`;
  analysis artifacts stay in the corresponding analysis folder.

## Diagram Generator Input

Use these nodes for a high-level component diagram:

- `scripts/generators`: procedural scenario generation.
- `SimInstanceGenerator`: procedural generation contract.
- `backends/csim/configs`: named typed `SimConfig` declarations.
- `backends/csim/bindings/types`: typed boundary objects.
- `backends/csim/generator/instance_store`: `.csimin` persistence.
- `scripts/generators/sim_instances`: generated sample table storage.
- `SimGenerator`: disk-backed generated sample reader.
- `scripts/runners/control_sim`: control-sim CLI loaders, policy aliases, and
  run artifact writers over typed instances.
- `ai/rl/simengine_env/ScenarioTable`: RL table reader.
- `ai/rl/simengine_batch/BatchSimGenerator`: RL scenario index sampler.
- `backends/csim/bindings/puffer_c`: ctypes binding layer.
- `PufferSimEngineBackend`: single-engine Python adapter.
- `BatchPufferSimEngineBackend`: fixed-width batched Python adapter.
- `backends/csim/sim_engine`: native C SimEngine.
- `.runs`: control-sim run outputs.

Use these edges:

- `scripts/generators` -> `SimInstanceGenerator`: subclasses contract.
- `scripts/generators` -> `backends/csim/configs`: resolves named `SimConfig`.
- `scripts/generators` -> `backends/csim/bindings/types`: creates typed
  `SimInstance` records.
- `scripts/generators` -> `instance_store`: writes `.csimin`.
- `instance_store` -> `scripts/generators/sim_instances`: stores generated
  sample tables.
- `scripts/generators/sim_instances` -> `SimGenerator`: disk-backed sampling.
- `scripts/generators/sim_instances` -> `scripts/runners/control_sim`: CLI
  scenario table input.
- `scripts/generators/sim_instances` -> `ScenarioTable`: RL scenario input.
- `scripts/runners/control_sim` -> `backends/csim/runner`: run typed
  instances through `SimRunner`.
- `ScenarioTable` -> `BatchSimGenerator`: table index sampling.
- `BatchSimGenerator` -> `BatchSimRunner`: batch resets with typed instances.
- `PufferSimEngineBackend` -> `backends/csim/sim_engine`: single-engine C API
  calls.
- `BatchPufferSimEngineBackend` -> `backends/csim/sim_engine`: batched C API
  calls.
- `scripts/runners/control_sim` -> `.runs`: writes trials, summaries, and
  snapshots.
