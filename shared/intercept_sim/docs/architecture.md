# Intercept Sim Architecture

## Core Boundary

The simulator has two separate responsibilities:

```text
Experiment execution
  "Run this concrete config through the simulator."

Scenario evaluation
  "Generate related experiment configs and derive scenario-specific metrics."
```

This split keeps the closed-loop simulator reusable while allowing each scenario
to define its own sweep space and evaluation metrics.

## Experiment Execution

`run_experiment(config)` is the standard execution boundary. Everything below it
is scenario-agnostic.

```text
ExperimentConfig
    |
    v
build_runner(config)
    |
    v
InterceptionRunner.run(duration_s)
    |
    +--> RotorPy Multirotor dynamics
    +--> KinematicTarget
    +--> SceneSnapshot
    +--> GeometryCamera
    +--> FeaturePerceptionModel
    +--> Observer
    +--> Controller
    +--> CtbrCommand
    |
    v
RunnerStep[]
    |
    +--> ExperimentMetrics
    |
    v
ExperimentTelemetry
```

`ExperimentTelemetry` is the standard run artifact. It is generated downstream of
`run_experiment`, independent of whether the experiment came from a single YAML,
a benchmark, or a scenario sweep.

The telemetry contains:

- experiment id and comment
- config snapshot
- summary `ExperimentMetrics`
- per-step telemetry:
  - pursuer position, velocity, and attitude
  - target states
  - camera capture
  - delayed measurements
  - observer image feature and relative-state outputs
  - controller thrust and body-rate command

This telemetry is intentionally more stable than internal implementation
objects. It does not expose RotorPy instances, controller objects, or observer
objects.

## Scenario Boundary

A scenario owns setup and evaluation, but not the simulator internals.

```text
Scenario
    |
    +--> build_experiment_configs()
    |
    +--> run_scenario(scenario)
    |       |
    |       +--> run_experiment(config) for each config
    |       |
    |       v
    |   ExperimentTelemetry
    |
    +--> evaluate(telemetry)
    |
    v
ScenarioMetrics
```

The formal interface is intentionally small:

```text
Scenario
    name
    build_experiment_configs() -> list[ExperimentConfig]
    comment_for_config(config) -> str
    evaluate(telemetry) -> ScenarioMetrics
```

The constraints are:

- scenario code may generate configs
- scenario code may group telemetry
- scenario code may compute scenario-specific metrics
- scenario code should execute through `run_scenario`, which calls
  `run_experiment`
- scenario code should not reach into RotorPy, controller internals, or observer
  internals

## ScenarioMetrics

`ScenarioMetrics` are derived from `ExperimentTelemetry` plus scenario metadata.
They should not require rerunning the simulator.

```text
ExperimentTelemetry[]
    |
    v
Scenario-specific evaluator
    |
    v
ScenarioMetrics
```

This means persisted telemetry should be sufficient to regenerate scenario
tables later.

## Red Balloon Example

The red-balloon scenario currently varies:

- `distance_m`
- `closing_speed_mps`
- `seed`
- `los_azimuth_deg`
- `los_elevation_deg`

It expands these values into concrete `ExperimentConfig` runs:

```text
RedBalloonScenario
    |
    +-- distance = 10.4 m
    +-- speed    = 5, 10, 15, 19, 21 m/s
    +-- seed     = 1
    +-- bearing  = azimuth/elevation grid
    |
    v
ExperimentConfig[]
```

Each config then runs through the standard simulator:

```text
ExperimentConfig
    |
    v
run_experiment
    |
    v
ExperimentTelemetry
```

The red-balloon evaluator derives per-run and aggregate rows from telemetry:

```text
ExperimentTelemetry[]
    |
    +--> per-run miss distance
    +--> catch time
    +--> visibility fraction
    +--> feature availability fraction
    +--> average image error
    |
    v
RedBalloonScenarioMetrics
    |
    +--> run rows
    |    +--> one row per distance/speed/seed/azimuth/elevation run
    +--> aggregate rows
         +--> grouped by distance and speed
         +--> CEP50
         +--> CEP90
         +--> catch fraction
```

CEP is derived from telemetry because each telemetry step records pursuer and
target positions. The evaluator computes each run's miss distance from the
relative distance time series, then computes percentiles across grouped runs.
Bearing sweeps therefore preserve each individual run row while producing CEP
aggregates over all sampled bearings and seeds for a distance/speed condition.

## Persistence Model

Persistence should mirror the boundary:

```text
ExperimentTelemetry
    -> run_j/summary.json
    -> run_j/config.yaml
    -> run_j/telemetry.jsonl.gz  (only when detailed_trace=true)

ScenarioMetrics
    -> group.json
    -> run_j/scenario_metrics.json
```

Scenario runs are persisted as incremented groups:

```text
.runs/
  scenarios/
    <scenario_name>/
      {i}_{yy}_{mm}_{dd}/
        group.json
        run_1/
          summary.json
          config.yaml
          telemetry.jsonl.gz  (optional detailed trace)
          scenario_metrics.json
        run_2/
          summary.json
          config.yaml
          telemetry.jsonl.gz  (optional detailed trace)
          scenario_metrics.json
```

The group index `i` increments within each scenario directory. Run indices are
1-indexed within each group. `group.json` is intended as the entry point for
future visualization tooling; it records the scenario name, creation time, run
count, whether detailed traces were persisted, run file paths, and aggregate
metrics.

The default log root is `.runs/` relative to the current working directory. The
default persistence mode is trace-light: `summary.json`, `config.yaml`,
`scenario_metrics.json`, and `group.json` are written, while per-step telemetry
is omitted. Pass `detailed_trace=true` in code or `--detailed-trace` on the
red-balloon CLI to write compressed `telemetry.jsonl.gz`. This flag is also the
boundary for larger debug artifacts, such as covariance histories, if those are
added later.

For one-off debugging, a compact per-step log is still useful. For reproducible
scenario analysis, prefer persisting the standard telemetry and deriving metrics
from it when detailed traces are explicitly enabled.

## Regression Check After Split

The scenario-metric split was checked against the prior paper-derived
red-balloon comparison:

```text
distance_m = 10.4
duration_s = 5.0
speeds     = 5, 10, 15, 19, 21 m/s
seed       = 1
```

The new telemetry-derived red-balloon rows match the previous results to
rounding:

| observer | speed m/s | miss m | catch |
|---|---:|---:|---:|
| truth | 5 | 0.055 | yes |
| truth | 10 | 1.016 | no |
| truth | 15 | 1.657 | no |
| truth | 19 | 1.635 | no |
| truth | 21 | 1.564 | no |
| ekf | 5 | 0.631 | no |
| ekf | 10 | 1.885 | no |
| ekf | 15 | 2.282 | no |
| ekf | 19 | 2.135 | no |
| ekf | 21 | 2.044 | no |

The matching numbers indicate that the split changed how outputs are structured,
not the simulator behavior.
