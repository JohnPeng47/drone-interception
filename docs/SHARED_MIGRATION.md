# Shared Migration Plan

This note scopes the migration away from the broad `intercept_sim` shared tree
toward a small local surface owned by `gavin_puffer` / `beihang_paper_sim`.

## Current State

`beihang_paper_sim` currently imports `intercept_sim` from two possible places:

- `gavin_puffer/shared/intercept_sim`
- `../intercept_sim/src/intercept_sim`

The run scripts add both paths to `sys.path`. The two trees are currently the
same source content aside from cache files, but keeping both active makes the
actual import source path-dependent.

`beihang_paper_sim` is not using all of `intercept_sim`. It uses a small subset
as a shared data/model layer for:

- typed payloads passed through Drake abstract ports
- target and scene construction
- camera projection and delayed image measurements
- red-balloon scenario expansion
- log serialization, telemetry, and metrics
- CTBR command conversion / hover fallback helpers

The old standalone runner, manual simulator, legacy controllers, and legacy
observers are not part of the active `beihang_paper_sim` path.

## Keep

Keep or migrate these APIs into a local namespace such as
`gavin_puffer.sim_shared` or `gavin_puffer.paper_sim_core`.

Core types:

- `types.py`
- `SimulationTarget`
- `CameraIntrinsics`
- `CameraRig`
- `SceneSnapshot`
- `CameraCapture`
- `ImageFeatureMeasurement`
- `ObserverState`
- `CtbrCommand`

Scene and target model:

- `scene.make_scene_snapshot`
- `scene.visibility.project_target`
- `scene.visibility.target_position_camera`
- `targets.KinematicTarget`

Camera and perception:

- `sensors.GeometryCamera`
- `sensors.FeaturePerceptionModel`

Experiment/config helpers:

- `experiments.config.ExperimentConfig`
- `experiments.config.load_experiment_config` if direct YAML loading remains useful
- `experiments.red_balloon.RedBalloonScenario`
- `experiments.red_balloon.load_red_balloon_scenario`
- `experiments.red_balloon.build_red_balloon_config`
- selected scenario aggregation helpers only if still called by scripts

Runner helper subset:

- `experiments.runner.ExperimentResult`
- `experiments.runner.save_experiment_result`
- `experiments.runner._initial_rotorpy_state`
- `experiments.runner._target_from_config`
- `experiments.runner._camera_from_config`
- `experiments.runner._perception_from_config`

Analysis and telemetry:

- `analysis.ExperimentMetrics`
- `analysis.compute_metrics`
- `analysis.circular_error_probable`
- `experiments.telemetry.build_experiment_telemetry`
- telemetry dataclasses needed by run scripts

Adapters:

- `rotorpy_adapter.rotorpy_state_to_target`
- `rotorpy_adapter.ctbr_to_rotorpy`
- `rotorpy_adapter.hover_ctbr`

## Deprecate

Deprecate or remove these after callers are confirmed absent:

- `manual_sim/**`
- old shared controllers in `controllers/**`
- old shared observers in `observers/**`
- generic `runner.InterceptionRunner`
- `experiments.benchmark`
- `experiments.delay_benchmark`
- broad architecture/worklog docs tied to the old standalone package

If any of these still have experimental value, move them under a clearly marked
`deprecated/` area rather than keeping them in the supported import path.

## Camera Plan

The camera path is active and should be migrated, not deleted.

Current dataflow:

```text
SceneSnapshot
  -> GeometryCamera
  -> CameraCapture
  -> FeaturePerceptionModel
  -> tuple[ImageFeatureMeasurement]
  -> DkfObserver
```

`beihang_paper_sim/diagram.py` creates:

- `camera_rig = _camera_from_config(raw["camera"])`
- `perception = _perception_from_config(raw["perception"])`
- `geometry_camera = GeometryCamera(camera_rig)`

It then wires:

- `world["scene"] -> sensing["camera"]`
- `sensing["camera"].capture -> sensing["perception"]`
- `sensing["perception"].measurements -> estimation["core"]`
- `capture` and `measurements` into the logger

`DkfObserver` depends on the image measurement stream:

- it waits for a detected `uv_norm` before initialization
- it seeds its image-plane state from the first valid `uv_norm`
- it applies delayed corrections from later `uv_norm` measurements

The controller nuance matters: `ControlCore` currently does not directly form
the line-of-sight vector from `ObserverState.image_feature.uv_norm`. It forms
`n_t` from the DKF relative-position estimate:

```python
n_t = -p_r / norm_pr
```

So the camera path is still needed, but its role is upstream of the controller:
it initializes/corrects the DKF, drives visibility and feature-availability
metrics, supports pixel-noise sweeps, and fills telemetry.

Do not keep old camera-adjacent controllers or observers just because they use
`uv_norm`; keep only the camera/perception/measurement pieces that feed the
current DKF.

## Drake Compatibility

`beihang_paper_sim` also depends on nearby `drake_sims` utilities:

- `drake_sims.ports`
- `drake_sims.logger.RunnerStepLogger`
- `drake_sims.adapters`
- selected `codex_sim` Drake wrappers

Those modules currently import `intercept_sim.types` and related helpers. There
are two migration options.

Preferred option:

- vendor the small required Drake compatibility layer into `gavin_puffer`
- point it at the new local shared namespace
- remove the old `intercept_sim` path from `sys.path`

Transitional option:

- keep `drake_sims` external for now
- add an `intercept_sim` compatibility shim that re-exports the migrated local
  APIs
- emit `DeprecationWarning` from the shim
- remove the shim once all imports are updated

The preferred option is cleaner, but the transitional option reduces immediate
churn if the goal is to split the migration into smaller commits.

## Migration Steps

1. Create the new local shared namespace.
2. Copy only the supported files/APIs listed above.
3. Trim `experiments.runner` down to config factories, result saving, and state
   construction helpers.
4. Rewrite `beihang_paper_sim`, `backends`, and tests from `intercept_sim.*` to
   the new namespace.
5. Decide whether to vendor the Drake compatibility modules now or keep a short
   compatibility shim.
6. Remove `experiments_root / "intercept_sim" / "src"` from run-script path
   setup once imports no longer require it.
7. Add deprecation warnings or `DEPRECATED.md` in the old shared tree.
8. Run verification.

## Verification

Minimum checks after migration:

```bash
rg "from intercept_sim|import intercept_sim" .
python -m gavin_puffer.control_sims.beihang_paper_sim.run_50_trials --n-trials 1 --duration-s 0.2
pytest tests/test_puffer_backend_smoke.py tests/test_rotorpy_backend_copy.py
```

Expected result:

- no `intercept_sim` imports outside deprecated shims or external `drake_sims`
- one short `beihang_paper_sim` run builds and advances
- backend smoke tests still pass

