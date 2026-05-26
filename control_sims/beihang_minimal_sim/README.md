# Beihang Minimal Sim

This package is a deliberately small Drake benchmark inspired by the Beihang
interception controller. It is meant for heuristic strategy experiments, not
for high-fidelity validation.

The task keeps the core restrictions:

- the interceptor commands CTBR: collective thrust plus body rates;
- the target is observed through a pinhole image feature;
- the controller must balance two objectives: center the target in the image
  and close distance for capture.

The simulator intentionally omits the full paper sim's DKF, motor model, C
backend, perception delay, and detailed vehicle parameters. Those belong in
`beihang_paper_sim`; this package is the fast, inspectable search environment.

## System Layout

Each Drake `LeafSystem` lives in its own source file.

```text
world/
  point_mass_ctbr_plant.py
  target_motion_system.py
  scene_assembler.py

sensing/
  pinhole_camera_system.py
  image_feature_system.py

controller/
  heuristic_strategy_system.py
  beihang_baseline_strategy.py
  strategy_api.py

scoring/
  capture_status_system.py
  trial_logger.py
```

The file most LLM strategy experiments should mutate is:

```text
controller/beihang_baseline_strategy.py
```

The fixed Drake adapter around strategies is:

```text
controller/heuristic_strategy_system.py
```

## Run

```bash
python -m control_sims.beihang_minimal_sim.replay --duration 8
python -m control_sims.beihang_minimal_sim.run_trials --trials 10 --duration 8
```

Primary metrics are capture rate, capture time, minimum distance, final
distance, control effort, crash, and arena violation.

