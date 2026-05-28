Please do not recommend or make any decisions for "temporary compatibility"

All this does is to proliferate different implementations of the same function, which then drifts over time and leads to confusion when trying to understand the codebase

# SimEngine Interface
- All interactions with `SimEngine` must go through the C API and the Python binding layer in `backends/csim/bindings`.
- Python callers should pass typed objects from `backends/csim/bindings/types` (`SimInstance`, `SimConfig`, `PursuerInitialState`, `TargetConfig`, `CameraConfig`) rather than raw dicts or parallel adapter types.
- Do not add bypasses that call `sim_engine_*` directly outside the binding layer.
- `SimGenerator` means a disk-backed reader for generated `SimInstance` files. `PregeneratedSimGenerator` must not be reintroduced.
- Procedural generators must subclass `SimInstanceGenerator` and must be implemented within `scripts/generators`.
- Shared generator infrastructure that affects all `SimGenerator` or `SimInstanceGenerator` implementations belongs under `backends/csim/generator`.
- Generated simulation sample files (`*.csimin`) must be written under `scripts/generators/sim_instances`.
- Sim consumers, including control sims and RL runners, must consume generated `.csimin` files rather than sampling procedural generators directly.

# Drone Detection
- Anytime progress is made on drone image detection, document it in `docs/detection/worklog.md` with the date.
- Progress includes any tangible work towards improving or creating an image detection algorithm for drones.
- Progress also includes useful information that clarifies the goal, sets new goals, or records an experiment or attempt as a failure. Failed attempts are important ML evidence and should be recorded so future work can learn from them.

# Integration Tests
- After every significant change to sim logic, run this test
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py

# Scripts and Runnable Code
- *ALL* executable CLI code (except for analysis, and others explicitly defined in this doc) must follow these conventions
    - be implemented inside scripts/ -> 

# One-Off Analysis
- *All* analysis scripts that generate output should be implemented inside docs/analysis
- If the target of the analysis is tied to a specific source code folder, then you should create a subfolder for that source code inside docs/analysis/<source_folder>
- If the analysis generates artifacts, then put it into the analysis folder