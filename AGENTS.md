Please do not recommend or make any decisions for "temporary compatibility"

All this does is to proliferate different implementations of the same function, which then drifts over time and leads to confusion when trying to understand the codebase

# SimEngine Interface
- All interactions with `SimEngine` must go through the C API and the Python binding layer in `backends/csim/bindings`.
- Python callers should pass typed objects from `backends/csim/bindings/types` (`SimInstance`, `SimConfig`, `PursuerInitialState`, `TargetConfig`, `CameraConfig`) rather than raw dicts or parallel adapter types.
- Do not add bypasses that call `sim_engine_*` directly outside the binding layer.
- All generators that subclass `SimGenerator` or `PregeneratedSimGenerator` must be implemented within `scripts/generators`.

# Drone Detection
- Anytime progress is made on drone image detection, document it in `docs/detection/worklog.md` with the date.
- Progress includes any tangible work towards improving or creating an image detection algorithm for drones.
- Progress also includes useful information that clarifies the goal, sets new goals, or records an experiment or attempt as a failure. Failed attempts are important ML evidence and should be recorded so future work can learn from them.

# Integration Tests
- After every significant change to sim logic, run this test
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
