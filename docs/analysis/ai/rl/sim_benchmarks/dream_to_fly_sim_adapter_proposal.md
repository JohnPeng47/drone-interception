# Dream to Fly Simulator Adapter Proposal

Date: 2026-06-03

## Paper

Paper inspected: `papers/dream_to_fly.txt`

Downloaded PDF:

- `docs/analysis/ai/rl/sim_benchmarks/dream_to_fly_2501.14377.pdf`

The local text is `Dream to Fly: Model-Based Reinforcement Learning for Vision-Based Drone Flight`, arXiv `2501.14377v2`, dated 2026-04-10. The original arXiv identifier is from 2025.

## Simulator Stack Identified

The paper's simulation experiments use a high-fidelity environment combining:

- Flightmare for quadrotor simulation.
- Agilicious for the agile quadrotor platform, dynamics setup, and track-generation context.
- Habitat-Sim for fast RGB rendering in the training loop.

The paper's control and observation contract is:

- Observation: normalized RGB camera image, resized to `64 x 64 x 3`.
- Action: 4D CTBR command `[collective_thrust, body_rate_x, body_rate_y, body_rate_z]`, bounded in `[-1, 1]^4`.
- Reward: progress toward the active gate, body-rate penalty, collision penalty `-4.0`, gate-passed reward `+10.0`.

## Downloaded Artifacts

All artifacts are shallow clones in `docs/analysis/ai/rl/sim_benchmarks`:

- `flightmare`, commit `d4218aedac18cbe9364a0a0df10ab992c4b65e4f`
- `agilicious`, commit `50bba456156f6f438bea212c643b02e162442fa7`
- `habitat-sim`, commit `57ee4941dc4765240f0f91f70b2c97a919bf9038`

Important note: I did not find an official Dream-to-Fly implementation linked from Papers With Code. These are the public simulator components named by the paper, not the paper authors' complete training environment.

## Existing API Surface

Flightmare already has a vectorized RL API:

- `flightlib/include/flightlib/envs/vec_env.hpp`
- `flightlib/include/flightlib/envs/quadrotor_env/quadrotor_env.hpp`
- `flightrl/examples/run_drone_control.py`

The API has OpenAI Gym-style `reset()` and `step()` calls, vectorized environment support, and a Python `flightgym` binding. The default Flightmare quadrotor environment exposes a 12D state observation, so matching Dream-to-Fly requires adding the paper's racing task wrapper plus Habitat RGB rendering.

## Adapter Goal

Build a benchmark harness that can run the same benchmark shape as our existing intercept runner tests:

- Fixed scenario set.
- `reset()`.
- Batched `step(actions)`.
- Deterministic fake policy.
- JSON benchmark result with `mode`, `num_envs`, `steps`, `env_steps`, `elapsed_s`, `sim_sps`, `terminal_count`, and `obs_shape`.

This should be a simulator benchmark adapter, not a replacement for `SimEngine`. SimEngine interaction rules remain unchanged for our production interception code.

## Proposed Design

Create a small external-simulator benchmark module with three modes:

1. `flightmare_state`
   - Uses Flightmare `flightgym` vectorized env directly.
   - Observation shape is Flightmare's state vector.
   - Purpose: physics/control-loop throughput baseline without rendering.

2. `flightmare_rgb_habitat`
   - Wraps Flightmare/Agilicious dynamics and track state.
   - Uses Habitat-Sim to render `64 x 64 x 3` RGB frames at the quadrotor camera pose.
   - Implements the paper's race reward and gate terminal logic.
   - Purpose: closest benchmark to Dream-to-Fly.

3. `flightmare_rgb_unity` optional
   - Uses Flightmare's Unity renderer instead of Habitat.
   - Purpose: compare against upstream Flightmare rendering, but not required for the paper-replica harness.

## Proposed Files

Core adapter code:

- `ai/rl/external_sim_benchmarks/__init__.py`
- `ai/rl/external_sim_benchmarks/types.py`
- `ai/rl/external_sim_benchmarks/flightmare_backend.py`
- `ai/rl/external_sim_benchmarks/habitat_renderer.py`
- `ai/rl/external_sim_benchmarks/racing_task.py`
- `ai/rl/external_sim_benchmarks/benchmark.py`

CLI runner:

- `scripts/runners/benchmark_external_sim_envs.py`

Tests:

- `tests/test_external_sim_benchmark_modes.py`

Keep the downloaded upstream repos under `docs/analysis/ai/rl/sim_benchmarks` as references. Do not import from those paths in long-lived code; install or configure the upstream dependencies explicitly.

## Test Plan

Start with tests that do not require a GPU renderer:

- `flightmare_state` smoke benchmark with `num_envs=2`, `steps=1`.
- Deterministic reset and step for fixed actions.
- Terminal refill behavior on a tiny max-episode-steps setting.
- JSON schema parity with `ai.rl.simengine_batch.benchmark_modes.BenchmarkResult`.

Add renderer-gated tests:

- `flightmare_rgb_habitat` smoke test if `habitat_sim` is importable.
- Assert RGB observations have shape `(num_envs, 64, 64, 3)` and finite normalized values in `[0, 1]`.
- Measure separate timing buckets for physics step, render step, reward/terminal logic, and policy latency.

## Implementation Sequence

1. Package/build Flightmare `flightgym` in an isolated environment and run its bundled `run_drone_control.py` with rendering disabled.
2. Implement `FlightmareStateBackend` with the same `reset()` and `step(actions)` shape as `NativeInterceptBackend`.
3. Add `run_flightmare_state_benchmark()` beside the existing benchmark style and a CLI mode for it.
4. Implement a minimal racing task wrapper with fixed Circle/Kidney/Figure-8 gate definitions if the paper's exact tracks are not published.
5. Integrate Habitat rendering as a pluggable renderer.
6. Add `flightmare_rgb_habitat` benchmarks and renderer-gated tests.

## Risks

- The paper's exact training environment is not public as a single repo, so track geometry and task details may require reconstruction.
- Habitat-Sim and Flightmare have heavyweight native dependencies. The CI path should skip renderer tests unless dependencies are installed.
- Agilicious has a custom academic license flow; use it as a reference unless license terms are confirmed for direct integration.
- Renderer throughput may dominate the benchmark. Report physics-only and RGB-rendered modes separately.
