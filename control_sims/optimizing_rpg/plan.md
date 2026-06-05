# RPG Solver Reimplementation Plan

## Context

The current `rpg_time_optimal_motor_portfolio` runner is dominated by trajectory
planning, not simulation. A one-seed profile on
`scripts/generators/sim_instances/sobol_samples_512.csimin` showed:

- Full CLI wall time: about 25.7 s.
- `solve_portfolio_plan`: about 25.6 s.
- First candidate planner solve: about 24.7 s.
- CasADi NLP build: about 6.6 s.
- IPOPT optimizer: about 16.7 s over 84 iterations.
- Replay scoring: about 45 ms.
- SimRunner stepping/final metrics: about 70 ms.

The replacement solver should therefore focus on removing CasADi/IPOPT from the
hot path, reducing the decision variables, exploiting trajectory structure, and
parallelizing useful fixed-time probes. It should not create a long-lived
temporary compatibility fork of planner behavior. Every milestone must report
its seed `1` execution time difference and percent improvement versus the
Milestone 1 single-scenario reference. The final parallelization harness must
also benchmark more than one scenario execution. Milestones 3, 5, and 6 add an
explicit catch objective using `scripts/generators/sim_instances/sobol_samples_128.csimin`
or a documented subset of it. The default subset is seeds
`1,2,3,4,5,6,7,8`.

## Target Architecture

Build a specialized drone intercept solver around fixed-time feasibility:

```text
planner(instance)
  parallel time-probe scheduler
    solve_fixed_time(instance, T)
      rollout dynamics
      compute costs and constraints
      compute derivatives or local model
      solve structured trajectory update
      line search / trust region
    verify best candidates in SimEngine
  return fastest replay-valid motor plan
```

The core formulation should optimize motor commands only:

```text
N * 4 motor command variables
```

rather than the current full NLP:

```text
T + all states + all motor commands
```

For a 60-node horizon, that changes the primary decision vector from roughly
1278 variables to 240 controls. States should be produced by forward rollout.
Final acceptance should still be based on replay through the existing C API and
Python binding layer.

## Benchmark Rule

- Milestones 1 through 5 use seed `1` from `scripts/generators/sim_instances/sobol_samples_512.csimin` as the primary harness.
- Milestone 1 establishes the single-scenario reference wall time.
- Every milestone records elapsed wall time, baseline wall time, delta seconds,
  and percent improvement.
- Milestone 3 starts catch correctness: seed `1` must produce replay-valid catch
  commands at a fixed time, and the `sobol_samples_128.csimin` subset records
  caught/min-distance/failure-reason diagnostics without making catch fraction
  the primary tuning target.
- Milestone 5 starts catch-search tuning using parallel fixed-time probes on
  seed `1` and the `sobol_samples_128.csimin` subset.
- Milestone 6 also benchmarks more than one scenario execution to validate the
  final parallelization harness. It is the first milestone where catch fraction
  is a primary reported objective.

## Milestone 1: Baseline Harness

### Goal

Create a repeatable benchmark and correctness harness before writing the new
solver.

### Implementation Guidance

- Keep benchmark code and generated artifacts under `control_sims/optimizing_rpg`.
- Use `scripts/generators/sim_instances/sobol_samples_512.csimin`.
- Start with a fixed seed set covering easy, medium, and hard scenarios. Include
  seed `1`, because it has an existing profile.
- Record current portfolio behavior:
  - catch result
  - selected plan total time
  - min distance
  - final distance
  - wall time
  - selected candidate
  - per-candidate solve traces
- Add a single comparison command that can run the current planner and later the
  experimental planner on the same seeds.
- Report parallel scaling separately for:
  - one-scenario latency
  - many-scenario throughput
  - `--workers` scaling
  - CPU utilization where practical

### Acceptance Criteria

- One command benchmarks the current IPOPT planner on a fixed seed set.
- Output includes per-seed wall time, catch status, min distance, final distance,
  selected plan total time, selected candidate, and failure reason.
- Output includes aggregate catch fraction, median/p90 wall time, and total wall
  time.
- Seed `1` reproduces the known baseline shape: about 25 s, first candidate
  early-accepted, caught.
- The harness makes one-scenario latency distinct from many-scenario throughput.
- No direct `sim_engine_*` bypasses are added outside the binding layer.

## Milestone 2: Numeric Rollout Core

### Goal

Replace symbolic CasADi graph construction with a direct numeric trajectory
rollout suitable for the custom solver.

### Implementation Guidance

- Implement:

```python
rollout(instance, controls, total_time) -> trajectory
```

- Match the current planner state layout:

```text
position_w      3
velocity_w      3
quat_wxyz       4
body_rates_b    3
motor_rpm       4
```

- Start with the same planning dynamics currently used in
  `control_sims/rpg_time_optimal/planner.py`:
  - motor lag
  - thrust from RPM
  - attitude quaternion dynamics
  - body-rate dynamics
  - state clamps
- Preallocate trajectory arrays for states, commands, distances, and violation
  metrics.
- Keep the first version in Python/NumPy for math validation.
- Design the API to be pure and reentrant so multiple rollouts can run
  concurrently.

### Acceptance Criteria

- Given an existing IPOPT plan's motor commands, the new rollout predicts a
  trajectory close enough to the current planner rollout for planning use.
- Rollout reports terminal position, min distance, RPM ranges, body-rate ranges,
  altitude range, and violation metrics.
- One 60-node rollout is milliseconds-scale, not seconds-scale.
- The trajectory can be converted into a motor plan and replayed through
  `BatchPufferSimEngineBackend`.
- The rollout function does not allocate large arrays inside the node loop.
- Concurrent rollout calls do not share mutable solver state.

## Milestone 3: Fixed-Time Feasibility Solver

### Goal

Build `solve_fixed_time(instance, T)` before optimizing over time.

### Implementation Guidance

- Implement:

```python
solve_fixed_time(instance, total_time, initial_controls) -> candidate_plan
```

- Optimize only motor commands for a fixed horizon.
- Use a merit objective such as:

```text
terminal miss penalty
+ capture-window miss penalty
+ motor smoothness penalty
+ command saturation penalty
+ body-rate violation penalty
+ altitude violation penalty
+ thrust violation penalty
```

- Treat SimEngine replay as the final accept/reject oracle.
- Start with a simple optimizer if needed, but keep it clearly marked as an
  experimental stepping stone, not a permanent alternative implementation.
- Make the fixed-time solver pure/reentrant so many `T` values can be solved in
  parallel.
- Add structured diagnostics for each solve.

### Acceptance Criteria

- For seed `1` and a baseline-like intercept time, fixed-time solve produces a
  motor command trajectory that catches in SimEngine replay.
- A catch-diagnostic pass runs on seeds `1,2,3,4,5,6,7,8` from
  `scripts/generators/sim_instances/sobol_samples_128.csimin`, recording
  caught, min distance, final distance, replay validity, and failure reason per
  seed.
- The solver returns diagnostics:
  - objective terms
  - terminal miss
  - min distance
  - max violations
  - iteration count
  - wall time
  - replay result
  - failure reason
- It distinguishes "optimization failed" from "optimized but replay-infeasible."
- Multiple fixed-time solves for different `T` values can run concurrently with
  deterministic results.
- The fixed-time API is stable enough for a parallel time-probe scheduler.

## Milestone 4: Structured Trajectory Update

### Goal

Replace generic optimization with a trajectory-structured iLQR/SQP-style update.

### Implementation Guidance

- Implement the iteration structure:

```text
rollout current controls
linearize dynamics along trajectory
quadraticize objective
backward pass
forward rollout with line search
accept or reject update
```

- Use fixed dimensions:

```text
state:   17
control: 4
horizon: N
```

- Initially allow finite-difference derivatives to validate the algorithm.
- Add batched/parallel finite-difference derivative evaluation as an explicit
  stepping stone.
- Then replace the hot derivative path with analytic or semi-analytic
  derivatives.
- Exploit trajectory time-chain structure. Do not assemble one giant generic
  dense optimization problem.
- Keep constraints mostly as penalties/barriers at this milestone.

### Acceptance Criteria

- On seed `1`, the structured solver reaches a replay-valid intercept faster
  than the current IPOPT baseline by a meaningful margin.
- Per-iteration timing is broken down into:
  - rollout
  - derivatives
  - backward pass
  - line search
  - replay verification
- Finite-difference and analytic/semi-analytic derivative modes agree within a
  documented tolerance on a small trajectory.
- Parallel derivative evaluation produces the same accepted result as serial
  evaluation.
- Merit score improves monotonically or the solver reports a clear stall reason.
- Solver output is deterministic for the same seed/config.

## Milestone 5: Parallel Time Search, Warm Starts, And Early Exit

### Goal

Turn fixed-time feasibility into a practical minimum-time intercept planner with
explicit parallelism.

### Implementation Guidance

- Implement:

```python
find_fastest_intercept(instance) -> plan
```

- Use a parallel probe scheduler:

```text
launch fixed-time solves for several T values
collect feasible/infeasible results
refine bracket
launch next batch
return fastest replay-valid plan
```

- Use bracketing:

```text
T_low: likely impossible
T_high: feasible or conservative
```

- Use binary search, safeguarded line search, or batched frontier refinement.
- Warm-start each new `T` from the nearest prior solution.
- Resample controls when changing time or horizon.
- Add early exits:
  - replay catch margin is good enough
  - time resolution is below useful sim tick resolution
  - fixed-time solve stalls
  - iteration budget is exhausted
  - a robust plan already satisfies acceptance gates

### Acceptance Criteria

- On the benchmark seed set, the planner returns the fastest found replay-valid
  plan, not just the first feasible plan.
- Seed `1` catches with comparable catch time and min distance to the baseline.
- The catch-search strategy runs on seeds `1,2,3,4,5,6,7,8` from
  `scripts/generators/sim_instances/sobol_samples_128.csimin` and records
  per-seed catch-search outcomes.
- Diagnostics report:
  - time probes launched
  - probes solved serially vs in parallel
  - feasible/infeasible bracket
  - number of fixed-time solves
  - warm-start source for each probe
  - early-exit reason
- Parallel time probing improves one-scenario wall time versus serial probing on
  at least one nontrivial benchmark case.
- Parallel execution is deterministic at the planner-result level.
- Failure cases include useful reasons:
  - no feasible bracket
  - fixed-time infeasible
  - replay mismatch
  - constraint violation
  - iteration budget exceeded

## Milestone 6: Performance Hardening And Runner Integration

### Goal

Make the custom solver production-shaped and compare it against the current
portfolio runner.

### Implementation Guidance

- Profile the custom solver before moving hot code out of Python.
- Move only proven hot kernels to a native implementation path.
- Pick one implementation strategy for hot kernels, not several:
  - C++ extension
  - Cython
  - Numba
  - existing C backend style
- Preallocate all trajectory, derivative, and line-search buffers.
- Use contiguous fixed-shape arrays.
- Avoid Python object creation inside rollout, derivative, backward-pass, and
  line-search loops.
- Keep final replay through the existing binding-backed SimEngine path.
- Integrate as a real policy/runner path, not a temporary compatibility shim.
- Add focused tests and benchmark outputs.

### Acceptance Criteria

- Benchmark report compares:
  - current IPOPT portfolio wall time
  - custom solver serial one-scenario wall time
  - custom solver parallel one-scenario wall time
  - many-scenario throughput
  - catch fraction
  - catch time
  - min distance
  - failure reasons
  - CPU utilization where practical
- The multi-scenario benchmark uses at least two seeds from
  `scripts/generators/sim_instances/sobol_samples_128.csimin`; the default target
  is seeds `1,2,3,4,5,6,7,8`, with optional expansion to all 128 scenarios.
- For seed `1`, custom solver wall time is dramatically below the about 25 s
  baseline while still catching in replay.
- On the selected seed set, catch quality meets explicit thresholds for catch
  fraction, min distance, and replay validity.
- The implementation exposes documented configuration knobs:
  - horizon nodes
  - fixed-time iteration budget
  - time-search tolerance
  - penalty weights
  - line-search settings
  - parallel probe count
  - replay acceptance gates
- Significant sim logic changes pass:

```bash
python -m pytest -q tests/test_sim_instance_store.py tests/test_puffer_backend_smoke.py
```

## Optimization Sources By Layer

1. Problem formulation:
   optimize motor commands only, not states plus controls plus time.

2. Time handling:
   move intercept time into a parallel fixed-time feasibility search.

3. Dynamics rollout:
   replace symbolic graph construction with direct numeric rollout.

4. Derivatives:
   use tailored finite-difference, batched finite-difference, then analytic or
   semi-analytic derivatives.

5. Local update solve:
   exploit trajectory Riccati/block structure instead of generic sparse NLP
   machinery.

6. Constraints:
   use cheap penalties/barriers during solve and exact replay gates for final
   acceptance.

7. Warm starts:
   reuse controls between time probes, nearby horizons, and portfolio variants.

8. Early exit:
   stop when the plan is replay-valid and robust enough, not when a general NLP
   optimum is polished.

9. Implementation:
   preallocate buffers, use fixed shapes, remove Python inner-loop overhead only
   after profiling.

10. Parallelism:
   use scenario-level workers for throughput, parallel time probes for
   one-scenario latency, and kernel-level parallelism for derivatives and line
   search where profiling supports it.
