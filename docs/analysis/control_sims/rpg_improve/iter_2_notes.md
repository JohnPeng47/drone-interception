# Iteration 2 Notes

## Observation From Review Agent

- The main failure is now execution-grid/dynamics fidelity for motor-feedforward, not CTBR tuning.
- Diagnostics were unnecessarily expensive because motor-feedforward policy runs re-solved plans already produced by the diagnostic planner pass.
- Seed 6 showed that IPOPT budget can recover a valid `0.1 m` plan, but direct SimEngine RPM replay still misses, so the remaining issue is not just planner feasibility.

## Implemented This Iteration

- Added a cached motor-feedforward policy inside `run_diagnostics.py` so motor execution reuses the already-solved diagnostic plans.
- Added `--policies` to skip CTBR during hard-seed sweeps.
- Added `--rollout-tail-s` and `--post-plan-command-mode` for direct plan-rollout horizon diagnostics.
- Added `motor_command_mode` and `plan_time_scale` CLI/config support for command-sampling experiments.
- Fixed terminal tolerance satisfaction to allow small numerical epsilon.

## Results

- Hard missed seeds `1,4,5,6` with `cpc=0.1`, `nodes=60`, `ipopt=300`, motor-only cached execution:
  - all solved successfully to approximately `0.1 m`;
  - direct SimEngine planned-RPM rollout missed all four;
  - cached motor execution matched direct rollout min distances exactly.
- Seed 6 with linear motor command interpolation got worse: direct rollout miss increased to `4.41 m`.
- Seed 6 with `0.2 s` hold-last tail did not improve: direct rollout min distance stayed `1.724 m`.
- Seed 6 with `plan_time_scale=1.4` caught at `0.484 m`, but applying the same scale to hard seeds `1,4,5,6` only caught seed 6.
- Hard-seed timing sweep:
  - scale `1.0`: seed mins `1=2.417`, `4=1.789`, `5=1.822`, `6=1.724`;
  - scale `1.2`: seed mins `1=3.279`, `4=1.465`, `5=1.841`, `6=1.163`;
  - scale `1.4`: seed mins `1=1.557`, `4=1.919`, `5=2.674`, `6=0.484`.

## Suggestions

- Keep ZOH command sampling; it matches the planner assumption and outperforms linear interpolation.
- Do not focus on post-horizon command behavior yet; seed 6 is already missing before any tail behavior helps.
- Do not use a single global time scale as the main fix; it is seed-dependent and can trade one miss for another.
- Continue with dynamics-fidelity experiments using the cached motor-only path:
  - `dynamics_substeps=2` and `3`;
  - possibly lower `terminal_nodes` to offset solve cost;
  - tighter `cpc_tolerance_m` only after direct rollout fidelity improves.
