# Beihang Minimal Sim

This package contains the minimal Beihang image-centering controller strategy
adapted to the shared C `SimEngine` runner.

The runtime path is:

```text
.csimin scenarios
  -> scripts/runners/control_sim/beihang_minimal_sim.py
  -> backends.csim.runner.SimRunner
  -> BeihangMinimalSimControlPolicy
  -> BatchPufferSimEngineBackend
  -> C SimEngine
```

The deleted Drake point-mass replay path is not part of this interface. Control
sim consumers must run generated `.csimin` scenarios through `SimRunner`.

Primary implementation files:

```text
config.py
types.py
policy.py
controller/beihang_baseline_strategy.py
```

Run through the control-sim CLI:

```bash
python scripts/runners/control_sim/beihang_minimal_sim.py \
  --scenario-table scripts/generators/sim_instances/controller_regression_6/sobol_samples.csimin \
  --samples 6 \
  --workers 1
```
