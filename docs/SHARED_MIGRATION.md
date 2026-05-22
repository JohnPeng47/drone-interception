# Shared Migration Status

The broad `shared/intercept_sim` tree has been removed.

The active Beihang control-sim code now owns the small local surface it uses:

- control payload and scene types live in `control_sims/beihang_paper_sim/types.py`
- camera/perception helpers live in `control_sims/beihang_paper_sim/sensing/`
- target and scene construction live in `control_sims/beihang_paper_sim/world/`
- the shared generator contract lives in `backends/generator.py`
- red-balloon scenario generation lives in
  `control_sims/beihang_paper_sim/sim/generator/`
- metrics used by the generator runners live in
  `control_sims/beihang_paper_sim/sim/generator/base.py`

Useful verification commands:

```bash
rg "from intercept_sim|import intercept_sim" .
python -m control_sims.beihang_paper_sim.sim.generator.red_balloon_trials --n-trials 1 --duration-s 0.2
pytest tests/test_puffer_backend_smoke.py tests/test_rotorpy_backend_copy.py
```
