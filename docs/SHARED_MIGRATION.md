# Shared Migration Status

The broad `shared/intercept_sim` tree has been removed.

The active Beihang control-sim code now owns the small local surface it uses:

- control payload and scene types live in `control_sims/beihang_paper_sim/types.py`
- camera/perception helpers live in `control_sims/beihang_paper_sim/sensing/`
- target and scene construction live in `control_sims/beihang_paper_sim/world/`
- the shared generator contract lives in `backends/csim/generator/generator.py`
- reusable scenario generators live in `scripts/generators/`

Useful verification commands:

```bash
rg "from intercept_sim|import intercept_sim" .
pytest tests/test_puffer_backend_smoke.py tests/test_rotorpy_backend_copy.py
```
