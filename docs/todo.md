# TODO

- Deferred: implement RotorPy-style wind-relative airspeed effects in the C backend. This includes updating/consuming world-frame wind, computing body-frame airspeed from `R.T @ (v - wind)`, and using local rotor airspeeds for aero terms when `aero=True`. Ignored for now because the matched `paper_sim` replay uses `aero=False` and no wind dynamics.
- Deferred: implement or approximate RotorPy's adaptive `solve_ivp(method="RK45")` integration behavior. The current C backend uses fixed RK4 substeps; this matches the replayed paper cases closely enough, so adaptive local-error-controlled integration is ignored for now.
> 
