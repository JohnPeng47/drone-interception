# TODO

- Deferred: implement RotorPy-style wind-relative airspeed effects in the C backend. This includes updating/consuming world-frame wind, computing body-frame airspeed from `R.T @ (v - wind)`, and using local rotor airspeeds for aero terms when `aero=True`. Ignored for now because the matched `paper_sim` replay uses `aero=False` and no wind dynamics.
- Deferred: implement or approximate RotorPy's adaptive `solve_ivp(method="RK45")` integration behavior. The current C backend uses fixed RK4 substeps; this matches the replayed paper cases closely enough, so adaptive local-error-controlled integration is ignored for now.

# 2026/05/29
- Consider adding sample_interval parameter to Python API to get lower resolution frames

# 2026/06/04
- • Not explicitly, at least not in the public Flightmare repo/docs I checked.

  What Flightmare does have is a configurable quadratic motor-speed-to-thrust map:

  thrust_map: [1.3298253500372892e-06, 0.0038360810526746033, -1.7689986848125325]

  from docs/analysis/ai/rl/sim_benchmarks/flightmare/flightlib/configs/quadrotor_env.yaml:13, used as:

  return omega_poly * thrust_map_;

  in docs/analysis/ai/rl/sim_benchmarks/flightmare/flightlib/src/dynamics/quadrotor_dynamics.cpp:106.

  That strongly looks like a fitted/calibrated curve, because it is:

  thrust = a * omega^2 + b * omega + c

  rather than a pure physical simplification like:

  thrust = k * omega^2

  But I did not find a comment, doc, or paper sentence saying “we fit this motor model from bench data.”

  The Flightmare paper describes the dynamics at the rotor-thrust level, not the motor-omega fitting level. It says Flightmare uses “classical quadrotor
  dynamics” and gives the mapping from individual rotor thrusts [f1, f2, f3, f4] to total thrust and torques. It also says the physics block can switch
  between basic dynamics and “real-world dynamics,” but it does not document how the default thrust_map coefficients were obtained.

  So the accurate answer is: probably calibrated/fitted, based on the shape and numeric coefficients, but not documented as fitted in the public Flightmare
  sources I found.