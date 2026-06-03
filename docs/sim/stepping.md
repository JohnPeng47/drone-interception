• Think of it as two API levels over the same engine.

C API: Primitive Operations
C exposes small operations separately:

sim_engine_step_motor_speeds_dt(...);  // advance state/time
sim_engine_get_snapshot(...);          // read state + collect camera outputs

The C step at backends/csim/sim_engine.c:367 does only this:

1. Step pursuer dynamics using dt and substeps.
2. Step each target once with the full dt.
3. Increment engine->t.
4. Update intercept metrics.

It does not collect camera outputs or render frames. Snapshot collection at backends/csim/sim_engine.c:395 is a separate C call, and that is where
sim_engine_collect_camera_outputs() runs.

So a C caller could do:

for 10 steps:
    sim_engine_step_motor_speeds_dt(...)

sim_engine_get_snapshot(...)
That would run physics 10 times but check camera/rendering only once.

Python Scalar API: Step Means Step Then Snapshot
The scalar Python API chooses a higher-level contract:

snapshot = backend.step_ctbr(snapshot, command)

That means:

1. Read the previous Python snapshot.
2. Rehydrate/configure the C SimEngine.
3. Convert CTBR command to motor RPMs.
4. Call C sim_engine_step_motor_speeds_dt(...).
5. Immediately call C sim_engine_get_snapshot(...).
6. Return a new Python snapshot dict.

That is in backends/csim/bindings/puffer_c.py:533.

So Python scalar stepping is not separate from C stepping. It is a wrapper that always does:

C step -> C snapshot -> Python dict

Why The Difference Exists
The C interface is granular because it is the engine boundary: step, inspect, collect cameras, configure rendering, and batch operations are separate
building blocks.

The Python scalar interface is user-facing and snapshot-oriented. It makes each Python step_ctbr() return all state needed by controllers, tests, and
render consumers. That is why snapshots happen every Python step.

The batch Python backend is similar in behavior but different internally: it keeps C engines resident, calls batch C step, then batch C snapshot every
Python batch step. See backends/csim/bindings/puffer_c.py:751.

Rendering Consequence
Because rendering is triggered during snapshot/camera-output collection, current Python scalar stepping checks capture/rendering after every Python step.
But that is a Python API behavior, not an unavoidable property of the C engine.