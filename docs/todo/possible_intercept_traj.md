Goal:
# Current Implementation
For each candidate intercept time T within the episode horizon:

1. Predict target position at time T

For a straight-line target:

p_t_T = p_t0 + v_t * T

2. Compute the thrust direction the pursuer would need

Start from current pursuer state:

p0 = pursuer position
v0 = pursuer velocity
R0 = pursuer attitude

If the pursuer could accelerate for the full T, required world acceleration would be:

a_req = 2 * (p_t_T - p0 - v0 * T) / T**2

But because the pursuer must first rotate its thrust axis toward the needed acceleration direction, we estimate the rotation delay.

3. Convert required acceleration into required thrust direction

Gravity is not optional, so the thrust-produced acceleration must be:

thrust_accel_req = a_req - gravity_w

where:

gravity_w = [0, 0, -9.81]

The required thrust direction is:

n_req_w = normalize(thrust_accel_req)

4. Compute how long the pursuer needs to rotate toward that direction

Current thrust axis:

n_now_w = R0 @ thrust_axis_b

Then:

theta = acos(dot(n_now_w, n_req_w))
t_rotate_min = theta / max_rate_rps

T_accel = T - t_rotate_min

If T_accel <= 0, this intercept time is infeasible.

6. Recompute required acceleration using only useful acceleration time

a_req = 2 * (p_t_T - p0 - v0 * T) / T_accel**2

Then recompute:

thrust_accel_req = a_req - gravity_w
thrust_required_n = mass_kg * norm(thrust_accel_req)

7. Accept if thrust is within limits

feasible = thrust_required_n <= max_thrust_n

If any candidate T passes, the sample is considered reachable. If none pass, reject the sample.

Assumptions:

- Target trajectory is deterministic and independent of pursuer action.
- Target moves in a straight line during the validation horizon.
- Pursuer is approximated as a point mass after the initial attitude slew.
- Pursuer can rotate at max_rate_rps immediately and exactly.
- Pursuer does not meaningfully accelerate toward the final desired direction while rotating.
- After rotation, pursuer can hold the required thrust direction perfectly.
- Motor lag, drag, angular acceleration limits, and thrust ramp dynamics are ignored.
- This is not a proof of interceptability. It is a conservative-ish rejector for obviously bad samples.
- It may still produce false positives, but should avoid many physically unreasonable cases.


# Relevant Proposal on using a Idealized MPC controller to Rule out Bad Trajectories
https://claude.ai/chat/7072284c-0b3d-4989-8d3a-ebd530b8355e