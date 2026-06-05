# Intercept PPO Reward Shaping Notes

Date: 2026-06-04

## Corpus

PDFs were downloaded to `research/intercept-ppo/pdfs` and converted with `pdftotext -layout` to `research/intercept-ppo/txt`.

| Paper | Local PDF | Local text | Source |
|---|---|---|---|
| Agile Interception of an Agile Drone with Reinforcement Learning | `pdfs/agile_interception_competitive_rl.pdf` | `txt/agile_interception_competitive_rl.txt` | https://arxiv.org/pdf/2603.16279 |
| Learned Controllers for Agile Quadrotor Pursuit-Evasion | `pdfs/learned_controllers_agile_quadrotors.pdf` | `txt/learned_controllers_agile_quadrotors.txt` | https://arxiv.org/pdf/2506.02849 |
| Online Planning for Multi-UAV Pursuit-Evasion in Unknown Environments Using Deep Reinforcement Learning | `pdfs/online_planning_multi_uav_pursuit_evasion.pdf` | `txt/online_planning_multi_uav_pursuit_evasion.txt` | https://arxiv.org/pdf/2409.15866 |
| Deep Reinforcement Learning-Based Guidance Law for Intercepting Low-Slow-Small UAVs | `pdfs/deep_rl_guidance_lss_uav_rppo.pdf` | `txt/deep_rl_guidance_lss_uav_rppo.txt` | https://www.mdpi.com/2226-4310/12/11/968 |
| Active Interception for Multi-Target Encirclement by Heterogeneous UAVs | `pdfs/active_interception_lippo_heterogeneous_uavs.pdf` | `txt/active_interception_lippo_heterogeneous_uavs.txt` | https://www.mdpi.com/2411-9660/10/2/26 |
| Dynamic Decoupling Mechanism for Multi-UAV Cooperative Pursuit Based on MA2PPO | `pdfs/dynamic_decoupling_ma2ppo_cooperative_pursuit.pdf` | `txt/dynamic_decoupling_ma2ppo_cooperative_pursuit.txt` | https://www.techscience.com/cmc/online/detail/24135 |

## Current ai/rl Setup

The current PPO task is a single pursuer navigating to a target using low-level collective thrust plus body-rate actions. The native reward is in `ai/rl/puffer_intercept/c/puffer_intercept_binding.c`:

```text
r_t = 28 * (exp(-d_t / 250) - exp(-d_{t-1} / 250))
      - 0.001 * d_t / 532.67
      - 2e-4 * ||omega_t||
      - 30 * I_fail
```

The reward has no terminal catch bonus. PPO currently clamps raw rollout rewards to `[-1, 1]` before computing advantages in `ai/rl/puffer_intercept/puffer_ppo.py`, so any coefficient transfer from papers should be checked against the post-clamp reward distribution.

Current observations are 25 raw values: pursuer position, velocity, rotation matrix, previous action, target position, and target velocity. They are not currently expressed as normalized relative position/velocity in the Python observation helper.

## Most Relevant Findings

### 1. Agile drone-on-drone PPO is closest to our task

`agile_interception_competitive_rl` is directly relevant: one quadrotor catches another, both trained with PPO, and policy actions are collective thrust plus body rates. Its pursuer reward is:

```text
r_P = r_catch - r_dist - r_coll - r_fail - r_cmd
```

The paper's coefficients are close to our current constants:

| Term | Paper value | Current equivalent |
|---|---:|---:|
| distance penalty | `lambda_dist = 0.001` | `distance_weight = 0.001` |
| command/body-rate penalty | `lambda_cmd = 2e-4` | `rate_weight = 2e-4` |
| failure penalty | `lambda_fail = 30` | `fail_penalty = 30` |
| catch reward | `lambda_catch = 10` | none |
| collision/contact penalty | `lambda_coll = 0.1` | none |
| boundary penalty | `lambda_bnd = 1.0` | hard fail only |

Useful transpositions:

- Our rate and fail penalty magnitudes are already aligned with the closest paper.
- The biggest missing shaping term from that paper is not more distance shaping, but soft safety shaping: soft contact/collision penalties and exponential boundary-buffer penalties before hard failure.
- Their observations use relative position and relative velocity to the opponent, plus boundary/ground distances, normalized by range and max velocity. Our current observations include absolute positions and velocities, so translation/generalization may be worse than necessary.
- They used terminal catch reward. We intentionally removed the catch bonus, so do not add it blindly. Treat it as a diagnostic-backed change only if eval rollouts show terminal loitering or near-target non-termination.

### 2. Prediction/closing guidance helps when targets move, but only lightly applies to stationary targets

`deep_rl_guidance_lss_uav_rppo`, `online_planning_multi_uav_pursuit_evasion`, and `active_interception_lippo_heterogeneous_uavs` all emphasize prediction:

- RPPO designs reward around velocity prediction and overload constraints to address sparse rewards in 3D interception.
- OPEN uses an evader-prediction-enhanced network and a two-stage reward refinement process.
- LIPPO predicts future target positions and pursues interception points instead of current target locations.

For the current stationary-target task, target prediction collapses to the current target position. The useful part is therefore one-step lookahead / closing-rate shaping, not a full LSTM predictor:

```text
closing_t = (d_{t-1} - d_t) / dt
r_close = w_c * clip(closing_t / v_ref, -1, 1)
```

This overlaps with the current potential-progress term, so it should only be added if logs show the exponential potential is too sparse early in training or flattened by reward clipping. A less redundant variant is velocity alignment:

```text
r_align = w_align * clip(dot(v_pursuer, target_pos - pursuer_pos)
                         / (||v_pursuer|| * ||target_pos - pursuer_pos|| + eps),
                         -1, 1)
```

That rewards pointing motion toward the target even before substantial distance closure occurs.

### 3. Smoothness is a sim-to-real issue, not just cosmetics

OPEN explicitly reports that training only with task rewards produced aggressive policies that failed in real-world deployment, while adding smoothness too early blocked exploration; their successful setup used two-stage reward refinement.

For our current setup, a command-delta penalty is worth testing after the agent reliably reaches the target:

```text
r_smooth = -w_da * ||a_t - a_{t-1}||^2
```

This differs from the existing body-rate penalty: current `rate_weight` discourages large angular-rate commands, while `r_smooth` discourages oscillatory command changes. Because our observation already includes previous action, this term is easy to expose to the policy.

Do not introduce it as a permanent default until we can measure whether it slows initial learning. The likely pattern is task reward first, then smoothness refinement.

### 4. Boundary shaping is the highest-confidence immediate addition

The agile interception PPO paper calls out boundary exploitation: agents learn to use arena walls as strategic artifacts unless discouraged. Our current env only applies hard failure when the pursuer exits bounds.

A soft boundary buffer would give PPO a gradient before catastrophic failure:

```text
r_bound = -w_b * sum_i exp(-(d_i - margin) / sigma_b) * I[d_i < margin]
```

where `d_i` is distance to each wall/ground limit. This is consistent with the closest drone-on-drone PPO paper and should reduce cliff-edge policies.

### 5. Multi-agent formation rewards do not apply yet

The MA2PPO and LIPPO papers include formation, encirclement, target assignment, and multi-agent credit assignment rewards. These are not useful for the current single-pursuer stationary target objective.

The transferable lesson is methodological: ablate reward terms and avoid irrelevant interactions that increase advantage variance. For our codebase, that means logging reward components and measuring term contribution before adding more shaping.

## Recommended Next Reward Experiments

1. Add native reward-component logging before changing coefficients.
   - Log progress, distance, rate, fail, raw reward, and PPO-clipped reward.
   - Current training logs aggregate episode return, length, and min distance only.

2. Check whether PPO reward clipping is flattening the intended shaping.
   - If many raw rewards hit `+1` or `-1`, the effective objective is not the paper/objective formula.

3. Add soft boundary/ground penalty.
   - This is the clearest directly supported transposition from drone-on-drone PPO.

4. Convert observations toward relative, normalized target features.
   - Use relative target position and velocity, preserve rotation matrix and previous action, add boundary distances if bounds exist.
   - This is not reward shaping, but the closest papers consistently normalize relative state.

5. Add command-delta smoothness only after basic catch/navigation is reliable.
   - Use staged training or a scheduled coefficient so exploration is not suppressed early.

6. Add a small terminal catch bonus only if diagnostics show near-target loitering.
   - The closest paper uses `lambda_catch = 10`, but our current objective intentionally removed the intercept bonus.
   - A smaller bonus or thresholded terminal-distance penalty should be justified by terminal distance histograms, not copied blindly.

## Practical Objective Candidate

For the current stationary target, the safest candidate objective is:

```text
r_t =
    28 * (exp(-d_t / 250) - exp(-d_{t-1} / 250))
  - 0.001 * d_t / 532.67
  - 2e-4 * ||omega_t||
  - w_b * boundary_buffer_t
  - 30 * I_fail
```

Optional second-stage refinement:

```text
r_t -= w_da * ||a_t - a_{t-1}||^2
```

Optional only if loitering is observed:

```text
r_t += w_success * I_intercept
```

The next engineering step should be instrumentation, because without component logs we cannot tell whether a new term changes behavior or is simply clipped away before PPO sees it.
