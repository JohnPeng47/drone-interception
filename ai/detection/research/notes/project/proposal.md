# Vision-Based Drone Interception via Synthetic Perception + Competitive RL

## Background and gap

Two recent lines of work define the state of the art in drone-vs-drone interception, and both punt on perception in the same way:

- **Yan 2024 (Beihang, IEEE TIE 2025; arXiv:2409.17497)** — analytical IBVS+PNG controller with a delayed Kalman filter. Real outdoor flights at 80% interception success in <4 m/s wind. Uses YOLOv7 on a tethered red balloon as the "target" — a trivially detectable stand-in. Predecessor: Yang 2024 (arXiv:2404.08296) introduced the IBVS-on-SO(3) formulation and the DKF.
- **Gavin 2025 (Thales/LAAS/ENAC, AMIAD 2025; hal-05554100)** — competitive multi-agent PPO with co-evolved pursuer and evader, low-level control (collective thrust + body rates), JAX simulator, real indoor flight with net catching. Pursuer's opponent observation is ground-truth 3D relative position+velocity from motion capture. Limitations section explicitly: *"perception of the target in high-velocity flights and state estimation are required but were not addressed."*

In parallel, the perception literature has produced credible synthetic-data drone detectors:

- **Sim2Air (Barisic 2022)**, **SynDroneVision (Lenhard 2024)**, **Drone Detection from Pure Synthetic Data (Wisniewski 2024)**, **SimD3 (2026)**, **Track Boosting (Akyon 2021)** — all train CNN drone detectors on rendered/simulated imagery and evaluate detection performance on real flight data.

These two communities don't cite each other. No published work plugs a synthetic-data-trained drone detector into a closed-loop interception controller, and no work uses image-plane observations of an actively-evading drone target inside a learned RL controller.

This proposal closes that seam.

## Phase 1: Synthetic perception pipeline for drone detection

### Goal
Build a synthetic data generation pipeline that produces labeled images of a target drone as seen from an interceptor drone along realistic approach trajectories. Train a CNN detector on the synthetic data alone. Characterize its failure modes on real footage.

### Lineage
- **Pipeline recipe**: Sim2Air's Blender + texture randomization template (S-UAV-T models, runnable code at larics/synthetic-UAV).
- **Background and lighting**: SynDroneVision's HDRI + Unreal-Engine-5 approach for realistic skies; Wisniewski's finding that pure-DR backgrounds underperform realistic ones.
- **Augmentation**: Lenhard's post-capture motion blur + Gaussian noise + JPEG compression chain; Akyon's *selective* synthetic-mixing finding (more synthetic ≠ better).
- **Edge cases**: Symeonidis's visible-vertex bbox projection (avoids occluded-region label noise); SimD3's tiny-target convention for sub-pixel ambiguity at 3-5 px target sizes.

### Distinguishing scope
Existing synthetic-drone-detection work generates random poses against random backgrounds. We sample poses **along simulated interception trajectories** — the camera is on a pursuer drone closing on a target, so the pose distribution matches what a real interceptor will see at deployment. This is the dataset-distribution choice that prior work doesn't make.

### Deliverables
1. Synthetic dataset (≥50k labeled frames) with per-frame metadata (relative pose, range, lighting, target type).
2. Trained YOLO-class drone detector with reported mAP on a held-out real-flight evaluation set (two-drone footage we collect).
3. **Detector failure-mode characterization**: bbox jitter distribution as a function of range, drop rate vs. target size, false-positive rate against birds/clutter. This characterization is the input to Phase 2.

### Evaluation
- mAP on real flight footage of two drones (one pursuer, one target).
- Comparison against a baseline trained on a public real-drone dataset (e.g., MAV-Vid, Drone-vs-Bird) at matched dataset size.

### Outcome
A drone detector that works on real footage, plus a quantitative model of its noise characteristics. We do **not** deploy this detector in Phase 2 — Phase 1's role is to (a) prove synthetic data alone is sufficient and (b) parameterize the noise distribution that Phase 2 trains against.

---

## Phase 2: Image-plane competitive RL for active-evader interception

### Goal
Train a pursuer policy via competitive multi-agent RL that takes **image-plane observations** of the target (bbox-style) plus its own inertial state, and outputs low-level control (collective thrust + body rates). Co-evolve against a learned evader. Demonstrate sim-to-real on physical drones.

### Lineage
- **RL framework**: Gavin 2025's competitive PPO co-evolution with low-level control; their JAX simulator as starting fork.
- **Observation modality**: Yang 2024 / Yan 2024's bearing-only IBVS as evidence that image-plane information is sufficient for interception (good observability properties); we use the same input modality but replace the analytical guidance law with a learned policy.
- **Sim-to-real**: Swift's (Kaufmann 2023) residual-modeling recipe — Gaussian processes for perception residuals, k-NN for dynamics residuals, fit from a small amount of real-world rollout data.

### Key departure from Gavin 2025
Gavin's pursuer observes `[po - pi, vo - vi]` — ground-truth 3D opponent state from motion capture. We replace this with `[bbox_cx, bbox_cy, bbox_w, bbox_h]` (image-plane, with optional history/recurrence) injected with **realistic detector noise sampled from the Phase 1 characterization**. Self-observation (own attitude + velocity from IMU/VIO) is unchanged.

### Why image-plane, not 3D state
- No range estimation required (range from bbox size is noisy and target-dimension-dependent).
- No velocity differencing (numerical differentiation of noisy bbox amplifies noise).
- Domain randomization during RL training is honest — we randomize over the actual failure modes a CNN detector exhibits (jitter, drops, sub-pixel error), measured in Phase 1.
- Aligns with the IBVS observability evidence from Yang/Yan — bearing alone is sufficient information.

### Deliverables
1. Pursuer + evader policies trained in JAX simulator with Phase-1-parameterized bbox noise.
2. Sim-to-real transfer to physical drones using Phase 1's actual detector at inference (closing the loop on the noise model).
3. Comparison against three baselines: (a) Gavin's mocap-state policy, (b) IBVS-PNG controller from Yan 2024 with our detector plugged in, (c) PN/pure-pursuit heuristics.

### Evaluation
- Catch rate, time-to-catch, crash rate in simulation (matching Gavin's metrics).
- Real-world catch rate against a manually piloted evader and against the learned evader policy in an indoor arena.
- Ablation: trained policy vs. detector-output baseline (Phase 2 path A) to quantify what the learned controller buys over analytical IBVS.

### Outcome
First demonstrated drone-vs-drone interception system trained with synthetic perception only, deployed against an actively evading target, with a learned controller that consumes image-plane observations directly.

---

## Lineage summary

```
Phase 1 inputs:                          Phase 2 inputs:
  Sim2Air (pipeline)                       Gavin 2025 (RL framework, sim)
  SynDroneVision (HDRI/UE5)                Yang/Yan (bearing observability)
  Wisniewski (DR vs realistic)             Swift (residual sim2real)
  Lenhard (augmentation)                   Phase 1 (detector + noise model)
  Akyon (selective mixing)
  Symeonidis (bbox projection)
  SimD3 (tiny-target labels)

Phase 1 outputs:                         Phase 2 outputs:
  - Trained drone detector                 - Pursuer + evader policies
  - mAP on real flight                     - Sim-to-real demo with real detector
  - Detector noise characterization  ────► - Comparisons vs Gavin/Yan/heuristic baselines
```

The contribution is the seam: Phase 1's noise characterization parameterizes Phase 2's training distribution, and Phase 1's detector deploys at Phase 2's inference. Neither phase reproduces existing work; each addresses a gap the cited prior work explicitly leaves open.
