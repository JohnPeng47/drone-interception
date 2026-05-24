# Trajectory-Conditioned Synthetic Drone Detection Pipeline

## Distinguishing claim

Existing synthetic-drone-detection datasets sample **random poses against random backgrounds** ([Sim2Air](sim2air_barisic_2022.txt), [SynDroneVision](syndronevision_lenhard_2024.txt), [Wisniewski](pure_synthetic_wisniewski_2024.txt), [SimD3](simd3_2026.txt)). The pose distribution at deployment for an interceptor is not random — it is the joint distribution of (pursuer pose, target pose) traced out by an actively-closing pursuer against an evading target. We sample frames **along simulated interception trajectories**, so the training distribution matches the test distribution.

This is the only contribution unique to this pipeline. Every other component is a recombination of prior-art recipes; the table at the end of this doc credits each one.

---

## Pipeline architecture

```diagram
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 1: TRAJECTORY GENERATOR                                                   │
│   Run interception sim → stream of (t, pursuer_pose_W, target_pose_W) tuples    │
│     - Pursuer policy: seed with PNG/IBVS analytical guidance, later swap RL     │
│     - Evader policy: scripted (banks/jukes) for Phase 1; learned later          │
│     - Sample rate: 30 Hz, sim length 4-8 s per engagement, ~10k engagements     │
│     - Output: pursuer_pose, target_pose, target_attitude, range, closing_rate   │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 2: SCENE COMPOSER (Blender Cycles, headless)                              │
│   For each (pursuer_pose, target_pose) sample:                                  │
│     a. Camera ← pursuer_pose, intrinsics matching deployment camera             │
│     b. Place target drone model at target_pose (relative to world)              │
│     c. Sample HDRI from PolyHaven set (lighting + sky background)               │
│     d. Sample drone model from N=10 fleet (Sim2Air S-UAV-T set + custom)        │
│     e. Sample texture from mixed shape-bias texture pool [→ sim2air_pose]       │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 3: RENDER (Cycles, 512 path-trace samples, 1920×1080)                     │
│     - Realism ON: PBR shaders, Lumen-equivalent global illum via HDRI emission  │
│     - Wisniewski finding: realistic context outperforms pure DR backgrounds     │
│       [→ wisniewski_sdr]                                                        │
│     - SynDroneVision finding: UE5/Lumen-class lighting closes the sim2real gap  │
│       on out-of-distribution real data [→ syndronevision_ue5]                   │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 4: ANNOTATION (per-frame, automated)                                      │
│     - Project visible mesh vertices of target model into image plane            │
│     - bbox = tight axis-aligned hull of visible-vertex projections              │
│       [→ symeonidis_bbox]  [→ simd3_bbox]                                       │
│     - Per-frame metadata: range, relative_attitude, target_size_px,             │
│       lighting_id, occlusion_fraction                                           │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 5: POST-CAPTURE AUGMENTATION                                              │
│     - Motion blur: Gaussian/box kernel, M ∈ {13,15,17,19,21}, applied to ~10%   │
│       of frames per SynDroneVision recipe [→ syndronevision_blur]               │
│     - Akyon recipe: film-grain noise + JPEG compression + matched-blur          │
│       (blur magnitude is the most important feature) [→ akyon_features]         │
│     - Per-frame: stochastic kernel choice, no global post-process               │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ Stage 6: CURATION (selective mixing)                                            │
│     - Akyon: more synthetic ≠ better; sub-sample optimally                      │
│       [→ akyon_selective]                                                       │
│     - Hold trajectory diversity > frame count: 10k engagements × 4-8 s ≫        │
│       1 long engagement × many frames                                           │
│     - 7% background-only frames (no target visible) per SynDroneVision          │
└────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ OUTPUT: ≥50k labeled frames + per-frame metadata + detector noise model         │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Concrete implementation choices

### Trajectory source (Stage 1)
- **Engine**: Gavin 2025's JAX competitive PPO sim is the right substrate — already built for the Phase 2 controller, so the trajectory distribution Phase 1 trains on matches the deployment distribution exactly.
- **Pursuer policy for Phase 1 trajectories**: do *not* use the eventual learned RL policy yet (chicken-and-egg). Use Yang/Yan-style IBVS+PNG analytical guidance with ground-truth target state. This is conservative — produces straight, fast-closing engagements. Augment with random closing geometries (offset/lead) so the dataset is not collapsed onto one guidance law.
- **Evader policy**: scripted bank/juke maneuvers + straight-line + sinusoidal-altitude. The detector should not be tuned to a specific evader.
- **Coverage check**: bin (range, bearing-rate, target_size_px) and reject samples that exceed per-bin quota — guarantees the dataset is not dominated by the easy mid-range regime.

### Renderer (Stages 2–3)
- **Blender + Cycles** over Unreal/AirSim, despite SynDroneVision/SimD3 using UE5. Reasoning:
  - Sim2Air's open-source S-UAV-T pipeline ([larics/synthetic-UAV](sim2air_barisic_2022.txt:246)) is Blender-based; we get a working starting point.
  - Headless batch rendering is mature.
  - The SynDroneVision lighting advantage is reproducible in Cycles via HDRI emission + path tracing — Lumen is not load-bearing.
- **Drone fleet**: start with Sim2Air's 10 models (S-UAV-T set), add 2-3 custom models matching the actual evader hardware once selected.
- **Textures**: Sim2Air-style randomized texture pool (32 textures: diffuse/glossy/glass/translucent BSDFs + atypical patterns + 1 photoreal carbon). Wisniewski's "realistic + structured randomization" beats both pure-DR and pure-photoreal — we keep both buckets.
- **HDRI library**: PolyHaven 50+ outdoor HDRIs covering daylight, overcast, twilight, dawn, dusk. Sample uniformly per engagement (not per frame — keep lighting consistent across an engagement).

### Annotation (Stage 4)
- **Visible-vertex bbox projection** (Symeonidis-style): per frame, project each mesh vertex through the camera, ray-cast against the depth buffer to test visibility, take axis-aligned hull of visible projections. Avoids the standard bbox bug where a tight hull around the *full* mesh extends outside the silhouette when part of the drone is occluded by a closer object (e.g. its own arms self-occluding at oblique angles).
- **Sub-pixel targets**: at long range the target is 2-5 px. Follow SimD3's convention: clamp bbox to ≥1 px and emit `target_size_px` in metadata so a downstream curriculum can stratify.

### Augmentation (Stage 5)
- **Blur**: SynDroneVision's stochastic kernel sweep, applied to 10% of frames. Akyon's finding that blur is the single highest-impact synthetic feature suggests this should be tuned to match the *real* deployment camera's motion-blur signature (capture a calibration sequence early).
- **Noise + JPEG**: film-grain noise (Akyon) + JPEG quantization at q ∈ {70, 85, 95} matches CMOS sensor noise + radio-link compression artifacts.
- **No mosaic / no random-crop**: those break the trajectory-conditioned pose assumption. The pose is the label-bearing variable.

### Curation (Stage 6)
- **Akyon's selective-mixing finding**: 500 well-chosen synthetic frames beat 50k random ones. We respect this: target ~50k frames *after* per-bin coverage filtering, not 500k unfiltered.
- **Train/val split by engagement**, not by frame, to prevent leakage.

---

## What this pipeline does *not* include (and why)

- **Birds / payload distractors** (SimD3 contribution): Phase 1's failure mode of interest is bbox jitter / drops on the target, not false-positive discrimination against birds. The Phase 2 RL controller is robust to false-positive blips because it consumes a tracker output, not raw detections. Defer SimD3's bird-distractor extension to Phase 1.5 if FP rate becomes the binding evaluation metric on real footage.
- **Adverse weather** (SimD3): the Phase 2 deployment scenario is indoor or fair-weather outdoor. Don't pay the rendering cost for fog/snow.
- **Texture-only photorealism**: Wisniewski shows pure-photoreal underperforms. We deliberately keep Sim2Air's atypical-texture mix.

---

## Lineage table

| Pipeline component | Inherited from | What we change |
|---|---|---|
| Blender + Cycles + texture randomization | [Sim2Air](sim2air_barisic_2022.txt) | Pose sampling — random → trajectory-conditioned |
| HDRI library + lighting realism | [SynDroneVision](syndronevision_lenhard_2024.txt) | Cycles+HDRI instead of UE5+Lumen |
| Structured domain randomization (realistic context) | [Wisniewski](pure_synthetic_wisniewski_2024.txt) | Adopt SDR over pure DR, but keep Sim2Air texture pool |
| Post-capture blur sweep | [SynDroneVision](syndronevision_lenhard_2024.txt) | Tune kernel to match real camera motion-blur signature |
| Film-grain noise + matched blur as primary features | [Akyon](track_boosting_akyon_2021.txt) | Apply per-frame, not as global post-process |
| Selective synthetic mixing | [Akyon](track_boosting_akyon_2021.txt) | Sub-sample by trajectory-coverage bins |
| Visible-vertex bbox projection | [Symeonidis](symeonidis_2021.txt) | Apply to drone meshes; emit per-frame occlusion fraction |
| Tight bbox at sub-pixel scale | [SimD3](simd3_2026.txt) | Emit `target_size_px` for curriculum stratification |
| **Trajectory-conditioned pose distribution** | **(novel)** | Camera lives on simulated interceptor closing on simulated evader |

---

## Per-paper detail

For supporting evidence, parameter values, and quantitative results behind each component above, see:

- [Sim2Air (Barisic 2022)](sim2air_barisic_2022.txt) — texture randomization recipe, S-UAV-T dataset spec, mAP gains
- [SynDroneVision (Lenhard 2024)](syndronevision_lenhard_2024.txt) — UE5/Lumen pipeline, post-capture blur, OOD generalization
- [Wisniewski (2024)](pure_synthetic_wisniewski_2024.txt) — SDR vs DR comparison, pure-synthetic Faster-RCNN result
- [SimD3 (2026)](simd3_2026.txt) — AirSim+UE5+weather, multi-camera rig, payload/bird distractors
- [Akyon (2021)](track_boosting_akyon_2021.txt) — selective synthetic mixing, blur as dominant feature
- [Symeonidis (2021)](symeonidis_2021.txt) — projected-vertex annotation method
