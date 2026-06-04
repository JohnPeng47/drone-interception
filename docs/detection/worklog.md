# Drone Detection Worklog

## 2026-06-04 - Traditional CV Blob Detector For IVBS

- Added a deterministic `control_sims/ivbs` image measurement path based on
  traditional CV, not SimEngine truth.
- Implemented dark-blob segmentation, connected-component selection, centroid
  extraction, equivalent-area apparent radius, and confidence scoring.
- Wired the IVBS observer to optionally fuse detector centroid and apparent
  size as a noisy range cue using known target radius and camera focal length.
- Added tests covering blob detection and proving that apparent-size
  measurements can affect IVBS commands while forbidden target truth mutations
  do not.
- Tightened the image-measurement provider boundary to accept only a slot id,
  so providers are not handed `SimInstance`, `SimSnapshot`, or full runner
  state. A supplied CV miss now suppresses fallback to simulator camera
  projection for that tick.
- Treated `None` from a configured image provider as an explicit detector miss
  and made telemetry report provider detection state when a provider is
  configured.
- Extended detector validation to reject non-finite principal points.

## 2026-06-04 - IVBS Apparent-Size Range Cue Planning

- Identified detector-derived apparent target size as the most direct way to
  improve IVBS range observability without using simulator truth.
- Added an IVBS improvement milestone to expose apparent size as an image
  measurement derived from rendered pixels or detector output, not from
  `SimEngine` truth fields such as `range_m` or `target_pos_c`.
- Planned EKF fusion of apparent size using the known target radius and camera
  focal length as a noisy scalar range cue.

## 2026-06-04 - Traditional CV Constraint For IVBS Detection

- Clarified that IVBS image-based detection/range-cue work should not require
  SimEngine changes.
- Updated the IVBS improvement plan to use deterministic traditional CV
  techniques outside SimEngine, such as frame differencing, segmentation,
  contour extraction, connected components, morphology, blob/ellipse fitting,
  and temporal filtering.
- The planned apparent-size measurement must come from pixels or an existing
  image stream, not simulator truth fields.

## 2026-05-23 - Liftoff Drone Variants And Detector Probe

- Added five Liftoff-derived Vortex-frame target drone variants:
  - `vortex_dal_xnova_runcam`
  - `vortex_racekraft_xnova_hs1177`
  - `vortex_gemfan_xnova_actioncam`
  - `vortex_dal_heavy_actioncam`
  - `vortex_racekraft_low_cam`
- Exported local mesh caches under `.runs/liftoff_assets/variants/`.
- Rendered one centered `640x480` image per variant with the native software renderer.
- Ran the downloaded Hugging Face YOLO checkpoint:
  `/home/john/drone-interception/ai/detection/models/doguilmak__Drone-Detection-YOLOv11x/weight/best.pt`
- Detection result: 5/5 variants detected.

| Variant | Detections | Best Confidence |
| --- | ---: | ---: |
| `vortex_dal_xnova_runcam` | 1 | 0.642 |
| `vortex_racekraft_xnova_hs1177` | 1 | 0.587 |
| `vortex_gemfan_xnova_actioncam` | 1 | 0.592 |
| `vortex_dal_heavy_actioncam` | 2 | 0.722 |
| `vortex_racekraft_low_cam` | 1 | 0.637 |

Artifacts:

- Raw rendered images: `.runs/liftoff_variant_yolo_probe/images/`
- Annotated images: `.runs/liftoff_variant_yolo_probe/annotated/`
- CSV: `.runs/liftoff_variant_yolo_probe/detections.csv`
- JSON summary: `.runs/liftoff_variant_yolo_probe/summary.json`
- Throwaway probe script: `docs/analysis/ai/detection/probe_liftoff_variant_yolo.py`

Note: inference completed on CPU. PyTorch printed a CUDA driver warning because the installed CUDA runtime is newer than the system driver.

- Downloaded open-source datasets:
  - UAV-Eagle
  - DUT Anti-UAV
  - Drone-vs-Bird annotations only
  - MAV-VID as bare repo/archive source
- Checked Hugging Face for paper-linked checkpoints. No clear paper reproduction model found. Used `doguilmak/Drone-Detection-YOLOv11x` as a generic baseline.
- HF benchmark results:
  - 80-image sample, `imgsz=1280`, CPU: ~3.0-3.1 sec/image
  - 64 detections across 49/80 images
  - 5-image sample, `imgsz=640`, CPU after move: 8.52 sec total
- Set up `ai/detection/` pipeline and moved image-detection assets out of `papers/image_detect/`.
- Generated YOLO MVP dataset:
  - train: 5710
  - val: 2600
  - test: 2200
- Blocker: CUDA unavailable due NVIDIA driver/PyTorch mismatch.
