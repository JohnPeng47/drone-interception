# Worklog

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

# 2026-05-28
- Created structured Generator workflow
- Now Generators only write samples to disk, decoupled from run