# Drone Detection Worklog

## 2026-05-23

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
- Set up `detection/` pipeline and moved image-detection assets out of `papers/image_detect/`.
- Generated YOLO MVP dataset:
  - train: 5710
  - val: 2600
  - test: 2200
- Blocker: CUDA unavailable due NVIDIA driver/PyTorch mismatch.
