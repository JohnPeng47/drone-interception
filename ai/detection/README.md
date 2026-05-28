# Drone Detection Pipeline

This folder contains the local setup for drone detector training and evaluation.

## Layout

- `scripts/prepare_yolo_real_mvp.py`: builds a YOLO-format training dataset from the downloaded real-drone datasets.
- `scripts/prepare_validation_sample.py`: creates a small validation image set for quick inference checks.
- `scripts/download_hf_model.py`: downloads a Hugging Face YOLO checkpoint.
- `scripts/run_inference.py`: runs a YOLO checkpoint on a folder of images and writes predictions.
- `data/`: generated datasets and validation samples.
- `models/`: downloaded checkpoints.
- `runs/`: inference/training outputs.

## Quick Baseline

Create validation sample:

```bash
python ai/detection/scripts/prepare_validation_sample.py
```

Download the community HF drone detector:

```bash
python ai/detection/scripts/download_hf_model.py
```

Run inference:

```bash
python ai/detection/scripts/run_inference.py
```

## Training Dataset Prep

Build the real-drone MVP dataset:

```bash
python ai/detection/scripts/prepare_yolo_real_mvp.py
```

This creates:

```text
ai/detection/data/yolo_real_mvp/
  images/{train,val,test}
  labels/{train,val,test}
  data.yaml
```

The dataset has one class: `drone`.

Smoke training command:

```bash
yolo detect train \
  model=yolo11n.pt \
  data=ai/detection/data/yolo_real_mvp/data.yaml \
  imgsz=640 \
  epochs=5 \
  batch=8 \
  device=cpu \
  project=ai/detection/runs/train \
  name=yolo11n_real_mvp_smoke
```

Use a CUDA-enabled PyTorch install before attempting larger training runs.
