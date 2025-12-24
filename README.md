# FPQC-Net

FPQC-Net is a lightweight multi-task CNN for fetal ultrasound plane classification and plane-specific standard-plane adequacy assessment.

> This repository is a release intended to support reproducibility of the proposed framework.
> **No clinical data is provided** due to institutional policies and privacy regulations.
> The code is designed to run end-to-end on **any input image**.

## Key Features
- Multi-task architecture:
  - 6-way anatomical plane classification
  - 5 plane-specific binary quality heads (AC/HC/FL/4CH/CL)
- Task-aware masking for quality heads (only the relevant head contributes to loss)

## Training
```bash
python scripts/train_dummy.py --epochs 2 --batch_size 16
```

## Batch Inference
```bash
python scripts/batch_infer.py --input_dir ./images --output_csv ./preds.csv --weights ./optional_ckpt.pth
```

## Installation
```bash
pip install -r requirements.txt
pip install -e .
