# Thai Snake Classification

This repository contains the code for training snake species classification models using PyTorch, supporting both CNNs (MobileNetV3, EfficientNet, ResNet) and Vision Transformers (ViT, Swin).

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Dataset Structure:

   ```
   dataset_root/
       Species_A/
           img1.jpg
           ...
       Species_B/
           ...
   ```

## Usage

The `train_unified.py` script handles the entire pipeline: loading data, filtering by threshold, augmenting, and training.

### Example Commands

**Train MobileNetV3-Small (CNN) with Threshold 100:**

```bash
python train_unified.py --model MobileNetV3-Small --threshold 100 --aug_intensity medium --data_dir /path/to/dataset
```

**Train ViT-Base (Transformer) with Threshold 500:**

```bash
python train_unified.py --model ViT-Base/16 --threshold 500 --aug_intensity high --data_dir /path/to/dataset
```

### Arguments

- `--model`: Model architecture. Choices: `MobileNetV3-Small`, `EfficientNet-B0`, `ResNet-50`, `ViT-Base/16`, `Swin-Base`.
- `--threshold`: Minimum number of images per species to include class (100, 200, 300, 400, 500).
- `--aug_intensity`: Augmentation intensity (`none`, `low`, `medium`, `high`).
- `--data_dir`: Path to dataset root.
- `--output_dir`: Directory for outputs (checkpoints, logs).
- `--epochs`: Number of epochs (default: 50).
- `--batch_size`: Batch size (default: 16).

## Experimental Setup Details

- **Hardware**: optimized for RTX 4090 (AMP enabled).
- **Preprocessing**: Resize to 224x224, Normalized (ImageNet stats).
- **Splits**:
    1. Filter classes < Threshold.
    2. Split 80/20 (Train+Val / Test).
    3. Split Train+Val 80/20 (Train / Val).
- **Optimizer**: AdamW (wd=0.05).
- **Scheduler**: Cosine Annealing (50 epochs).
- **Freezing**: Backbone frozen for first 10 epochs.
