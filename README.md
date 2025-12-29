# Thai Snake Classification: A Unified Benchmarking Pipeline

This repository provides a unified PyTorch-based pipeline for classifying Thai snake species. It supports a variety of modern architectures, including Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), with a standardized experimental protocol designed for academic reproducibility.

## Features

- **Unified Training Loop**: Standardized training script (`train_unified.py`) for both CNN and Transformer families.
- **Modern Architectures**:
  - **CNNs**: MobileNetV3-Small, EfficientNet-B0, ResNet-50.
  - **Transformers**: ViT-Base/16, Swin-Base, DeiT-Base (distilled).
- **Reproducible Methodology**: Fixed seeds, standardized data splits, and precise augmentation profiles.
- **Hardware Optimized**: Integrated support for Automatic Mixed Precision (AMP) and optimized data loading for NVIDIA GPUs (e.g., RTX 4090).

---

## Experimental Protocol

### 1. Data Management

The pipeline implements a rigorous data selection and splitting strategy:

- **Species Filtering**: Only species with at least $T$ images are retained. Supported thresholds $T \in \{100, 200, 300, 400, 500\}$.
- **Splitting Strategy**:
  - Initial **80:20** split into **Train+Val** and **Test** sets.
  - Secondary **80:20** split of the **Train+Val** portion into **Training** and **Validation** sets.
- **Preprocessing**: All images are resized to $224 \times 224$ pixels and normalized using standard ImageNet mean and variance.

### 2. Augmentation Strategy

On-the-fly geometric augmentations follow four intensity profiles (**none**, **low**, **medium**, **high**), composed of:

- Rotation, Horizontal Flip, Shear, Width/Height Shifts, and Zoom.
- Precise Keras-style emulation using a Pad-Affine-Crop sequence.

### 3. Optimization

A unified optimization protocol is applied to all models:

- **Optimizer**: AdamW with a weight decay of 0.05.
- **Schedule**: Cosine learning-rate schedule over 50 epochs.
- **Gradient Clipping**: Global-norm clipping at 1.0.
- **Warm-up**: 10-epoch freeze-unfreeze schedule. The backbone remains frozen for the first 10 epochs, with only the classification head updated.
- **Two-Group Learning Rates**:
  - **CNNs**: Backbone ($1 \times 10^{-4}$), Head ($1 \times 10^{-3}$).
  - **Transformers**: Backbone ($2 \times 10^{-5}$), Head ($2 \times 10^{-4}$).

---

## Getting Started

### Installation

```bash
conda activate your_env
pip install -r requirements.txt
```

### Usage

Use `train_unified.py` to launch experiments:

```bash
python train_unified.py \
    --model MobileNetV3-Small \
    --threshold 100 \
    --aug_intensity medium \
    --data_dir /path/to/dataset
```

#### Arguments

- `--model`: Architecture selection (`MobileNetV3-Small`, `EfficientNet-B0`, `ResNet-50`, `ViT-Base/16`, `Swin-Base`, `DeiT-Base`).
- `--threshold`: Minimum image count per species (`100`, `200`, `300`, `400`, `500`).
- `--aug_intensity`: Augmentation level (`none`, `low`, `medium`, `high`).
- `--data_dir`: Root directory of the dataset.

---

## Inference

### Local Inference

The `inference_local.py` script supports both standard PyTorch checkpoints (`.pth`) and Hugging Face Transformers model folders.

**Using a standard checkpoint:**

```bash
python inference_local.py \
    --image /path/to/test_image.jpg \
    --model_path experiments/EfficientNet-B0_medium_T100/checkpoint_last.pth \
    --class_indices experiments/EfficientNet-B0_medium_T100/class_indices.json \
    --model_name EfficientNet-B0
```

**Using a Transformers model (e.g., Swin):**

```bash
python inference_local.py \
    --image /path/to/test_image.jpg \
    --model_path transformer_experiment_results_swin_exact/Swin-Base_augnone_fineT_thres500
```

*(Note: For Transformers models, class mapping and architecture are inferred from `config.json` inside the directory).*

### Model Upload to Hugging Face Hub

Use `huggingface_upload.py` to upload your experiment results to the Hugging Face Hub:

```bash
python huggingface_upload.py \
    --folder experiments/EfficientNet-B0_medium_T100 \
    --repo_id kritaphatson/thai_snake_image_classifier \
    --token YOUR_HF_TOKEN
```

### Hugging Face hosting

To host the model on Hugging Face Spaces:

1. Create a new Space with the **Gradio** SDK.
2. Upload `app.py`, `models.py`, `data_manager.py`, and `requirements.txt`.
3. Upload your trained model checkpoint (e.g., `checkpoint_last.pth`) and `class_indices.json`.
4. Update the paths in `app.py` to point to your uploaded files.
5. The Space will automatically build and launch the Gradio interface.

For local web UI testing:

```bash
python app.py
```

---

## Repository Structure

- `train_unified.py`: Main entry point for training and evaluation.
- `data_manager.py`: Dataset loading, filtering, splitting, and augmentation logic.
- `models.py`: Unified model factory and LR configuration.
- `requirements.txt`: Environment dependencies.
- `inference_local.py`: Command-line interface for local testing.
- `huggingface_upload.py`: Script to upload models to Hugging Face Hub.
- `app.py`: Gradio web interface for Hugging Face Spaces.

---

## Citation

If you use this codebase in your research, please cite the original project.

```bibtex
@misc{thai_snake_classification_2025,
  author = {Snake Research Team},
  title = {Thai Snake Classification: A Unified Benchmarking Pipeline},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/user/thai_snake_classification}}
}
```
