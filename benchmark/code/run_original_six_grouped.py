"""Re-run the original six-model design on the fixed, leakage-safe dataset.

The manifest is authoritative: no images are re-split. By default the primary
scope retains the original >=100 images-per-class rule. An independent-group
threshold is available for sensitivity analyses. `validate` checks the
held-out partition and all cross-validation folds without loading a model.
`cv` fits one architecture/regime/augmentation configuration on five grouped
folds of train+val. `final` requires its completed CV record, refits on all
development observations for the CV-selected epoch count, and tests once.

This is a new protocol. The earlier class-folder training code used image-level
stratification, and its historical scores must not be copied into results from
this manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
MODELS = {
    "MobileNetV3-Small": "mobilenetv3_small_100",
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet-50": "resnet50",
    "ViT-Base/16": "vit_base_patch16_224",
    "DeiT-Base/16": "deit_base_distilled_patch16_224",
    "Swin-Base": "swin_base_patch4_window7_224",
}
TRANSFORMERS = {"ViT-Base/16", "DeiT-Base/16", "Swin-Base"}
REGIMES = ("scratch", "frozen", "fine_tune")
AUGMENTATIONS = ("none", "low", "medium", "high")
PRIMARY_RECIPES = (
    ("frozen", "none"),
    ("fine_tune", "none"),
    ("fine_tune", "medium"),
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
REQUIRED_COLUMNS = {
    "relative_path", "split", "class_label", "row_id",
    "leakage_group_id", "content_sha256",
}


@dataclass(frozen=True)
class Config:
    mode: str
    dataset_root: Path
    output_root: Path
    model: str
    regime: str
    augmentation: str
    threshold_unit: str
    min_images_per_label: int
    min_groups_per_label: int
    folds: int
    epochs: int
    warmup_epochs: int
    early_stopping_patience: int
    early_stopping_min_epochs: int
    batch_size: int
    workers: int
    seed: int
    class_weighting: str
    bootstrap_replicates: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("validate", "plan", "cv", "final"), default="validate")
    parser.add_argument(
        "--dataset-root", type=Path,
        default=ROOT / "1 คัดแล้ว" / "combined_training_top10_leakage_safe",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper_experiments" / "original_six_grouped")
    parser.add_argument("--model", choices=tuple(MODELS), default="Swin-Base")
    parser.add_argument("--regime", choices=REGIMES, default="frozen")
    parser.add_argument("--augmentation", choices=AUGMENTATIONS, default="none")
    parser.add_argument("--threshold-unit", choices=("images", "groups"), default="images")
    parser.add_argument("--min-images-per-label", type=int, default=100)
    parser.add_argument("--min-groups-per-label", type=int, default=100)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="CV epochs without validation-accuracy improvement before stopping; 0 disables")
    parser.add_argument("--early-stopping-min-epochs", type=int, default=10,
                        help="Minimum CV epochs, also required after the fine-tuning warm-up")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--class-weighting", choices=("none", "balanced"), default="none")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    args = parser.parse_args()
    for key in ("min_images_per_label", "min_groups_per_label", "folds", "epochs", "batch_size", "bootstrap_replicates"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if args.warmup_epochs < 0:
        parser.error("--warmup-epochs cannot be negative")
    if args.early_stopping_patience < 0 or args.early_stopping_min_epochs < 1:
        parser.error("Early-stopping patience must be nonnegative and minimum epochs positive")
    return Config(**vars(args))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def audit_disjoint(frame: pd.DataFrame, partition_col: str) -> None:
    for key in ("row_id", "leakage_group_id", "content_sha256"):
        if frame.groupby(key, dropna=False)[partition_col].nunique().max() != 1:
            raise AssertionError(f"{key} crosses {partition_col} partitions")


def read_manifest(config: Config) -> tuple[pd.DataFrame, list[str], str, dict]:
    root = config.dataset_root.resolve()
    manifest = root / "manifest.csv"
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if frame.empty or not frame["split"].isin(("train", "val", "test")).all():
        raise ValueError("Manifest is empty or has an unexpected split value")
    if frame[list(REQUIRED_COLUMNS)].eq("").any().any():
        raise ValueError("A required manifest field is empty")
    if frame["relative_path"].duplicated().any():
        raise ValueError("Repeated relative paths in manifest")
    if frame.groupby("leakage_group_id")["class_label"].nunique().max() != 1:
        raise AssertionError("A leakage group contains multiple source labels")
    audit_disjoint(frame, "split")

    # Check that every listed image lives inside this dataset root.
    paths = [(root / relative).resolve() for relative in frame["relative_path"]]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Manifest path escapes the dataset root")
    missing_files = [str(path) for path in paths if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} listed media files are missing; first: {missing_files[0]}")
    frame["absolute_path"] = [str(path) for path in paths]

    if config.threshold_unit == "images":
        counts = frame.groupby("class_label").size()
        threshold = config.min_images_per_label
    else:
        counts = frame.groupby("class_label")["leakage_group_id"].nunique()
        threshold = config.min_groups_per_label
    labels = sorted(counts[counts >= threshold].index.tolist())
    if not labels:
        raise ValueError(f"No labels meet the {config.threshold_unit} threshold")
    selected = frame.loc[frame["class_label"].isin(labels)].copy()
    selected["label_id"] = pd.Categorical(selected["class_label"], categories=labels).codes
    for split in ("train", "val", "test"):
        missing_labels = set(labels) - set(selected.loc[selected["split"] == split, "class_label"])
        if missing_labels:
            raise ValueError(f"Labels missing from {split}: {sorted(missing_labels)}")
    selected = selected.sort_values(
        ["split", "class_label", "leakage_group_id", "relative_path"], kind="stable"
    ).reset_index(drop=True)
    audit = {
        "manifest_sha256": sha256(manifest),
        "all_images": len(frame),
        "all_labels": frame["class_label"].nunique(),
        "all_groups": frame["leakage_group_id"].nunique(),
        "selected_images": len(selected),
        "selected_labels": len(labels),
        "selected_groups": selected["leakage_group_id"].nunique(),
        "selected_split_images": selected["split"].value_counts().to_dict(),
        "selected_split_groups": selected.groupby("split")["leakage_group_id"].nunique().to_dict(),
        "threshold_unit": config.threshold_unit,
        "minimum_count_per_label": threshold,
        "labels": labels,
    }
    return selected, labels, audit["manifest_sha256"], audit


def make_folds(frame: pd.DataFrame, config: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    dev = frame.loc[frame["split"].isin(("train", "val"))].reset_index(drop=True)
    group_labels = dev.groupby("leakage_group_id")["label_id"].first()
    if group_labels.value_counts().min() < config.folds:
        raise ValueError("Some labels have fewer independent development groups than CV folds")
    splitter = StratifiedGroupKFold(n_splits=config.folds, shuffle=True, random_state=config.seed)
    folds = []
    for train_idx, val_idx in splitter.split(dev, dev["label_id"], dev["leakage_group_id"]):
        tagged = dev.copy()
        tagged["cv_partition"] = "train"
        tagged.loc[val_idx, "cv_partition"] = "val"
        audit_disjoint(tagged, "cv_partition")
        if set(dev.iloc[train_idx]["label_id"]) != set(dev["label_id"]):
            raise AssertionError("A CV training fold lacks a label")
        if set(dev.iloc[val_idx]["label_id"]) != set(dev["label_id"]):
            raise AssertionError("A CV validation fold lacks a label")
        folds.append((train_idx, val_idx))
    return folds


def configuration_dir(config: Config) -> Path:
    model_name = config.model.replace("/", "-")
    threshold = config.min_images_per_label if config.threshold_unit == "images" else config.min_groups_per_label
    return config.output_root / f"{config.threshold_unit}{threshold}" / model_name / config.regime / config.augmentation


def protocol_signature(config: Config, manifest_hash: str, labels: list[str]) -> dict:
    return {
        "manifest_sha256": manifest_hash,
        "model": config.model,
        "timm_model": MODELS[config.model],
        "regime": config.regime,
        "augmentation": config.augmentation,
        "threshold_unit": config.threshold_unit,
        "min_images_per_label": config.min_images_per_label,
        "min_groups_per_label": config.min_groups_per_label,
        "labels": labels,
        "folds": config.folds,
        "epochs": config.epochs,
        "warmup_epochs": config.warmup_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "early_stopping_min_epochs": config.early_stopping_min_epochs,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "class_weighting": config.class_weighting,
        "image_size": 224,
        "preprocessing": "RGB, direct 224x224 resize, ImageNet mean/std",
        "optimizer": "AdamW, weight_decay=0.05, cosine learning rate, gradient clipping=1.0",
        "head_to_backbone_learning_rate_ratio": 10,
        "selection_metric": "validation accuracy",
        "checkpoint_rule": "best validation-accuracy checkpoint in CV; last epoch of full-development refit in final",
        "final_epoch_rule": "median CV best epoch, rounded to nearest integer",
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def worker_seed(_worker_id: int) -> None:
    worker_seed_value = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed_value)
    random.seed(worker_seed_value)


def image_transforms(level: str):
    resize = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
    normalize = (transforms.ToTensor(), transforms.Normalize(MEAN, STD))
    evaluation = transforms.Compose((resize, *normalize))
    if level == "none":
        return evaluation, evaluation
    strength = {"low": 0.5, "medium": 1.0, "high": 1.5}[level]
    # Match the old settings: 10-degree rotation, 10% shift/zoom,
    # 0.1-radian shear at medium intensity; multiply all magnitudes together.
    degrees = 10 * strength
    translation = 0.1 * strength
    zoom = 0.1 * strength
    shear_degrees = math.degrees(0.1 * strength)
    training = transforms.Compose((
        resize,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Pad(48, padding_mode="edge"),
        transforms.RandomAffine(
            degrees=(-degrees, degrees),
            translate=(translation, translation),
            scale=(1 - zoom, 1 + zoom),
            shear=(-shear_degrees, shear_degrees),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop((224, 224)),
        *normalize,
    ))
    return training, evaluation


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths = frame["absolute_path"].tolist()
        self.labels = frame["label_id"].astype(int).tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as raw:
            image = self.transform(raw.convert("RGB"))
        return image, self.labels[index]


def loader(frame: pd.DataFrame, transform, config: Config, train: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        ManifestDataset(frame, transform), batch_size=config.batch_size,
        shuffle=train, num_workers=config.workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0, generator=generator, worker_init_fn=worker_seed,
    )


def build_model(config: Config, n_classes: int, device: torch.device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    pretrained = config.regime != "scratch"
    model = timm.create_model(MODELS[config.model], pretrained=pretrained, num_classes=n_classes)
    head = []
    backbone = []
    for name, parameter in model.named_parameters():
        if name.split(".")[0] in {"head", "head_dist", "fc", "classifier"}:
            head.append(parameter)
        else:
            backbone.append(parameter)
    if not head or not backbone:
        raise RuntimeError(f"Cannot identify head and backbone for {config.model}")
    family_backbone_lr = 2e-5 if config.model in TRANSFORMERS else 1e-4
    optimizer = torch.optim.AdamW(
        ({"params": backbone, "lr": family_backbone_lr},
         {"params": head, "lr": family_backbone_lr * 10}),
        weight_decay=0.05,
    )
    return model.to(device), backbone, head, optimizer


def logits_only(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    # timm's distilled DeiT can return two heads during training.
    if isinstance(output, (tuple, list)):
        return torch.stack(tuple(output), dim=0).mean(dim=0)
    return output


def scores(truth: np.ndarray, probabilities: np.ndarray) -> dict:
    prediction = probabilities.argmax(axis=1)
    top3 = np.argpartition(probabilities, -min(3, probabilities.shape[1]), axis=1)[:, -min(3, probabilities.shape[1]):]
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(truth, prediction)),
        "top3_accuracy": float(np.mean((top3 == truth[:, None]).any(axis=1))),
    }


def predict(model: nn.Module, batches: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_true = []
    all_probs = []
    with torch.inference_mode():
        for images, labels in batches:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = logits_only(model(images))
            all_true.append(labels.numpy())
            all_probs.append(logits.float().softmax(dim=1).cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_probs)


def fit(
    config: Config, train_frame: pd.DataFrame, val_frame: pd.DataFrame | None,
    n_classes: int, seed: int, output_dir: Path, final_epochs: int | None = None,
) -> dict:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backbone, head, optimizer = build_model(config, n_classes, device)
    train_transform, eval_transform = image_transforms(config.augmentation)
    train_loader = loader(train_frame, train_transform, config, train=True, seed=seed)
    val_loader = loader(val_frame, eval_transform, config, train=False, seed=seed) if val_frame is not None else None
    weights = None
    if config.class_weighting == "balanced":
        counts = np.bincount(train_frame["label_id"].to_numpy(dtype=int), minlength=n_classes)
        weights = torch.as_tensor(len(train_frame) / (n_classes * counts), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    epochs = final_epochs if final_epochs is not None else config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        # Give the unfrozen backbone a full patience window after warm-up.
        if config.regime == "fine_tune" and epoch == config.warmup_epochs + 1:
            epochs_without_improvement = 0
        train_backbone = config.regime == "scratch" or (
            config.regime == "fine_tune" and epoch > config.warmup_epochs
        )
        for parameter in backbone:
            parameter.requires_grad_(train_backbone)
        for parameter in head:
            parameter.requires_grad_(True)
        # A frozen backbone must also stop updating batch-normalization state
        # and must not apply training-time stochastic operations.
        if train_backbone:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = logits_only(model(images))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": total_loss / seen, "backbone_trainable": train_backbone}
        if val_loader is not None:
            y_true, probs = predict(model, val_loader, device)
            row.update({f"val_{key}": value for key, value in scores(y_true, probs).items()})
            current = row["val_accuracy"]
            if current > best_score:
                best_score = current
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(model.state_dict(), output_dir / "best.pt")
            else:
                epochs_without_improvement += 1
        history.append(row)
        print(f"{config.model} {config.regime} {config.augmentation}: epoch {epoch}/{epochs}, loss={row['train_loss']:.4f}", flush=True)
        minimum_epochs = config.early_stopping_min_epochs + (
            config.warmup_epochs if config.regime == "fine_tune" else 0
        )
        if (val_loader is not None and config.early_stopping_patience > 0
                and epoch >= minimum_epochs
                and epochs_without_improvement >= config.early_stopping_patience):
            stopped_early = True
            print(f"Early stopping at epoch {epoch}; best validation accuracy at epoch {best_epoch}", flush=True)
            break
    if val_loader is None:
        best_epoch = epochs
        torch.save(model.state_dict(), output_dir / "final.pt")
    else:
        model.load_state_dict(torch.load(output_dir / "best.pt", map_location=device, weights_only=True))
    atomic_json(output_dir / "history.json", history)
    return {
        "model": model, "device": device, "best_epoch": best_epoch,
        "epochs_trained": len(history), "stopped_early": stopped_early,
        "elapsed_seconds": time.perf_counter() - start,
        "parameters": sum(p.numel() for p in model.parameters()),
        "evaluation_transform": eval_transform,
    }


def prediction_table(frame: pd.DataFrame, probs: np.ndarray, labels: list[str]) -> pd.DataFrame:
    columns = ["relative_path", "split", "class_label", "label_id", "row_id", "leakage_group_id", "content_sha256"]
    table = frame[columns].reset_index(drop=True).copy()
    table["predicted_label"] = [labels[i] for i in probs.argmax(axis=1)]
    for idx in range(probs.shape[1]):
        table[f"probability_{idx:03d}"] = probs[:, idx]
    return table


def fold_fingerprint(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> str:
    """Tie a fold artifact to its exact observations and assignment order."""
    digest = hashlib.sha256()
    fields = ("relative_path", "class_label", "label_id", "row_id", "leakage_group_id", "content_sha256")
    for partition, frame in (("train", train_frame), ("val", val_frame)):
        digest.update(partition.encode("utf-8"))
        for row in frame[list(fields)].itertuples(index=False, name=None):
            digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def completed_fold(
    fold_dir: Path, number: int, signature: dict, fingerprint: str,
    val_frame: pd.DataFrame, labels: list[str],
) -> tuple[dict, pd.DataFrame] | None:
    """Return a verified fold result, or None if its commit artifacts are absent.

    A partial fold is retrained. A complete but mismatched fold is rejected so
    predictions from another manifest/protocol cannot enter this CV summary.
    """
    result_path = fold_dir / "fold_result.json"
    predictions_path = fold_dir / "oof_predictions.csv"
    if not result_path.is_file() or not predictions_path.is_file():
        return None
    record = json.loads(result_path.read_text(encoding="utf-8"))
    if record.get("protocol") != signature or record.get("fold") != number or record.get("fingerprint") != fingerprint:
        raise ValueError(f"Completed fold {number} does not match the current manifest, settings, or assignment")
    result = record.get("result")
    if not isinstance(result, dict) or result.get("fold") != number:
        raise ValueError(f"Completed fold {number} has invalid metrics")
    if not 1 <= result.get("best_epoch", 0) <= signature["epochs"]:
        raise ValueError(f"Completed fold {number} has an invalid selected epoch")
    if not result["best_epoch"] <= result.get("epochs_trained", 0) <= signature["epochs"]:
        raise ValueError(f"Completed fold {number} has an invalid training duration")
    if not isinstance(result.get("stopped_early"), bool):
        raise ValueError(f"Completed fold {number} lacks a valid early-stopping flag")
    probability_columns = [f"probability_{idx:03d}" for idx in range(len(labels))]
    expected_columns = [
        "relative_path", "split", "class_label", "label_id", "row_id",
        "leakage_group_id", "content_sha256", "predicted_label",
        *probability_columns, "fold",
    ]
    part = pd.read_csv(predictions_path, keep_default_na=False)
    if list(part.columns) != expected_columns or len(part) != len(val_frame):
        raise ValueError(f"Completed fold {number} has incomplete prediction rows or columns")
    for key in ("relative_path", "split", "class_label", "row_id", "leakage_group_id", "content_sha256"):
        if not np.array_equal(part[key].astype(str).to_numpy(), val_frame[key].astype(str).to_numpy()):
            raise ValueError(f"Completed fold {number} prediction {key} differs from manifest")
    if not np.array_equal(part["label_id"].to_numpy(dtype=int), val_frame["label_id"].to_numpy(dtype=int)):
        raise ValueError(f"Completed fold {number} prediction label IDs differ from manifest")
    if not np.all(part["fold"].to_numpy(dtype=int) == number):
        raise ValueError(f"Completed fold {number} prediction fold IDs differ")
    probabilities = part[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError(f"Completed fold {number} has invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-5):
        raise ValueError(f"Completed fold {number} probabilities do not sum to one")
    predicted = np.asarray([labels[idx] for idx in probabilities.argmax(axis=1)])
    if not np.array_equal(predicted, part["predicted_label"].astype(str).to_numpy()):
        raise ValueError(f"Completed fold {number} predicted labels do not match probabilities")
    recomputed = scores(val_frame["label_id"].to_numpy(dtype=int), probabilities)
    for key, value in recomputed.items():
        if not np.isclose(value, result.get(key, float("nan")), rtol=0, atol=1e-8):
            raise ValueError(f"Completed fold {number} {key} differs from predictions")
    return result, part


def prune_verified_cv_checkpoint(
    fold_dir: Path, verified: tuple[dict, pd.DataFrame] | None,
) -> None:
    """Keep resumable CV evidence while discarding its redundant model weights.

    The selected final model is refitted from the CV-selected epoch count, so
    CV fold weights are not used after predictions and metrics are verified.
    """
    if verified is None:
        raise ValueError("Cannot discard a CV checkpoint before its fold artifacts are verified")
    checkpoint = fold_dir / "best.pt"
    if checkpoint.is_file():
        checkpoint.unlink()
        print(f"Removed verified fold checkpoint: {checkpoint}", flush=True)


def grouped_intervals(frame: pd.DataFrame, probabilities: np.ndarray, replicates: int, seed: int) -> dict:
    group_to_indices = list(frame.groupby("leakage_group_id", sort=True).indices.values())
    rng = np.random.default_rng(seed)
    truth = frame["label_id"].to_numpy(dtype=int)
    keys = tuple(scores(truth, probabilities))
    draws = {key: [] for key in keys}
    for _ in range(replicates):
        groups = rng.integers(0, len(group_to_indices), size=len(group_to_indices))
        indices = np.concatenate([group_to_indices[index] for index in groups])
        result = scores(truth[indices], probabilities[indices])
        for key in keys:
            draws[key].append(result[key])
    return {key: [float(np.percentile(draws[key], 2.5)), float(np.percentile(draws[key], 97.5))] for key in keys}


def main() -> None:
    config = parse_args()
    selected, labels, manifest_hash, audit = read_manifest(config)
    folds = make_folds(selected, config)
    signature = protocol_signature(config, manifest_hash, labels)
    output_dir = configuration_dir(config)
    fold_counts = []
    dev = selected.loc[selected["split"].isin(("train", "val"))].reset_index(drop=True)
    for number, (train_idx, val_idx) in enumerate(folds, 1):
        fold_counts.append({
            "fold": number, "train_images": len(train_idx), "val_images": len(val_idx),
            "train_groups": dev.iloc[train_idx]["leakage_group_id"].nunique(),
            "val_groups": dev.iloc[val_idx]["leakage_group_id"].nunique(),
        })
    if config.mode == "plan":
        threshold = config.min_images_per_label if config.threshold_unit == "images" else config.min_groups_per_label
        plan = pd.DataFrame([
            {"model": model, "regime": regime, "augmentation": augmentation,
             "threshold_unit": config.threshold_unit, "minimum_count_per_label": threshold}
            for model in MODELS for regime, augmentation in PRIMARY_RECIPES
        ])
        config.output_root.mkdir(parents=True, exist_ok=True)
        plan.to_csv(config.output_root / f"{config.threshold_unit}{threshold}_plan.csv", index=False)
        print(json.dumps({"configurations": len(plan), "audit": audit}, indent=2))
        return
    if config.mode == "validate":
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / "dry_run_audit.json", {"audit": audit, "folds": fold_counts, "protocol": signature})
        print(json.dumps({"audit": audit, "folds": fold_counts}, indent=2))
        return

    if config.mode == "cv":
        output_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = output_dir / "protocol.json"
        if protocol_path.exists():
            previous = json.loads(protocol_path.read_text(encoding="utf-8"))
            if previous != signature:
                raise ValueError(f"Existing CV protocol differs from the requested run: {protocol_path}")
        else:
            atomic_json(protocol_path, signature)
        summary_path = output_dir / "cv_summary.json"
        fold_results = []
        prediction_parts = []
        for number, (train_idx, val_idx) in enumerate(folds, 1):
            train_frame = dev.iloc[train_idx].reset_index(drop=True)
            val_frame = dev.iloc[val_idx].reset_index(drop=True)
            fold_dir = output_dir / f"fold_{number}"
            fingerprint = fold_fingerprint(train_frame, val_frame)
            completed = completed_fold(fold_dir, number, signature, fingerprint, val_frame, labels)
            if completed is not None:
                result, part = completed
                prune_verified_cv_checkpoint(fold_dir, completed)
                print(f"Fold {number}/{config.folds}: verified completed artifacts; skipping training", flush=True)
                fold_results.append(result)
                prediction_parts.append(part)
                continue
            if summary_path.exists():
                raise ValueError(f"CV summary exists but fold {number} lacks complete artifacts: {fold_dir}")
            trained = fit(config, train_frame, val_frame, len(labels), config.seed + number,
                          fold_dir)
            eval_loader = loader(val_frame, trained["evaluation_transform"], config, False, config.seed + number)
            truth, probs = predict(trained["model"], eval_loader, trained["device"])
            if not np.array_equal(truth, val_frame["label_id"].to_numpy(dtype=int)):
                raise AssertionError("Validation prediction order changed")
            metrics = scores(truth, probs)
            result = {"fold": number, "best_epoch": trained["best_epoch"],
                      "epochs_trained": trained["epochs_trained"],
                      "stopped_early": trained["stopped_early"],
                      "seconds": trained["elapsed_seconds"], "parameters": trained["parameters"], **metrics}
            part = prediction_table(val_frame, probs, labels)
            part["fold"] = number
            # The JSON record is the commit marker. Write OOF rows first, then
            # save metrics; only a verified pair is eligible for resume.
            atomic_csv(fold_dir / "oof_predictions.csv", part)
            atomic_json(fold_dir / "fold_result.json", {
                "protocol": signature, "fold": number, "fingerprint": fingerprint, "result": result,
            })
            verified = completed_fold(fold_dir, number, signature, fingerprint, val_frame, labels)
            if verified is None:
                raise AssertionError(f"Fold {number} did not commit successfully")
            fold_results.append(verified[0])
            prediction_parts.append(verified[1])
            del trained
            prune_verified_cv_checkpoint(fold_dir, verified)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = {
            "protocol": signature, "folds": fold_results,
            "mean": {key: float(np.mean([row[key] for row in fold_results])) for key in (
                "accuracy", "balanced_accuracy", "macro_f1", "kappa", "top3_accuracy"
            )},
            "sd": {key: float(np.std([row[key] for row in fold_results], ddof=1)) for key in (
                "accuracy", "balanced_accuracy", "macro_f1", "kappa", "top3_accuracy"
            )},
            "selected_final_epochs": max(1, int(np.rint(np.median([row["best_epoch"] for row in fold_results])))),
        }
        if summary_path.exists():
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            if previous != summary:
                raise ValueError(f"Existing CV summary differs from verified fold results: {summary_path}")
        else:
            atomic_csv(output_dir / "development_oof_predictions.csv", pd.concat(prediction_parts, ignore_index=True))
            atomic_csv(output_dir / "cross_validation_metrics.csv", pd.DataFrame(fold_results))
            atomic_json(summary_path, summary)
        print(json.dumps(summary["mean"], indent=2))
        return

    # Final evaluation is allowed only after this exact configuration's CV.
    cv_path = output_dir / "cv_summary.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"Run CV first: {cv_path}")
    test_marker = output_dir / "locked_test_started.json"
    if (output_dir / "final_metrics.json").exists() or test_marker.exists():
        raise FileExistsError(f"Locked test evaluation was already started in {output_dir}")
    cv = json.loads(cv_path.read_text(encoding="utf-8"))
    if cv["protocol"] != signature:
        raise ValueError("CV settings, labels, or manifest hash differ from this final run")
    final_epochs = cv["selected_final_epochs"]
    trained = fit(config, dev, None, len(labels), config.seed + 1000,
                  output_dir / "final_model", final_epochs=final_epochs)
    test = selected.loc[selected["split"] == "test"].reset_index(drop=True)
    atomic_json(test_marker, {"protocol": signature, "test_images": len(test)})
    test_loader = loader(test, trained["evaluation_transform"], config, False, config.seed + 1000)
    truth, probs = predict(trained["model"], test_loader, trained["device"])
    if not np.array_equal(truth, test["label_id"].to_numpy(dtype=int)):
        raise AssertionError("Test prediction order changed")
    prediction_table(test, probs, labels).to_csv(output_dir / "test_predictions.csv", index=False)
    result = {
        "protocol": signature, "test_images": len(test),
        "test_groups": test["leakage_group_id"].nunique(),
        "selected_epochs": final_epochs, "parameters": trained["parameters"],
        "training_seconds": trained["elapsed_seconds"],
        "point": scores(truth, probs),
        "group_bootstrap_95ci": grouped_intervals(test, probs, config.bootstrap_replicates, config.seed + 2000),
    }
    atomic_json(output_dir / "final_metrics.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
