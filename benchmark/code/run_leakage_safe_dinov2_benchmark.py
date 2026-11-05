"""Leakage-safe DINOv2 linear-probe benchmark for the revised snake dataset.

The script uses the fixed train/validation/test assignments in manifest.csv.
The confirmatory label set is selected by independent leakage-group count, not
raw image count. Cross-validation is performed only on train+validation data;
the locked test split is evaluated once after the linear-probe configuration is
fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image, ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Config:
    dataset_root: str
    output_dir: str
    model_name: str = "vit_base_patch14_reg4_dinov2.lvd142m"
    min_groups_per_label: int = 100
    image_size: int = 224
    batch_size: int = 64
    workers: int = 4
    folds: int = 5
    seed: int = 20260925
    bootstrap_replicates: int = 2000
    logistic_c: float = 1.0


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default=r"C:\Users\ADMIN\PycharmProjects\snake_research\1 คัดแล้ว\combined_training_top10_leakage_safe",
    )
    parser.add_argument(
        "--output-dir",
        default=r"C:\Users\ADMIN\PycharmProjects\snake_research\paper_experiments\dinov2_linear_common32",
    )
    parser.add_argument("--model-name", default="vit_base_patch14_reg4_dinov2.lvd142m")
    parser.add_argument("--min-groups-per-label", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--logistic-c", type=float, default=1.0)
    return Config(**vars(parser.parse_args()))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_confirmatory_manifest(config: Config) -> tuple[pd.DataFrame, list[str], str]:
    root = Path(config.dataset_root)
    manifest_path = root / "manifest.csv"
    frame = pd.read_csv(manifest_path, low_memory=False)
    required = {
        "relative_path",
        "split",
        "class_label",
        "row_id",
        "leakage_group_id",
        "content_sha256",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if not frame["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unexpected split value in manifest")

    group_counts = frame.groupby("class_label")["leakage_group_id"].nunique()
    labels = sorted(group_counts[group_counts >= config.min_groups_per_label].index.tolist())
    selected = frame[frame["class_label"].isin(labels)].copy()
    selected["absolute_path"] = selected["relative_path"].map(lambda p: str(root / Path(p)))
    selected["label_id"] = pd.Categorical(
        selected["class_label"], categories=labels, ordered=True
    ).codes

    if selected["label_id"].lt(0).any():
        raise AssertionError("Label encoding failed")
    if selected["absolute_path"].map(lambda p: Path(p).suffix.lower() in IMAGE_EXTENSIONS).eq(False).any():
        raise ValueError("Unexpected image extension")

    # Audit the locked partition before touching model code.
    for key in ["row_id", "leakage_group_id", "content_sha256"]:
        split_counts = selected.groupby(key)["split"].nunique()
        if split_counts.max() != 1:
            raise AssertionError(f"Leakage detected: {key} crosses fixed partitions")

    if selected.groupby("leakage_group_id")["class_label"].nunique().max() != 1:
        raise AssertionError("A retained leakage group has conflicting labels")

    for split in ["train", "val", "test"]:
        absent = sorted(set(labels) - set(selected.loc[selected["split"] == split, "class_label"]))
        if absent:
            raise AssertionError(f"Confirmatory labels absent from {split}: {absent}")

    selected = selected.sort_values(
        ["split", "class_label", "leakage_group_id", "relative_path"], kind="stable"
    ).reset_index(drop=True)
    return selected, labels, sha256(manifest_path)


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[str], transform: transforms.Compose):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index]
        try:
            with Image.open(path) as image:
                return self.transform(image.convert("RGB"))
        except Exception as exc:
            raise RuntimeError(f"Could not load image: {path}") from exc


def build_model_and_transform(config: Config, device: torch.device):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_kwargs = {"pretrained": True, "num_classes": 0}
    try:
        model = timm.create_model(
            config.model_name, img_size=config.image_size, **model_kwargs
        )
    except TypeError as exc:
        # Convolutional timm models have no img_size constructor argument.
        if "img_size" not in str(exc):
            raise
        model = timm.create_model(config.model_name, **model_kwargs)
    model.eval().to(device)
    pretrained_cfg = getattr(model, "pretrained_cfg", {})
    mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
    std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
    resize_size = int(round(config.image_size / 0.875))
    transform = transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return model, transform


def extract_or_load_features(
    config: Config,
    frame: pd.DataFrame,
    manifest_hash: str,
    output_dir: Path,
) -> tuple[np.ndarray, dict]:
    feature_path = output_dir / "features.npy"
    metadata_path = output_dir / "feature_metadata.json"
    signature = {
        "manifest_sha256": manifest_hash,
        "relative_paths_sha256": hashlib.sha256(
            "\n".join(frame["relative_path"].astype(str)).encode("utf-8")
        ).hexdigest(),
        "record_count": int(len(frame)),
        "model_name": config.model_name,
        "image_size": config.image_size,
    }
    if feature_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(metadata.get(key) == value for key, value in signature.items()):
            features = np.load(feature_path)
            if features.shape[0] == len(frame):
                print(f"Loaded cached features: {features.shape}", flush=True)
                return features, metadata

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_model_and_transform(config, device)
    dataset = ImagePathDataset(frame["absolute_path"].tolist(), transform)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    feature_blocks: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader, start=1):
            batch = batch.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                embeddings = model(batch)
            feature_blocks.append(embeddings.float().cpu().numpy())
            if batch_index % 25 == 0 or batch_index == len(loader):
                done = min(batch_index * config.batch_size, len(dataset))
                print(f"Features: {done:,}/{len(dataset):,}", flush=True)
    elapsed = time.perf_counter() - started
    features = np.concatenate(feature_blocks, axis=0).astype(np.float32, copy=False)
    if len(features) != len(frame):
        raise AssertionError("Feature count does not match manifest")
    np.save(feature_path, features)

    metadata = {
        **signature,
        "feature_shape": list(features.shape),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "extraction_seconds": elapsed,
        "images_per_second": len(frame) / elapsed,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "batch_size": config.batch_size,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved features: {features.shape} in {elapsed:.1f} s", flush=True)
    return features, metadata


def make_classifier(config: Config):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.logistic_c,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            tol=1e-5,
            random_state=config.seed,
        ),
    )


def metric_bundle(y_true: np.ndarray, probabilities: np.ndarray, label_ids: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=label_ids, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, predictions, labels=label_ids)),
        "top3_accuracy": float(top_k_accuracy_score(y_true, probabilities, k=3, labels=label_ids)),
        "log_loss": float(log_loss(y_true, probabilities, labels=label_ids)),
        "n_images": int(len(y_true)),
    }


def group_bootstrap_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    label_ids: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    unique_groups = np.unique(groups)
    group_to_indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        indices = np.concatenate([group_to_indices[group] for group in sampled])
        metrics = metric_bundle(y_true[indices], probabilities[indices], label_ids)
        for key, value in metrics.items():
            if key != "n_images" and np.isfinite(value):
                values.setdefault(key, []).append(float(value))
    return {
        key: {
            "lower_95": float(np.percentile(samples, 2.5)),
            "upper_95": float(np.percentile(samples, 97.5)),
            "bootstrap_replicates": len(samples),
        }
        for key, samples in values.items()
    }


def run_benchmark(config: Config) -> None:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, labels, manifest_hash = load_confirmatory_manifest(config)
    label_ids = np.arange(len(labels), dtype=np.int64)
    print(
        f"Confirmatory scope: {len(labels)} labels, {len(frame):,} images, "
        f"{frame['leakage_group_id'].nunique():,} groups",
        flush=True,
    )
    frame.drop(columns=["absolute_path"]).to_csv(output_dir / "selected_manifest.csv", index=False)
    (output_dir / "class_names.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    features, feature_metadata = extract_or_load_features(
        config, frame, manifest_hash, output_dir
    )
    y = frame["label_id"].to_numpy(dtype=np.int64)
    groups = frame["leakage_group_id"].astype(str).to_numpy()
    is_test = frame["split"].eq("test").to_numpy()
    dev_indices = np.flatnonzero(~is_test)
    test_indices = np.flatnonzero(is_test)

    splitter = StratifiedGroupKFold(
        n_splits=config.folds, shuffle=True, random_state=config.seed
    )
    fold_rows = []
    oof_probabilities = np.full((len(dev_indices), len(labels)), np.nan, dtype=np.float64)
    for fold, (fit_local, val_local) in enumerate(
        splitter.split(features[dev_indices], y[dev_indices], groups[dev_indices]), start=1
    ):
        classifier = make_classifier(config)
        fit_started = time.perf_counter()
        classifier.fit(features[dev_indices[fit_local]], y[dev_indices[fit_local]])
        fit_seconds = time.perf_counter() - fit_started
        probabilities = classifier.predict_proba(features[dev_indices[val_local]])
        oof_probabilities[val_local] = probabilities
        metrics = metric_bundle(y[dev_indices[val_local]], probabilities, label_ids)
        metrics.update(
            {
                "fold": fold,
                "fit_seconds": fit_seconds,
                "train_images": int(len(fit_local)),
                "validation_images": int(len(val_local)),
                "train_groups": int(len(np.unique(groups[dev_indices[fit_local]]))),
                "validation_groups": int(len(np.unique(groups[dev_indices[val_local]]))),
            }
        )
        fold_rows.append(metrics)
        print(
            f"Fold {fold}/{config.folds}: accuracy={metrics['accuracy']:.4f}, "
            f"macro-F1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    if np.isnan(oof_probabilities).any():
        raise AssertionError("Cross-validation did not produce one prediction per development image")

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(output_dir / "cross_validation_metrics.csv", index=False)
    oof_predictions = frame.iloc[dev_indices][
        ["relative_path", "split", "class_label", "row_id", "leakage_group_id"]
    ].copy()
    oof_predictions["true_label_id"] = y[dev_indices]
    oof_predictions["predicted_label_id"] = oof_probabilities.argmax(axis=1)
    oof_predictions["predicted_label"] = [labels[index] for index in oof_predictions["predicted_label_id"]]
    oof_predictions["confidence"] = oof_probabilities.max(axis=1)
    oof_predictions.to_csv(output_dir / "development_oof_predictions.csv", index=False)

    final_classifier = make_classifier(config)
    final_fit_started = time.perf_counter()
    final_classifier.fit(features[dev_indices], y[dev_indices])
    final_fit_seconds = time.perf_counter() - final_fit_started
    inference_started = time.perf_counter()
    test_probabilities = final_classifier.predict_proba(features[test_indices])
    test_inference_seconds = time.perf_counter() - inference_started
    test_metrics = metric_bundle(y[test_indices], test_probabilities, label_ids)
    intervals = group_bootstrap_intervals(
        y[test_indices],
        test_probabilities,
        groups[test_indices],
        label_ids,
        config.bootstrap_replicates,
        config.seed + 1,
    )

    test_predictions = frame.iloc[test_indices][
        ["relative_path", "class_label", "row_id", "leakage_group_id", "source_type"]
    ].copy()
    test_predictions["true_label_id"] = y[test_indices]
    test_predictions["predicted_label_id"] = test_probabilities.argmax(axis=1)
    test_predictions["predicted_label"] = [labels[index] for index in test_predictions["predicted_label_id"]]
    test_predictions["confidence"] = test_probabilities.max(axis=1)
    test_predictions["correct"] = (
        test_predictions["true_label_id"] == test_predictions["predicted_label_id"]
    )
    test_predictions.to_csv(output_dir / "test_predictions.csv", index=False)

    precision, recall, f1, support = precision_recall_fscore_support(
        y[test_indices],
        test_predictions["predicted_label_id"].to_numpy(),
        labels=label_ids,
        zero_division=0,
    )
    pd.DataFrame(
        {
            "label_id": label_ids,
            "class_label": labels,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).to_csv(output_dir / "test_per_class_metrics.csv", index=False)
    pd.DataFrame(
        confusion_matrix(
            y[test_indices], test_predictions["predicted_label_id"].to_numpy(), labels=label_ids
        ),
        index=labels,
        columns=labels,
    ).to_csv(output_dir / "test_confusion_matrix.csv")

    summary = {
        "status": "complete",
        "manifest_sha256": manifest_hash,
        "confirmatory_scope": {
            "minimum_independent_groups_per_label": config.min_groups_per_label,
            "labels": len(labels),
            "images": int(len(frame)),
            "groups": int(frame["leakage_group_id"].nunique()),
            "development_images": int(len(dev_indices)),
            "test_images": int(len(test_indices)),
            "development_groups": int(len(np.unique(groups[dev_indices]))),
            "test_groups": int(len(np.unique(groups[test_indices]))),
        },
        "feature_extraction": feature_metadata,
        "cross_validation": {
            "folds": config.folds,
            "metrics_mean": {
                column: float(fold_table[column].mean())
                for column in [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "cohen_kappa",
                    "top3_accuracy",
                    "log_loss",
                ]
            },
            "metrics_standard_deviation": {
                column: float(fold_table[column].std(ddof=1))
                for column in [
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "cohen_kappa",
                    "top3_accuracy",
                    "log_loss",
                ]
            },
            "pooled_oof_metrics": metric_bundle(y[dev_indices], oof_probabilities, label_ids),
        },
        "locked_test": {
            "metrics": test_metrics,
            "group_bootstrap_95_ci": intervals,
            "groups": int(len(np.unique(groups[test_indices]))),
            "bootstrap_replicates": config.bootstrap_replicates,
        },
        "linear_probe_efficiency": {
            "fit_seconds": final_fit_seconds,
            "test_inference_seconds": test_inference_seconds,
            "test_images_per_second": len(test_indices) / test_inference_seconds,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["locked_test"], indent=2), flush=True)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    run_benchmark(parse_args())
