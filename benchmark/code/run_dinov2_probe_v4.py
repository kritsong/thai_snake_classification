"""Frozen DINOv2 linear probe on the full-species v4 split (same folds and test as the main grid).

Settings repeat the earlier verified supplement without tuning: timm
vit_base_patch14_reg4_dinov2.lvd142m (frozen, 768-d CLS features), RGB, bicubic
shorter side 256, center crop 224, pretrained mean/std, float16 autocast;
StandardScaler fitted on the training part only, then balanced multinomial
logistic regression (C=1, L2, lbfgs, max_iter=2000, tol=1e-5). Features are
extracted once from the v4 image cache. A species absent from a CV training
fold receives probability 0 in that fold. After the taxonomy revision of 20 Oct 2026,
which changed labels only, the features extracted under the v4 labels are reused
after checking every file hash.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

import run_full_species_grouped as full
import run_original_six_grouped as base
from run_leakage_safe_dinov2_benchmark import ImagePathDataset, build_model_and_transform

ROOT = Path(__file__).resolve().parent
MODEL = "vit_base_patch14_reg4_dinov2.lvd142m"
OUTPUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "dinov2_probe"
V4_RESULTS = ROOT / "paper_experiments" / "full_species_fresh_v4"  # results under the v4 labels
SEED, FOLDS = 20260925, 5


def reuse_features(previous: Path, cache: Path, order: Path, frame: pd.DataFrame) -> bool:
    """Save features extracted under the v4 labels in this manifest's order, if they exist.

    The image files and their hashes are unchanged by the taxonomy revision; only the sort
    order of relabelled records differs.
    """
    if not (previous / cache.name).exists():
        return False
    saved = pd.read_csv(previous / order.name, keep_default_na=False).reset_index().set_index("relative_path")
    assert len(saved) == len(frame) and set(saved.index) == set(frame.relative_path), "Image set changed"
    assert (saved.cached_sha256.loc[frame.relative_path].to_numpy() == frame.cached_sha256.to_numpy()).all(), "Image changed"
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, np.load(previous / cache.name)[saved["index"].loc[frame.relative_path].to_numpy()])
    frame[["relative_path", "cached_sha256"]].to_csv(order, index=False)
    return True


def features(frame: pd.DataFrame) -> np.ndarray:
    cache = OUTPUT / "features.npy"
    order = OUTPUT / "feature_order.csv"
    if cache.exists() or reuse_features(V4_RESULTS / OUTPUT.name, cache, order, frame):
        saved = pd.read_csv(order, keep_default_na=False)
        assert saved.relative_path.tolist() == frame.relative_path.tolist(), "Feature order changed"
        return np.load(cache)
    device = torch.device("cuda")
    model, transform = build_model_and_transform(SimpleNamespace(model_name=MODEL, image_size=224), device)
    loader = DataLoader(ImagePathDataset(frame.absolute_path.tolist(), transform), batch_size=64, num_workers=6)
    chunks = []
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                chunks.append(model(images.to(device)).float().cpu().numpy())
    x = np.concatenate(chunks)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    frame[["relative_path", "cached_sha256"]].to_csv(order, index=False)
    return x


def fit_predict(x_train, y_train, x_eval, n_classes):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, class_weight="balanced", solver="lbfgs", max_iter=2000, tol=1e-5, random_state=SEED))
    with threadpool_limits(8):
        model.fit(x_train, y_train)
    probs = np.zeros((len(x_eval), n_classes))
    probs[:, model[-1].classes_] = model.predict_proba(x_eval)
    return probs, int(model[-1].n_iter_.max())


def main():
    frame, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=FOLDS))
    start = time.perf_counter()
    x = features(frame)
    extract_seconds = time.perf_counter() - start
    dev_mask = frame.split.isin(("train", "val")).to_numpy()
    dev, x_dev = frame[dev_mask].reset_index(drop=True), x[dev_mask]
    y_dev = dev.label_id.to_numpy(int)
    records, parts = [], []
    for number, (train_idx, val_idx) in enumerate(full.make_folds(frame, SimpleNamespace(folds=FOLDS)), start=1):
        probs, iterations = fit_predict(x_dev[train_idx], y_dev[train_idx], x_dev[val_idx], len(labels))
        val = dev.iloc[val_idx].reset_index(drop=True)
        part = base.prediction_table(val, probs, labels); part["fold"] = number
        parts.append(part)
        records.append({"fold": number, "lbfgs_iterations": iterations, **base.scores(y_dev[val_idx], probs)})
        print(json.dumps(records[-1]), flush=True)
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "top3_accuracy")
    test = frame[~dev_mask].reset_index(drop=True)
    probs, iterations = fit_predict(x_dev, y_dev, x[~dev_mask], len(labels))
    truth = test.label_id.to_numpy(int)
    base.prediction_table(test, probs, labels).to_csv(OUTPUT / "test_predictions.csv", index=False)
    pd.concat(parts, ignore_index=True).to_csv(OUTPUT / "development_oof_predictions.csv", index=False)
    summary = {
        "manifest_sha256": manifest_hash, "model": MODEL, "labels": labels,
        "input": "v4 cache; RGB; bicubic shorter side 256; center crop 224; pretrained mean/std; float16 autocast",
        "classifier": "StandardScaler (training part only) + balanced multinomial logistic regression C=1, lbfgs, max_iter=2000, tol=1e-5",
        "scikit_learn": sklearn.__version__, "feature_seconds": extract_seconds,
        "cv": {"folds": records, "mean": {k: float(np.mean([r[k] for r in records])) for k in keys},
               "sd": {k: float(np.std([r[k] for r in records], ddof=1)) for k in keys}},
        "final": {"lbfgs_iterations": iterations, "test_images": len(test), "test_groups": int(test.leakage_group_id.nunique()),
                  "point": base.scores(truth, probs),
                  "group_bootstrap_95ci": base.grouped_intervals(test, probs, 2000, SEED + 2000)},
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"cv_mean": summary["cv"]["mean"], "test": summary["final"]["point"]}, indent=2))


if __name__ == "__main__":
    main()
