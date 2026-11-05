"""Full-species, post-aware variant of run_original_six_grouped.py.

Training, evaluation, checkpointing, resume checks, and bootstrap code are the
original runner's, unchanged. Only three things differ, all fixed by the v3
dataset build (scripts/build_full_species_dataset.py):
  * every species in the manifest is a class (no image-count threshold), so a
    species may be absent from a CV validation fold or from the test set;
  * groups are post-aware components (manifest column `component`);
  * the five CV folds are read from the manifest's `cv_fold` column.
The original runner file is imported, not edited, so its pinned hash holds.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import run_original_six_grouped as base

ROOT = Path(__file__).resolve().parent
# v4 split with the taxonomy revision of 20 Oct 2026 (scripts/build_taxonomy_revision_v4.py).
DEFAULT_DATASET = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
REQUIRED = {"relative_path", "split", "cv_fold", "class_label", "row_id", "component", "content_sha256"}
_original_signature = base.protocol_signature


def read_manifest(config):
    root = config.dataset_root.resolve()
    manifest = root / "manifest.csv"
    frame = pd.read_csv(manifest, dtype=str, keep_default_na=False)
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")
    if not frame["split"].isin(("development", "test")).all():
        raise ValueError("Unexpected split value")
    if frame[list(REQUIRED)].eq("").any().any():
        raise ValueError("A required manifest field is empty")
    if frame["relative_path"].duplicated().any():
        raise ValueError("Repeated relative paths in manifest")
    expected_folds = {str(i) for i in range(1, config.folds + 1)}
    if set(frame.loc[frame.split == "development", "cv_fold"]) != expected_folds:
        raise ValueError("Development folds do not match --folds")
    if not frame.loc[frame.split == "test", "cv_fold"].eq("test").all():
        raise ValueError("A test record carries a CV fold")
    # The original runner pools train+val as development; keep its vocabulary.
    frame["split"] = frame["split"].replace({"development": "train"})
    frame["leakage_group_id"] = frame["component"]
    base.audit_disjoint(frame, "split")
    paths = [(root / relative).resolve() for relative in frame["relative_path"]]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Manifest path escapes the dataset root")
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"{len(absent)} listed media files are missing; first: {absent[0]}")
    frame["absolute_path"] = [str(path) for path in paths]
    labels = sorted(frame["class_label"].unique().tolist())
    frame["label_id"] = pd.Categorical(frame["class_label"], categories=labels).codes
    frame = frame.sort_values(["split", "class_label", "leakage_group_id", "relative_path"],
                              kind="stable").reset_index(drop=True)
    test = frame[frame.split == "test"]
    audit = {
        "manifest_sha256": base.sha256(manifest),
        "label_scope": "all species in the manifest; no threshold",
        "images": len(frame), "labels": len(labels), "groups": frame["leakage_group_id"].nunique(),
        "split_images": frame["split"].value_counts().to_dict(),
        "split_groups": frame.groupby("split")["leakage_group_id"].nunique().to_dict(),
        "labels_with_test_images": int(test["class_label"].nunique()),
        "labels": labels,
    }
    return frame, labels, audit["manifest_sha256"], audit


def make_folds(frame, config):
    dev = frame.loc[frame["split"].isin(("train", "val"))].reset_index(drop=True)
    folds = []
    for number in range(1, config.folds + 1):
        val_idx = np.flatnonzero(dev["cv_fold"].to_numpy() == str(number))
        train_idx = np.flatnonzero(dev["cv_fold"].to_numpy() != str(number))
        tagged = dev.copy()
        tagged["cv_partition"] = "train"
        tagged.loc[val_idx, "cv_partition"] = "val"
        base.audit_disjoint(tagged, "cv_partition")
        folds.append((train_idx, val_idx))
    return folds


def protocol_signature(config, manifest_hash, labels):
    signature = _original_signature(config, manifest_hash, labels)
    signature.update({
        "label_scope": "all species; no threshold (min_images_per_label must be 1)",
        "grouping": "post-aware components of v2 leakage groups, spreadsheet rows, explicit Facebook post keys",
        "fold_source": "manifest cv_fold (StratifiedGroupKFold over components, fixed at dataset build); fresh 80/20 grouped test split (dataset v4)",
        "runner_wrapper": "run_full_species_grouped.py",
        "input_images": "dataset image cache (shorter side 448 px, Lanczos, JPEG q95), then the runner transforms",
    })
    return signature


def configuration_dir(config):
    return config.output_root / "full_species" / config.model.replace("/", "-") / config.regime / config.augmentation


base.read_manifest = read_manifest
base.make_folds = make_folds
base.protocol_signature = protocol_signature
base.configuration_dir = configuration_dir

if __name__ == "__main__":
    import sys
    if "--min-images-per-label" in sys.argv and sys.argv[sys.argv.index("--min-images-per-label") + 1] != "1":
        raise SystemExit("Full-species runs use --min-images-per-label 1")
    if "--dataset-root" not in sys.argv:
        sys.argv += ["--dataset-root", str(DEFAULT_DATASET)]
    base.main()
