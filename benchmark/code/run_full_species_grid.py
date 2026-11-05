"""Grid controller for the full-species study on the fresh 80/20 split (dataset v4).

Reuses run_original_six_grid.py unchanged (selection rule, restart and adoption
logic) and points it at run_full_species_grouped.py with
--min-images-per-label 1. Author decision (7 October 2026): two recipes per
architecture, frozen + low augmentation and fine-tuning + low augmentation,
giving 12 configurations.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import run_original_six_grid as grid

ROOT = Path(__file__).resolve().parent
# v4 split with the taxonomy revision of 20 Oct 2026 (scripts/build_taxonomy_revision_v4.py).
grid.DATASET = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
grid.OUTPUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
grid.PLAN = grid.OUTPUT / "full_species_plan.csv"
grid.RUNNER = ROOT / "run_full_species_grouped.py"
grid.RECIPES = (("fine_tune", "low"), ("frozen", "low"))
_original_command = grid.command
# Scheduling order only (results do not depend on it): fine-tuning first, longest jobs first.
_LONGEST_FIRST = ("Swin-Base", "ResNet-50", "DeiT-Base/16", "EfficientNet-B0", "ViT-Base/16", "MobileNetV3-Small")
# Runner wrapper recorded in each configuration's protocol; checked on resume.
RUNNER_WRAPPER = "run_full_species_grouped.py"


def config_dir(model, regime, augmentation):
    return grid.OUTPUT / "full_species" / model.replace("/", "-") / regime / augmentation


def command(mode, model, regime, augmentation):
    parts = _original_command(mode, model, regime, augmentation)
    parts[parts.index("--min-images-per-label") + 1] = "1"
    return parts


def load_plan():
    with grid.PLAN.open("r", encoding="utf-8-sig", newline="") as handle:
        configs = [(r["model"], r["regime"], r["augmentation"]) for r in csv.DictReader(handle)]
    expected = {(m, r, a) for m in grid.MODELS for r, a in grid.RECIPES}
    if len(configs) != len(expected) or set(configs) != expected:
        raise ValueError(f"The saved plan differs from the {len(expected)} approved configurations")
    return sorted(configs, key=lambda c: (c[1] != "fine_tune", _LONGEST_FIRST.index(c[0])))


def check_summary(path, model, regime, augmentation, manifest_hash):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    expected = {
        "manifest_sha256": manifest_hash, "model": model, "regime": regime, "augmentation": augmentation,
        "min_images_per_label": 1, "folds": grid.FOLDS, "epochs": grid.EPOCHS,
        "early_stopping_patience": grid.EARLY_STOPPING_PATIENCE,
        "early_stopping_min_epochs": grid.EARLY_STOPPING_MIN_EPOCHS,
        "batch_size": grid.BATCH_SIZE, "seed": grid.SEED, "runner_wrapper": RUNNER_WRAPPER,
    }
    bad = {key: (protocol.get(key), value) for key, value in expected.items() if protocol.get(key) != value}
    if bad:
        raise ValueError(f"Existing output has mismatched protocol: {path}: {bad}")
    if len(data.get("folds", [])) != grid.FOLDS:
        raise ValueError(f"Existing CV output does not contain {grid.FOLDS} folds: {path}")
    for required in ("cross_validation_metrics.csv", "development_oof_predictions.csv"):
        if not (path.parent / required).is_file():
            raise FileNotFoundError(f"Completed CV summary is missing {required}: {path}")
    return data


def write_plan():
    grid.OUTPUT.mkdir(parents=True, exist_ok=True)
    if not grid.PLAN.exists():
        rows = ["model,regime,augmentation,threshold_unit,minimum_count_per_label"]
        rows += [f"{m},{r},{a},images,1" for m in grid.MODELS for r, a in grid.RECIPES]
        grid.PLAN.write_text("\n".join(rows) + "\n", encoding="utf-8")


grid.config_dir = config_dir
grid.command = command
grid.check_summary = check_summary
grid.load_plan = load_plan

if __name__ == "__main__":
    import sys
    write_plan()
    sys.argv += ["--dataset-root", str(grid.DATASET), "--output-root", str(grid.OUTPUT), "--plan-path", str(grid.PLAN)]
    grid.main()
