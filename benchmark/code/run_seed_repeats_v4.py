"""Training-seed repeats of the final refits for the key full-species v4 systems.

For one system and repeat k (1 or 2), the final model is refitted on all
development data exactly as in the original run (same configuration, verified
against the saved protocol and CV summary; same refit epochs) except for the
training seed, which is the original final seed plus 1000*k. The refit is
evaluated once on the locked test set. Outputs go to
paper_experiments/full_species_v4_taxonomy/seed_repeats/<system>/repeat_<k>/ and never touch the
original runs. Checkpoints are deleted after prediction; the seed reproduces them.

Usage: python run_seed_repeats_v4.py --system "<name>" --repeat 1
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent
PE = ROOT / "paper_experiments"
OUT = PE / "full_species_v4_taxonomy" / "seed_repeats"
GRID = {  # name: (runner wrapper module, grid output root, selected regime)
    "Swin-Base": ("run_full_species_grouped", PE / "full_species_v4_taxonomy", "fine_tune"),
    "ConvNeXt-V2-Base": ("run_full_species_grouped_added", PE / "full_species_v4_taxonomy" / "grid_added", "fine_tune"),
    "DINOv2-Base/14": ("run_full_species_grouped_added", PE / "full_species_v4_taxonomy" / "grid_added", "fine_tune"),
    "BioCLIP-2": ("run_full_species_grouped_bioclip2", PE / "full_species_v4_taxonomy" / "grid_bioclip2", "frozen"),
}
STYLE = {"ConvNeXt-V2-B 384, cross-entropy": (384, "ce"), "ConvNeXt-V2-B 384, seesaw": (384, "seesaw"),
         "ConvNeXt-V2-B 224, cross-entropy": (224, "ce")}
STRONG = {"BioCLIP-2 (strong recipe)": "BioCLIP-2", "DINOv2-Base/14 (strong recipe)": "DINOv2-Base/14"}
CHECK_ONLY = False
STYLE_DEFAULTS = dict(epochs=15, warmup_epochs=1, lr=2e-5, warmup_lr=2e-7, head_lr_ratio=10.0, weight_decay=2e-5,
                      batch_size=16, accumulate=2, drop_path=0.1, cutmix_prob=0.5, workers=6, seed=20260925, folds=5)


def slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def save(out, system, repeat, seed, test, probs, labels, base, extra):
    base.prediction_table(test, probs, labels).to_csv(out / "test_predictions.csv", index=False)
    truth = test.label_id.to_numpy(int)
    base.atomic_json(out / "final_metrics.json", {"system": system, "repeat": repeat, "seed": seed,
                                                  "point": base.scores(truth, probs), **extra})
    for checkpoint in (out / "final_model").glob("*.pt"):
        checkpoint.unlink()
    print(json.dumps({"system": system, "repeat": repeat, "seed": seed, "accuracy": base.scores(truth, probs)["accuracy"]}))


def grid_repeat(system, repeat):
    wrapper, grid_root, regime = GRID[system]
    module = importlib.import_module(wrapper)
    base = module.base
    import run_full_species_grouped as full
    sys.argv = [wrapper, "--mode", "final", "--dataset-root", str(full.DEFAULT_DATASET),
                "--output-root", str(grid_root), "--model", system, "--regime", regime, "--augmentation", "low",
                "--threshold-unit", "images", "--min-images-per-label", "1", "--folds", "5", "--epochs", "50",
                "--early-stopping-patience", "10", "--early-stopping-min-epochs", "10", "--batch-size", "16",
                "--workers", "4", "--seed", "20260925", "--bootstrap-replicates", "2000"]
    config = base.parse_args()
    selected, labels, manifest_hash, _ = base.read_manifest(config)
    signature = base.protocol_signature(config, manifest_hash, labels)
    original = base.configuration_dir(config)
    cv = json.loads((original / "cv_summary.json").read_text(encoding="utf-8"))
    assert cv["protocol"] == signature, "configuration differs from the original run"
    assert (original / "final_metrics.json").exists()
    if CHECK_ONLY:
        print("protocol verified:", system)
        return
    seed = config.seed + 1000 + 1000 * repeat
    out = OUT / slug(system) / f"repeat_{repeat}"
    if (out / "final_metrics.json").exists():
        print("already done:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = selected.loc[selected["split"].isin(("train", "val"))].reset_index(drop=True)
    test = selected.loc[selected["split"] == "test"].reset_index(drop=True)
    trained = base.fit(config, dev, None, len(labels), seed, out / "final_model", final_epochs=cv["selected_final_epochs"])
    truth, probs = base.predict(trained["model"], base.loader(test, trained["evaluation_transform"], config, False, seed),
                                trained["device"])
    assert np.array_equal(truth, test.label_id.to_numpy(int))
    save(out, system, repeat, seed, test, probs, labels, base,
         {"refit_epochs": cv["selected_final_epochs"], "training_seconds": trained["elapsed_seconds"]})


def style_repeat(system, repeat):
    import run_full_species_grouped as full
    import run_original_six_grouped as base
    import run_snakeclef_style_convnext as style
    size, loss = STYLE[system]
    style.configure(size)
    args = SimpleNamespace(**STYLE_DEFAULTS, loss=loss, image_size=size)
    frame, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=5))
    original = PE / "full_species_v4_taxonomy" / "snakeclef_style" / f"convnextv2_base_{size}_{loss}"
    saved = json.loads((original / "protocol.json").read_text(encoding="utf-8"))
    assert style.protocol(args, manifest_hash, labels) == saved, "configuration differs from the original run"
    assert (original / "final_metrics.json").exists()
    if CHECK_ONLY:
        print("protocol verified:", system)
        return
    seed = args.seed + 1000 + 1000 * repeat
    out = OUT / slug(system) / f"repeat_{repeat}"
    if (out / "final_metrics.json").exists():
        print("already done:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = frame.loc[frame.split.isin(("train", "val"))].reset_index(drop=True)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    model, seconds = style.train_model(dev, len(labels), args, seed, out / "final_model")
    truth, probs = style.predict_tta(model, test, args, seed)
    assert np.array_equal(truth, test.label_id.to_numpy(int))
    save(out, system, repeat, seed, test, probs, labels, base, {"refit_epochs": args.epochs, "training_seconds": seconds})


def strong_repeat(system, repeat):
    import run_full_species_grouped as full
    import run_original_six_grouped as base
    import run_strong_recipe_v4 as strong
    model_name = STRONG[system]
    strong.style.configure(strong.RECIPE["image_size"])
    args = SimpleNamespace(**strong.RECIPE)
    frame, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=5))
    original = strong.OUTPUT / model_name.replace("/", "-")
    saved = json.loads((original / "protocol.json").read_text(encoding="utf-8"))
    _, _, _, has_drop_path = strong.build(model_name, len(labels))
    assert strong.protocol(model_name, manifest_hash, labels, has_drop_path) == saved, "configuration differs from the original run"
    assert (original / "final_metrics.json").exists()
    if CHECK_ONLY:
        print("protocol verified:", system)
        return
    seed = args.seed + 1000 + 1000 * repeat
    out = OUT / slug(system) / f"repeat_{repeat}"
    if (out / "final_metrics.json").exists():
        print("already done:", out)
        return
    out.mkdir(parents=True, exist_ok=True)
    dev = frame.loc[frame.split.isin(("train", "val"))].reset_index(drop=True)
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    model, seconds = strong.train_model(model_name, dev, len(labels), args, seed, out / "final_model")
    truth, probs = strong.style.predict_tta(model, test, args, seed)
    assert np.array_equal(truth, test.label_id.to_numpy(int))
    save(out, system, repeat, seed, test, probs, labels, base, {"refit_epochs": args.epochs, "training_seconds": seconds})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", required=True, choices=sorted([*GRID, *STYLE, *STRONG]))
    parser.add_argument("--repeat", type=int, choices=(1, 2), required=True)
    parser.add_argument("--check-only", action="store_true", help="verify the configuration, do not train")
    a = parser.parse_args()
    global CHECK_ONLY
    CHECK_ONLY = a.check_only
    runner = grid_repeat if a.system in GRID else strong_repeat if a.system in STRONG else style_repeat
    runner(a.system, a.repeat)


if __name__ == "__main__":
    main()
