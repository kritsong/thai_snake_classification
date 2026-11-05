"""External SnakeCLEF Thailand evaluation of the full-species v4 systems.

Scores the frozen scope (paper_experiments/external_snakeclef2024/full_species_v4_taxonomy_scope)
with each final system and all 139 outputs; no fitting on external data, no
class masking. Image probabilities are averaged within each observation; metrics
are observation-level, with 2,000 observation resamples shared by all systems.
Systems: the nine CV-selected grid models (final.pt; the original six plus
ConvNeXt-V2-Base, DINOv2-Base/14 and BioCLIP 2), the DINOv2 and BioCLIP 2 probes (each
final logistic regression is refitted deterministically on the cached
development features and checked against the saved test predictions), BioCLIP 2
zero-shot (saved class text embeddings), and the SnakeCLEF-style ConvNeXt runs
when their final checkpoints exist (seesaw and CE at 384 px, CE at 224 px; flip TTA). BioCLIP 2 was trained on
TreeOfLife-200M, which draws on GBIF/iNaturalist, the same sources as SnakeCLEF,
so its external rows are marked as possibly overlapping with its training data.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

import run_full_species_grouped as full
import run_full_species_grouped_bioclip2  # noqa: F401  registers ConvNeXt-V2-Base, DINOv2-Base/14, BioCLIP-2
import run_original_six_grouped as base

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
ADDED = RUNS / "grid_added"
BIOCLIP_GRID = RUNS / "grid_bioclip2"
OVERLAP_NOTE = "possible training overlap: BioCLIP 2 training data (TreeOfLife-200M) includes GBIF/iNaturalist images"
SCOPE = ROOT / "paper_experiments" / "external_snakeclef2024" / "full_species_v4_taxonomy_scope"
OUT = RUNS / "external_snakeclef"
SEED, REPS = 20260925, 2000


class Images(Dataset):
    def __init__(self, records, transform):
        self.records, self.transform = records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        with Image.open(ROOT / self.records[i]["relative_extracted_path"]) as raw:
            return self.transform(raw.convert("RGB"))


def predict(model, records, transform, tta=False):
    model.eval().cuda()
    out = []
    with torch.inference_mode():
        for images in DataLoader(Images(records, transform), batch_size=32, num_workers=4):
            images = images.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                probs = base.logits_only(model(images)).float().softmax(1)
                if tta:
                    probs = (probs + base.logits_only(model(images.flip(3))).float().softmax(1)) / 2
            out.append(probs.cpu().numpy())
    return np.concatenate(out)


def grid_systems(labels, records):
    for grid in (RUNS, ADDED, BIOCLIP_GRID):
        selection = grid / "selected_final_configurations.json"
        if selection.exists():
            yield from grid_models(grid, json.loads(selection.read_text(encoding="utf-8"))["choices"], labels, records)


def grid_models(grid, selection, labels, records):
    for model_name, c in selection.items():
        folder = grid / "full_species" / model_name.replace("/", "-") / c["regime"] / c["augmentation"]
        if not (folder / "final_metrics.json").exists():
            continue
        model = timm.create_model(base.MODELS[model_name], pretrained=False, num_classes=len(labels))
        model.load_state_dict(torch.load(folder / "final_model" / "final.pt", map_location="cpu", weights_only=True))
        _, transform = base.image_transforms(c["augmentation"])
        identity = base.sha256(folder / "final_model" / "final.pt")
        if model_name == "BioCLIP-2":
            identity += "; " + OVERLAP_NOTE
        yield f"{model_name} ({c['regime']}+{c['augmentation']})", predict(model, records, transform), identity


def dinov2_system(labels, records):
    import run_dinov2_probe_v4 as dino
    from run_leakage_safe_dinov2_benchmark import build_model_and_transform
    frame, _, _, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=5))
    x = dino.features(frame)
    dev = frame.split.isin(("train", "val")).to_numpy()
    model, transform = build_model_and_transform(SimpleNamespace(model_name=dino.MODEL, image_size=224), torch.device("cuda"))
    feats = []
    with torch.inference_mode():
        for images in DataLoader(Images(records, transform), batch_size=32, num_workers=4):
            with torch.autocast("cuda", dtype=torch.float16):
                feats.append(model(images.cuda()).float().cpu().numpy())
    probs, _ = dino.fit_predict(x[dev], frame.label_id.to_numpy(int)[dev], np.concatenate(feats), len(labels))
    # The refit must reproduce the saved locked-test probabilities exactly.
    saved = pd.read_csv(dino.OUTPUT / "test_predictions.csv")
    test_probs, _ = dino.fit_predict(x[dev], frame.label_id.to_numpy(int)[dev], x[~dev], len(labels))
    cols = [c for c in saved.columns if c.startswith("probability_")]
    assert np.allclose(saved[cols].to_numpy(float), test_probs, atol=1e-6), "DINOv2 refit differs from saved test predictions"
    return "DINOv2 probe (frozen)", probs, "refit verified against saved test predictions"


def bioclip2_systems(labels, records):
    import run_bioclip2_v4 as bioclip
    if not (bioclip.OUTPUT / "summary.json").exists():
        return
    frame, _, _, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=5))
    model, preprocess, _ = bioclip.load_model(torch.device("cuda"))
    x = bioclip.image_features(frame, model, preprocess, torch.device("cuda"))
    dev = frame.split.isin(("train", "val")).to_numpy()
    feats = []
    with torch.inference_mode():
        for images in DataLoader(Images(records, preprocess), batch_size=32, num_workers=4):
            with torch.autocast("cuda", dtype=torch.float16):
                feats.append(model.encode_image(images.cuda(), normalize=True).float().cpu().numpy())
    external = np.concatenate(feats)
    y = frame.label_id.to_numpy(int)[dev]
    probs, _ = bioclip.fit_predict(x[dev], y, external, len(labels))
    saved = pd.read_csv(bioclip.OUTPUT / "test_predictions.csv")
    test_probs, _ = bioclip.fit_predict(x[dev], y, x[~dev], len(labels))
    cols = [c for c in saved.columns if c.startswith("probability_")]
    assert np.allclose(saved[cols].to_numpy(float), test_probs, atol=1e-6), "BioCLIP 2 refit differs from saved test predictions"
    yield "BioCLIP 2 probe (frozen)", probs, "refit verified against saved test predictions; " + OVERLAP_NOTE
    weights = np.load(bioclip.OUTPUT / "zero_shot_text_weights.npy")
    yield "BioCLIP 2 zero-shot", bioclip.zero_shot_probs(external, weights, float(model.logit_scale.exp())), OVERLAP_NOTE


def strong_systems(labels, records):
    """Grid architectures under the third recipe (SnakeCLEF-style at 224 px, flip TTA)."""
    import run_snakeclef_style_convnext as style
    import run_strong_recipe_v4 as strong
    style.configure(224)
    for model_name in strong.MODELS:
        folder = strong.OUTPUT / model_name.replace("/", "-")
        if not (folder / "final_metrics.json").exists():
            continue
        model, _, _, _ = strong.build(model_name, len(labels))
        model.load_state_dict(torch.load(folder / "final_model" / "final.pt", map_location="cpu", weights_only=True))
        identity = base.sha256(folder / "final_model" / "final.pt")
        if model_name == "BioCLIP-2":
            identity += "; " + OVERLAP_NOTE
        yield f"{model_name} (strong)", predict(model, records, style.transforms(False), tta=True), identity


def convnext_systems(labels, records):
    import run_snakeclef_style_convnext as style
    for size, loss, name in ((384, "seesaw", "ConvNeXt-V2-B 384, seesaw"), (384, "ce", "ConvNeXt-V2-B 384, cross-entropy"),
                             (224, "ce", "ConvNeXt-V2-B 224, cross-entropy")):
        ckpt = RUNS / "snakeclef_style" / f"convnextv2_base_{size}_{loss}" / "final_model" / "final.pt"
        if not (ckpt.parent.parent / "final_metrics.json").exists():
            continue
        style.configure(size)
        model = timm.create_model(style.TIMM_NAME, pretrained=False, num_classes=len(labels))
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        yield name, predict(model, records, style.transforms(False), tta=True), base.sha256(ckpt)


def observation_level(records, probs, labels):
    groups = defaultdict(list)
    for i, r in enumerate(records):
        groups[r["observation_id"]].append(i)
    keys = sorted(groups)
    truth = np.array([labels.index(records[groups[k][0]]["model_class"]) for k in keys])
    return keys, truth, np.stack([probs[groups[k]].mean(0) for k in keys])


def main():
    scope_summary = json.loads((SCOPE / "summary.json").read_text(encoding="utf-8"))
    assert hashlib.sha256((SCOPE / "scope.jsonl").read_bytes()).hexdigest() == scope_summary["scope_sha256"]
    records = [json.loads(l) for l in (SCOPE / "scope.jsonl").read_text(encoding="utf-8").splitlines() if l]
    _, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=5))
    OUT.mkdir(parents=True, exist_ok=True)
    systems = (list(grid_systems(labels, records)) + [dinov2_system(labels, records)]
               + list(bioclip2_systems(labels, records)) + list(convnext_systems(labels, records))
               + list(strong_systems(labels, records)))
    rng = np.random.default_rng(SEED)
    keys, truth, _ = observation_level(records, systems[0][1], labels)
    samples = rng.integers(0, len(truth), size=(REPS, len(truth)))
    observed = np.unique(truth)
    rows, correct = [], {}
    for name, probs, identity in systems:
        _, t, obs = observation_level(records, probs, labels)
        assert np.array_equal(t, truth)
        pred = obs.argmax(1)
        top3 = (np.argsort(-obs, axis=1)[:, :3] == truth[:, None]).any(1)
        _, rec, f1, _ = precision_recall_fscore_support(truth, pred, labels=np.arange(len(labels)), zero_division=0)
        correct[name] = pred == truth
        slug = "".join(ch if ch.isalnum() else "_" for ch in name)
        pd.DataFrame({"observation_id": keys, "true": [labels[i] for i in truth], "predicted": [labels[i] for i in pred],
                      **{f"probability_{i:03d}": obs[:, i] for i in range(len(labels))}}).to_csv(OUT / f"observations_{slug}.csv", index=False)
        rows.append({"system": name, "identity": identity, "observations": len(truth), "images": len(records),
                     "species": int(len(observed)), "accuracy": float(correct[name].mean()),
                     "accuracy_ci_low": float(np.quantile(correct[name][samples].mean(1), .025)),
                     "accuracy_ci_high": float(np.quantile(correct[name][samples].mean(1), .975)),
                     "top3_accuracy": float(top3.mean()), "macro_f1_external_species": float(f1[observed].mean()),
                     "balanced_accuracy_external_species": float(rec[observed].mean())})
    table = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    # Systems with possible pretraining overlap are scored but excluded from comparisons.
    overlap = set(table.loc[table.identity.str.contains(OVERLAP_NOTE, regex=False), "system"])
    table["possible_pretraining_overlap"] = table.system.isin(overlap)
    best = table.loc[~table.possible_pretraining_overlap, "system"].iloc[0]
    paired = [{"system_a": best, "system_b": n, "diff_accuracy": float(correct[best].mean() - correct[n].mean()),
               "ci_low": float(np.quantile((correct[best][samples].mean(1) - correct[n][samples].mean(1)), .025)),
               "ci_high": float(np.quantile((correct[best][samples].mean(1) - correct[n][samples].mean(1)), .975))}
              for n in correct if n != best and n not in overlap]
    table.to_csv(OUT / "external_systems.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "external_paired.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"manifest_sha256": manifest_hash, "scope": scope_summary,
                                                  "systems": rows, "paired_vs_best": paired}, indent=2), encoding="utf-8")
    print(table.drop(columns=["identity"]).round(4).to_string(index=False))
    print(pd.DataFrame(paired).round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
