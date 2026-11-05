"""BioCLIP 2 on the full-species v4 split: zero-shot and frozen linear probe.

Model: hf-hub:imageomics/bioclip-2 (open_clip, ViT-L/14, MIT licence), frozen,
with its own evaluation preprocessing (bicubic shorter side 224, center crop
224, CLIP mean/std), float16 autocast. Image features are the L2-normalised
768-d image embeddings, extracted once from the v4 image cache (after the taxonomy
revision of 20 Oct 2026, which changed labels only, reused after checking every file hash).

Zero-shot: one text embedding per class, the normalised mean over the 80
OpenAI ImageNet prompt templates applied to the class's TreeOfLife-200M
taxonomic string plus its common name when TreeOfLife lists one ("taxonomic +
common" text). The text form was fixed before any evaluation; nothing is tuned.
Bungarus wanghaotingi uses the TreeOfLife entry for its GBIF synonym B.
multicinctus (the author-approved external mapping), and Asthenodipsas malaccana
the entry for A. malaccanus, the masculine ending of the same name. Classes absent from
TreeOfLife borrow the higher ranks of a congener and get no common name.

Probe: identical protocol to run_dinov2_probe_v4.py (StandardScaler fitted on
the training part only, balanced multinomial logistic regression C=1), same
five folds and locked test.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import open_clip
import pandas as pd
import sklearn
import torch
from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES
from torch.utils.data import DataLoader

import run_full_species_grouped as full
import run_original_six_grouped as base
from run_dinov2_probe_v4 import V4_RESULTS, fit_predict, reuse_features
from run_leakage_safe_dinov2_benchmark import ImagePathDataset

ROOT = Path(__file__).resolve().parent
HUB_ID = "hf-hub:imageomics/bioclip-2"
OUTPUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "bioclip2"
TREEOFLIFE_SPECIES = Path.home() / ".cache/huggingface/hub/datasets--imageomics--TreeOfLife-200M/snapshots" \
    / "5f2dc493b3dc0e544438a04038ab15faa646b749/embeddings/txt_emb_species.json"
TEXT_SYNONYMS = {"Bungarus wanghaotingi": "Bungarus multicinctus", "Asthenodipsas malaccana": "Asthenodipsas malaccanus"}
SEED, FOLDS = 20260925, 5


def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(HUB_ID)
    return model.eval().to(device), preprocess, open_clip.get_tokenizer(HUB_ID)


def image_features(frame, model, preprocess, device):
    cache, order = OUTPUT / "image_features.npy", OUTPUT / "feature_order.csv"
    if cache.exists() or reuse_features(V4_RESULTS / OUTPUT.name, cache, order, frame):
        saved = pd.read_csv(order, keep_default_na=False)
        assert saved.relative_path.tolist() == frame.relative_path.tolist(), "Feature order changed"
        return np.load(cache)
    loader = DataLoader(ImagePathDataset(frame.absolute_path.tolist(), preprocess), batch_size=64, num_workers=6)
    chunks = []
    with torch.inference_mode():
        for images in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                chunks.append(model.encode_image(images.to(device), normalize=True).float().cpu().numpy())
    x = np.concatenate(chunks)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    np.save(cache, x)
    frame[["relative_path", "cached_sha256"]].to_csv(order, index=False)
    return x


def class_texts(labels):
    entries, genus_ranks = {}, {}
    for ranks, common in json.loads(TREEOFLIFE_SPECIES.read_text(encoding="utf-8")):
        genus, epithet = ranks[5].strip(), ranks[6].strip()
        if genus and epithet:
            entries.setdefault(f"{genus} {epithet.split()[0]}", (ranks, common))
            genus_ranks.setdefault(genus, ranks[:6])
    rows = []
    for label in labels:
        lookup = TEXT_SYNONYMS.get(label, label)
        if lookup in entries:
            ranks, common = entries[lookup]
            source = "treeoflife" if lookup == label else "treeoflife_synonym"
        else:
            genus, epithet = label.split()[:2]
            ranks, common, source = [*genus_ranks[genus], epithet], "", "congener_ranks"
        taxonomic = " ".join(r.strip().split()[0] if i == 6 else r.strip() for i, r in enumerate(ranks) if r.strip())
        text = f"{taxonomic} with common name {common}" if common else taxonomic
        rows.append({"class_label": label, "text": text, "source": source})
    return pd.DataFrame(rows)


def text_classifier(texts, model, tokenizer, device):
    weights = []
    with torch.inference_mode():
        for text in texts:
            tokens = tokenizer([template(text) for template in OPENAI_IMAGENET_TEMPLATES]).to(device)
            emb = model.encode_text(tokens, normalize=True).float().mean(dim=0)
            weights.append(emb / emb.norm())
    return torch.stack(weights).cpu().numpy()


def zero_shot_probs(x, weights, logit_scale):
    logits = logit_scale * (x @ weights.T)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def main():
    frame, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=FOLDS))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_model(device)
    start = time.perf_counter()
    x = image_features(frame, model, preprocess, device)
    extract_seconds = time.perf_counter() - start
    texts = class_texts(labels)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    texts.to_csv(OUTPUT / "zero_shot_class_texts.csv", index=False)
    weights = text_classifier(texts.text.tolist(), model, tokenizer, device)
    np.save(OUTPUT / "zero_shot_text_weights.npy", weights)
    logit_scale = float(model.logit_scale.exp())

    keys = ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "top3_accuracy")
    dev_mask = frame.split.isin(("train", "val")).to_numpy()
    dev, x_dev = frame[dev_mask].reset_index(drop=True), x[dev_mask]
    test, x_test = frame[~dev_mask].reset_index(drop=True), x[~dev_mask]
    y_dev, truth = dev.label_id.to_numpy(int), test.label_id.to_numpy(int)

    zs_test = zero_shot_probs(x_test, weights, logit_scale)
    base.prediction_table(test, zs_test, labels).to_csv(OUTPUT / "zero_shot_test_predictions.csv", index=False)
    zs_dev = zero_shot_probs(x_dev, weights, logit_scale)

    records, parts = [], []
    for number, (train_idx, val_idx) in enumerate(full.make_folds(frame, SimpleNamespace(folds=FOLDS)), start=1):
        probs, iterations = fit_predict(x_dev[train_idx], y_dev[train_idx], x_dev[val_idx], len(labels))
        part = base.prediction_table(dev.iloc[val_idx].reset_index(drop=True), probs, labels)
        part["fold"] = number
        parts.append(part)
        records.append({"fold": number, "lbfgs_iterations": iterations, **base.scores(y_dev[val_idx], probs)})
        print(json.dumps(records[-1]), flush=True)
    pd.concat(parts, ignore_index=True).to_csv(OUTPUT / "development_oof_predictions.csv", index=False)
    probs, iterations = fit_predict(x_dev, y_dev, x_test, len(labels))
    base.prediction_table(test, probs, labels).to_csv(OUTPUT / "test_predictions.csv", index=False)

    summary = {
        "manifest_sha256": manifest_hash, "model": HUB_ID, "open_clip": open_clip.__version__, "labels": labels,
        "input": "v4 cache; RGB; BioCLIP 2 eval preprocessing (bicubic shorter side 224, center crop 224, CLIP mean/std); float16 autocast",
        "features": "L2-normalised 768-d image embeddings", "feature_seconds": extract_seconds,
        "zero_shot": {
            "text": "TreeOfLife-200M taxonomic string + common name when listed; mean of 80 OpenAI ImageNet templates",
            "text_sources": texts.source.value_counts().to_dict(), "logit_scale": logit_scale,
            "development_all_images": base.scores(y_dev, zs_dev),
            "test": {"point": base.scores(truth, zs_test),
                     "group_bootstrap_95ci": base.grouped_intervals(test, zs_test, 2000, SEED + 3000)},
        },
        "probe": {
            "classifier": "StandardScaler (training part only) + balanced multinomial logistic regression C=1, lbfgs, max_iter=2000, tol=1e-5",
            "scikit_learn": sklearn.__version__,
            "cv": {"folds": records, "mean": {k: float(np.mean([r[k] for r in records])) for k in keys},
                   "sd": {k: float(np.std([r[k] for r in records], ddof=1)) for k in keys}},
            "final": {"lbfgs_iterations": iterations, "test_images": len(test),
                      "test_groups": int(test.leakage_group_id.nunique()), "point": base.scores(truth, probs),
                      "group_bootstrap_95ci": base.grouped_intervals(test, probs, 2000, SEED + 2000)},
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"zero_shot_test": summary["zero_shot"]["test"]["point"],
                      "probe_cv_mean": summary["probe"]["cv"]["mean"],
                      "probe_test": summary["probe"]["final"]["point"]}, indent=2))


if __name__ == "__main__":
    main()
