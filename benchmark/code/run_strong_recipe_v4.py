"""SnakeCLEF-style recipe at 224 px for every grid architecture (the third recipe).

Exactly the recipe of run_snakeclef_style_convnext.py --image-size 224 --loss ce
(the run reported as "ConvNeXt-V2 224, cross-entropy"): 224 px input; random
resized crop, flips, affine, colour jitter, elastic distortion, random erasing;
CutMix on half the batches; cross-entropy; AdamW, backbone learning rate 2e-5,
weight decay 2e-5, ten-fold head learning rate, effective batch 32, drop-path
0.1, 15 epochs with a one-epoch warm-up and cosine decay; flip test-time
augmentation. Only the backbone changes. Pretrained weights, the ImageNet input
normalization and the classifier layers that receive the head learning rate are
those of the grid (run_original_six_grouped.build_model's rule). BioCLIP 2's
open_clip image tower has no drop-path setting, so it trains without drop-path.
CV on the five development folds, then one refit on all development data for the
same 15 epochs and one locked-test evaluation.

Usage: python run_strong_recipe_v4.py --model "Swin-Base" --mode cv|final
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import v2

import run_full_species_grouped_bioclip2  # noqa: F401  registers ConvNeXt-V2, DINOv2 and BioCLIP 2
import run_full_species_grouped as full
import run_original_six_grouped as base
import run_snakeclef_style_convnext as style

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "strong"
HEAD_NAMES = {"head", "head_dist", "fc", "classifier"}  # same rule as base.build_model
MODELS = ("MobileNetV3-Small", "EfficientNet-B0", "ResNet-50", "ViT-Base/16", "DeiT-Base/16", "Swin-Base",
          "DINOv2-Base/14", "BioCLIP-2")
RECIPE = dict(image_size=224, epochs=15, warmup_epochs=1, lr=2e-5, warmup_lr=2e-7, head_lr_ratio=10.0,
              weight_decay=2e-5, batch_size=16, accumulate=2, drop_path=0.1, cutmix_prob=0.5, workers=6,
              seed=20260925, folds=5, bootstrap_replicates=2000)


def build(name, n_classes):
    model = timm.create_model(base.MODELS[name], pretrained=True, num_classes=n_classes, drop_path_rate=RECIPE["drop_path"])
    head = [p for n, p in model.named_parameters() if n.split(".")[0] in HEAD_NAMES]
    backbone = [p for n, p in model.named_parameters() if n.split(".")[0] not in HEAD_NAMES]
    if not head or not backbone:
        raise RuntimeError(f"Cannot identify head and backbone for {name}")
    has_drop_path = any(m.__class__.__name__ == "DropPath" for m in model.modules())
    return model, backbone, head, has_drop_path


def train_model(name, train_frame, n_classes, args, seed, out_dir: Path):
    base.seed_everything(seed)
    device = torch.device("cuda")
    model, backbone, head, _ = build(name, n_classes)
    model = model.to(device, memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW([{"params": backbone, "lr_scale": 1.0}, {"params": head, "lr_scale": args.head_lr_ratio}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    loader = style.make_loader(train_frame, True, args, seed)
    cutmix = v2.CutMix(num_classes=n_classes, alpha=1.0)
    scaler = torch.amp.GradScaler("cuda")
    steps_per_epoch, step, history, start = len(loader) // args.accumulate, 0, [], time.perf_counter()
    rng = np.random.default_rng(seed)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for batch, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            mixed = rng.random() < args.cutmix_prob
            if mixed:
                labels_b = labels.roll(1, 0)
                images, soft = cutmix(images, labels)
                differ = labels != labels_b
                lam = float(soft[differ, labels[differ]].mean()) if differ.any() else 1.0
            with torch.autocast("cuda", dtype=torch.float16):
                logits = base.logits_only(model(images))
            logits = logits.float()
            loss = (lam * F.cross_entropy(logits, labels) + (1 - lam) * F.cross_entropy(logits, labels_b)
                    if mixed else F.cross_entropy(logits, labels))
            scaler.scale(loss / args.accumulate).backward()
            if (batch + 1) % args.accumulate == 0:
                for group in optimizer.param_groups:
                    group["lr"] = style.lr_at(step, steps_per_epoch, args) * group["lr_scale"]
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            total += float(loss.detach()) * len(labels)
            seen += len(labels)
        history.append({"epoch": epoch, "train_loss": total / seen, "lr_end": optimizer.param_groups[0]["lr"]})
        print(f"{name}: epoch {epoch}/{args.epochs}, loss={total / seen:.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_json(out_dir / "history.json", history)
    return model, time.perf_counter() - start


def protocol(name, manifest_hash, labels, has_drop_path):
    return {"manifest_sha256": manifest_hash, "model": name, "timm_model": base.MODELS[name],
            "recipe": "SnakeCLEF-style at 224 px, cross-entropy (as run_snakeclef_style_convnext.py --image-size 224 --loss ce)",
            **RECIPE, "drop_path_applied": has_drop_path, "head_rule": sorted(HEAD_NAMES), "loss": "ce",
            "labels": labels, "tta": "mean of original and horizontal flip",
            "torch_version": torch.__version__, "timm_version": timm.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("cv", "final", "check"), required=True)
    a = parser.parse_args()
    style.configure(RECIPE["image_size"])
    args = SimpleNamespace(**RECIPE)
    frame, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=full.DEFAULT_DATASET, folds=args.folds))
    folds = full.make_folds(frame, SimpleNamespace(folds=args.folds))
    dev = frame.loc[frame.split.isin(("train", "val"))].reset_index(drop=True)
    _, _, _, has_drop_path = build(a.model, len(labels))
    signature = protocol(a.model, manifest_hash, labels, has_drop_path)
    out = OUTPUT / a.model.replace("/", "-")
    if a.mode == "check":
        print(json.dumps({"model": a.model, "drop_path_applied": has_drop_path}))
        return
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        if json.loads((out / "protocol.json").read_text(encoding="utf-8")) != signature:
            raise ValueError("Existing protocol differs")
    else:
        base.atomic_json(out / "protocol.json", signature)

    if a.mode == "cv":
        records, parts = [], []
        for number, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_dir = out / f"fold_{number}"
            train_frame, val_frame = dev.iloc[train_idx].reset_index(drop=True), dev.iloc[val_idx].reset_index(drop=True)
            fingerprint = base.fold_fingerprint(train_frame, val_frame)
            result_path = fold_dir / "fold_result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                assert saved["protocol"] == signature and saved["fingerprint"] == fingerprint, f"Fold {number} mismatch"
                records.append(saved["result"])
                parts.append(pd.read_csv(fold_dir / "oof_predictions.csv", keep_default_na=False))
                print(f"Fold {number}: verified saved result; skipping", flush=True)
                continue
            model, seconds = train_model(a.model, train_frame, len(labels), args, args.seed + number, fold_dir)
            truth, probs = style.predict_tta(model, val_frame, args, args.seed + number)
            assert np.array_equal(truth, val_frame.label_id.to_numpy(int))
            part = base.prediction_table(val_frame, probs, labels)
            part["fold"] = number
            base.atomic_csv(fold_dir / "oof_predictions.csv", part)
            record = {"fold": number, "epochs_trained": args.epochs, "seconds": seconds, **base.scores(truth, probs)}
            base.atomic_json(result_path, {"protocol": signature, "fold": number, "fingerprint": fingerprint, "result": record})
            records.append(record)
            parts.append(part)
            del model
            torch.cuda.empty_cache()
        keys = ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "top3_accuracy")
        summary = {"protocol": signature, "folds": records,
                   "mean": {k: float(np.mean([r[k] for r in records])) for k in keys},
                   "sd": {k: float(np.std([r[k] for r in records], ddof=1)) for k in keys}}
        base.atomic_csv(out / "development_oof_predictions.csv", pd.concat(parts, ignore_index=True))
        base.atomic_csv(out / "cross_validation_metrics.csv", pd.DataFrame(records))
        base.atomic_json(out / "cv_summary.json", summary)
        print(json.dumps(summary["mean"], indent=2))
        return

    if not (out / "cv_summary.json").exists():
        raise SystemExit("Run CV first")
    if (out / "locked_test_started.json").exists():
        raise SystemExit("Locked test already started for this configuration")
    model, seconds = train_model(a.model, dev, len(labels), args, args.seed + 1000, out / "final_model")
    torch.save(model.state_dict(), out / "final_model" / "final.pt")
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    base.atomic_json(out / "locked_test_started.json", {"protocol": signature, "test_images": len(test)})
    truth, probs = style.predict_tta(model, test, args, args.seed + 1000)
    assert np.array_equal(truth, test.label_id.to_numpy(int))
    base.prediction_table(test, probs, labels).to_csv(out / "test_predictions.csv", index=False)
    result = {"protocol": signature, "test_images": len(test), "test_groups": int(test.leakage_group_id.nunique()),
              "training_seconds": seconds, "parameters": sum(p.numel() for p in model.parameters()),
              "point": base.scores(truth, probs),
              "group_bootstrap_95ci": base.grouped_intervals(test, probs, args.bootstrap_replicates, args.seed + 2000)}
    base.atomic_json(out / "final_metrics.json", result)
    print(json.dumps(result["point"], indent=2))


if __name__ == "__main__":
    main()
