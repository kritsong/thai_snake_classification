"""SnakeCLEF-winner-style baseline on the full-species v4 split.

Follows the published SnakeCLEF 2023/2024 winning recipe (Hu et al. 2023;
Wang et al. 2024) as far as it applies to image-only training, using
torchvision equivalents of their Albumentations pipeline:
  * ConvNeXt-V2-Base, ImageNet-22k fine-tuned at 384 px (timm
    convnextv2_base.fcmae_ft_in22k_in1k_384, CC-BY-NC-4.0 weights);
  * 384x384 input; random resized crop, flips, shift/scale/rotate, colour
    jitter, elastic distortion, random erasing (Cutout), CutMix (p=0.5);
  * Seesaw loss (p=0.8, q=2.0, published defaults) or cross-entropy (--loss),
    the only difference between the two runs of the ablation;
  * AdamW, backbone LR 2e-5 with weight decay 2e-5, 15 epochs, first epoch
    linear warm-up from 2e-7, cosine decay; flip test-time augmentation.
The classifier head uses 10x the backbone LR, as in the main runner. No
setting is tuned: CV reports the fixed recipe and the final model trains on all
development data for the same 15 epochs, then is tested once.
Not used (not applicable here): location metadata, venomous post-processing,
multi-branch venom experts, test-set pseudo-labels, ensembles.
--image-size 224 is the resolution control: the same recipe at the 224 px used
by the grid models, with the 224 px weights of the same checkpoint
(convnextv2_base.fcmae_ft_in22k_in1k); everything else is unchanged.
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

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode, v2

import run_full_species_grouped as full  # patches the base runner's manifest/fold readers
import run_original_six_grouped as base

ROOT = Path(__file__).resolve().parent
TIMM_NAMES = {384: "convnextv2_base.fcmae_ft_in22k_in1k_384", 224: "convnextv2_base.fcmae_ft_in22k_in1k"}
TIMM_NAME = TIMM_NAMES[384]
SIZE, MEAN, STD = 384, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def configure(size: int) -> None:
    """Select input resolution and the matching pretrained weights."""
    global SIZE, TIMM_NAME
    SIZE, TIMM_NAME = size, TIMM_NAMES[size]


class SeesawLoss(nn.Module):
    """Seesaw loss (Wang et al., CVPR 2021), following the mmdetection implementation."""

    def __init__(self, n_classes: int, p: float = 0.8, q: float = 2.0, eps: float = 1e-2):
        super().__init__()
        self.p, self.q, self.eps = p, q, eps
        self.register_buffer("cum_samples", torch.zeros(n_classes, dtype=torch.float64))

    def update(self, labels: torch.Tensor, weight: float) -> None:
        self.cum_samples += torch.bincount(labels, minlength=len(self.cum_samples)).double() * weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        onehot = F.one_hot(labels, logits.shape[1]).float()
        counts = self.cum_samples.clamp(min=1).float()
        ratio = counts[None, :] / counts[:, None]                      # N_j / N_i
        mitigation = torch.where(ratio < 1, ratio.pow(self.p), torch.ones_like(ratio))[labels]
        scores = logits.detach().float().softmax(dim=1)
        self_scores = scores.gather(1, labels[:, None]).clamp(min=self.eps)
        score_ratio = scores / self_scores
        compensation = torch.where(score_ratio > 1, score_ratio.pow(self.q), torch.ones_like(score_ratio))
        weights = mitigation * compensation
        adjusted = logits.float() + weights.log() * (1 - onehot)
        return F.cross_entropy(adjusted, labels)


class Images(Dataset):
    def __init__(self, frame: pd.DataFrame, transform):
        self.paths, self.labels, self.transform = frame.absolute_path.tolist(), frame.label_id.astype(int).tolist(), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as raw:
            return self.transform(raw.convert("RGB")), self.labels[index]


def transforms(train: bool):
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)]
    if not train:
        return v2.Compose([v2.Resize((SIZE, SIZE), interpolation=InterpolationMode.BICUBIC, antialias=True), *to_tensor])
    return v2.Compose([
        v2.RandomResizedCrop(SIZE, scale=(0.35, 1.0), interpolation=InterpolationMode.BICUBIC, antialias=True),
        v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
        v2.RandomAffine(degrees=15, translate=(0.0625, 0.0625), scale=(0.9, 1.1)),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.RandomApply([v2.ElasticTransform(alpha=40.0, sigma=5.0)], p=0.2),
        *to_tensor,
        v2.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])


def make_loader(frame, train, args, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Images(frame, transforms(train)), batch_size=args.batch_size, shuffle=train,
                      num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
                      generator=generator, worker_init_fn=base.worker_seed, drop_last=train)


def lr_at(step, steps_per_epoch, args):
    warm = steps_per_epoch * args.warmup_epochs
    if step < warm:
        return args.warmup_lr + (args.lr - args.warmup_lr) * step / max(1, warm)
    progress = (step - warm) / max(1, steps_per_epoch * args.epochs - warm)
    return 0.5 * args.lr * (1 + math.cos(math.pi * progress))


def train_model(train_frame, n_classes, args, seed, out_dir: Path):
    base.seed_everything(seed)
    device = torch.device("cuda")
    model = timm.create_model(TIMM_NAME, pretrained=True, num_classes=n_classes, drop_path_rate=args.drop_path)
    model = model.to(device, memory_format=torch.channels_last)
    head = [p for n, p in model.named_parameters() if n.startswith("head.fc")]
    backbone = [p for n, p in model.named_parameters() if not n.startswith("head.fc")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr_scale": 1.0}, {"params": head, "lr_scale": args.head_lr_ratio}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    criterion = SeesawLoss(n_classes).to(device) if args.loss == "seesaw" else None
    loader = make_loader(train_frame, True, args, seed)
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
                # CutMix returns soft targets; recover the two sources and their area weight.
                # torchvision pairs each image with the batch rolled by one; one lambda per batch.
                labels_b = labels.roll(1, 0)
                images, soft = cutmix(images, labels)
                differ = labels != labels_b
                lam = float(soft[differ, labels[differ]].mean()) if differ.any() else 1.0
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images)
            if criterion is not None:
                criterion.update(labels, lam if mixed else 1.0)
                if mixed:
                    criterion.update(labels_b, 1 - lam)
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)
                else:
                    loss = criterion(logits, labels)
            else:
                logits = logits.float()
                loss = (lam * F.cross_entropy(logits, labels) + (1 - lam) * F.cross_entropy(logits, labels_b)
                        if mixed else F.cross_entropy(logits, labels))
            scaler.scale(loss / args.accumulate).backward()
            if (batch + 1) % args.accumulate == 0:
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(step, steps_per_epoch, args) * group["lr_scale"]
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
            total += float(loss.detach()) * len(labels)
            seen += len(labels)
        history.append({"epoch": epoch, "train_loss": total / seen, "lr_end": optimizer.param_groups[0]["lr"]})
        print(f"{args.loss}: epoch {epoch}/{args.epochs}, loss={total / seen:.4f}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    base.atomic_json(out_dir / "history.json", history)
    return model, time.perf_counter() - start


def predict_tta(model, frame, args, seed):
    model.eval()
    probabilities, truth = [], []
    with torch.inference_mode():
        for images, labels in make_loader(frame, False, args, seed):
            images = images.to("cuda", non_blocking=True, memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = (model(images).float().softmax(1) + model(images.flip(3)).float().softmax(1)) / 2
            probabilities.append(logits.cpu().numpy())
            truth.append(labels.numpy())
    return np.concatenate(truth), np.concatenate(probabilities)


def protocol(args, manifest_hash, labels):
    return {"manifest_sha256": manifest_hash, "timm_model": TIMM_NAME, "loss": args.loss,
            "seesaw": {"p": 0.8, "q": 2.0, "eps": 0.01, "counts": "accumulated online per fold, floored at 1"} if args.loss == "seesaw" else None,
            "image_size": SIZE, "epochs": args.epochs, "warmup_epochs": args.warmup_epochs, "lr": args.lr,
            "warmup_lr": args.warmup_lr, "head_lr_ratio": args.head_lr_ratio, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "accumulate": args.accumulate, "drop_path": args.drop_path,
            "cutmix_prob": args.cutmix_prob, "seed": args.seed, "labels": labels, "tta": "mean of original and horizontal flip",
            "torch_version": torch.__version__, "timm_version": timm.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("cv", "final"), required=True)
    parser.add_argument("--loss", choices=("seesaw", "ce"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=full.DEFAULT_DATASET)
    parser.add_argument("--output-root", type=Path, default=ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "snakeclef_style")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-lr", type=float, default=2e-7)
    parser.add_argument("--head-lr-ratio", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--drop-path", type=float, default=0.1)
    parser.add_argument("--cutmix-prob", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--image-size", type=int, choices=tuple(TIMM_NAMES), default=384)
    args = parser.parse_args()
    configure(args.image_size)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    frame, labels, manifest_hash, _ = full.read_manifest(SimpleNamespace(dataset_root=args.dataset_root, folds=args.folds))
    folds = full.make_folds(frame, SimpleNamespace(folds=args.folds))
    dev = frame.loc[frame.split.isin(("train", "val"))].reset_index(drop=True)
    signature = protocol(args, manifest_hash, labels)
    out = args.output_root / f"convnextv2_base_{SIZE}_{args.loss}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        if json.loads((out / "protocol.json").read_text(encoding="utf-8")) != signature:
            raise ValueError("Existing protocol differs")
    else:
        base.atomic_json(out / "protocol.json", signature)

    if args.mode == "cv":
        records, parts = [], []
        for number, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_dir = out / f"fold_{number}"
            train_frame, val_frame = dev.iloc[train_idx].reset_index(drop=True), dev.iloc[val_idx].reset_index(drop=True)
            fingerprint = base.fold_fingerprint(train_frame, val_frame)
            result_path = fold_dir / "fold_result.json"
            if result_path.exists():
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                assert saved["protocol"] == signature and saved["fingerprint"] == fingerprint, f"Fold {number} mismatch"
                part = pd.read_csv(fold_dir / "oof_predictions.csv", keep_default_na=False)
                records.append(saved["result"]); parts.append(part)
                print(f"Fold {number}: verified saved result; skipping", flush=True)
                continue
            model, seconds = train_model(train_frame, len(labels), args, args.seed + number, fold_dir)
            truth, probs = predict_tta(model, val_frame, args, args.seed + number)
            assert np.array_equal(truth, val_frame.label_id.to_numpy(int))
            part = base.prediction_table(val_frame, probs, labels); part["fold"] = number
            base.atomic_csv(fold_dir / "oof_predictions.csv", part)
            record = {"fold": number, "epochs_trained": args.epochs, "seconds": seconds, **base.scores(truth, probs)}
            base.atomic_json(result_path, {"protocol": signature, "fold": number, "fingerprint": fingerprint, "result": record})
            records.append(record); parts.append(part)
            del model; torch.cuda.empty_cache()
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
    model, seconds = train_model(dev, len(labels), args, args.seed + 1000, out / "final_model")
    torch.save(model.state_dict(), out / "final_model" / "final.pt")
    test = frame.loc[frame.split == "test"].reset_index(drop=True)
    base.atomic_json(out / "locked_test_started.json", {"protocol": signature, "test_images": len(test)})
    truth, probs = predict_tta(model, test, args, args.seed + 1000)
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
