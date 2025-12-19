# train_vit_exact_aug.py
# ------------------------------------------------------------
# ViT-Base fine-tuning on your snake folders — uses ONLY the
# augmentations you listed (rotation/shift/zoom/flip/shear) and nothing else.
#
# Requested updates:
# - Mode: ONLY fine-tune from pretrained.
# - "Warm up" = freeze backbone for the first 10 epochs, then unfreeze.
# - Keep SAME base dir & summary CSV; experiment folder name uses "correct_fine_tune".
#
# Other rules preserved:
# - One class per subfolder under DATA_DIR
# - Filters classes by THRESHOLDS; stratified train/val split
# - NO class/sample weights; NO early stopping
# - NO checkpoint saving (save_strategy="no"); skip run if final_metrics.json exists
# - Saves: history.json, metrics.json, classification_report.txt, final_metrics.json
# - Global CSV: OUTPUT_BASE_DIR/experiments_summary_vit.csv
#
# Tested with:
#   torch 2.6.0+cu124, torchvision 0.21.0+cu124, transformers 4.56.2 (Windows)
# ------------------------------------------------------------

import os, json, time, random, csv
from collections import defaultdict
from typing import List, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    top_k_accuracy_score, classification_report
)

from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names


# ---------------- Configuration ----------------
DATA_DIR = r"C:\Users\ADMIN\Downloads\snake_datasetNew"

MODEL_NAME = "ViT-Base"
MODEL_ID   = "google/vit-base-patch16-224-in21k"

OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "transformer_experiment_results_vit")
SUMMARY_CSV = os.path.join(OUTPUT_BASE_DIR, "experiments_summary_vit.csv")

THRESHOLDS = [100, 200, 300, 400, 500]
VAL_FRACTION = 0.20
SEED = 42

EPOCHS = 50
FREEZE_EPOCHS = 10           # freeze backbone for first 10 epochs, then unfreeze
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE   = 32

NUM_WORKERS = 8
PERSISTENT_WORKERS = True
USE_TORCH_COMPILE = False     # keep False on Windows

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ====== Augmentation — EXACTLY as requested ======
BASE_AUG = {
    "rotation_range": 10,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "zoom_range": 0.1,
    "horizontal_flip": True,
    "shear_range": 0.1,
    "fill_mode": "nearest",   # mimic via interpolation=NEAREST in RandomAffine
}
AUG_LEVELS = {"low": 0.5, "medium": 1.0, "high": 1.5}

def scaled_aug(base_aug, scale):
    out = {}
    for k, v in base_aug.items():
        out[k] = (v * scale) if isinstance(v, (int, float)) else v
    return out

AUG_SETTINGS = {
    "none": {},
    "low":    scaled_aug(BASE_AUG, AUG_LEVELS["low"]),
    "medium": scaled_aug(BASE_AUG, AUG_LEVELS["medium"]),
    "high":   scaled_aug(BASE_AUG, AUG_LEVELS["high"]),
}


# ---------------- Data helpers ----------------
def list_images_recursive(root: str) -> List[str]:
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f.lower())[1] in IMG_EXTS:
                out.append(os.path.join(d, f))
    return out

def top_level_class(root: str, filepath: str) -> str:
    rel = os.path.relpath(filepath, root)
    return rel.split(os.sep)[0]

def scan_dataset(root: str) -> Dict[str, List[str]]:
    species_to_files = defaultdict(list)
    for p in list_images_recursive(root):
        sp = top_level_class(root, p)
        if sp and sp != "." and not sp.startswith("._"):
            species_to_files[sp].append(p)
    return species_to_files

def filter_by_threshold(species_to_files: Dict[str, List[str]], threshold: int):
    kept_species = sorted([sp for sp, files in species_to_files.items() if len(files) >= threshold])
    if not kept_species:
        return [], [], []
    kept = set(kept_species)
    X, y_names = [], []
    for sp, files in species_to_files.items():
        if sp in kept:
            for p in files:
                X.append(p)
                y_names.append(sp)
    return X, y_names, kept_species

def stratified_split(X: List[str], y_names: List[str], val_frac: float, seed: int):
    y = np.array(y_names); X = np.array(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(sss.split(np.zeros_like(y), y))
    return X[tr].tolist(), y[tr].tolist(), X[va].tolist(), y[va].tolist()


# ---------------- Dataset (uses ONLY the requested augs) ----------------
class ImgDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], aug_params: dict, processor, resize_to=224):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.resize_to = int(resize_to)
        self.aug = self._build_aug_pipeline(aug_params)

    def _build_aug_pipeline(self, aug_params: dict):
        ops = [transforms.Resize((self.resize_to, self.resize_to), interpolation=InterpolationMode.BILINEAR)]

        if aug_params:
            deg = float(aug_params.get("rotation_range", 0.0))
            tx  = float(aug_params.get("width_shift_range", 0.0))
            ty  = float(aug_params.get("height_shift_range", 0.0))
            zm  = float(aug_params.get("zoom_range", 0.0))
            sh  = float(aug_params.get("shear_range", 0.0))
            do_flip = bool(aug_params.get("horizontal_flip", False))

            if do_flip:
                ops.append(transforms.RandomHorizontalFlip(p=0.5))

            if any([deg, tx, ty, zm, sh]):
                scale_low  = max(0.7, 1.0 - zm)
                scale_high = 1.0 + zm
                translate  = (max(0.0, min(tx, 0.45)), max(0.0, min(ty, 0.45)))
                ops.append(
                    transforms.RandomAffine(
                        degrees=deg,
                        translate=translate if (tx > 0 or ty > 0) else None,
                        scale=(scale_low, scale_high) if zm > 0 else None,
                        shear=sh if sh > 0 else None,
                        interpolation=InterpolationMode.NEAREST,  # mimic Keras 'nearest' fill behavior
                    )
                )

        return transforms.Compose(ops)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])
        img = Image.open(p).convert("RGB")
        img = self.aug(img)

        proc = self.processor(
            images=img,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        px = proc["pixel_values"].squeeze(0)  # [3, 224, 224]
        return {"pixel_values": px, "labels": y}


# ---------------- Metrics ----------------
def compute_metrics_builder(id2label: Dict[int, str]):
    def softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def fn(eval_pred):
        if isinstance(eval_pred, EvalPrediction):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
            logits, labels = eval_pred

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        preds = np.argmax(logits, axis=-1)
        prob = softmax_np(logits)

        acc = accuracy_score(labels, preds)
        try:
            top3 = top_k_accuracy_score(labels, prob, k=3, labels=list(range(len(id2label))))
        except Exception:
            top3 = float("nan")

        return {
            "accuracy": acc,
            "top3": top3,
            "macro_f1": f1_score(labels, preds, average="macro"),
            "kappa": cohen_kappa_score(labels, preds),
        }
    return fn


# ---------------- Freeze / Trainer helpers ----------------
def set_backbone_trainable(model: torch.nn.Module, train_backbone: bool):
    for name, p in model.named_parameters():
        if name.startswith("classifier") or ".classifier." in name:
            p.requires_grad = True
        else:
            p.requires_grad = train_backbone

def count_trainable(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total

class SafeCETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(pixel_values=inputs["pixel_values"], labels=labels)
        loss = outputs.loss if hasattr(outputs, "loss") and outputs.loss is not None else \
               F.cross_entropy(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        # Include ALL params in the optimizer so unfreezing later doesn't require rebuilding it.
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            param_dict = {n: p for n, p in self.model.named_parameters()}
            optimizer_grouped_parameters = [
                {
                    "params": [param_dict[n] for n in param_dict.keys() if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [param_dict[n] for n in param_dict.keys() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            opt_cls, opt_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = opt_cls(optimizer_grouped_parameters, **opt_kwargs)
        return self.optimizer


class WarmFreezeUnfreezeCallback(TrainerCallback):
    """Freeze backbone for the first N epochs, then unfreeze."""
    def __init__(self, freeze_epochs: int = 10):
        self.freeze_epochs = int(freeze_epochs)
        self.unfroze = False

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        set_backbone_trainable(model, train_backbone=False)
        trp, tot = count_trainable(model)
        print(f"[WarmStart] Backbone FROZEN for first {self.freeze_epochs} epochs. "
              f"Trainable params initially: {trp:,} / {tot:,}")

    def on_epoch_begin(self, args, state, control, model=None, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        if (not self.unfroze) and (current_epoch >= self.freeze_epochs):
            set_backbone_trainable(model, train_backbone=True)
            self.unfroze = True
            trp, tot = count_trainable(model)
            print(f"[WarmStart] UNFROZE backbone at epoch {current_epoch}. "
                  f"Trainable params now: {trp:,} / {tot:,}")


# ---------------- CSV summary ----------------
SUMMARY_FIELDS = [
    "exp_name","model_name","model_id","threshold","aug","mode","lr","epochs",
    "train_samples","val_samples","num_classes","eval_accuracy","eval_top3",
    "eval_macro_f1","eval_kappa","eval_loss","precision","seed","output_dir","runtime_sec"
]
def append_summary_row(csv_path: str, row: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if not file_exists: w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


# ---------------- Main runner ----------------
def run():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)

    print(f"Torch: {torch.__version__}")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"[Data] Dir={DATA_DIR}")
    species_to_files = scan_dataset(DATA_DIR)
    print(f"[Data] Found {len(species_to_files)} species total.")

    # precision prefs
    precision_kwargs = {}; precision_name = "cpu"
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            precision_kwargs["bf16"] = True; precision_name = "bf16"
        else:
            precision_kwargs["fp16"] = True; precision_name = "fp16"

    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)

    for thr in THRESHOLDS:
        X, y_names, kept_species = filter_by_threshold(species_to_files, thr)
        if not X:
            print(f"[Skip] No species meet threshold={thr}."); continue

        print(f"[Data] Keeping {len(kept_species)} classes with ≥{thr} images.")
        print(f"[Data] Total images: {len(X)}  |  Classes: {len(kept_species)}")

        label2id = {name: i for i, name in enumerate(kept_species)}
        id2label = {i: name for name, i in label2id.items()}

        X_tr, ytr_names, X_va, yva_names = stratified_split(X, y_names, VAL_FRACTION, SEED)
        y_tr = [label2id[n] for n in ytr_names]; y_va = [label2id[n] for n in yva_names]
        print(f"[Split] train={len(X_tr)}  val={len(X_va)}")

        val_ds = ImgDataset(X_va, y_va, aug_params={}, processor=processor, resize_to=224)

        for aug_name, aug_params in AUG_SETTINGS.items():
            train_ds = ImgDataset(X_tr, y_tr, aug_params=aug_params, processor=processor, resize_to=224)

            # === ONLY fine-tune from pretrained, with warm-up freeze stage ===
            ft_tag, lr0 = "correct_fine_tune", 2e-5
            exp_name = f"{MODEL_NAME.replace(' ', '')}_aug{aug_name}_{ft_tag}_thres{thr}"
            exp_dir  = os.path.join(OUTPUT_BASE_DIR, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            # ---- Skip if already done ----
            if os.path.exists(os.path.join(exp_dir, "final_metrics.json")):
                print(f"[Skip] Already finished: {exp_name}")
                continue

            print(f"\n--- {exp_name} ---")
            run_start = time.time()

            # model (pretrained), start frozen; callback will unfreeze later
            print("  > Init from pretrained (warm-up freeze -> unfreeze)")
            model = ViTForImageClassification.from_pretrained(
                MODEL_ID,
                num_labels=len(id2label),
                id2label=id2label,
                label2id=label2id,
                torch_dtype=(
                    torch.bfloat16 if precision_kwargs.get("bf16")
                    else (torch.float16 if precision_kwargs.get("fp16") else None)
                ),
                ignore_mismatched_sizes=True
            )
            set_backbone_trainable(model, train_backbone=False)
            trp, tot = count_trainable(model)
            print(f"  > Trainable params (start): {trp:,} / {tot:,}")

            # Training args — NO checkpoint; use cosine schedule; LR warmup disabled
            targs = TrainingArguments(
                output_dir=exp_dir,
                per_device_train_batch_size=TRAIN_BATCH_SIZE,
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                learning_rate=lr0,
                num_train_epochs=EPOCHS,
                weight_decay=0.05,

                eval_strategy="epoch",
                save_strategy="no",
                load_best_model_at_end=False,

                lr_scheduler_type="cosine",
                warmup_ratio=0.0,
                warmup_steps=0,

                optim="adamw_torch_fused",
                dataloader_num_workers=NUM_WORKERS,
                dataloader_persistent_workers=PERSISTENT_WORKERS,
                remove_unused_columns=False,
                label_smoothing_factor=0.0,
                report_to=["tensorboard"],
                torch_compile=USE_TORCH_COMPILE,
                logging_steps=50,
                **precision_kwargs,
            )

            trainer = SafeCETrainer(
                model=model,
                args=targs,
                train_dataset=train_ds,
                eval_dataset=val_ds,
                processing_class=processor,
                data_collator=DefaultDataCollator(),
                compute_metrics=compute_metrics_builder(id2label),
                callbacks=[WarmFreezeUnfreezeCallback(freeze_epochs=FREEZE_EPOCHS)],
            )

            # Save label maps
            with open(os.path.join(exp_dir, "label2id.json"), "w", encoding="utf-8") as f:
                json.dump(label2id, f, indent=2, ensure_ascii=False)
            with open(os.path.join(exp_dir, "id2label.json"), "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in id2label.items()}, f, indent=2, ensure_ascii=False)

            # Train
            trainer.train()

            # Logs & metrics
            with open(os.path.join(exp_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(trainer.state.log_history, f, indent=2)

            eval_metrics = trainer.evaluate()
            with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}, f, indent=2)

            # Report
            preds = trainer.predict(val_ds)
            logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
            y_true = preds.label_ids
            y_pred = np.argmax(logits, axis=1)
            rep = classification_report(
                y_true, y_pred,
                target_names=[id2label[i] for i in range(len(id2label))],
                digits=4
            )
            with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(rep)

            # Final metrics JSON + CSV
            run_end = time.time()
            final = {
                "exp_name": exp_name, "model_name": MODEL_NAME, "model_id": MODEL_ID,
                "threshold": thr, "aug": aug_name, "mode": ft_tag, "lr": lr0, "epochs": EPOCHS,
                "train_samples": len(X_tr), "val_samples": len(X_va), "num_classes": len(kept_species),
                "eval_accuracy": float(eval_metrics.get("eval_accuracy", np.nan)),
                "eval_top3": float(eval_metrics.get("eval_top3", np.nan)),
                "eval_macro_f1": float(eval_metrics.get("eval_macro_f1", np.nan)),
                "eval_kappa": float(eval_metrics.get("eval_kappa", np.nan)),
                "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
                "precision": precision_name,
                "seed": SEED,
                "output_dir": os.path.abspath(exp_dir),
                "runtime_sec": float(run_end - run_start),
            }
            with open(os.path.join(exp_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(final, f, indent=2)
            append_summary_row(SUMMARY_CSV, final)

            print(f"Saved metrics -> {exp_dir}")
            print(f"Appended summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    run()
