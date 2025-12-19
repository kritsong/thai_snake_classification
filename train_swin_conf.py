# retrain_swin_best_and_confusion.py
# ----------------------------------------------------------------------
# Retrains the BEST Swin-Base configuration(s) from your summary table,
# saves the full model & processor, and writes metrics + confusion matrix.
#
# Defaults to Threshold=500, aug='none', mode='fineT'.
# You can run multiple thresholds by editing RUN_THRESHOLDS.
#
# Outputs per run (exp_dir):
#   - config.json, pytorch_model.bin               (model)
#   - preprocessor_config.json                     (processor)
#   - label2id.json, id2label.json
#   - history.json, metrics.json, final_metrics.json
#   - classification_report.txt
#   - confusion_matrix_100.csv
#   - confusion_matrix.png, confusion_matrix_normalized.png
#   - predictions.csv  (filepath, true, pred, correct, top1_prob, top3)
#
# Tested with: torch 2.x, torchvision 0.15+, transformers 4.3x–4.5x (Windows)
# ----------------------------------------------------------------------

import os, json, time, random, csv, math
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    top_k_accuracy_score, classification_report, confusion_matrix
)

import matplotlib.pyplot as plt

from transformers import (
    AutoImageProcessor,
    SwinForImageClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    DefaultDataCollator,
    set_seed
)
from transformers.trainer_utils import EvalPrediction


# ====================== USER CONFIG ======================

DATA_DIR = r"C:\Users\ADMIN\Downloads\snake_datasetNew"

# Base HF model
MODEL_ID   = "microsoft/swin-base-patch4-window7-224"
MODEL_NAME = "Swin-Base"

# Where to save runs (each run gets a subfolder like Swin-Base_augnone_fineT_thres500)
OUTPUT_BASE_DIR = os.path.join(os.getcwd(), "transformer_experiment_results_swin_exact")

# Which thresholds to retrain. Default is the *best overall* you reported.
# You can set RUN_THRESHOLDS = [100, 200, 300, 400, 500] to retrain all best-Swin combos.
RUN_THRESHOLDS = [400, 300, 200, 100]

# Train schedule
EPOCHS = 50
VAL_FRACTION = 0.20
SEED = 42
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_WORKERS = 8
PERSISTENT_WORKERS = True
USE_TORCH_COMPILE = False  # keep False on Windows

# Aug settings for Swin bests (from your table: Swin used 'none' for all thresholds)
AUG_FOR_SWIN = "none"

# CSV summary of these retrains
SUMMARY_CSV = os.path.join(OUTPUT_BASE_DIR, "experiments_summary_swin_best.csv")

# Image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ====================== Best Swin mapping (from your table) ======================

# For Swin-Base best per threshold:
# 100 -> none + frozen
# 200 -> none + frozen
# 300 -> none + fineT
# 400 -> none + fineT
# 500 -> none + fineT
BEST_SWINS = {
    100: dict(aug="none", mode="frozen", lr=1e-3),
    200: dict(aug="none", mode="frozen", lr=1e-3),
    300: dict(aug="none", mode="fineT",  lr=2e-5),
    400: dict(aug="none", mode="fineT",  lr=2e-5),
    500: dict(aug="none", mode="fineT",  lr=2e-5),
}


# ====================== Data helpers ======================

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

def scan_dataset(root: str):
    species_to_files: Dict[str, List[str]] = defaultdict(list)
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
                X.append(p); y_names.append(sp)
    return X, y_names, kept_species

def stratified_split(X, y_names, val_frac: float, seed: int):
    y = np.array(y_names); X = np.array(X)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(sss.split(np.zeros_like(y), y))
    return X[tr].tolist(), y[tr].tolist(), X[va].tolist(), y[va].tolist()


# ====================== Dataset ======================

class ImgDataset(Dataset):
    def __init__(self, paths, labels, aug_params: dict, processor, resize_to=224):
        self.paths = paths
        self.labels = labels
        self.processor = processor
        self.resize_to = int(resize_to)
        self.aug = self._build_aug_pipeline(aug_params)

    def _build_aug_pipeline(self, aug_params: dict):
        ops = [transforms.Resize((self.resize_to, self.resize_to), interpolation=InterpolationMode.BILINEAR)]
        # Swin best uses 'none' aug; pipeline remains just Resize.
        # (Kept structure for future toggles)
        return transforms.Compose(ops)

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]; y = int(self.labels[idx])
        img = Image.open(p).convert("RGB"); img = self.aug(img)
        proc = self.processor(images=img, return_tensors="pt", do_resize=False, do_center_crop=False)
        px = proc["pixel_values"].squeeze(0)  # [3, H, W]
        return {"pixel_values": px, "labels": y, "path": p}


# ====================== Metrics ======================

def compute_metrics_builder(id2label: Dict[int, str]):
    def softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)

    def fn(eval_pred: EvalPrediction):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, (tuple, list)) else eval_pred.predictions
        labels = eval_pred.label_ids
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


# ====================== Helpers ======================

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


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], out_dir: str, prefix: str = "confusion_matrix"):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    # Save CSV
    csv_fp = os.path.join(out_dir, f"{prefix}.csv")
    with open(csv_fp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())

    # Plot helper
    def _plot_cm(matrix: np.ndarray, normalize: bool, fname: str, cmap="Blues"):
        if normalize:
            with np.errstate(invalid="ignore"):
                matrix = matrix.astype(np.float64)
                row_sums = matrix.sum(axis=1, keepdims=True) + 1e-12
                matrix = matrix / row_sums
        n = len(class_names)
        fig_h = max(6, min(20, 0.35 * n))   # scale fig size with classes
        fig_w = fig_h
        plt.figure(figsize=(fig_w, fig_h))
        plt.imshow(matrix, interpolation="nearest", cmap=cmap)
        plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
        plt.colorbar(fraction=0.046, pad=0.04)
        tick_marks = np.arange(n)
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=6)
        plt.yticks(tick_marks, class_names, fontsize=6)

        # annotate values sparsely if not too many classes
        if n <= 50:
            fmt = ".2f" if normalize else "d"
            thresh = (matrix.max() + matrix.min()) / 2.0
            for i in range(n):
                for j in range(n):
                    val = matrix[i, j]
                    if (not normalize and val == 0):
                        continue
                    plt.text(j, i, format(val, fmt),
                             ha="center", va="center",
                             color="white" if val > thresh else "black",
                             fontsize=6)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        out_fp = os.path.join(out_dir, fname)
        plt.savefig(out_fp, dpi=300)
        plt.close()

    # Save figures
    _plot_cm(cm, normalize=False, fname=f"{prefix}.png")
    _plot_cm(cm, normalize=True,  fname=f"{prefix}_normalized.png")


def append_summary_row(csv_path: str, row: dict, fields: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


SUMMARY_FIELDS = [
    "exp_name","model_name","model_id","threshold","aug","mode","lr","epochs",
    "train_samples","val_samples","num_classes","eval_accuracy","eval_top3",
    "eval_macro_f1","eval_kappa","eval_loss","precision","seed","output_dir","runtime_sec"
]


# ====================== Runner ======================

def run_one_threshold(thr: int):
    if thr not in BEST_SWINS:
        raise RuntimeError(f"No best-Swin setting registered for threshold={thr}.")
    cfg = BEST_SWINS[thr]
    aug = cfg["aug"]; mode = cfg["mode"]; lr0 = float(cfg["lr"])

    # Compose experiment directory name consistent with your previous naming
    exp_name = f"{MODEL_NAME.replace(' ', '')}_aug{aug}{'_' + mode}_thres{thr}"
    exp_dir  = os.path.join(OUTPUT_BASE_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    set_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    print(f"\n=== Retrain Swin best @ threshold={thr} | aug={aug} | mode={mode} | lr={lr0} ===")
    print(f"Output: {exp_dir}")

    # Torch perf
    print(f"Torch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # Data
    species_to_files = scan_dataset(DATA_DIR)
    X, y_names, kept_species = filter_by_threshold(species_to_files, thr)
    if not X:
        raise RuntimeError(f"No species meet threshold={thr} in {DATA_DIR}.")

    label2id = {name: i for i, name in enumerate(kept_species)}
    id2label = {i: name for name, i in label2id.items()}

    X_tr, ytr_names, X_va, yva_names = stratified_split(X, y_names, VAL_FRACTION, SEED)
    y_tr = [label2id[n] for n in ytr_names]; y_va = [label2id[n] for n in yva_names]

    # Precision prefs
    precision_kwargs = {}; precision_name = "cpu"
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            precision_kwargs["bf16"] = True; precision_name = "bf16"
        else:
            precision_kwargs["fp16"] = True; precision_name = "fp16"

    # Processor
    processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    train_ds = ImgDataset(X_tr, y_tr, {}, processor, resize_to=224)
    val_ds   = ImgDataset(X_va, y_va, {}, processor, resize_to=224)

    # Model
    if mode == "fineT":
        model = SwinForImageClassification.from_pretrained(
            MODEL_ID,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=(torch.bfloat16 if precision_kwargs.get("bf16")
                         else (torch.float16 if precision_kwargs.get("fp16") else None)),
            ignore_mismatched_sizes=True
        )
        set_backbone_trainable(model, True)
    elif mode == "frozen":
        model = SwinForImageClassification.from_pretrained(
            MODEL_ID,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            torch_dtype=(torch.bfloat16 if precision_kwargs.get("bf16")
                         else (torch.float16 if precision_kwargs.get("fp16") else None)),
            ignore_mismatched_sizes=True
        )
        set_backbone_trainable(model, False)
    else:
        raise ValueError(f"Unknown mode={mode}")

    trp, tot = count_trainable(model)
    print(f"Trainable params: {trp:,} / {tot:,}")

    # Training args — no checkpoint saving (we save final model explicitly)
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
        warmup_ratio=0.10,
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
    )

    # Save label maps early
    with open(os.path.join(exp_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, indent=2, ensure_ascii=False)
    with open(os.path.join(exp_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2, ensure_ascii=False)

    # Train
    t0 = time.time()
    trainer.train()

    # Logs & metrics
    with open(os.path.join(exp_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    eval_metrics = trainer.evaluate()
    with open(os.path.join(exp_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in eval_metrics.items() if isinstance(v, (int, float))}, f, indent=2)

    # Predictions on val for confusion matrix
    preds = trainer.predict(val_ds)
    logits = preds.predictions[0] if isinstance(preds.predictions, tuple) else preds.predictions
    y_true = preds.label_ids
    y_pred = np.argmax(logits, axis=1)

    # Save classification report
    rep = classification_report(
        y_true, y_pred,
        target_names=[id2label[i] for i in range(len(id2label))],
        digits=4
    )
    with open(os.path.join(exp_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)

    # Save confusion matrix (CSV + PNGs)
    class_names = [id2label[i] for i in range(len(id2label))]
    save_confusion(y_true, y_pred, class_names, out_dir=exp_dir, prefix="confusion_matrix")

    # Save predictions CSV (with top-1 prob and top-3)
    def softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=1, keepdims=True)
    probs = softmax_np(logits)
    # We didn't pass file paths through Trainer; re-collect from val_ds
    val_paths = [val_ds.paths[i] for i in range(len(val_ds))]
    top1_prob = probs[np.arange(len(y_pred)), y_pred]
    top3_idx = np.argsort(-probs, axis=1)[:, :3]
    pred_rows = []
    for i in range(len(y_pred)):
        pred_rows.append([
            val_paths[i],
            id2label[int(y_true[i])],
            id2label[int(y_pred[i])],
            int(y_true[i]) == int(y_pred[i]),
            float(top1_prob[i]),
            ";".join(id2label[int(k)] for k in top3_idx[i])
        ])
    with open(os.path.join(exp_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath","true","pred","correct","top1_prob","top3_labels"])
        w.writerows(pred_rows)

    # ---- SAVE MODEL + PROCESSOR (final) ----
    trainer.model.save_pretrained(exp_dir)
    processor.save_pretrained(exp_dir)

    # Final metrics JSON + summary row
    final = {
        "exp_name": os.path.basename(exp_dir),
        "model_name": MODEL_NAME, "model_id": MODEL_ID,
        "threshold": thr, "aug": aug, "mode": mode, "lr": lr0, "epochs": EPOCHS,
        "train_samples": len(train_ds), "val_samples": len(val_ds), "num_classes": len(id2label),
        "eval_accuracy": float(eval_metrics.get("eval_accuracy", np.nan)),
        "eval_top3": float(eval_metrics.get("eval_top3", np.nan)),
        "eval_macro_f1": float(eval_metrics.get("eval_macro_f1", np.nan)),
        "eval_kappa": float(eval_metrics.get("eval_kappa", np.nan)),
        "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
        "precision": "bf16" if precision_kwargs.get("bf16") else ("fp16" if precision_kwargs.get("fp16") else "fp32"),
        "seed": SEED,
        "output_dir": os.path.abspath(exp_dir),
        "runtime_sec": float(time.time() - t0),
    }
    with open(os.path.join(exp_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    append_summary_row(SUMMARY_CSV, final, SUMMARY_FIELDS)

    print("\nSaved model + processor + metrics + confusion matrix to:", exp_dir)
    return exp_dir


def main():
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    for thr in RUN_THRESHOLDS:
        try:
            run_one_threshold(thr)
        except Exception as e:
            print(f"[ERROR] Threshold {thr}: {e}")


if __name__ == "__main__":
    main()
