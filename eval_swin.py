# eval_swin_from_saved_preds.py
# Evaluate Swin runs using saved NPZ predictions (val_predictions_th{threshold}.npz)

import os, re, json, csv, argparse
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    log_loss,
    top_k_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
)

DEFAULT_ROOT = os.path.join(os.getcwd(), "swin_transformer_retrained_best_fixed")
NPZ_NAME_RE  = re.compile(r"val_predictions_th(?P<th>\d+)\.npz$", re.IGNORECASE)

def load_id2label(run_dir):
    p = os.path.join(run_dir, "id2label.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    try:
        max_id = max(int(k) for k in d.keys())
        id2label = [""] * (max_id + 1)
        for k, v in d.items():
            id2label[int(k)] = v
        return id2label
    except Exception:
        return list(d.values())

def safe_topk(probs, y_true, k):
    try:
        if probs.shape[1] < k:
            return None
        return float(top_k_accuracy_score(y_true, probs, k=k))
    except Exception:
        return None

def try_roc_auc(y_true, probs):
    try:
        if probs.shape[1] < 2:
            return None
        return float(roc_auc_score(y_true, probs, multi_class="ovr"))
    except Exception:
        return None

def write_confusion_csv(cm, class_names, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + list(map(int, row)))

def write_per_class_csv(report_dict, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for key, v in report_dict.items():
            if key in ("accuracy", "macro avg", "weighted avg", "micro avg"):
                continue
            if isinstance(v, dict) and {"precision", "recall", "f1-score", "support"} <= v.keys():
                w.writerow([key, f"{v['precision']:.6f}", f"{v['recall']:.6f}",
                            f"{v['f1-score']:.6f}", int(v["support"])])

def write_misclassifications(paths, y_true, y_pred, probs, class_names, out_csv, topn=3):
    if paths is None:
        return
    C = probs.shape[1]
    topn = min(topn, C)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        headers = ["image_path", "true_label", "pred_label", "pred_confidence"]
        for i in range(topn):
            headers += [f"top{i+1}_label", f"top{i+1}_prob"]
        w.writerow(headers)
        top_idx = np.argsort(-probs, axis=1)[:, :topn]
        for i, pth in enumerate(paths):
            row = [
                pth,
                class_names[y_true[i]] if class_names else int(y_true[i]),
                class_names[y_pred[i]] if class_names else int(y_pred[i]),
                float(probs[i, y_pred[i]]),
            ]
            for j in range(topn):
                cls = top_idx[i, j]
                row += [class_names[cls] if class_names else int(cls), float(probs[i, cls])]
            w.writerow(row)

def evaluate_npz(npz_path, run_dir, agg_writer):
    data = np.load(npz_path, allow_pickle=True)
    probs = data["probs"]
    y_true = data["y_true"]
    paths = data["paths"] if "paths" in data.files else None
    y_pred = np.argmax(probs, axis=1)

    id2label = load_id2label(run_dir)
    if id2label is None:
        id2label = [str(i) for i in range(probs.shape[1])]
    class_names = id2label

    acc = float(accuracy_score(y_true, y_pred))
    pr_macro, rc_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    pr_weight, rc_weight, f1_weight, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    ll = float(log_loss(y_true, probs))
    top3 = safe_topk(probs, y_true, 3)
    top5 = safe_topk(probs, y_true, 5)
    auc_ovr = try_roc_auc(y_true, probs)
    kappa = float(cohen_kappa_score(y_true, y_pred))

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    run_name = os.path.basename(run_dir.rstrip(os.sep))
    m = NPZ_NAME_RE.search(os.path.basename(npz_path))
    th = m.group("th") if m else "NA"
    out_dir = os.path.join(run_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        "run": run_name,
        "threshold": th,
        "num_samples": int(len(y_true)),
        "num_classes": int(probs.shape[1]),
        "accuracy": acc,
        "macro_precision": float(pr_macro),
        "macro_recall": float(rc_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(pr_weight),
        "weighted_recall": float(rc_weight),
        "weighted_f1": float(f1_weight),
        "log_loss": ll,
        "top3_acc": top3,
        "top5_acc": top5,
        "roc_auc_ovr": auc_ovr,
        "cohen_kappa": kappa,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    write_per_class_csv(report, os.path.join(out_dir, "per_class_metrics.csv"))
    write_confusion_csv(cm, class_names, os.path.join(out_dir, "confusion_matrix_100.csv"))
    write_misclassifications(paths, y_true, y_pred, probs, class_names, os.path.join(out_dir, "misclassifications.csv"))

    agg_writer.writerow([
        run_name, th, len(y_true), probs.shape[1],
        f"{acc:.6f}", f"{f1_macro:.6f}", f"{f1_weight:.6f}", f"{ll:.6f}",
        f"{top3:.6f}" if top3 is not None else "",
        f"{top5:.6f}" if top5 is not None else "",
        f"{auc_ovr:.6f}" if auc_ovr is not None else "",
        f"{kappa:.6f}",
        os.path.relpath(out_dir, start=os.path.dirname(os.path.abspath(out_dir)))
    ])

    print(f"[OK] {run_name} (th={th})  acc={acc:.4f} top3={top3:.4f} kappa={kappa:.4f}  macroF1={f1_macro:.4f}  saved -> {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Evaluate Swin runs from saved NPZ predictions.")
    ap.add_argument("--root", type=str, default=DEFAULT_ROOT,
                    help="Root folder containing retrained runs (default: ./swin_transformer_retrained_best)")
    args = ap.parse_args()

    root = args.root
    if not os.path.isdir(root):
        raise SystemExit(f"Root not found: {root}")

    run_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    npz_files = []
    for rd in run_dirs:
        for f in os.listdir(rd):
            if NPZ_NAME_RE.match(f):
                npz_files.append((os.path.join(rd, f), rd))

    if not npz_files:
        raise SystemExit("No val_predictions_th*.npz files found under the root.")

    agg_path = os.path.join(root, "aggregate_eval_summary.csv")
    with open(agg_path, "w", newline="", encoding="utf-8") as agg_f:
        agg_w = csv.writer(agg_f)
        agg_w.writerow([
            "run", "threshold", "num_samples", "num_classes",
            "accuracy", "macro_f1", "weighted_f1", "log_loss",
            "top3_acc", "top5_acc", "roc_auc_ovr", "cohen_kappa",
            "eval_dir_rel"
        ])
        for npz_path, run_dir in sorted(npz_files):
            evaluate_npz(npz_path, run_dir, agg_w)

    print(f"\n[Done] Aggregate summary -> {agg_path}")

if __name__ == "__main__":
    main()
