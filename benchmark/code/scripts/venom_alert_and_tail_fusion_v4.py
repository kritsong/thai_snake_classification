"""Two post-hoc decision analyses on saved predictions (no training), tuned on development data only.

1. Venom alert. A system flags an image as potentially medically important venomous (MIVS) when the summed
   probability of the 20 MIVS species is at least t, independently of the species it names. Operating points:
   the species rule of the paper (alert when the top species is MIVS), t = 0.5 (no tuning), and the largest t
   whose out-of-fold (OOF) sensitivity on development images reaches 90% and 95%. Each rule is then scored once
   on the locked test set: sensitivity (share of MIVS images flagged), miss rate and false-alarm rate (share of
   harmless images flagged), with 95% group-bootstrap intervals.

2. Zero-shot fusion. Trained-model probabilities are combined with BioCLIP 2 zero-shot probabilities as
   log p = log p_trained + lambda * log p_zero-shot (renormalized). Lambda is chosen on the grid 0, 0.05, ..., 1.5
   to maximize OOF top-1 accuracy over all development images, then the fused model is scored once on the test
   set, overall and by the number of development images of the true species.

Zero-shot development probabilities are recomputed from the cached BioCLIP 2 image features and text weights
(logit scale 100); the script checks that this reproduces the saved zero-shot test predictions, and saves the
development probabilities as bioclip2/zero_shot_development_predictions.csv (same columns as the test file).
Writes paper_experiments/full_species_v4_taxonomy/analysis/{venom_alert*.csv, tail_fusion*.csv,
post_hoc_decisions.json}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402

RUNS = full.RUNS
OUT = full.OUT
SNAKECLEF = RUNS / "snakeclef_style"
SYSTEMS = {
    "ConvNeXt-V2 384, cross-entropy": SNAKECLEF / "convnextv2_base_384_ce",
    "ConvNeXt-V2 384, Seesaw": SNAKECLEF / "convnextv2_base_384_seesaw",
    "BioCLIP 2 (224)": RUNS / "strong" / "BioCLIP-2",
    "DINOv2 (224)": RUNS / "strong" / "DINOv2-Base-14",
    "ConvNeXt-V2 (224)": SNAKECLEF / "convnextv2_base_224_ce",
}
FUSED = ("ConvNeXt-V2 384, cross-entropy", "ConvNeXt-V2 384, Seesaw")
LAMBDAS = np.round(np.arange(0, 1.5001, 0.05), 2)
TARGETS = (0.90, 0.95)
SEED = full.SEED + 5000


def read(path, order=None):
    frame = pd.read_csv(path, keep_default_na=False)
    if order is not None:
        frame = frame.set_index("relative_path").loc[order].reset_index()
    cols = [c for c in frame.columns if c.startswith("probability_")]
    assert len(cols) == 139
    return frame, frame[cols].to_numpy(float)


def zero_shot(paths):
    base = RUNS / "bioclip2"
    order = pd.read_csv(base / "feature_order.csv", keep_default_na=False)
    position = pd.Series(np.arange(len(order)), index=order.relative_path)
    x = np.load(base / "image_features.npy")[position.loc[paths].to_numpy()]
    logits = 100.0 * (x.astype(np.float64) @ np.load(base / "zero_shot_text_weights.npy").astype(np.float64).T)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True)


def save_zero_shot_development(labels):
    """Zero-shot probabilities of every development image, in manifest order, as a prediction table."""
    base = RUNS / "bioclip2"
    meta = pd.read_csv(base / "development_oof_predictions.csv", keep_default_na=False,
                       usecols=["relative_path", "split", "class_label", "label_id", "row_id", "leakage_group_id",
                                "content_sha256"], dtype={"row_id": str, "leakage_group_id": str})
    order = pd.read_csv(base / "feature_order.csv", keep_default_na=False).relative_path
    meta = meta.set_index("relative_path").loc[order[order.isin(meta.relative_path)]].reset_index()
    probs = zero_shot(meta.relative_path.tolist())
    meta["predicted_label"] = [labels[i] for i in probs.argmax(1)]
    table = pd.concat([meta, pd.DataFrame(probs, columns=[f"probability_{i:03d}" for i in range(len(labels))])], axis=1)
    table.to_csv(base / "zero_shot_development_predictions.csv", index=False)
    return len(table), float((probs.argmax(1) == meta.label_id.to_numpy(int)).mean())


def fuse(p_trained, p_zero, lam):
    logp = np.log(np.clip(p_trained, 1e-12, None)) + lam * np.log(np.clip(p_zero, 1e-12, None))
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    manifest = pd.read_csv(full.V4 / "manifest.csv", dtype=str, keep_default_na=False)
    labels = sorted(manifest.class_label.unique())
    assert len(labels) == 139
    venom = pd.read_csv(RUNS / "species_venom_status.csv", dtype=str, keep_default_na=False)
    status = dict(zip(venom.species, venom.mivs))
    mivs = np.array([status.get(label, "") == "1" for label in labels])
    harmless = np.array([status.get(label, "") == "0" for label in labels])
    assert mivs.sum() == 20
    dev_counts = manifest[manifest.split != "test"].class_label.value_counts().reindex(labels, fill_value=0).to_numpy()

    reference, _ = read(SYSTEMS["ConvNeXt-V2 384, cross-entropy"] / "test_predictions.csv")
    test_paths = reference.relative_path.tolist()
    truth = reference.label_id.to_numpy(int)
    groups = reference.leakage_group_id.to_numpy()
    tested = np.unique(truth)

    # Zero-shot probabilities: check the recomputation against the saved test predictions.
    zs_test = zero_shot(test_paths)
    _, saved = read(RUNS / "bioclip2" / "zero_shot_test_predictions.csv", test_paths)
    assert np.abs(zs_test - saved).max() < 1e-4 and (zs_test.argmax(1) == saved.argmax(1)).all()
    print("zero-shot development predictions (images, accuracy):", save_zero_shot_development(labels))

    rng_samples = list(full.bootstrap_indices(groups, truth, np.random.default_rng(SEED), stratified=False))
    strat_samples = list(full.bootstrap_indices(groups, truth, np.random.default_rng(SEED + 1), stratified=True))
    report = {"lambda_grid": LAMBDAS.tolist(), "targets": list(TARGETS), "bootstrap_resamples": full.REPS}

    # ---------------- 1. Venom alert ----------------
    true_v, true_h = mivs[truth], harmless[truth]
    eligible = true_v | true_h
    alert_rows, curve_rows = [], []
    for name, folder in SYSTEMS.items():
        oof, p_oof = read(folder / "development_oof_predictions.csv")
        y_oof = oof.label_id.to_numpy(int)
        s_oof = p_oof[:, mivs].sum(1)
        oof_v, oof_h = mivs[y_oof], harmless[y_oof]
        _, p_test = read(folder / "test_predictions.csv", test_paths)
        s_test = p_test[:, mivs].sum(1)
        species_rule = mivs[p_test.argmax(1)]
        rules = {"species (top-1 is MIVS)": (species_rule, None, float(mivs[p_oof.argmax(1)][oof_v].mean())),
                 "P(MIVS) >= 0.5": (s_test >= 0.5, 0.5, float((s_oof[oof_v] >= 0.5).mean()))}
        for target in TARGETS:
            # Largest threshold whose OOF sensitivity reaches the target.
            t = float(np.quantile(s_oof[oof_v], 1 - target, method="lower"))
            assert (s_oof[oof_v] >= t).mean() >= target
            rules[f"OOF sensitivity {int(target * 100)}%"] = (s_test >= t, t, float((s_oof[oof_v] >= t).mean()))
        for rule, (flag, t, oof_sens) in rules.items():
            sens = flag[true_v].mean()
            fa = flag[true_h].mean()
            boot_s, boot_f = [], []
            for idx in rng_samples:
                v, h = true_v[idx], true_h[idx]
                boot_s.append(flag[idx][v].mean())
                boot_f.append(flag[idx][h].mean())
            alert_rows.append({"system": name, "rule": rule, "threshold": t, "oof_sensitivity": oof_sens,
                               "oof_false_alarm": float(((mivs[p_oof.argmax(1)] if t is None else s_oof >= t))[oof_h].mean()),
                               "test_sensitivity": float(sens), "sensitivity_ci": interval(boot_s),
                               "test_miss_rate": float(1 - sens), "test_false_alarm": float(fa),
                               "false_alarm_ci": interval(boot_f),
                               "venomous_images": int(true_v.sum()), "harmless_images": int(true_h.sum())})
        # ROC area on the test set (MIVS vs harmless images).
        pos, neg = s_test[true_v], s_test[true_h]
        ranks = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
        report.setdefault("venom_auc", {})[name] = float(auc)
        for t in np.round(np.arange(0.01, 1.0, 0.01), 2):
            curve_rows.append({"system": name, "threshold": t, "test_sensitivity": float((s_test[true_v] >= t).mean()),
                               "test_false_alarm": float((s_test[true_h] >= t).mean()),
                               "oof_sensitivity": float((s_oof[oof_v] >= t).mean()),
                               "oof_false_alarm": float((s_oof[oof_h] >= t).mean())})
    alerts = pd.DataFrame(alert_rows)
    alerts.to_csv(OUT / "venom_alert.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(OUT / "venom_alert_curve.csv", index=False)
    print(alerts.drop(columns=["venomous_images", "harmless_images"]).round(4).to_string(index=False))
    print("venom AUC:", {k: round(v, 4) for k, v in report["venom_auc"].items()})
    assert eligible.sum() == true_v.sum() + true_h.sum()

    # ---------------- 2. Zero-shot fusion ----------------
    band_of = np.select([dev_counts[truth] >= lo for lo, _, _ in full.BANDS], [b for _, _, b in full.BANDS], default="")
    fusion_rows, band_rows = [], []
    for name in FUSED:
        folder = SYSTEMS[name]
        oof, p_oof = read(folder / "development_oof_predictions.csv")
        y_oof = oof.label_id.to_numpy(int)
        zs_oof = zero_shot(oof.relative_path.tolist())
        oof_acc = [float((fuse(p_oof, zs_oof, lam).argmax(1) == y_oof).mean()) for lam in LAMBDAS]
        lam = float(LAMBDAS[int(np.argmax(oof_acc))])
        report.setdefault("fusion_oof_accuracy", {})[name] = dict(zip(map(str, LAMBDAS), oof_acc))
        _, p_test = read(folder / "test_predictions.csv", test_paths)
        p_fused = fuse(p_test, zs_test, lam)
        base_m, fused_m = full.metrics(truth, p_test, tested), full.metrics(truth, p_fused, tested)
        ok_b, ok_f = p_test.argmax(1) == truth, p_fused.argmax(1) == truth
        diffs = [ok_f[idx].mean() - ok_b[idx].mean() for idx in rng_samples]
        bal = []
        for idx in strat_samples:
            y = truth[idx]
            bal.append(full.balanced_accuracy_score(y, p_fused[idx].argmax(1))
                       - full.balanced_accuracy_score(y, p_test[idx].argmax(1)))
        vh_b = harmless[p_test.argmax(1)][true_v].mean()
        vh_f = harmless[p_fused.argmax(1)][true_v].mean()
        fusion_rows.append({"system": name, "lambda": lam, "oof_accuracy_base": oof_acc[0],
                            "oof_accuracy_fused": max(oof_acc),
                            **{f"test_{k}_base": v for k, v in base_m.items()},
                            **{f"test_{k}_fused": v for k, v in fused_m.items()},
                            "accuracy_difference": float(ok_f.mean() - ok_b.mean()),
                            "accuracy_difference_ci": interval(diffs),
                            "accuracy_clustered_mcnemar_p": full.clustered_mcnemar(ok_f, ok_b, groups)[1],
                            "balanced_accuracy_difference_ci": interval(bal),
                            "venom_to_harmless_base": float(vh_b), "venom_to_harmless_fused": float(vh_f)})
        for _, _, band in full.BANDS:
            mask = band_of == band
            d = [ok_f[idx][mask[idx]].mean() - ok_b[idx][mask[idx]].mean() for idx in rng_samples]
            band_rows.append({"system": name, "band": band, "test_images": int(mask.sum()),
                              "species": int(len(np.unique(truth[mask]))),
                              "accuracy_base": float(ok_b[mask].mean()), "accuracy_fused": float(ok_f[mask].mean()),
                              "difference": float(ok_f[mask].mean() - ok_b[mask].mean()), "difference_ci": interval(d),
                              "only_fused_correct": int((ok_f & ~ok_b & mask).sum()),
                              "only_base_correct": int((ok_b & ~ok_f & mask).sum()),
                              "clustered_mcnemar_p": full.clustered_mcnemar(ok_f[mask], ok_b[mask], groups[mask])[1]})
    fusion = pd.DataFrame(fusion_rows)
    bands = pd.DataFrame(band_rows)
    fusion.to_csv(OUT / "tail_fusion.csv", index=False)
    bands.to_csv(OUT / "tail_fusion_bands.csv", index=False)
    pd.set_option("display.width", 250)
    print(fusion.T.to_string())
    print(bands.round(4).to_string(index=False))
    report["venom_alert"] = alert_rows
    report["fusion"] = fusion_rows
    report["fusion_bands"] = band_rows
    (OUT / "post_hoc_decisions.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
