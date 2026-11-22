"""Post-hoc location prior on saved predictions (20 Nov 2026), in the spirit of the location metadata used by the
winning SnakeCLEF solutions. No training and no inference: only saved probabilities and the province of each report.

For an image from province l, the species probabilities p(y|x) of a trained model are reweighted by how much more
common each species is in that province than overall, among development reports:
    p(y|x, l) proportional to p(y|x) * (q(y|l) / pi(y)) ** beta,
    pi(y) = (n_y + 1) / (N + K),      q(y|l) = (n_ly + alpha * pi(y)) / (n_l + alpha),
where n counts development reports (not images, so a video does not count ten times), K = 139 and alpha shrinks
provinces with few reports toward the national frequency. Images without a province keep p(y|x).
alpha (1, 3, 10, 30, 100) and beta (0 to 1.5 in steps of 0.1) are chosen to maximize out-of-fold development
accuracy. For the out-of-fold images of fold f the counts come from the other four folds only, so no image informs
its own prior. The test images use counts from all development reports and are scored once.
Systems: the five systems of the venom-alert analysis, and the two 384-pixel models fused with BioCLIP 2 zero-shot
predictions (weights from post_hoc_decisions.json).
Inputs: paper_experiments/source_post_metadata_audit/row_province_private.csv (scripts/extract_row_province_v4.py).
Outputs (aggregate only, no locations): analysis/location_prior.csv, analysis/location_prior_bands.csv,
analysis/location_prior_decisions.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402
from venom_alert_and_tail_fusion_v4 import SYSTEMS, fuse, read  # noqa: E402

RUNS, OUT = full.RUNS, full.OUT
PROVINCE = full.ROOT / "paper_experiments" / "source_post_metadata_audit" / "row_province_private.csv"
ALPHAS = (1, 3, 10, 30, 100)
BETAS = np.round(np.arange(0, 1.5001, 0.1), 1)
SEED = full.SEED + 7000


def prior_table(rows, alpha, k):
    """rows: DataFrame of development reports with label_id and province. Returns {province: log ratio vector}."""
    n_y = np.bincount(rows.label_id, minlength=k).astype(float)
    pi = (n_y + 1) / (n_y.sum() + k)
    table = {}
    for province, part in rows.dropna(subset=["province"]).groupby("province"):
        n_ly = np.bincount(part.label_id, minlength=k).astype(float)
        q = (n_ly + alpha * pi) / (n_ly.sum() + alpha)
        table[province] = np.log(q / pi)
    return table


def adjust(probs, provinces, table, beta):
    logp = np.log(np.clip(probs, 1e-12, None))
    if beta:
        for i, province in enumerate(provinces):
            if province in table:
                logp[i] += beta * table[province]
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    manifest = pd.read_csv(full.V4 / "manifest.csv", dtype=str, keep_default_na=False)
    province = pd.read_csv(PROVINCE, dtype={"row_id": str}, keep_default_na=False).replace("", np.nan)
    row_province = dict(zip(province.row_id, province.province))
    reference, _ = read(SYSTEMS["ConvNeXt-V2 384, cross-entropy"] / "test_predictions.csv")
    labels = pd.Series(sorted(manifest.class_label.unique()))
    k = len(labels)
    assert k == 139
    label_of = {name: int(i) for i, name in labels.items()}
    assert all(label_of[c] == int(i) for c, i in zip(reference.class_label, reference.label_id))
    manifest["label_id"] = manifest.class_label.map(label_of)
    manifest["province"] = manifest.row_id.map(row_province)
    dev_rows = manifest[manifest.split == "development"].drop_duplicates("row_id")[["row_id", "cv_fold", "label_id", "province"]]
    fold_of = dict(zip(manifest.relative_path, manifest.cv_fold))

    truth = reference.label_id.to_numpy(int)
    groups = reference.leakage_group_id.to_numpy()
    test_prov = reference.row_id.astype(str).map(row_province).to_numpy(object)
    has_prov = pd.notna(test_prov)
    poster = dict(zip(province.row_id, province.poster_key_sha256.fillna("")))
    dev_posters = {poster[r] for r in dev_rows.row_id} - {""}
    disjoint = ~reference.row_id.astype(str).map(poster).isin(dev_posters).to_numpy()
    assert disjoint.sum() == 2606, disjoint.sum()  # as in benchmark_poster_overlap_sensitivity.csv
    tested = sorted(set(truth))
    dev_counts = manifest[manifest.split != "test"].class_label.value_counts().reindex(labels.values, fill_value=0).to_numpy()
    band_of = np.select([dev_counts[truth] >= lo for lo, _, _ in full.BANDS], [b for _, _, b in full.BANDS], default="")
    venom = pd.read_csv(RUNS / "species_venom_status.csv").set_index("species").mivs.reindex(labels.values)
    mivs, harmless = (venom == 1).to_numpy(), (venom == 0).to_numpy()
    samples = list(full.bootstrap_indices(groups, truth, np.random.default_rng(SEED), stratified=False))

    decisions = json.loads((OUT / "post_hoc_decisions.json").read_text(encoding="utf-8"))
    lam = {d["system"]: d["lambda"] for d in decisions["fusion"]}
    _, zs_test = read(RUNS / "bioclip2" / "zero_shot_test_predictions.csv", reference.relative_path.tolist())

    systems = [(name, folder, None) for name, folder in SYSTEMS.items()]
    systems += [(f"{name} + zero-shot", SYSTEMS[name], lam[name]) for name in lam]
    rows, band_rows, report = [], [], {"alphas": list(ALPHAS), "betas": BETAS.tolist(), "systems": {}}
    full_table = {a: prior_table(dev_rows, a, k) for a in ALPHAS}
    fold_tables = {(f, a): prior_table(dev_rows[dev_rows.cv_fold != f], a, k)
                   for f in sorted(dev_rows.cv_fold.unique()) for a in ALPHAS}
    for name, folder, fusion in systems:
        oof, p_oof = read(folder / "development_oof_predictions.csv")
        _, p_test = read(folder / "test_predictions.csv", reference.relative_path.tolist())
        if fusion is not None:
            _, zs_oof = read(RUNS / "bioclip2" / "zero_shot_development_predictions.csv", oof.relative_path.tolist())
            p_oof, p_test = fuse(p_oof, zs_oof, fusion), fuse(p_test, zs_test, fusion)
        y_oof = oof.label_id.to_numpy(int)
        oof_prov = oof.row_id.astype(str).map(row_province).to_numpy(object)
        oof_fold = oof.relative_path.map(fold_of).to_numpy()
        grid = {}
        for a in ALPHAS:
            for b in BETAS:
                correct = np.zeros(len(oof), bool)
                for f in np.unique(oof_fold):
                    m = oof_fold == f
                    correct[m] = adjust(p_oof[m], oof_prov[m], fold_tables[(f, a)], b).argmax(1) == y_oof[m]
                grid[(a, float(b))] = float(correct.mean())
        best = max(grid.values())
        alpha, beta = min((key for key, v in grid.items() if v == best), key=lambda key: (key[1], -key[0]))
        p_adj = adjust(p_test, test_prov, full_table[alpha], beta)
        base, adj = p_test.argmax(1), p_adj.argmax(1)
        c_base, c_adj = base == truth, adj == truth
        diffs = [c_adj[s].mean() - c_base[s].mean() for s in samples]
        _, p_value = full.clustered_mcnemar(c_adj, c_base, groups)

        def top5(p):
            return float((np.argsort(-p, axis=1)[:, :5] == truth[:, None]).any(1).mean())

        def venom_miss(pred):
            v = mivs[truth]
            return float(harmless[pred][v].mean())

        rows.append({"system": name, "alpha": alpha, "beta": beta,
                     "oof_accuracy_base": grid[(alpha, 0.0)], "oof_accuracy_adjusted": best,
                     "test_accuracy_base": float(c_base.mean()), "test_accuracy_adjusted": float(c_adj.mean()),
                     "accuracy_difference": float(c_adj.mean() - c_base.mean()), "accuracy_difference_ci": interval(diffs),
                     "clustered_mcnemar_p": p_value,
                     "balanced_accuracy_base": balanced_accuracy_score(truth, base),
                     "balanced_accuracy_adjusted": balanced_accuracy_score(truth, adj),
                     "macro_f1_base": f1_score(truth, base, labels=tested, average="macro", zero_division=0),
                     "macro_f1_adjusted": f1_score(truth, adj, labels=tested, average="macro", zero_division=0),
                     "top5_base": top5(p_test), "top5_adjusted": top5(p_adj),
                     "mivs_as_non_mivs_base": venom_miss(base), "mivs_as_non_mivs_adjusted": venom_miss(adj),
                     "test_images_with_province": int(has_prov.sum()),
                     "accuracy_with_province_base": float(c_base[has_prov].mean()),
                     "accuracy_with_province_adjusted": float(c_adj[has_prov].mean()),
                     "poster_disjoint_images": int(disjoint.sum()),
                     "accuracy_poster_disjoint_base": float(c_base[disjoint].mean()),
                     "accuracy_poster_disjoint_adjusted": float(c_adj[disjoint].mean()),
                     "poster_disjoint_mcnemar_p": full.clustered_mcnemar(c_adj[disjoint], c_base[disjoint],
                                                                         groups[disjoint])[1]})
        for _, _, band in full.BANDS:
            m = band_of == band
            band_rows.append({"system": name, "band": band, "test_images": int(m.sum()),
                              "accuracy_base": float(c_base[m].mean()), "accuracy_adjusted": float(c_adj[m].mean())})
        report["systems"][name] = {"alpha": alpha, "beta": beta, "fusion_lambda": fusion,
                                   "oof_grid": {f"{a}|{b}": v for (a, b), v in grid.items()}}
        print(f"{name:45s} alpha={alpha:<4} beta={beta:.1f} acc {c_base.mean():.4f} -> {c_adj.mean():.4f} "
              f"(diff {c_adj.mean() - c_base.mean():+.4f}, p={p_value:.3g})")

    report.update({"development_reports": int(len(dev_rows)),
                   "development_reports_with_province": int(dev_rows.province.notna().sum()),
                   "test_images": int(len(truth)), "test_images_with_province": int(has_prov.sum()),
                   "provinces": int(dev_rows.province.nunique())})
    pd.DataFrame(rows).to_csv(OUT / "location_prior.csv", index=False)
    pd.DataFrame(band_rows).to_csv(OUT / "location_prior_bands.csv", index=False)
    (OUT / "location_prior_decisions.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print({k: v for k, v in report.items() if k != "systems"})


if __name__ == "__main__":
    main()
