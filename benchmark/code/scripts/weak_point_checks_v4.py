"""Two checks on saved predictions for weak points W6 and W10 of the internal review (15 Nov 2026). No training.

W6. The SnakeCLEF MIVS list leaves 16 dangerously venomous Thai species (front-fanged elapids and vipers, and
Rhabdophis keelbacks; list in venom_detail_v4.py) among the "harmless" species. For each venom-alert rule of
post_hoc_decisions.json, split the harmless test images into these 16 species and non-venomous species, and report
how many alerts fall on each: the false-alarm rate among non-venomous images, and the share of the 16 dangerous
species' images flagged.

W10. BioCLIP 2 may have seen some benchmark photographs during pretraining. Video frames were extracted by the
authors from videos and are unlikely to have been posted to public portals as photographs, so the zero-shot accuracy
and the fusion gain are reported separately for still photographs and video frames of the test set (group bootstrap
intervals and clustered McNemar p-values, as in the paper).

Writes paper_experiments/full_species_v4_taxonomy/analysis/{venom_alert_dangerous.csv, fusion_by_source.csv}.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402
import venom_alert_and_tail_fusion_v4 as post  # noqa: E402
from venom_detail_v4 import VENOMOUS_NON_MIVS  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250)
    manifest = pd.read_csv(full.V4 / "manifest.csv", dtype=str, keep_default_na=False)
    labels = sorted(manifest.class_label.unique())
    venom = pd.read_csv(full.RUNS / "species_venom_status.csv", dtype=str, keep_default_na=False)
    status = dict(zip(venom.species, venom.mivs))
    mivs = np.array([status.get(label, "") == "1" for label in labels])
    harmless = np.array([status.get(label, "") == "0" for label in labels])
    dangerous = np.array([label in VENOMOUS_NON_MIVS for label in labels])
    assert dangerous.sum() == 16 and not (dangerous & ~harmless).any()
    decisions = json.loads((full.OUT / "post_hoc_decisions.json").read_text(encoding="utf-8"))

    reference, _ = post.read(post.SYSTEMS["ConvNeXt-V2 384, cross-entropy"] / "test_predictions.csv")
    paths = reference.relative_path.tolist()
    truth = reference.label_id.to_numpy(int)
    groups = reference.leakage_group_id.to_numpy()
    source = manifest.set_index("relative_path").loc[paths, "source_type"].to_numpy()
    samples = list(full.bootstrap_indices(groups, truth, np.random.default_rng(post.SEED + 10), stratified=False))

    # ---------------- W6 ----------------
    true_v, true_d, true_n = mivs[truth], dangerous[truth], harmless[truth] & ~dangerous[truth]
    rows = []
    for name, folder in post.SYSTEMS.items():
        _, p = post.read(folder / "test_predictions.csv", paths)
        score = p[:, mivs].sum(1)
        rules = {"species (top-1 is MIVS)": mivs[p.argmax(1)]}
        for row in decisions["venom_alert"]:
            if row["system"] == name and row["threshold"] is not None:
                rules[row["rule"]] = score >= row["threshold"]
        for rule, flag in rules.items():
            alerts_on_harmless = flag & (true_d | true_n)
            rows.append({"system": name, "rule": rule,
                         "venomous_flagged": float(flag[true_v].mean()),
                         "harmless_flagged": float(flag[true_d | true_n].mean()),
                         "dangerous_non_mivs_images": int(true_d.sum()),
                         "dangerous_non_mivs_flagged": float(flag[true_d].mean()),
                         "non_venomous_images": int(true_n.sum()),
                         "non_venomous_flagged": float(flag[true_n].mean()),
                         "alerts_on_harmless": int(alerts_on_harmless.sum()),
                         "of_which_dangerous_non_mivs": int((flag & true_d).sum())})
    w6 = pd.DataFrame(rows)
    w6.to_csv(full.OUT / "venom_alert_dangerous.csv", index=False)
    print(w6.round(4).to_string(index=False))
    species = pd.Series([labels[t] for t in truth[true_d]]).value_counts()
    print("dangerous non-MIVS test images by species:", species.to_dict())

    # ---------------- W10 ----------------
    zs = post.zero_shot(paths)
    rows = []
    for name in post.FUSED:
        lam = next(r["lambda"] for r in decisions["fusion"] if r["system"] == name)
        _, p = post.read(post.SYSTEMS[name] / "test_predictions.csv", paths)
        ok_b = p.argmax(1) == truth
        ok_f = post.fuse(p, zs, lam).argmax(1) == truth
        ok_z = zs.argmax(1) == truth
        for kind in ("all", "still_image", "video_frame"):
            mask = np.ones(len(truth), bool) if kind == "all" else source == kind
            assert mask.any(), kind
            diffs = [ok_f[i][mask[i]].mean() - ok_b[i][mask[i]].mean() for i in samples]
            rows.append({"system": name, "lambda": lam, "source": kind, "test_images": int(mask.sum()),
                         "groups": int(len(np.unique(groups[mask]))),
                         "zero_shot_accuracy": float(ok_z[mask].mean()), "accuracy_base": float(ok_b[mask].mean()),
                         "accuracy_fused": float(ok_f[mask].mean()),
                         "difference": float(ok_f[mask].mean() - ok_b[mask].mean()),
                         "difference_ci": post.interval(diffs),
                         "clustered_mcnemar_p": full.clustered_mcnemar(ok_f[mask], ok_b[mask], groups[mask])[1]})
    w10 = pd.DataFrame(rows)
    w10.to_csv(full.OUT / "fusion_by_source.csv", index=False)
    print(w10.round(4).to_string(index=False))

    # ---------------- W6b: alert over every dangerously venomous species ----------------
    # The 20 MIVS species, the 16 dangerously venomous species outside the MIVS list, and the two front-fanged
    # species without a SnakeCLEF flag (Calliophis gracilis, Trimeresurus lanna): 38 species. Thresholds are set on
    # out-of-fold development predictions exactly as for the MIVS alert.
    unflagged_venomous = {"Calliophis gracilis", "Trimeresurus lanna"}
    assert all(status.get(s, "") == "" for s in unflagged_venomous)
    all_venomous = mivs | dangerous | np.array([label in unflagged_venomous for label in labels])
    assert all_venomous.sum() == 38
    true_all, true_other = all_venomous[truth], ~all_venomous[truth]
    rows = []
    for name, folder in post.SYSTEMS.items():
        oof, p_oof = post.read(folder / "development_oof_predictions.csv")
        y_oof = oof.label_id.to_numpy(int)
        _, p = post.read(folder / "test_predictions.csv", paths)
        for summed, mask in (("MIVS (20)", mivs), ("all venomous (38)", all_venomous)):
            s_oof, s_test = p_oof[:, mask].sum(1), p[:, mask].sum(1)
            rules = {"species (top-1 in list)": (mask[p.argmax(1)], None)}
            for target in post.TARGETS:
                t = float(np.quantile(s_oof[all_venomous[y_oof]], 1 - target, method="lower"))
                rules[f"OOF sensitivity {int(target * 100)}% (all venomous)"] = (s_test >= t, t)
            for rule, (flag, t) in rules.items():
                boot_s = [flag[i][true_all[i]].mean() for i in samples]
                boot_f = [flag[i][true_other[i]].mean() for i in samples]
                rows.append({"system": name, "summed_species": summed, "rule": rule, "threshold": t,
                             "venomous_images": int(true_all.sum()), "other_images": int(true_other.sum()),
                             "sensitivity_all_venomous": float(flag[true_all].mean()),
                             "sensitivity_ci": post.interval(boot_s),
                             "sensitivity_mivs": float(flag[true_v].mean()),
                             "sensitivity_other_venomous": float(flag[true_all & ~true_v].mean()),
                             "false_alarm_other": float(flag[true_other].mean()),
                             "false_alarm_ci": post.interval(boot_f)})
    w6b = pd.DataFrame(rows)
    w6b.to_csv(full.OUT / "venom_alert_all_venomous.csv", index=False)
    print(w6b.drop(columns=["sensitivity_ci", "false_alarm_ci"]).round(4).to_string(index=False))

    # ---------------- F2: report-level accuracy ----------------
    # Probabilities averaged over all test images of one report (spreadsheet row), as SnakeCLEF scores observations.
    rows_id = reference.row_id.astype(str).to_numpy()
    report = pd.Series(np.arange(len(rows_id))).groupby(rows_id)
    first = report.first().to_numpy()
    rows = []
    systems = {name: post.read(folder / "test_predictions.csv", paths)[1] for name, folder in post.SYSTEMS.items()}
    for name in post.FUSED:
        lam = next(r["lambda"] for r in decisions["fusion"] if r["system"] == name)
        systems[f"{name} + zero-shot"] = post.fuse(systems[name], zs, lam)
    systems["BioCLIP 2 zero-shot"] = zs
    for name, p in systems.items():
        mean = np.vstack([p[idx].mean(0) for idx in report.indices.values()])
        y = truth[[idx[0] for idx in report.indices.values()]]
        assert all((truth[idx] == truth[idx[0]]).all() for idx in report.indices.values())
        rows.append({"system": name, "images": len(truth), "reports": len(y),
                     "image_accuracy": float((p.argmax(1) == truth).mean()),
                     "report_accuracy": float((mean.argmax(1) == y).mean())})
    f2 = pd.DataFrame(rows)
    f2.to_csv(full.OUT / "report_level_accuracy.csv", index=False)
    print(f2.round(4).to_string(index=False), len(first))


if __name__ == "__main__":
    main()
