"""Benchmark tables for the single-protocol manuscript (reads saved predictions and analysis outputs only).

The manuscript reports one training protocol: the SnakeCLEF-style recipe at 224 px for all nine
architectures, the winning SnakeCLEF configuration (ConvNeXt-V2 at 384 px, cross-entropy and Seesaw),
and BioCLIP 2 without training on benchmark images as a reference. This script selects those twelve
systems from the full analysis (scripts/analyze_full_species_v4.py), recomputes the paired tests with
Holm adjustment over the comparisons reported in the manuscript only, and writes benchmark_*.csv next
to the full analysis outputs. The other recipes stay in the full analysis files.
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402

ROOT = full.ROOT
A = full.OUT
EXTERNAL = full.RUNS / "external_snakeclef"

# Analysis name -> manuscript label, in table order.
SYSTEMS = {
    "MobileNetV3-Small (strong)": "MobileNetV3-Small",
    "EfficientNet-B0 (strong)": "EfficientNet-B0",
    "ResNet-50 (strong)": "ResNet-50",
    "ViT-Base/16 (strong)": "ViT-Base/16",
    "DeiT-Base/16 (strong)": "DeiT-Base/16",
    "Swin-Base (strong)": "Swin-Base",
    "ConvNeXt-V2-B 224, cross-entropy": "ConvNeXt-V2-Base",
    "DINOv2-Base/14 (strong)": "DINOv2 ViT-B/14",
    "BioCLIP-2 (strong)": "BioCLIP 2 ViT-L/14",
    "ConvNeXt-V2-B 384, cross-entropy": "ConvNeXt-V2-Base, 384 px",
    "ConvNeXt-V2-B 384, seesaw": "ConvNeXt-V2-Base, 384 px, Seesaw",
    "BioCLIP 2 zero-shot": "BioCLIP 2 zero-shot",
}
# CV rows (model, augmentation) of cv_configurations.csv for each trained system.
CV_KEYS = {name: (name.split(" (")[0], "strong") for name in SYSTEMS if name.endswith("(strong)")}
CV_KEYS.update({"ConvNeXt-V2-B 224, cross-entropy": ("ConvNeXt-V2-B 224", "snakeclef_style_ce"),
                "ConvNeXt-V2-B 384, cross-entropy": ("ConvNeXt-V2-B 384", "snakeclef_style_ce"),
                "ConvNeXt-V2-B 384, seesaw": ("ConvNeXt-V2-B 384", "snakeclef_style_seesaw")})
# Seed-repeat system names (scripts/seed_repeats_summary_v4.py) for the systems repeated with three seeds.
SEED_NAMES = {"ConvNeXt-V2-B 224, cross-entropy": "ConvNeXt-V2-B 224, cross-entropy",
              "DINOv2-Base/14 (strong)": "DINOv2-Base/14 (strong recipe)",
              "BioCLIP-2 (strong)": "BioCLIP-2 (strong recipe)",
              "ConvNeXt-V2-B 384, seesaw": "ConvNeXt-V2-B 384, seesaw",
              "ConvNeXt-V2-B 384, cross-entropy": "ConvNeXt-V2-B 384, cross-entropy"}
ADJACENT = [("BioCLIP-2 (strong)", "DINOv2-Base/14 (strong)"),
            ("DINOv2-Base/14 (strong)", "ConvNeXt-V2-B 224, cross-entropy"),
            ("ConvNeXt-V2-B 224, cross-entropy", "Swin-Base (strong)")]


def main():
    manifest = pd.read_csv(full.V4 / "manifest.csv", dtype=str, keep_default_na=False)
    found = {name: path for name, path in full.systems().items() if name in SYSTEMS}
    missing = set(SYSTEMS) - set(found)
    assert not missing, missing
    reference, _ = full.load(found["ConvNeXt-V2-B 384, cross-entropy"])
    probs = {name: full.load(found[name], reference)[1] for name in SYSTEMS}
    truth = reference.label_id.to_numpy(int)
    groups = reference.leakage_group_id.to_numpy()
    correct = {name: p.argmax(1) == truth for name, p in probs.items()}

    # Main table: CV mean/SD, test metrics with intervals, genus accuracy.
    test = pd.read_csv(A / "test_systems.csv").set_index("system")
    cv = pd.read_csv(A / "cv_configurations.csv")
    genus = pd.read_csv(A / "genus_level.csv").set_index("system")
    rows = []
    for name, label in SYSTEMS.items():
        row = {"system": name, "label": label}
        if name in CV_KEYS:
            model, aug = CV_KEYS[name]
            hit = cv[(cv.model == model) & (cv.augmentation == aug)]
            assert len(hit) == 1, name
            row.update(cv_accuracy=hit.cv_accuracy_mean.iloc[0], cv_accuracy_sd=hit.cv_accuracy_sd.iloc[0])
        row.update(test.loc[name].to_dict())
        row["genus_accuracy"] = genus.loc[name, "genus_accuracy"]
        assert abs(row["accuracy"] - correct[name].mean()) < 1e-12
        rows.append(row)
    pd.DataFrame(rows).to_csv(A / "benchmark_systems.csv", index=False)

    # Paired tests: the most accurate system against every other, and adjacent pairs among the four
    # best architectures. Accuracy differences use the shared unstratified group bootstrap.
    best = max(SYSTEMS, key=lambda n: correct[n].mean())
    pairs = [(best, other) for other in SYSTEMS if other != best] + ADJACENT
    samples = list(full.bootstrap_indices(groups, truth, np.random.default_rng(full.SEED), stratified=False))
    out = []
    for a, b in pairs:
        diff = correct[a].astype(float) - correct[b].astype(float)
        draws = np.array([diff[i].mean() for i in samples])
        chi2, p = full.clustered_mcnemar(correct[a], correct[b], groups)
        out.append({"system_a": a, "system_b": b, "diff_accuracy": diff.mean(),
                    "diff_accuracy_ci_low": np.quantile(draws, .025), "diff_accuracy_ci_high": np.quantile(draws, .975),
                    "clustered_mcnemar_chi2": chi2, "clustered_mcnemar_p": p})
    pairs_frame = pd.DataFrame(out)
    pairs_frame["clustered_mcnemar_holm_p"] = full.holm(pairs_frame.clustered_mcnemar_p)
    # Cross-check against the full analysis where the pair exists there (same resamples).
    previous = pd.read_csv(A / "paired_comparisons.csv")
    for row in pairs_frame.itertuples():
        hit = previous[(previous.system_a == row.system_a) & (previous.system_b == row.system_b)]
        if len(hit):
            assert abs(hit.diff_accuracy_ci_low.iloc[0] - row.diff_accuracy_ci_low) < 1e-12, row
            assert abs(hit.clustered_mcnemar_p.iloc[0] - row.clustered_mcnemar_p) < 1e-12, row
    pairs_frame.to_csv(A / "benchmark_paired_comparisons.csv", index=False)

    # Subsets of the band, venom, poster and external tables.
    for source, key in (("frequency_bands.csv", "system"), ("venom_confusions.csv", "system"),
                        ("poster_overlap_sensitivity.csv", "system")):
        frame = pd.read_csv(A / source)
        frame = frame[frame[key].isin(SYSTEMS)].copy()
        frame["label"] = frame[key].map(SYSTEMS)
        frame.to_csv(A / f"benchmark_{source}", index=False)
    external = pd.read_csv(EXTERNAL / "external_systems.csv")
    external = external[external.system.isin(SYSTEMS)].copy()
    external["label"] = external.system.map(SYSTEMS)
    assert len(external) == len(SYSTEMS)
    external.to_csv(A / "benchmark_external_systems.csv", index=False)

    # Seed repeats: the five repeated systems and every pair of them, over all 3 x 3 seed combinations.
    seeds = pd.read_csv(A / "seed_repeats.csv").set_index("system")
    seed_rows, seed_pairs = [], []
    complete = {name: seed_name for name, seed_name in SEED_NAMES.items() if seed_name in seeds.index}
    print("seed repeats incomplete:", sorted(set(SEED_NAMES) - set(complete)))
    for name, seed_name in complete.items():
        r = seeds.loc[seed_name]
        seed_rows.append({"system": name, "label": SYSTEMS[name],
                          **{f"accuracy_seed{k}": r[f"accuracy_seed{k}"] for k in range(3)},
                          "accuracy_mean": r.accuracy_mean, "accuracy_sd": r.accuracy_sd,
                          "balanced_accuracy_mean": r.balanced_accuracy_mean, "balanced_accuracy_sd": r.balanced_accuracy_sd})
    names = list(complete)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for metric in ("accuracy", "balanced_accuracy"):
                ra, rb = seeds.loc[SEED_NAMES[a]], seeds.loc[SEED_NAMES[b]]
                diffs = [ra[f"{metric}_seed{x}"] - rb[f"{metric}_seed{y}"] for x, y in product(range(3), range(3))]
                seed_pairs.append({"system_a": a, "system_b": b, "metric": metric, "min_difference": min(diffs),
                                   "max_difference": max(diffs), "same_sign_all_9": min(diffs) * max(diffs) > 0})
    pd.DataFrame(seed_rows).to_csv(A / "benchmark_seed_repeats.csv", index=False)
    pd.DataFrame(seed_pairs).to_csv(A / "benchmark_seed_pairs.csv", index=False)

    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[["label", "cv_accuracy", "cv_accuracy_sd", "accuracy", "accuracy_ci_low", "accuracy_ci_high",
                              "macro_f1_tested_species", "balanced_accuracy", "top3_accuracy", "genus_accuracy"]].round(4).to_string(index=False))
    print(pairs_frame.round(4).to_string(index=False))
    print(pd.DataFrame(seed_pairs).round(4).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
