"""Training-seed variation of the key systems (original final refit + two seed repeats).

Reads saved test predictions only. For each system: test accuracy, balanced
accuracy and macro-F1 (species with test images) per seed, with mean, sample SD
and range. For the comparisons the paper relies on: the difference for all 3 x 3
seed combinations and whether its sign is the same in all of them.
Writes analysis/seed_repeats.csv and analysis/seed_repeat_pairs.csv.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PE = ROOT / "paper_experiments"
RUNS = PE / "full_species_v4_taxonomy"
ADDED, BIO = RUNS / "grid_added", RUNS / "grid_bioclip2"
ORIGINAL = {
    "Swin-Base": RUNS / "full_species/Swin-Base/fine_tune/low",
    "ConvNeXt-V2-Base": ADDED / "full_species/ConvNeXt-V2-Base/fine_tune/low",
    "DINOv2-Base/14": ADDED / "full_species/DINOv2-Base-14/fine_tune/low",
    "BioCLIP-2": BIO / "full_species/BioCLIP-2/frozen/low",
    "ConvNeXt-V2-B 224, cross-entropy": RUNS / "snakeclef_style/convnextv2_base_224_ce",
    "ConvNeXt-V2-B 384, cross-entropy": RUNS / "snakeclef_style/convnextv2_base_384_ce",
    "ConvNeXt-V2-B 384, seesaw": RUNS / "snakeclef_style/convnextv2_base_384_seesaw",
    "BioCLIP-2 (strong recipe)": RUNS / "strong/BioCLIP-2",
    "DINOv2-Base/14 (strong recipe)": RUNS / "strong/DINOv2-Base-14",
}
PAIRS = [("ConvNeXt-V2-B 384, cross-entropy", "ConvNeXt-V2-B 384, seesaw"),
         ("ConvNeXt-V2-B 384, cross-entropy", "BioCLIP-2"),
         ("BioCLIP-2", "DINOv2-Base/14"),
         ("ConvNeXt-V2-Base", "Swin-Base"),
         ("ConvNeXt-V2-B 384, cross-entropy", "ConvNeXt-V2-B 224, cross-entropy"),
         ("ConvNeXt-V2-B 224, cross-entropy", "ConvNeXt-V2-Base"),
         ("BioCLIP-2 (strong recipe)", "DINOv2-Base/14 (strong recipe)"),
         ("BioCLIP-2 (strong recipe)", "BioCLIP-2")]


def slug(name):
    return "".join(ch if ch.isalnum() else "_" for ch in name)


def scores(path, reference=None):
    frame = pd.read_csv(path, keep_default_na=False)
    if reference is not None:
        assert frame.relative_path.tolist() == reference, f"test order differs: {path}"
    probs = frame[[c for c in frame.columns if c.startswith("probability_")]].to_numpy(float)
    truth, pred = frame.label_id.to_numpy(int), probs.argmax(1)
    tested = sorted(set(truth))
    return frame.relative_path.tolist(), {
        "accuracy": float(np.mean(pred == truth)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_f1_tested_species": float(f1_score(truth, pred, labels=tested, average="macro", zero_division=0))}


def main():
    reference, rows, per_seed = None, [], {}
    for system, folder in ORIGINAL.items():
        paths = [folder / "test_predictions.csv"] + [RUNS / "seed_repeats" / slug(system) / f"repeat_{k}" / "test_predictions.csv" for k in (1, 2)]
        if not all(p.exists() for p in paths):
            print("incomplete:", system)
            continue
        results = []
        for path in paths:
            order, result = scores(path, reference)
            reference = reference or order
            results.append(result)
        per_seed[system] = results
        row = {"system": system, "seeds": len(results)}
        for key in results[0]:
            values = np.array([r[key] for r in results])
            row.update({f"{key}_seed{i}": v for i, v in enumerate(values)})
            row.update({f"{key}_mean": values.mean(), f"{key}_sd": values.std(ddof=1),
                        f"{key}_min": values.min(), f"{key}_max": values.max()})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(RUNS / "analysis/seed_repeats.csv", index=False)
    pair_rows = []
    for a, b in PAIRS:
        if a not in per_seed or b not in per_seed:
            continue
        for key in ("accuracy", "balanced_accuracy"):
            diffs = [x[key] - y[key] for x, y in itertools.product(per_seed[a], per_seed[b])]
            pair_rows.append({"system_a": a, "system_b": b, "metric": key,
                              "mean_difference": float(np.mean([x[key] for x in per_seed[a]]) - np.mean([y[key] for y in per_seed[b]])),
                              "min_difference": min(diffs), "max_difference": max(diffs),
                              "same_sign_all_9": bool(min(diffs) > 0 or max(diffs) < 0)})
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(RUNS / "analysis/seed_repeat_pairs.csv", index=False)
    show = ["system"] + [c for c in table.columns if c.startswith("accuracy_")]
    print((table[show].set_index("system") * 100).round(2).to_string())
    print(pairs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
