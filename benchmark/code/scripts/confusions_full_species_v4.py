"""Most frequent species confusions on the locked test set (reads saved predictions only).

For the most accurate system and for BioCLIP 2 under the reported protocol: the most
frequent (true, predicted) pairs, and the most frequent errors in which a medically
important venomous species (SnakeCLEF MIVS flag) is predicted as a harmless one.
Writes analysis/top_confusions.csv and prints a summary.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
SYSTEMS = {
    "ConvNeXt-V2-B 384, cross-entropy": RUNS / "snakeclef_style" / "convnextv2_base_384_ce" / "test_predictions.csv",
    "BioCLIP-2 (strong)": RUNS / "strong" / "BioCLIP-2" / "test_predictions.csv",
}
TOP = 10


def main():
    venom = pd.read_csv(RUNS / "species_venom_status.csv", dtype=str, keep_default_na=False)
    status = dict(zip(venom.species, venom.mivs))
    rows = []
    for system, path in SYSTEMS.items():
        pred = pd.read_csv(path, usecols=["class_label", "predicted_label", "leakage_group_id"])
        errors = pred[pred.class_label != pred.predicted_label]
        pairs = (errors.groupby(["class_label", "predicted_label"])
                 .agg(images=("leakage_group_id", "size"), groups=("leakage_group_id", "nunique"))
                 .reset_index())
        support = pred.class_label.value_counts()
        pairs["true_test_images"] = pairs.class_label.map(support)
        pairs["true_status"] = pairs.class_label.map(status).fillna("")
        pairs["predicted_status"] = pairs.predicted_label.map(status).fillna("")
        pairs["same_genus"] = pairs.class_label.str.split().str[0] == pairs.predicted_label.str.split().str[0]
        top = pairs.sort_values(["images", "groups"], ascending=False).head(TOP).assign(kind="most_frequent")
        v2h = pairs[(pairs.true_status == "1") & (pairs.predicted_status == "0")]
        v2h_top = v2h.sort_values(["images", "groups"], ascending=False).head(TOP).assign(kind="venomous_to_harmless")
        rows += [top.assign(system=system), v2h_top.assign(system=system)]
        print(f"\n{system}: {len(errors)} errors; same-genus share {pairs.loc[pairs.same_genus, 'images'].sum() / len(errors):.3f}")
        print(top[["class_label", "predicted_label", "images", "groups", "true_test_images"]].to_string(index=False))
        print("venomous -> harmless:", int(v2h.images.sum()), "images")
        print(v2h_top[["class_label", "predicted_label", "images", "groups"]].to_string(index=False))
    pd.concat(rows, ignore_index=True).to_csv(RUNS / "analysis" / "top_confusions.csv", index=False)


if __name__ == "__main__":
    main()
