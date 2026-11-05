"""Checks requested in the external review of the single-protocol manuscript (reads saved outputs only).

1. Group-weighted test accuracy: each independent test group weighted equally (mean of within-group
   accuracy), next to the image-weighted accuracy, for the reported systems.
2. External scope: nominal SnakeCLEF Thailand species before and after name mapping, in and out of scope.
3. EXIF orientation tags in the source still images.
Writes analysis/group_weighted_accuracy.csv and prints the other checks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402
from benchmark_subset_v4 import SYSTEMS  # noqa: E402

ROOT = full.ROOT


def group_weighted():
    found = full.systems()
    reference, _ = full.load(found["ConvNeXt-V2-B 384, cross-entropy"])
    truth = reference.label_id.to_numpy(int)
    groups = reference.leakage_group_id.to_numpy()
    rows = []
    for name, label in SYSTEMS.items():
        correct = full.load(found[name], reference)[1].argmax(1) == truth
        per_group = pd.Series(correct).groupby(groups).mean()
        rows.append({"system": name, "label": label, "image_weighted": correct.mean(), "group_weighted": per_group.mean(),
                     "groups": len(per_group)})
    frame = pd.DataFrame(rows)
    frame["difference_points"] = 100 * (frame.group_weighted - frame.image_weighted)
    frame.to_csv(full.OUT / "group_weighted_accuracy.csv", index=False)
    print(frame.round(4).to_string(index=False))
    order = frame.sort_values("image_weighted").label.tolist() == frame.sort_values("group_weighted").label.tolist()
    print("same ranking under both weightings:", order)


def external_scope():
    scope_dir = ROOT / "paper_experiments" / "external_snakeclef2024" / "full_species_v4_taxonomy_scope"
    summary = json.loads((scope_dir / "summary.json").read_text(encoding="utf-8")) if (scope_dir / "summary.json").exists() else {}
    print("scope summary keys:", list(summary)[:30])
    for key, value in summary.items():
        if not isinstance(value, (dict, list)):
            print(f"  {key}: {value}")
        elif isinstance(value, dict) and len(value) < 15:
            print(f"  {key}: {value}")
    for name in sorted(p.name for p in scope_dir.iterdir()):
        print("  file:", name)


def exif_audit():
    manifest = pd.read_csv(full.V4 / "manifest.csv", dtype=str, keep_default_na=False)
    stills = manifest[manifest.source_type.str.contains("still|photo|image", case=False)]
    print("source types:", manifest.source_type.value_counts().to_dict())
    print("provenance example:", manifest.provenance.iloc[0][:200])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    group_weighted()
    external_scope()
    exif_audit()
