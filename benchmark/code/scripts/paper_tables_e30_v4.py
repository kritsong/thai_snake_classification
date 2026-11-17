"""LaTeX rows for the tables added or changed on 16 Nov 2026 (edit e30): the top-5 columns of Tables 5 and 10,
the zero-shot fusion table (Results 3.5) and the shared-contributor table (Results 3.6).
Reads analysis outputs only; prints rows to paste and checks overlaps with numbers already in the paper.
"""
from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parents[1] / "paper_experiments" / "full_species_v4_taxonomy" / "analysis"
ORDER = ["MobileNetV3-Small", "EfficientNet-B0", "ResNet-50", "ViT-Base/16", "DeiT-Base/16", "Swin-Base",
         "ConvNeXt-V2-Base", "DINOv2 ViT-B/14", "BioCLIP 2 ViT-L/14", "ConvNeXt-V2-Base, 384 px",
         "ConvNeXt-V2-Base, 384 px, Seesaw", "BioCLIP 2 zero-shot"]


def pct(x):
    return f"{100 * x:.1f}"


top5 = pd.read_csv(A / "top5_accuracy.csv")
for scope in ("test", "external"):
    t = top5[top5.scope == scope].set_index("system")
    print(f"% top-5, {scope}: " + ", ".join(f"{s}={pct(t.loc[s, 'top5'])}" for s in ORDER))

# Fusion table
fusion = pd.read_csv(A / "tail_fusion.csv").set_index("system")
source = pd.read_csv(A / "fusion_by_source.csv")
report = pd.read_csv(A / "report_level_accuracy.csv").set_index("system")
bands = pd.read_csv(A / "tail_fusion_bands.csv") if (A / "tail_fusion_bands.csv").exists() else None
fused5 = top5[top5.scope == "test fused"].set_index("system")
import json  # noqa: E402
decisions = json.loads((A / "post_hoc_decisions.json").read_text(encoding="utf-8"))
venom = {d["system"]: d for d in decisions["fusion"]}
rare = {(b["system"]): b for b in decisions["fusion_bands"] if b["band"] == "<20"}
models = [("ConvNeXt-V2 384, cross-entropy", "ConvNeXt-V2-Base, 384 px", "ConvNeXt-V2-B 384, cross-entropy"),
          ("ConvNeXt-V2 384, Seesaw", "ConvNeXt-V2-Base, 384 px, Seesaw", "ConvNeXt-V2 384, Seesaw")]
rows = {"Accuracy (\\%)": [], "Balanced accuracy": [], "Macro-F1": [], "Top-5 (\\%)": [], "Report-level (\\%)": [],
        "Still photographs (\\%)": [], "Video frames (\\%)": [], "Fewer than 20 dev.\\ images (\\%)": [],
        "MIVS as non-MIVS (\\%)": []}
for name, label, _ in models:
    f = fusion.loc[name]
    t = top5[top5.scope == "test"].set_index("system")
    rows["Accuracy (\\%)"] += [pct(f.test_accuracy_base), pct(f.test_accuracy_fused)]
    rows["Balanced accuracy"] += [f"{f.test_balanced_accuracy_base:.3f}", f"{f.test_balanced_accuracy_fused:.3f}"]
    rows["Macro-F1"] += [f"{f.test_macro_f1_tested_species_base:.3f}", f"{f.test_macro_f1_tested_species_fused:.3f}"]
    rows["Top-5 (\\%)"] += [pct(t.loc[label, "top5"]), pct(fused5.loc[name + " + zero-shot", "top5"])]
    rows["Report-level (\\%)"] += [pct(report.loc[name, "report_accuracy"]), pct(report.loc[name + " + zero-shot", "report_accuracy"])]
    for src, key in (("still_image", "Still photographs (\\%)"), ("video_frame", "Video frames (\\%)")):
        r = source[(source.system == name) & (source.source == src)].iloc[0]
        rows[key] += [pct(r.accuracy_base), pct(r.accuracy_fused)]
    rows["Fewer than 20 dev.\\ images (\\%)"] += [pct(rare[name]["accuracy_base"]), pct(rare[name]["accuracy_fused"])]
    rows["MIVS as non-MIVS (\\%)"] += [pct(venom[name]["venom_to_harmless_base"]), pct(venom[name]["venom_to_harmless_fused"])]
assert rows["Accuracy (\\%)"] == ["70.8", "73.1", "69.5", "72.8"], rows["Accuracy (\\%)"]
assert rows["Report-level (\\%)"][:2] == ["77.0", "79.9"] and rows["Fewer than 20 dev.\\ images (\\%)"] == ["12.1", "22.4", "21.6", "38.8"]
print("% fusion table")
for k, v in rows.items():
    print(f"{k} & " + " & ".join(v) + " \\\\")

# Shared-contributor table
poster = pd.read_csv(A / "benchmark_poster_overlap_sensitivity.csv")
poster["label"] = poster.label.str.replace("BioCLIP 2", "BioCLIP 2", regex=False)
p = poster.set_index("label")
print("% shared-contributor table (all test, poster-disjoint, difference in points)")
for s in ORDER:
    a, b = p.loc[s, "accuracy_all"], p.loc[s, "accuracy_poster_disjoint"]
    d = 100 * (b - a)
    print(f"{s} & {pct(a)} & {pct(b)} & {d:+.1f} \\\\")
n_all, n_disjoint = int(p.iloc[0].all_test_images), int(p.iloc[0].poster_disjoint_images)
print(f"% images: all {n_all}, poster-disjoint {n_disjoint}, removed {n_all - n_disjoint} ({100 * (n_all - n_disjoint) / n_all:.1f}%)")
