"""LaTeX rows of the location-prior table (edit e31, 20 Nov 2026), from analysis/location_prior*.csv."""
from pathlib import Path

import pandas as pd

A = Path(__file__).resolve().parents[1] / "paper_experiments" / "full_species_v4_taxonomy" / "analysis"
ORDER = [("ConvNeXt-V2 (224)", "ConvNeXt-V2-Base"), ("DINOv2 (224)", "DINOv2 ViT-B/14"), ("BioCLIP 2 (224)", "BioCLIP~2"),
         ("ConvNeXt-V2 384, cross-entropy", "ConvNeXt-V2 384"), ("ConvNeXt-V2 384, Seesaw", "ConvNeXt-V2 384, Seesaw"),
         ("ConvNeXt-V2 384, cross-entropy + zero-shot", "ConvNeXt-V2 384 + ZS"),
         ("ConvNeXt-V2 384, Seesaw + zero-shot", "ConvNeXt-V2 384, Seesaw + ZS")]
loc = pd.read_csv(A / "location_prior.csv").set_index("system")
bands = pd.read_csv(A / "location_prior_bands.csv")
rare = bands[bands.band == "<20"].set_index("system")


def p1(x):
    return f"{100 * x:.1f}"


for name, label in ORDER:
    r = loc.loc[name]
    lo, hi = [float(v) for v in r.accuracy_difference_ci.strip("[]").split(",")]
    d_disjoint = 100 * (r.accuracy_poster_disjoint_adjusted - r.accuracy_poster_disjoint_base)
    print(f"{label} & {int(r.alpha)} & {r.beta:.1f} & {p1(r.test_accuracy_base)} & {p1(r.test_accuracy_adjusted)} & "
          f"\\(+{100 * r.accuracy_difference:.1f}\\) [{100 * lo:.1f}, {100 * hi:.1f}] & "
          f"{r.balanced_accuracy_base:.3f} & {r.balanced_accuracy_adjusted:.3f} & "
          f"{p1(rare.loc[name, 'accuracy_base'])} & {p1(rare.loc[name, 'accuracy_adjusted'])} & "
          f"{p1(r.mivs_as_non_mivs_base)} & {p1(r.mivs_as_non_mivs_adjusted)} & \\(+{d_disjoint:.1f}\\) \\\\")
    print(f"%   p={r.clustered_mcnemar_p:.2g}, poster-disjoint p={r.poster_disjoint_mcnemar_p:.2g}, "
          f"top-5 {p1(r.top5_base)} -> {p1(r.top5_adjusted)}, macro-F1 {r.macro_f1_base:.3f} -> {r.macro_f1_adjusted:.3f}")
