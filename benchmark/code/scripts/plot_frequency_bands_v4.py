"""Figure: test accuracy by species frequency, and the seesaw-minus-CE difference per band.

Colors: validated categorical slots 1-4 (adjacent CVD dE >= 9.1, light surface);
slots 3-4 are below 3:1 contrast, so values are also given in the manuscript table.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "analysis"
OUT = ROOT / "snake_paper" / "figures" / "full_species_frequency_bands"
BANDS = [">=500", "100-499", "20-99", "<20"]
BAND_LABELS = ["≥500", "100–499", "20–99", "<20"]
SYSTEMS = [("ConvNeXt-V2-B 384, cross-entropy", "ConvNeXt-V2 384 px", "#2a78d6"),
           ("ConvNeXt-V2-B 384, seesaw", "ConvNeXt-V2 384 px, Seesaw", "#eb6834"),
           ("BioCLIP-2 (strong)", "BioCLIP 2", "#1baf7a"),
           ("BioCLIP 2 zero-shot", "BioCLIP 2 zero-shot", "#eda100")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

bands = pd.read_csv(A / "frequency_bands.csv")
ablation = pd.read_csv(A / "seesaw_vs_ce_by_band.csv").set_index("band_dev_images").loc[BANDS]
n_test = bands.groupby("band_dev_images").test_images.first().loc[BANDS]
n_species = bands.groupby("band_dev_images").species.first().loc[BANDS]

plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.65, 1]})
width = 0.19
for i, (key, label, color) in enumerate(SYSTEMS):
    values = 100 * bands[bands.system == key].set_index("band_dev_images").accuracy.loc[BANDS]
    xs = [b + (i - 1.5) * (width + 0.012) for b in range(len(BANDS))]
    ax.bar(xs, values, width=width, color=color, edgecolor="white", linewidth=1.0, label=label, zorder=3)
ax.set_xticks(range(len(BANDS)))
ax.set_xticklabels([f"{b}\n{s} species\n{n:,} images" for b, s, n in zip(BAND_LABELS, n_species, n_test)], fontsize=7)
ax.set_xlabel("Development images of the true species", color=INK)
ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 100)
ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right", labelcolor=INK)
ax.set_title("(a) Accuracy by species frequency", loc="left", fontsize=9, color=INK)

diff = 100 * ablation["diff"]
low, high = 100 * ablation.diff_ci_low, 100 * ablation.diff_ci_high
xs = range(len(BANDS))
bx.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
bx.errorbar(xs, diff, yerr=[diff - low, high - diff], fmt="o", color=INK, ecolor=MUTED, elinewidth=1.2,
            capsize=3, markersize=5, zorder=3)
for x, d in zip(xs, diff):
    bx.annotate(f"{d:+.1f}".replace("-", "−"), (x, d), xytext=(7, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK)
bx.set_xticks(list(xs))
bx.set_xticklabels(BAND_LABELS, fontsize=7.5)
bx.set_xlabel("Development images of the true species", color=INK)
bx.set_ylabel("Seesaw − CE accuracy (points)")
bx.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
bx.set_title("(b) Loss ablation, 95% CI", loc="left", fontsize=9, color=INK)
for a in (ax, bx):
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=200)
print("saved", OUT.with_suffix(".pdf"))
