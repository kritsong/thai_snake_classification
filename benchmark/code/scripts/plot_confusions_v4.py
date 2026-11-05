"""Figure: confusion matrix for the 20 most abundant test species and top species/venom confusions.

Reads saved locked-test predictions of the most accurate system (ConvNeXt-V2-Base at 384 px,
cross-entropy) and species_venom_status.csv. Writes snake_paper/figures/full_species_confusions.pdf
and .png.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
PRED_PATH = RUNS / "snakeclef_style" / "convnextv2_base_384_ce" / "test_predictions.csv"
VENOM_PATH = RUNS / "species_venom_status.csv"
OUT = ROOT / "snake_paper" / "figures" / "full_species_confusions"

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, CORAL = "#2a78d6", "#eb6834"

GENUS_ABBR = {
    "Chrysopelea": "Ch.",
    "Coelognathus": "Co.",
    "Cylindrophis": "Cy.",
    "Calloselasma": "Ca.",
}


def short_name(sp: str, mivs_set: set[str], full_genus: bool = False) -> str:
    if full_genus:
        return f"{sp}*" if sp in mivs_set else sp
    parts = sp.split()
    if len(parts) == 2:
        g_abbr = GENUS_ABBR.get(parts[0], f"{parts[0][0]}.")
        abbr = f"{g_abbr} {parts[1]}"
    else:
        abbr = sp
    return f"{abbr}*" if sp in mivs_set else abbr


def main() -> None:
    pred = pd.read_csv(PRED_PATH, usecols=["class_label", "predicted_label", "leakage_group_id"])
    venom = pd.read_csv(VENOM_PATH, dtype=str, keep_default_na=False)
    status = dict(zip(venom.species, venom.mivs))
    mivs_set = {sp for sp, flag in status.items() if flag == "1"}

    # Panel (a): 20 most frequent test species (alphabetical order to group congenerics) + Other column
    top20_freq = pred.class_label.value_counts().head(20).index.tolist()
    top20 = sorted(top20_freq)
    other_col = "Other (119 spp.)"

    sub = pred[pred.class_label.isin(top20)].copy()
    sub["pred_col"] = sub.predicted_label.where(sub.predicted_label.isin(top20), other_col)
    cols = top20 + [other_col]
    cm_counts = (
        pd.crosstab(sub.class_label, sub.pred_col)
        .reindex(index=top20, columns=cols, fill_value=0)
    )
    row_sums = cm_counts.sum(axis=1).values[:, None]
    cm_norm = cm_counts.values / np.maximum(row_sums, 1)

    # Panel (b) & (c): Top 8 overall confusions and top 8 MIVS -> non-MIVS (V -> H) confusions
    errors = pred[pred.class_label != pred.predicted_label]
    pairs = (
        errors.groupby(["class_label", "predicted_label"])
        .agg(images=("leakage_group_id", "size"), groups=("leakage_group_id", "nunique"))
        .reset_index()
    )
    pairs["true_status"] = pairs.class_label.map(status).fillna("")
    pairs["pred_status"] = pairs.predicted_label.map(status).fillna("")

    top_all = pairs.sort_values(["images", "groups"], ascending=False).head(8).iloc[::-1]
    top_v2h = (
        pairs[(pairs.true_status == "1") & (pairs.pred_status == "0")]
        .sort_values(["images", "groups"], ascending=False)
        .head(8)
        .iloc[::-1]
    )

    plt.rcParams.update({
        "font.size": 8.0,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    })

    fig = plt.figure(figsize=(12.2, 5.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.48, 1.0], height_ratios=[1.0, 1.0], wspace=0.68, hspace=0.40)
    ax_cm = fig.add_subplot(gs[:, 0])
    ax_all = fig.add_subplot(gs[0, 1])
    ax_v2h = fig.add_subplot(gs[1, 1])

    # Plot Panel (a)
    im = ax_cm.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    n_rows, n_cols = cm_counts.shape
    for r in range(n_rows):
        for c in range(n_cols):
            val = int(cm_counts.iat[r, c])
            if val > 0:
                prop = cm_norm[r, c]
                txt_color = "white" if prop >= 0.48 else INK
                ax_cm.text(
                    c, r, str(val),
                    ha="center", va="center",
                    fontsize=6.3, color=txt_color,
                    fontweight="bold" if r == c else "normal",
                )

    ax_cm.axvline(n_rows - 0.5, color=MUTED, linewidth=0.8, linestyle="--")

    row_labels = [short_name(sp, mivs_set, full_genus=True) for sp in top20]
    col_labels = [short_name(sp, mivs_set, full_genus=False) for sp in top20] + ["Other (119 spp.)"]

    ax_cm.set_xticks(range(n_cols))
    ax_cm.set_xticklabels(col_labels, rotation=50, ha="right", fontsize=7.0, style="italic")
    ax_cm.get_xticklabels()[-1].set_style("normal")

    ax_cm.set_yticks(range(n_rows))
    ax_cm.set_yticklabels(row_labels, fontsize=7.1, style="italic")
    ax_cm.set_xlabel("Predicted species", fontsize=8.5, color=INK, labelpad=4)
    ax_cm.set_ylabel("True species", fontsize=8.5, color=INK)
    ax_cm.set_title(
        f"(a) Confusion matrix for the 20 most frequent test species ({100 * len(sub) / len(pred):.1f}% of test images)",
        loc="left", fontsize=9.0, color=INK, pad=8,
    )

    cbar = fig.colorbar(im, ax=ax_cm, fraction=0.030, pad=0.022)
    cbar.set_label("Row-normalized proportion", fontsize=7.8, color=INK)
    cbar.ax.tick_params(labelsize=7.0)

    # Plot Panel (b) top: Overall confusions
    y_all = np.arange(len(top_all))
    bars_all = ax_all.barh(y_all, top_all.images, height=0.68, color=BLUE, edgecolor="white", zorder=3)
    labels_all = [
        f"{short_name(t, mivs_set)} → {short_name(p, mivs_set)}"
        for t, p in zip(top_all.class_label, top_all.predicted_label)
    ]
    ax_all.set_yticks(y_all)
    ax_all.set_yticklabels(labels_all, fontsize=7.2, style="italic")
    ax_all.set_xlim(0, 42)
    ax_all.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax_all.spines[["top", "right"]].set_visible(False)
    ax_all.set_xlabel("Test images (independent groups in parentheses)", fontsize=7.8, color=INK)
    ax_all.set_title("(b) Most frequent species confusions (all 139 species)", loc="left", fontsize=9.0, color=INK, pad=6)
    for bar, img_cnt, grp_cnt in zip(bars_all, top_all.images, top_all.groups):
        ax_all.text(
            bar.get_width() + 0.7, bar.get_y() + bar.get_height() / 2,
            f"{img_cnt} ({grp_cnt}g)", va="center", ha="left", fontsize=6.8, color=INK,
        )

    # Plot Panel (c) bottom: Venomous -> harmless (MIVS -> non-MIVS) confusions
    y_v2h = np.arange(len(top_v2h))
    bars_v2h = ax_v2h.barh(y_v2h, top_v2h.images, height=0.68, color=CORAL, edgecolor="white", zorder=3)
    labels_v2h = [
        f"{short_name(t, mivs_set)} → {short_name(p, mivs_set)}"
        for t, p in zip(top_v2h.class_label, top_v2h.predicted_label)
    ]
    ax_v2h.set_yticks(y_v2h)
    ax_v2h.set_yticklabels(labels_v2h, fontsize=7.2, style="italic")
    ax_v2h.set_xlim(0, 26)
    ax_v2h.grid(axis="x", color=GRID, linewidth=0.6, zorder=0)
    ax_v2h.spines[["top", "right"]].set_visible(False)
    ax_v2h.set_xlabel("Test images (independent groups in parentheses)", fontsize=7.8, color=INK)
    ax_v2h.set_title("(c) Most frequent venomous-to-harmless (MIVS → non-MIVS) errors", loc="left", fontsize=9.0, color=INK, pad=6)
    for bar, img_cnt, grp_cnt in zip(bars_v2h, top_v2h.images, top_v2h.groups):
        ax_v2h.text(
            bar.get_width() + 0.45, bar.get_y() + bar.get_height() / 2,
            f"{img_cnt} ({grp_cnt}g)", va="center", ha="left", fontsize=6.8, color=INK,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("Saved", OUT.with_suffix(".pdf"), "and", OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
