"""Compose the Grad-CAM figure from the saved overlays (snake_paper/grad_cam_v4/*_cam.png), marking
misidentified photographs with the predicted species (analysis/sample_predictions.csv).

Same 5 x 7 layout as scripts/generate_gradcam_v4.py; no model is run. Writes
snake_paper/figures/full_species_gradcam.pdf and .png.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_gradcam_v4 as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "analysis" / "sample_predictions.csv"


def short(name):
    genus, epithet = name.split()[:2]
    return f"{genus[0]}. {epithet}"


def main():
    preds = pd.read_csv(PRED).set_index("sample")
    assert list(preds.index) == list(g.SAMPLE_FILES)
    fig, axes = plt.subplots(5, 7, figsize=(10.5, 8.2))
    for ax, name in zip(axes.flat, g.SAMPLE_FILES):
        overlay = g.OUT_DIR / f"{Path(name).stem}_cam.png"
        with Image.open(overlay) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        row = preds.loc[name]
        label = short(row.species) if row.correct else f"{short(row.species)}\n→ {short(row.predicted)}"
        ax.set_xlabel(label, fontsize=7.2, style="italic", labelpad=2, color="#0b0b0b", linespacing=1.05)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.04, wspace=0.08, hspace=0.32)
    fig.savefig(g.OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(g.OUT_FIG.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("misidentified:", int((~preds.correct).sum()), "of", len(preds))


if __name__ == "__main__":
    main()
