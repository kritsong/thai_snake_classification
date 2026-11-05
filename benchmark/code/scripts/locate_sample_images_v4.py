"""Locate the 35 permissioned sample images (Figure 1 / Grad-CAM figure) in the v4 benchmark.

Each sample (a crop/resave of a benchmark photograph) is embedded with BioCLIP 2 using the same
preprocessing as the cached benchmark features, and matched to its nearest benchmark image of the
same species by cosine similarity. Writes paper_experiments/full_species_v4_taxonomy/analysis/sample_image_locations.csv
with the matched image, its split, and the similarity (CPU only; no training).
"""
import os
import sys
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
FEATS = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "bioclip2"
SAMPLES = ROOT / "snake_paper" / "figures" / "samples"
OUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "analysis" / "sample_image_locations.csv"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    model, _, preprocess = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip-2")
    model.eval()
    features = np.load(FEATS / "image_features.npy")
    order = pd.read_csv(FEATS / "feature_order.csv", keep_default_na=False)
    manifest = pd.read_csv(V4 / "manifest.csv", dtype=str, keep_default_na=False).set_index("relative_path")
    order = order.join(manifest[["split", "class_label", "component"]], on="relative_path")
    rows = []
    for path in sorted(SAMPLES.iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        species = " ".join(path.stem.split()[:2])
        with Image.open(path) as im, torch.inference_mode():
            x = model.encode_image(preprocess(im.convert("RGB"))[None], normalize=True).numpy()[0]
        sims = features @ x
        same = (order.class_label == species).to_numpy()
        best = int(np.argmax(np.where(same, sims, -1)))
        overall = int(np.argmax(sims))
        rows.append({"sample": path.name, "species": species, "matched_image": order.relative_path[best],
                     "split": order.split[best], "group": order.component[best], "similarity": float(sims[best]),
                     "second_best_same_species": float(np.sort(np.where(same, sims, -1))[-2]),
                     "best_overall_species": order.class_label[overall], "best_overall_similarity": float(sims[overall])})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT, index=False)
    print(frame[["sample", "split", "similarity", "second_best_same_species"]].to_string(index=False))
    print(frame.split.value_counts().to_string())


if __name__ == "__main__":
    main()
