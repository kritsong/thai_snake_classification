"""Model-size control for BioCLIP 2: the general CLIP ViT-L/14 it was initialized from.

BioCLIP 2 starts from a CLIP ViT-L/14 trained on LAION-2B (Gu et al., 2025,
App. E). This runs exactly the BioCLIP 2 protocol of run_bioclip2_v4.py
(L2-normalised image embeddings, the same class texts and 80 prompt templates
for zero-shot, the same logistic-regression probe, folds and locked test) on
OpenCLIP's LAION-2B ViT-L/14 (laion2b_s32b_b82k), so the difference between
the two isolates BioCLIP 2's biological training at fixed architecture and size.
"""
from __future__ import annotations

import open_clip

import run_bioclip2_v4 as bioclip

MODEL, PRETRAINED = "ViT-L-14", "laion2b_s32b_b82k"


def load_model(device):
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    return model.eval().to(device), preprocess, open_clip.get_tokenizer(MODEL)


bioclip.HUB_ID = f"open_clip:{MODEL}/{PRETRAINED}"
bioclip.OUTPUT = bioclip.ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "clip_vitl14_laion2b"
bioclip.load_model = load_model

if __name__ == "__main__":
    bioclip.OUTPUT.mkdir(parents=True, exist_ok=True)
    bioclip.main()
