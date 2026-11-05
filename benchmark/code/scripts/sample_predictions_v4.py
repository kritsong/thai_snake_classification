"""Flip-averaged predictions of the 384-px cross-entropy ConvNeXt-V2-Base final model on the 35 sample images
shown in the Grad-CAM figure (CPU inference only). Writes analysis/sample_predictions.csv, joined with the
benchmark split of each sample's nearest benchmark image (scripts/locate_sample_images_v4.py).
"""
import sys
from pathlib import Path

import pandas as pd
import timm
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_gradcam_v4 as g  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "analysis"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    labels = sorted(pd.read_csv(ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020" / "manifest.csv",
                                dtype=str, keep_default_na=False).class_label.unique())
    model = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k_384", pretrained=False, num_classes=len(labels))
    state = torch.load(g.MODEL_DIR / "final_model" / "final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model", state) if isinstance(state, dict) and "model" in state else state)
    model.eval()
    tf = transforms.Compose([transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                             transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    rows = []
    for name in g.SAMPLE_FILES:
        species = " ".join(name.split()[:2])
        with Image.open(g.SAMPLES_DIR / name) as im, torch.inference_mode():
            x = tf(im.convert("RGB"))[None]
            logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
        pred = labels[int(logits.argmax())]
        rows.append({"sample": name, "species": species, "predicted": pred, "correct": pred == species})
    frame = pd.DataFrame(rows).merge(pd.read_csv(ANALYSIS / "sample_image_locations.csv")[["sample", "split", "similarity", "second_best_same_species"]],
                                     on="sample", how="left")
    frame.to_csv(ANALYSIS / "sample_predictions.csv", index=False)
    print(frame.to_string(index=False))
    print("correct:", int(frame.correct.sum()), "of", len(frame))


if __name__ == "__main__":
    main()
