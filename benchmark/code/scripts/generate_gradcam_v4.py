"""Generate Grad-CAM overlays on the 35 permissioned sample images using the trained 139-species
ConvNeXt-V2-Base 384-px (cross-entropy) final model.

Saves individual overlays to snake_paper/grad_cam_v4/<Sample>_cam.png and a 5x7 composite figure
to snake_paper/figures/full_species_gradcam.pdf and .png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "snakeclef_style" / "convnextv2_base_384_ce"
SAMPLES_DIR = ROOT / "snake_paper" / "figures" / "samples"
OUT_DIR = ROOT / "snake_paper" / "grad_cam_v4"
OUT_FIG = ROOT / "snake_paper" / "figures" / "full_species_gradcam"

# Exact order of the 35 permissioned sample images in Figure 1 of main_full_species_v4.tex
SAMPLE_FILES = [
    "Ahaetulla nasuta 1.png",
    "Ahaetulla nasuta 2.png",
    "Ahaetulla nasuta 3.png",
    "Ahaetulla prasina 1.png",
    "Ahaetulla prasina 3.png",
    "Ahaetulla prasina 4.png",
    "Ahaetulla prasina 5.png",
    "Ahaetulla prasina 6.png",
    "Ahaetulla prasina 7.png",
    "Ahaetulla prasina 8.png",
    "Ahaetulla prasina 9.png",
    "Enhydris enhydris 1.png",
    "Enhydris enhydris 2.png",
    "Enhydris enhydris 3.png",
    "Enhydris enhydris 4.png",
    "Enhydris enhydris 5.png",
    "Enhydris enhydris 6.png",
    "Enhydris enhydris 7.png",
    "Enhydris enhydris 8.png",
    "Hypsiscopus plumbea 1.jpg",
    "Hypsiscopus plumbea 2.jpg",
    "Oligodon taeniatus 1.png",
    "Oligodon taeniatus 2.png",
    "Oligodon taeniatus 3.png",
    "Oligodon taeniatus 4.png",
    "Ptyas mucosa 1.png",
    "Ptyas mucosa 2.png",
    "Trimeresurus albolabris 1.png",
    "Trimeresurus albolabris 2.png",
    "Trimeresurus albolabris 3.png",
    "Trimeresurus macrops 1.png",
    "Trimeresurus macrops 2.png",
    "Trimeresurus macrops 3.png",
    "Trimeresurus macrops 4.png",
    "Trimeresurus macrops 5.png",
]


def short_binomial(sp: str) -> str:
    parts = sp.split()
    return f"{parts[0][0]}. {parts[1]}" if len(parts) == 2 else sp


def compute_gradcam(model: torch.nn.Module, x: torch.Tensor) -> tuple[np.ndarray, int, float]:
    """Compute Grad-CAM on the final convolutional stage (stages.3, 12x12 at 384 px) for the predicted class."""
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_mod, _inp, out):
        activations.append(out)

    def bwd_hook(_mod, _gin, gout):
        gradients.append(gout[0])

    h1 = model.stages[3].register_forward_hook(fwd_hook)
    h2 = model.stages[3].register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        probs = logits.softmax(dim=-1)[0]
        pred_idx = int(probs.argmax().item())
        conf = float(probs[pred_idx].item())
        logits[0, pred_idx].backward()

        act = activations[0][0]  # [C, H, W]
        grad = gradients[0][0]   # [C, H, W]
        weights = grad.mean(dim=(1, 2), keepdim=True)  # [C, 1, 1]
        cam = F.relu((weights * act).sum(dim=0))       # [H, W]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam, size=(384, 384), mode="bicubic", align_corners=False)[0, 0]
        cam_np = cam_up.detach().cpu().numpy()
        cam_np = np.maximum(cam_np, 0.0)
        denom = cam_np.max() - cam_np.min()
        if denom > 1e-8:
            cam_np = (cam_np - cam_np.min()) / denom
        return cam_np, pred_idx, conf
    finally:
        h1.remove()
        h2.remove()


def main() -> None:
    proto = json.loads((MODEL_DIR / "protocol.json").read_text(encoding="utf-8"))
    labels: list[str] = proto["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model(proto["timm_model"], pretrained=False, num_classes=len(labels))
    state = torch.load(MODEL_DIR / "final_model" / "final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jet = cm.get_cmap("jet")

    overlays: list[tuple[np.ndarray, str, str, bool]] = []
    for fname in SAMPLE_FILES:
        src_path = SAMPLES_DIR / fname
        true_sp = " ".join(src_path.stem.split()[:-1])
        img = Image.open(src_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)

        # Average Grad-CAM over original and horizontal flip (matching test-time flip averaging)
        cam_orig, pred_idx, _ = compute_gradcam(model, x)
        cam_flip, _, _ = compute_gradcam(model, torch.flip(x, dims=[3]))
        cam = 0.5 * (cam_orig + np.fliplr(cam_flip))
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        # Smooth mildly in 384x384 space (sigma=4 px at 384 px)
        cam_img = Image.fromarray(np.uint8(cam * 255)).filter(ImageFilter.GaussianBlur(radius=4.0))
        cam_smooth = np.asarray(cam_img, dtype=np.float32) / 255.0
        if cam_smooth.max() - cam_smooth.min() > 1e-8:
            cam_smooth = (cam_smooth - cam_smooth.min()) / (cam_smooth.max() - cam_smooth.min())

        # Render square 224x224 overlay (matching Figure 1 and original grad_cam layout)
        base_sq = img.resize((224, 224), Image.Resampling.BICUBIC)
        base_np = np.asarray(base_sq, dtype=np.float32) / 255.0
        cam_224 = np.asarray(
            Image.fromarray(np.uint8(cam_smooth * 255)).resize((224, 224), Image.Resampling.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        heat_rgb = jet(cam_224)[..., :3]
        blended = np.clip(0.58 * base_np + 0.42 * heat_rgb, 0.0, 1.0)
        blended_u8 = np.uint8(blended * 255)

        out_name = f"{src_path.stem}_cam.png"
        Image.fromarray(blended_u8).save(OUT_DIR / out_name)

        # Also get flip-averaged prediction
        with torch.no_grad():
            logits = 0.5 * (model(x) + model(torch.flip(x, dims=[3])))
            pred_sp = labels[int(logits.argmax(dim=-1).item())]
        overlays.append((blended_u8, true_sp, pred_sp, pred_sp == true_sp))

    # Build a clean 5x7 composite figure
    fig, axes = plt.subplots(5, 7, figsize=(10.5, 7.8))
    for idx, (ax, (arr, true_sp, pred_sp, ok)) in enumerate(zip(axes.flat, overlays)):
        ax.imshow(arr)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel(short_binomial(true_sp), fontsize=7.2, style="italic", labelpad=2, color="#0b0b0b")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.03, wspace=0.08, hspace=0.22)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print(f"Saved {len(overlays)} Grad-CAM overlays to {OUT_DIR} and composite to {OUT_FIG.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
