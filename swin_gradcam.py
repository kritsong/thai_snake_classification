# visualize_swin_gradcam_per_species_one_row.py
# -----------------------------------------------------------------------------
# Save ONE Grad-/LayerCAM ROW per species (10 tiles in a single row).
# No captions/titles inside figures. Small gaps between tiles.
# Files saved to: figures/gradcam_<Species>.png
# -----------------------------------------------------------------------------

import os
import math
import random
from typing import List, Tuple, Optional

import numpy as np
import cv2
import torch
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, SwinForImageClassification


# ===================== PATHS & CONFIG =====================
DATA_DIR   = r"C:\Users\ADMIN\Downloads\snake_datasetNew"
MODEL_DIR  = r"C:\Users\ADMIN\PycharmProjects\snake_research\transformer_experiment_results_swin_exact\Swin-Base_augnone_fineT_thres500"
OUTPUT_DIR = r"figures"                        # <— save directly in figures/
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE   = (224, 224)                        # must match Swin-Base training
THRESHOLD  = 500                               # only species with ≥ THRESHOLD images
TOP_K      = 5                                 # export exactly 5 species
SAMPLES    = 10                                # number of tiles (images) per row

# Layout for ONE ROW per species
GRID_COLS      = SAMPLES                        # 10 tiles wide
ROW_TILE_IN    = 1.6                            # per-tile width (inches)
ROW_HEIGHT_IN  = 1.6                            # row height (inches)
GAP_WSPACE     = 0.03                           # small horizontal gap between tiles

# Saliency / Grad-CAM settings
STAGE_INDEX   = 2                               # 0..3; 2 ≈ 14x14 tokens at 224px
AGG_METHOD    = "layercam"                      # "layercam" or "gradcam"
SMOOTH_PASSES = 8                               # 0 disables SmoothGrad
SMOOTH_SIGMA  = 0.10                            # noise std as fraction of dynamic range
POST_BLUR_SIG = 1.0                             # mild blur on heatmap (px); 0 disables

ALPHA      = 0.40                               # overlay opacity
SEED       = 1337
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ===================== Utilities =====================
def set_all_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_images(d: str) -> int:
    try:
        return sum(f.lower().endswith(IMG_EXTS) for f in os.listdir(d))
    except FileNotFoundError:
        return 0

def top_species_by_count(data_dir: str, threshold: int, k: int) -> List[str]:
    items = []
    for d in os.listdir(data_dir):
        p = os.path.join(data_dir, d)
        if not os.path.isdir(p):
            continue
        n = count_images(p)
        if n >= threshold:
            items.append((d, n))
    items.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in items[:k]]

def load_image(fp: str, size: Tuple[int,int]) -> Tuple[Image.Image, np.ndarray]:
    pil = Image.open(fp).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(pil)
    return pil, arr

def _square_hw(n_tokens: int) -> Tuple[int,int]:
    s = int(round(math.sqrt(n_tokens)))
    if s * s != n_tokens:
        s = int(math.sqrt(n_tokens))
    s = max(1, s)
    return s, s

def _gaussian_blur(u8: np.ndarray, sigma: float) -> np.ndarray:
    if sigma and sigma > 0:
        k = int(2 * round(3 * sigma) + 1)
        u8 = cv2.GaussianBlur(u8, (k, k), sigmaX=sigma, sigmaY=sigma)
    return u8

def _pick_stage_module(model: SwinForImageClassification, stage_index: int) -> nn.Module:
    layers = model.swin.encoder.layers
    stage_index = max(0, min(stage_index, len(layers)-1))
    return layers[stage_index]

def _aggregate_tokens(acts: torch.Tensor, grads: torch.Tensor, method: str) -> torch.Tensor:
    """
    acts, grads: (1, N, C)
    returns: (N,) normalized 0..1 (torch.cpu)
    """
    A = acts[0]   # (N,C)
    G = grads[0]  # (N,C)
    if method == "layercam":
        tokens = torch.relu(G) * torch.relu(A)
        tokens = tokens.sum(dim=1)             # (N,)
    else:  # "gradcam" classic
        w = G.mean(dim=0)                      # (C,)
        tokens = torch.relu(A @ w)             # (N,)
    tokens = tokens - tokens.min()
    if tokens.max() > 0:
        tokens = tokens / tokens.max()
    return tokens.detach().cpu()

def make_stage_cam(model,
                   px: torch.Tensor,                # (1,3,H,W)
                   stage_module: nn.Module,
                   method: str = AGG_METHOD,
                   smooth_passes: int = SMOOTH_PASSES,
                   smooth_sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """
    Compute token heatmap at the selected Swin stage via hooks.
    Returns heatmap resized to token grid (not yet to image size).
    """
    holder = {"acts": None, "grads": None, "dims": None}

    def fwd_hook(_m, _inp, out):
        # out may be tuple: (hidden_states, H, W) or (hidden_states, (H,W))
        hs = None; H = W = None
        if isinstance(out, (tuple, list)):
            hs = out[0]
            if len(out) >= 3 and isinstance(out[1], int) and isinstance(out[2], int):
                H, W = int(out[1]), int(out[2])
            elif len(out) >= 2 and isinstance(out[1], (tuple, list)) and len(out[1]) == 2:
                H, W = int(out[1][0]), int(out[1][1])
        else:
            hs = out
        if hs is None:
            raise RuntimeError("Failed to capture hidden_states at the selected stage.")
        hs.requires_grad_(True); hs.retain_grad()
        holder["acts"] = hs
        holder["dims"] = (H, W)
        hs.register_hook(lambda g: holder.__setitem__("grads", g))

    hook = stage_module.register_forward_hook(fwd_hook)

    def run_once(px_this: torch.Tensor):
        model.zero_grad(set_to_none=True)
        out = model(pixel_values=px_this)
        logits = out.logits
        idx = int(torch.argmax(logits, dim=-1)[0].item())
        score = logits[0, idx]
        score.backward(retain_graph=False)
        acts = holder["acts"]; grads = holder["grads"]
        if grads is None and acts is not None:
            grads = acts.grad
        if acts is None or grads is None:
            raise RuntimeError("Missing activations or gradients for CAM.")
        tokens = _aggregate_tokens(acts, grads, method=method)
        return tokens

    try:
        if smooth_passes and smooth_passes > 0:
            px_min = px.amin().item(); px_max = px.amax().item()
            scale  = (px_max - px_min) if px_max > px_min else 1.0
            acc = None
            for _ in range(smooth_passes):
                noise = torch.randn_like(px) * (smooth_sigma * scale)
                tokens = run_once(torch.clamp(px + noise, px_min, px_max))
                acc = tokens if acc is None else (acc + tokens)
            tokens = acc / float(smooth_passes)
        else:
            tokens = run_once(px)
    finally:
        hook.remove()

    # reshape to token grid
    n = tokens.shape[0]
    H, W = holder["dims"] if holder["dims"] not in (None, (None, None)) else _square_hw(n)
    heat_tokens = tokens.numpy().reshape(H, W).astype(np.float32)
    return heat_tokens

def to_inputs(pil_img: Image.Image, processor, device: str):
    return {k: v.to(device) for k, v in processor(
        images=pil_img, return_tensors="pt", do_resize=False, do_center_crop=False
    ).items()}

def overlay(np_img_uint8: np.ndarray, heatmap_tokens: np.ndarray,
            alpha: float = ALPHA, out_size: Tuple[int,int] = IMG_SIZE) -> np.ndarray:
    hmap = cv2.resize((heatmap_tokens * 255).astype(np.uint8), out_size, interpolation=cv2.INTER_CUBIC)
    if POST_BLUR_SIG and POST_BLUR_SIG > 0:
        hmap = _gaussian_blur(hmap, POST_BLUR_SIG)
    jet  = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
    jet  = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    base = cv2.resize(np_img_uint8, out_size, interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(jet, alpha, base, 1 - alpha, 0.0)


# ===================== Main =====================
def main():
    set_all_seeds(SEED)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    # choose top-K most represented species (≥ threshold)
    species = top_species_by_count(DATA_DIR, THRESHOLD, TOP_K)
    if not species:
        raise SystemExit(f"No species with ≥{THRESHOLD} images in: {DATA_DIR}")
    if len(species) < TOP_K:
        print(f"[Warn] Only {len(species)} species found ≥{THRESHOLD}: {species}")

    # load model & processor; pick stage
    print(f"[Info] Loading checkpoint: {MODEL_DIR}")
    processor = AutoImageProcessor.from_pretrained(MODEL_DIR)
    model = SwinForImageClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
    stage_module = _pick_stage_module(model, STAGE_INDEX)
    print(f"[Info] Hooking Swin stage {STAGE_INDEX}")

    # layout: a SINGLE ROW per species
    cols = GRID_COLS
    rows = 1

    for sp in species:
        sp_dir = os.path.join(DATA_DIR, sp)
        cand = [f for f in os.listdir(sp_dir) if f.lower().endswith(IMG_EXTS)]
        if not cand:
            print(f"[Skip] No images for {sp}")
            continue
        picks = (random.sample(cand, SAMPLES)
                 if len(cand) >= SAMPLES else
                 [random.choice(cand) for _ in range(SAMPLES)])

        fig, axes = plt.subplots(
            rows, cols,
            figsize=(cols * ROW_TILE_IN, ROW_HEIGHT_IN),
            gridspec_kw={"wspace": GAP_WSPACE, "hspace": 0.0}
        )
        # axes is 1D array when rows=1
        if isinstance(axes, np.ndarray):
            ax_list = axes.ravel().tolist()
        else:
            ax_list = [axes]

        for k, fname in enumerate(picks):
            fp = os.path.join(sp_dir, fname)
            try:
                pil, arr = load_image(fp, IMG_SIZE)
            except Exception as e:
                print(f"[Warn] Bad image skipped: {fp} ({e})")
                ax_list[k].axis("off")
                continue

            inputs = to_inputs(pil, processor, DEVICE)
            heat = make_stage_cam(
                model, inputs["pixel_values"], stage_module,
                method=AGG_METHOD, smooth_passes=SMOOTH_PASSES, smooth_sigma=SMOOTH_SIGMA
            )
            ov = overlay(arr, heat, alpha=ALPHA, out_size=IMG_SIZE)
            ax = ax_list[k]
            ax.imshow(ov); ax.axis("off")

        # hide any unused tiles if fewer images than SAMPLES
        for k in range(len(picks), rows * cols):
            ax_list[k].axis("off")

        # save: figures/gradcam_<Species>.png (spaces→underscores)
        fname_out = f"gradcam_{sp}.png".replace(" ", "_")
        out_path = os.path.join(OUTPUT_DIR, fname_out)
        plt.tight_layout(pad=0.02)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
