"""Warm batch-one inference cost of the full-species v4 final systems.

Same protocol as scripts/benchmark_selected_six.py, run after all training stopped.
All systems are timed in one session (written to inference_benchmark_all_systems.json;
the earlier inference_benchmark.json, which covered the first nine systems, is kept):
one fixed device-resident tensor; eval mode; GPU float16 autocast with
synchronized wall timing (20 warm-ups, 100 timed forwards); CPU float32 with four
threads (5 warm-ups, 30 timed forwards). Input is 224 px except the 384 px
SnakeCLEF-style ConvNeXt runs (timed for one forward; flip TTA doubles the
forward cost). The DINOv2 probe is timed as its frozen backbone; the
logistic-regression head is a single 768x139 matrix product and is not timed.
The BioCLIP 2 probe and zero-shot use the same image tower as the frozen BioCLIP 2
grid model, so that row covers them. Decoding, resizing, transfer, softmax
and model loading are excluded.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import timm
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import run_full_species_grouped_bioclip2  # noqa: E402,F401  registers the three added models
import run_original_six_grouped as base  # noqa: E402
from benchmark_selected_six import measure  # noqa: E402

RUNS = ROOT / "paper_experiments" / "full_species_fresh_v4"
GRIDS = (RUNS, ROOT / "paper_experiments" / "full_species_fresh_v4_added",
         ROOT / "paper_experiments" / "full_species_fresh_v4_bioclip2")
OUT = RUNS / "analysis" / "inference_benchmark_all_systems.json"
N_CLASSES = 139


def systems():
    for grid in GRIDS:
        selection = json.loads((grid / "selected_final_configurations.json").read_text(encoding="utf-8"))["choices"]
        for name, c in selection.items():
            ckpt = grid / "full_species" / name.replace("/", "-") / c["regime"] / c["augmentation"] / "final_model" / "final.pt"
            yield name, base.MODELS[name], ckpt, 224, N_CLASSES
    for size, loss, timm_name in ((384, "seesaw", "convnextv2_base.fcmae_ft_in22k_in1k_384"),
                                  (384, "ce", "convnextv2_base.fcmae_ft_in22k_in1k_384"),
                                  (224, "ce", "convnextv2_base.fcmae_ft_in22k_in1k")):
        ckpt = RUNS / "snakeclef_style" / f"convnextv2_base_{size}_{loss}" / "final_model" / "final.pt"
        yield f"ConvNeXt-V2-B {size}, {loss}", timm_name, ckpt, size, N_CLASSES
    yield "DINOv2 ViT-B/14 backbone (probe)", "vit_base_patch14_reg4_dinov2.lvd142m", None, 224, 0


def main():
    if OUT.exists():
        raise SystemExit("Existing benchmark retained; do not silently replace timing measurements")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260925)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    cpu, cuda = torch.device("cpu"), torch.device("cuda")
    rows = []
    for name, timm_name, ckpt, size, classes in systems():
        if ckpt is None:
            model = timm.create_model(timm_name, pretrained=True, num_classes=0, img_size=size)
        else:
            model = timm.create_model(timm_name, pretrained=False, num_classes=classes)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        tensor = torch.randn(1, 3, size, size)
        cpu_m = measure(model, tensor, cpu, 5, 30)
        model.to(cuda)
        gpu_m = measure(model, tensor.to(cuda), cuda, 20, 100)
        rows.append({"system": name, "timm_model": timm_name, "input_size": size,
                     "parameters": sum(p.numel() for p in model.parameters()),
                     "checkpoint_sha256": base.sha256(ckpt) if ckpt else "pretrained backbone (frozen)",
                     "checkpoint_mib": ckpt.stat().st_size / 1024 ** 2 if ckpt else None,
                     "cpu_float32": cpu_m, "gpu_autocast_float16": gpu_m})
        print(json.dumps({"system": name, "params_M": round(rows[-1]["parameters"] / 1e6, 2),
                          "cpu_median_ms": round(cpu_m["median_ms"], 2), "gpu_median_ms": round(gpu_m["median_ms"], 2),
                          "gpu_p95_ms": round(gpu_m["p95_ms"], 2), "gpu_peak_mib": round(gpu_m["peak_torch_allocated_mib"], 1)}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    record = {"status": "measured_after_all_training_finished",
              "hardware": {"gpu": torch.cuda.get_device_name(), "cpu": "12th Gen Intel Core i9-12900KF",
                           "cpu_threads": 4, "interop_threads": 1, "platform": platform.platform()},
              "software": {"python": platform.python_version(), "torch": torch.__version__, "timm": timm.__version__,
                           "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
              "protocol": __doc__, "systems": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    base.atomic_json(OUT, record)


if __name__ == "__main__":
    main()
