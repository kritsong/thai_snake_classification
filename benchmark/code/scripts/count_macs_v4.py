"""Multiply-accumulate counts of the benchmark models (architecture only; no weights are loaded).

Counts one forward pass of a 139-class model at its benchmark input size with
torch.utils.flop_counter (convolutions, matrix products and attention); MACs = FLOPs / 2.
Attention runs on the math backend so that its matrix products are counted, and the
CLIP tower runs in training mode so that nn.MultiheadAttention does not take its fused
inference path, which the counter does not see.
Writes paper_experiments/full_species_fresh_v4/analysis/model_macs.csv.
"""
from pathlib import Path

import open_clip
import pandas as pd
import timm
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_experiments" / "full_species_fresh_v4" / "analysis" / "model_macs.csv"
MODELS = [("MobileNetV3-Small", "mobilenetv3_small_100", 224, {}),
          ("EfficientNet-B0", "efficientnet_b0", 224, {}),
          ("ResNet-50", "resnet50", 224, {}),
          ("ViT-Base/16", "vit_base_patch16_224", 224, {}),
          ("DeiT-Base/16", "deit_base_distilled_patch16_224", 224, {}),
          ("Swin-Base", "swin_base_patch4_window7_224", 224, {}),
          ("ConvNeXt-V2-Base", "convnextv2_base", 224, {}),
          ("DINOv2-Base/14", "vit_base_patch14_reg4_dinov2", 224, {"img_size": 224}),
          ("ConvNeXt-V2-Base 384", "convnextv2_base", 384, {})]


class BioClip2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = open_clip.create_model("ViT-L-14", pretrained=None).visual
        self.head = torch.nn.Linear(768, 139)

    def forward(self, x):
        return self.head(self.visual(x))


def macs(model, size, train_mode=False):
    model.train(train_mode)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as counter:
        model(torch.zeros(1, 3, size, size))
    return counter.get_total_flops() / 2


rows = []
for name, arch, size, kwargs in MODELS:
    model = timm.create_model(arch, pretrained=False, num_classes=139, **kwargs)
    rows.append({"system": name, "input_size": size, "parameters": sum(p.numel() for p in model.parameters()),
                 "gmacs": macs(model, size) / 1e9})
model = BioClip2()
rows.append({"system": "BioCLIP-2", "input_size": 224, "parameters": sum(p.numel() for p in model.parameters()),
             "gmacs": macs(model, 224, train_mode=True) / 1e9})
frame = pd.DataFrame(rows)
frame.to_csv(OUT, index=False)
print(frame.to_string(index=False))
