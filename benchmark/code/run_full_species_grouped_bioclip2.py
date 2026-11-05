"""Adds BioCLIP 2 (open_clip ViT-L/14 image tower) to the full-species runner.

Everything else is the full-species runner (run_full_species_grouped.py via
run_full_species_grouped_added.py), so BioCLIP 2 gets the same 224 px
transforms, AdamW optimizer, 10-epoch head warm-up, cosine schedule, early
stopping and refit rule as the other eight models; as a transformer it takes the
transformer backbone learning rate (2e-5). The model is the pretrained BioCLIP 2
image tower (768-d projected embedding) followed by a new linear head with
PyTorch's default initialisation. The runner normalises inputs with ImageNet
mean/std; a fixed affine step inside the model converts them to the CLIP
mean/std BioCLIP 2 was trained with. The text tower is not used.
"""
from __future__ import annotations

import open_clip
import timm
import torch
from torch import nn

import run_full_species_grouped_added as added

base = added.base
NAME = "BioCLIP-2"
HUB_ID = "hf-hub:imageomics/bioclip-2"
TIMM_STANDIN = "open_clip:" + HUB_ID
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class OpenClipClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.visual = open_clip.create_model(HUB_ID).visual  # pretrained weights from the local hub cache
        self.head = nn.Linear(self.visual.output_dim, num_classes)
        in_mean, in_std = torch.tensor(base.MEAN), torch.tensor(base.STD)
        clip_mean, clip_std = torch.tensor(CLIP_MEAN), torch.tensor(CLIP_STD)
        # (x * in_std + in_mean - clip_mean) / clip_std, applied to ImageNet-normalised input.
        self.register_buffer("input_scale", (in_std / clip_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("input_shift", ((in_mean - clip_mean) / clip_std).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        return self.head(self.visual(x * self.input_scale + self.input_shift))


base.MODELS[NAME] = TIMM_STANDIN
base.TRANSFORMERS = base.TRANSFORMERS | {NAME}
_create_model = timm.create_model
_added_signature = base.protocol_signature


def create_model(name, pretrained=False, num_classes=1000, **kwargs):
    if name == TIMM_STANDIN:
        return OpenClipClassifier(num_classes)
    return _create_model(name, pretrained=pretrained, num_classes=num_classes, **kwargs)


def protocol_signature(config, manifest_hash, labels):
    signature = _added_signature(config, manifest_hash, labels)
    signature["runner_wrapper"] = "run_full_species_grouped_bioclip2.py"
    if config.model == NAME:
        signature["open_clip_version"] = open_clip.__version__
        signature["model_definition"] = "BioCLIP 2 image tower + linear head; ImageNet-to-CLIP input renormalisation"
    return signature


timm.create_model = create_model
base.protocol_signature = protocol_signature

if __name__ == "__main__":
    import sys
    if "--min-images-per-label" in sys.argv and sys.argv[sys.argv.index("--min-images-per-label") + 1] != "1":
        raise SystemExit("Full-species runs use --min-images-per-label 1")
    if "--dataset-root" not in sys.argv:
        sys.argv += ["--dataset-root", str(added.full.DEFAULT_DATASET)]
    base.main()
