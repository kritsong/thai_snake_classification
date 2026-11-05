"""Adds ConvNeXt-V2-Base and DINOv2-Base/14 to the full-species runner.

Everything else is run_full_species_grouped.py, i.e. the original runner with
the v4 dataset patches, so the two models get the same 224 px transforms,
AdamW optimizer, cosine schedule, warm-up, early stopping and refit rule as the
original six. The runner's family rule sets the backbone learning rate:
ConvNeXt-V2 is a convolutional network (1e-4), DINOv2 is a ViT (2e-5).
DINOv2's position embeddings are resampled by timm from 518 px to 224 px when
the pretrained weights are loaded.
"""
from __future__ import annotations

import timm

import run_full_species_grouped as full

base = full.base
ADDED_MODELS = {
    "ConvNeXt-V2-Base": "convnextv2_base.fcmae_ft_in22k_in1k",
    "DINOv2-Base/14": "vit_base_patch14_reg4_dinov2.lvd142m",
}
base.MODELS.update(ADDED_MODELS)
base.TRANSFORMERS = base.TRANSFORMERS | {"DINOv2-Base/14"}
_create_model = timm.create_model
_full_signature = base.protocol_signature


def create_model(name, **kwargs):
    if name == ADDED_MODELS["DINOv2-Base/14"]:
        kwargs.setdefault("img_size", 224)
    return _create_model(name, **kwargs)


def protocol_signature(config, manifest_hash, labels):
    signature = _full_signature(config, manifest_hash, labels)
    signature["runner_wrapper"] = "run_full_species_grouped_added.py"
    signature["backbone_learning_rate"] = 2e-5 if config.model in base.TRANSFORMERS else 1e-4
    return signature


timm.create_model = create_model
base.protocol_signature = protocol_signature

if __name__ == "__main__":
    import sys
    if "--min-images-per-label" in sys.argv and sys.argv[sys.argv.index("--min-images-per-label") + 1] != "1":
        raise SystemExit("Full-species runs use --min-images-per-label 1")
    if "--dataset-root" not in sys.argv:
        sys.argv += ["--dataset-root", str(full.DEFAULT_DATASET)]
    base.main()
