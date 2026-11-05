"""Learning-rate ablation for ConvNeXt-V2-Base under the shared grid recipe.

Identical to the grid's ConvNeXt-V2-Base run (run_full_species_grouped_added.py:
same weights, 224 px transforms, low augmentation, 10-epoch head warm-up,
50-epoch cosine schedule, early stopping and refit rule) except that it takes
the transformer learning-rate rule (backbone 2e-5, head 2e-4) instead of the
CNN rule (1e-4, 1e-3). With the SnakeCLEF-style run at 224 px, it splits the
recipe effect at equal resolution into the learning rate and everything else.
"""
from __future__ import annotations

import run_full_species_grouped_added as added

base = added.base
NAME = "ConvNeXt-V2-Base-lr2e-5"
base.MODELS[NAME] = added.ADDED_MODELS["ConvNeXt-V2-Base"]
base.TRANSFORMERS = base.TRANSFORMERS | {NAME}
_added_signature = base.protocol_signature


def protocol_signature(config, manifest_hash, labels):
    signature = _added_signature(config, manifest_hash, labels)
    signature["runner_wrapper"] = "run_full_species_grouped_ablation.py"
    signature["ablation"] = "ConvNeXt-V2-Base with the transformer learning-rate rule"
    return signature


base.protocol_signature = protocol_signature

if __name__ == "__main__":
    import sys
    if "--dataset-root" not in sys.argv:
        sys.argv += ["--dataset-root", str(added.full.DEFAULT_DATASET)]
    base.main()
