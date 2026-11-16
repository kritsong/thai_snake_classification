"""Frozen image embeddings of the benchmark for public release (16 Nov 2026).

Writes snake_paper/hf_release_v4/embeddings/ (uploaded to kritaphatson/thai_snake_image_classifier as
benchmark_v4/embeddings/):
  bioclip2_vitl14.npy, dinov2_vitb14_reg4.npy, clip_vitl14_laion2b.npy  float32, 26,368 x 768, one row per image
  index.csv          row i of every embedding file: path, content hash, split, CV fold, species, label id, source
                     type and the keyed record/group/video identifiers of the public manifest
  classes.json       the 139 species in label-id order
  bioclip2_zero_shot_text_weights.npy, bioclip2_zero_shot_class_texts.csv  BioCLIP 2 text classifier (139 x 768)
  example_probe.py   trains the paper's logistic-regression probe on the development images and scores the test set
  README.md, checksums.json
The embeddings are the cached features behind the paper's probes and zero-shot results; only one pooled vector per
image is released (no patch features, images, URLs or poster fields).
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
DATASET = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
PUBLIC_MANIFEST = ROOT / "snake_paper" / "public_release_v4_taxonomy" / "data" / "manifest.csv"
OUT = ROOT / "snake_paper" / "hf_release_v4" / "embeddings"
MANIFEST_SHA = "542e09c4bd9eec06dd8fed38e114b2c13c9b50d6da7662bd279e9d4a114c764b"
SETS = {  # file name: (cache, order file)
    "bioclip2_vitl14.npy": ("bioclip2/image_features.npy", "bioclip2/feature_order.csv"),
    "dinov2_vitb14_reg4.npy": ("dinov2_probe/features.npy", "dinov2_probe/feature_order.csv"),
    "clip_vitl14_laion2b.npy": ("clip_vitl14_laion2b/image_features.npy", "clip_vitl14_laion2b/feature_order.csv"),
}
PRIVATE = re.compile(r"facebook\.com|fbcdn|story_fbid|profile\.php|C:\\|C:/|/Users/|ADMIN", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assert sha256(DATASET / "manifest.csv") == MANIFEST_SHA, "manifest changed"
    local = pd.read_csv(DATASET / "manifest.csv", dtype=str, keep_default_na=False)
    public = pd.read_csv(PUBLIC_MANIFEST, dtype=str, keep_default_na=False)
    assert public.relative_path.tolist() == local.relative_path.tolist()
    labels = sorted(local.class_label.unique())
    assert len(labels) == 139 and len(local) == 26368
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for name, (cache, order) in SETS.items():
        # The caches follow the training scripts' frame order; put them in manifest order, matching every image by
        # path and checking the hash of the stored copy the features were computed from.
        saved = pd.read_csv(RUNS / order, dtype=str, keep_default_na=False)
        assert len(saved) == len(local) and saved.relative_path.is_unique, name
        position = pd.Series(np.arange(len(saved)), index=saved.relative_path).loc[local.relative_path].to_numpy()
        assert (saved.cached_sha256.to_numpy()[position] == local.cached_sha256.to_numpy()).all(), name
        x = np.load(RUNS / cache)
        assert x.shape == (26368, 768) and x.dtype == np.float32 and np.isfinite(x).all(), (name, x.shape, x.dtype)
        np.save(OUT / name, np.ascontiguousarray(x[position]))

    index = public[["relative_path", "content_sha256", "split", "cv_fold", "class_label", "source_type",
                    "row_key", "group_key", "video_key"]].copy()
    index.insert(5, "label_id", [labels.index(c) for c in index.class_label])
    index.to_csv(OUT / "index.csv", index=False)
    (OUT / "classes.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    shutil.copyfile(RUNS / "bioclip2" / "zero_shot_text_weights.npy", OUT / "bioclip2_zero_shot_text_weights.npy")
    texts = pd.read_csv(RUNS / "bioclip2" / "zero_shot_class_texts.csv", keep_default_na=False)
    assert texts.class_label.tolist() == labels
    texts.to_csv(OUT / "bioclip2_zero_shot_class_texts.csv", index=False)
    for name in ("example_probe.py", "README.md"):
        shutil.copyfile(ROOT / "snake_paper" / "hf_release_v4" / "embeddings_src" / name, OUT / name)

    for path in OUT.iterdir():
        if path.suffix in (".csv", ".json", ".md", ".py"):
            assert not PRIVATE.search(path.read_text(encoding="utf-8")), path
    header = (OUT / "index.csv").read_text(encoding="utf-8").split("\n", 1)[0]
    assert not {"row_id", "leakage_group_id", "video_id", "component"} & set(header.split(","))
    checksums = {p.name: sha256(p) for p in sorted(OUT.iterdir())}
    (OUT / "checksums.json").write_text(json.dumps(checksums, indent=1), encoding="utf-8")
    print(len(checksums), "files,", round(sum(p.stat().st_size for p in OUT.iterdir()) / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
