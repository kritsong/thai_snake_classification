"""Dataset v4: fresh 80/20 grouped split of the full-species corpus, with a resized image cache.

Author decisions (7 October 2026):
  * draw a fresh development/test split (the inherited v3 test is not reused);
  * 80/20 by images, grouped by post-aware components, balanced by species;
  * species seen in fewer than 3 independent components stay in development
    (they cannot be tested), as do the 152 recovered/name-fixed images;
  * every image is resized once (shorter side 448 px, Lanczos, JPEG q95,
    4:4:4) and all models read the cache.
Images come from v3 (26,363) plus the 5 development images v3 excluded only to
protect its inherited test; with a fresh component-level split they are valid.
EXIF orientation is not applied, matching the original runner's pixel order.
"""
from __future__ import annotations

import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "1 คัดแล้ว"
V2_ROOT = CORPUS / "combined_training_top10_similarity_audited_provisional_v2"
V3_ROOT = CORPUS / "full_species_post_aware_v3_20261007"
V3_SHA = "05d187722cc729c87ac9f726fc4c837b817c848038056286b24cb21256df132d"
OUTPUT = CORPUS / "full_species_fresh_split_v4_20261007"
SEED, FOLDS, TEST_SHARE, MIN_COMPONENTS, SHORT_SIDE = 20260925, 5, 0.20, 3, 448


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_one(job):
    source, target = job
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as raw:
        image = raw.convert("RGB")
    width, height = image.size
    scale = SHORT_SIDE / min(width, height)
    if scale < 1:
        image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.LANCZOS)
    image.save(target, format="JPEG", quality=95, subsampling=0)
    return sha256(target), image.size


def main():
    assert sha256(V3_ROOT / "manifest.csv") == V3_SHA, "v3 manifest changed"
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists; refusing to overwrite")
    v3 = pd.read_csv(V3_ROOT / "manifest.csv", dtype=str, keep_default_na=False)
    v3["source_file"] = [str(V3_ROOT / p) for p in v3.relative_path]
    back = pd.read_csv(V3_ROOT / "excluded_test_linked.csv", dtype=str, keep_default_na=False)
    back["source_file"] = [str(V2_ROOT / p) for p in back.v2_relative_path]
    for item in back.itertuples(index=False):
        assert sha256(Path(item.source_file)) == item.content_sha256, "Excluded image bytes changed"
    keep = ["source_file", "class_label", "source_label", "row_id", "component", "content_sha256",
            "source_type", "video_id", "provenance"]
    frame = pd.concat([v3[keep], back[keep]], ignore_index=True)

    # Which components may enter the test set.
    per_species = frame.groupby("class_label").component.nunique()
    rare = set(per_species[per_species < MIN_COMPONENTS].index)
    locked = set(frame.loc[frame.class_label.isin(rare), "component"])
    locked |= set(frame.loc[frame.provenance != "v2_manifest", "component"])
    frame["locked_reason"] = ""
    frame.loc[frame.component.isin(frame.loc[frame.provenance != "v2_manifest", "component"]), "locked_reason"] = "contains recovered or name-fixed image"
    frame.loc[frame.component.isin(frame.loc[frame.class_label.isin(rare), "component"]), "locked_reason"] = f"contains species with <{MIN_COMPONENTS} components"

    # Pre-registered rule: the first fold of a shuffled 5-fold StratifiedGroupKFold over
    # eligible components becomes the test set (about 20% of eligible images).
    eligible = frame[~frame.component.isin(locked)]
    splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_SHARE), shuffle=True, random_state=SEED)
    _, test_idx = next(splitter.split(eligible, eligible.class_label, eligible.component))
    test_components = set(eligible.iloc[test_idx].component)
    # Every tested species must also be learnable: return its smallest test
    # component to development until each tested species has development images.
    moved = []
    while True:
        in_test = frame.component.isin(test_components)
        dev_species = set(frame.loc[~in_test, "class_label"])
        orphan = sorted(set(frame.loc[in_test, "class_label"]) - dev_species)
        if not orphan:
            break
        species = orphan[0]
        sizes = frame[in_test & frame.class_label.eq(species)].groupby("component").size().sort_values(kind="stable")
        test_components.discard(sizes.index[0])
        moved.append({"species": species, "component": sizes.index[0], "direction": "test_to_development"})
    # Coverage rule: an eligible species (>= MIN_COMPONENTS components) left untested
    # receives its smallest unlocked development component, if the species keeps at
    # least two development components and every species in that component keeps one.
    eligible_species = sorted(per_species[per_species >= MIN_COMPONENTS].index)
    for species in eligible_species:
        in_test = frame.component.isin(test_components)
        if (in_test & frame.class_label.eq(species)).any():
            continue
        dev_rows = frame[~in_test & ~frame.component.isin(locked)]
        candidates = (dev_rows[dev_rows.class_label.eq(species)].groupby("component").size()
                      .reset_index(name="n").sort_values(["n", "component"], kind="stable"))
        for component in candidates.component:
            remaining = frame[~in_test & frame.component.ne(component)]
            members = set(frame.loc[frame.component.eq(component), "class_label"])
            if (remaining[remaining.class_label.eq(species)].component.nunique() >= 2
                    and members <= set(remaining.class_label)):
                test_components.add(component)
                moved.append({"species": species, "component": component, "direction": "development_to_test"})
                break
    frame["split"] = np.where(frame.component.isin(test_components), "test", "development")

    dev = frame[frame.split == "development"]
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    frame["cv_fold"] = "test"
    for number, (_, val_idx) in enumerate(folds.split(dev, dev.class_label, dev.component), start=1):
        frame.loc[dev.index[val_idx], "cv_fold"] = str(number)

    for key in ("component", "row_id", "content_sha256"):
        assert frame.groupby(key).split.nunique().max() == 1, f"{key} crosses development/test"
        assert frame[frame.split == "development"].groupby(key).cv_fold.nunique().max() == 1, f"{key} crosses CV folds"

    # Resized cache, one file per record (records are unique by source file).
    assert not frame.source_file.duplicated().any()
    frame["relative_path"] = [f"images/{h[:2]}/{h}_{i:05d}.jpg" for i, h in enumerate(frame.content_sha256)]
    jobs = [(Path(s), OUTPUT / r) for s, r in zip(frame.source_file, frame.relative_path)]
    OUTPUT.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=14) as pool:
        results = list(pool.map(cache_one, jobs, chunksize=64))
    frame["cached_sha256"] = [h for h, _ in results]
    frame["cached_width"] = [s[0] for _, s in results]
    frame["cached_height"] = [s[1] for _, s in results]
    frame["v3_or_v2_source"] = [str(Path(s).relative_to(CORPUS)) for s in frame.source_file]

    columns = ["relative_path", "split", "cv_fold", "class_label", "source_label", "row_id", "component",
               "content_sha256", "cached_sha256", "cached_width", "cached_height", "source_type", "video_id",
               "provenance", "locked_reason", "v3_or_v2_source"]
    frame[columns].to_csv(OUTPUT / "manifest.csv", index=False, encoding="utf-8")
    pd.DataFrame(moved).to_csv(OUTPUT / "split_adjustments.csv", index=False)

    test = frame[frame.split == "test"]
    summary = {
        "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": {"v3_manifest_sha256": V3_SHA, "readded_v3_test_linked_images": len(back)},
        "manifest_sha256": sha256(OUTPUT / "manifest.csv"),
        "images": len(frame), "species": int(frame.class_label.nunique()), "components": int(frame.component.nunique()),
        "split_images": frame.split.value_counts().to_dict(),
        "split_components": frame.groupby("split").component.nunique().to_dict(),
        "test_share_of_images": round(len(test) / len(frame), 4),
        "species_with_test_images": int(test.class_label.nunique()),
        "species_development_only": int(frame.class_label.nunique() - test.class_label.nunique()),
        "rare_species_locked_to_development": len(rare),
        "locked_components": len(locked), "locked_images": int(frame.component.isin(locked).sum()),
        "split_adjustments": pd.DataFrame(moved).direction.value_counts().to_dict() if moved else {},
        "fold_images": frame[frame.split == "development"].cv_fold.value_counts().sort_index().to_dict(),
        "split_rule": (f"first fold of StratifiedGroupKFold(n_splits={round(1 / TEST_SHARE)}, shuffle=True, random_state={SEED}) "
                       f"over eligible components, stratified by species, then each untested species with >={MIN_COMPONENTS} components gets its smallest unlocked development component when it keeps >=2 in development; components with a species seen in <{MIN_COMPONENTS} "
                       "components or with recovered/name-fixed images are development-only"),
        "fold_rule": f"StratifiedGroupKFold(n_splits={FOLDS}, shuffle=True, random_state={SEED}) over development components",
        "scikit_learn": __import__("sklearn").__version__,
        "image_cache": f"RGB, shorter side resized to {SHORT_SIDE} px with Lanczos when larger (never upscaled), JPEG quality 95, 4:4:4; EXIF orientation not applied",
        "checks": "source bytes verified in v3 build and for re-added images; no component, row, or content hash crosses development/test or CV folds",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
