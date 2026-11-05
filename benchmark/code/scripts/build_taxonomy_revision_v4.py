"""Taxonomy revision of dataset v4: same images, split and folds; two label corrections.

Expert decisions (N. Ratnarathorn, 20 October 2026):
  * Fowlea piscator is a species distinct from F. flavipunctatus. Records labelled
    Xenochrophis piscator, merged into F. flavipunctatus in v4, become F. piscator.
  * Pareas malaccanus is now Asthenodipsas malaccana; it is merged with the existing
    Asthenodipsas malaccanus class under the current name A. malaccana.
  * Bungarus wanghaotingi, Trimeresurus sabahi and Ovophis monticola are kept.
Only class_label changes; source_label keeps the original string. The split and the
CV folds of v4 are kept as drawn (no re-split): F. piscator has development and test
groups; the three A. malaccana groups remain in development.
Images are hard links to the v4 cache, so their bytes are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "1 คัดแล้ว"
V4 = CORPUS / "full_species_fresh_split_v4_20261007"
V4_SHA = "10c26f99fc31468475ec1099176d10e97d11427d71f61a99e40c11f62c856dc0"
OUTPUT = CORPUS / "full_species_fresh_split_v4_taxonomy_20261020"
SPLIT_PISCATOR = ("Fowlea flavipunctatus", "Xenochrophis piscator", "Fowlea piscator")
MERGE = {"Pareas malaccanus": "Asthenodipsas malaccana", "Asthenodipsas malaccanus": "Asthenodipsas malaccana"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assert sha256(V4 / "manifest.csv") == V4_SHA, "v4 manifest changed"
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} exists; refusing to overwrite")
    frame = pd.read_csv(V4 / "manifest.csv", dtype=str, keep_default_na=False)
    old = frame.class_label.copy()

    old_class, source, new_class = SPLIT_PISCATOR
    piscator = (frame.class_label == old_class) & (frame.source_label == source)
    frame.loc[piscator, "class_label"] = new_class
    merged = frame.class_label.isin(MERGE)
    frame.loc[merged, "class_label"] = frame.loc[merged, "class_label"].map(MERGE)
    changed = frame.class_label != old
    assert piscator.sum() == 49 and merged.sum() == 5 and changed.sum() == 54
    assert frame.class_label.nunique() == old.nunique() == 139

    # Groups stay single-species where they were, and no group, row or hash crosses the split or folds.
    assert (frame[changed].groupby("component").class_label.nunique() == 1).all()
    assert frame.groupby("component").class_label.nunique().max() == old.groupby(frame.component).nunique().max()
    for key in ("component", "row_id", "content_sha256"):
        assert frame.groupby(key).split.nunique().max() == 1, f"{key} crosses development/test"
        assert frame[frame.split == "development"].groupby(key).cv_fold.nunique().max() == 1, f"{key} crosses CV folds"

    OUTPUT.mkdir(parents=True)
    for relative in frame.relative_path:
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(V4 / relative, target)
    frame.to_csv(OUTPUT / "manifest.csv", index=False, encoding="utf-8")
    changes = frame.loc[changed, ["relative_path", "split", "cv_fold", "component", "source_label"]].assign(
        v4_class_label=old[changed], class_label=frame.class_label[changed])
    changes.to_csv(OUTPUT / "label_changes.csv", index=False, encoding="utf-8")

    test = frame[frame.split == "test"]
    summary = json.loads((V4 / "summary.json").read_text(encoding="utf-8"))
    summary.update({
        "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": {"v4_manifest_sha256": V4_SHA},
        "manifest_sha256": sha256(OUTPUT / "manifest.csv"),
        "species": int(frame.class_label.nunique()),
        "species_with_test_images": int(test.class_label.nunique()),
        "species_development_only": int(frame.class_label.nunique() - test.class_label.nunique()),
        "taxonomy_revision": {
            "decided_by": "N. Ratnarathorn, 20 October 2026",
            "split": f"{source} records ({int(piscator.sum())} images) from {old_class} to {new_class}",
            "merge": f"{' and '.join(MERGE)} ({int(merged.sum())} images) to {'Asthenodipsas malaccana'}",
            "kept": "Bungarus wanghaotingi, Trimeresurus sabahi, Ovophis monticola",
            "split_and_folds": "v4 split and CV folds kept unchanged (no re-split)",
        },
        "changed_images": {"development": int((changed & (frame.split == "development")).sum()),
                           "test": int((changed & (frame.split == "test")).sum())},
    })
    for key in ("species_with_test_images", "species_development_only"):
        summary[f"v4_{key}"] = json.loads((V4 / "summary.json").read_text(encoding="utf-8"))[key]
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(changes.groupby(["v4_class_label", "class_label", "split"]).agg(
        images=("relative_path", "size"), groups=("component", "nunique"),
        folds=("cv_fold", lambda f: ",".join(sorted(set(f))))).to_string())


if __name__ == "__main__":
    main()
