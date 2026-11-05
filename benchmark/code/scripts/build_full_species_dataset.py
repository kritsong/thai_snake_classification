"""Build the full-species, post-aware dataset (v3) from the provisional v2 manifest.

Author decisions applied (7 October 2026):
  * approved species mapping (scripts/species_mapping.py): genus-only labels
    removed, subspecies merged to species, spelling/synonym variants merged,
    Xenochrophis piscator merged into Fowlea flavipunctatus;
  * row-level label corrections (snake_paper/label_corrections_20261007.csv);
  * 129 photos recovered from multi-snake posts via each row's Facebook link;
  * set-aside images whose only conflict was a name variant are reinstated.

Grouping and partitions:
  * groups are connected components of the v2 leakage groups, spreadsheet
    rows, and explicit Facebook post keys, so a post never spans partitions;
  * the v2 test partition is protected: development images linked to a test
    group are excluded, and added images are never placed in test;
  * added images join their component's development partition, or training.
Five StratifiedGroupKFold folds are drawn over development groups. Files are
hard links, so no media are copied. Writes the manifest, exclusions, summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from species_mapping import MAPPING, REMOVE  # noqa: E402

CORPUS = ROOT / "1 คัดแล้ว"
V1_ROOT = CORPUS / "combined_training_top10_leakage_safe"
V2_ROOT = CORPUS / "combined_training_top10_similarity_audited_provisional_v2"
V2_SHA = "795b90f8cb7e44b3a1df096703383993fa208f00b52c6d4edc71179d85ead844"
WORKBOOK = ROOT / "snake_found_new.xlsx"
WORKBOOK_SHA = "d72b7a2ba62fde50b22a36976cee9f70837bd21b5abcb8b284019ecd2494c8f6"
CORRECTIONS = ROOT / "snake_paper" / "label_corrections_20261007.csv"
RECOVERED = ROOT / "snake_paper" / "album_match_20261007" / "recovered_photos.csv"
OUTPUT = CORPUS / "full_species_post_aware_v3_20261007"
SEED, FOLDS = 20260925, 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def post_keys(raw) -> set[str]:
    """Same definition as scripts/audit_source_post_metadata.py."""
    keys = set()
    for text in re.findall(r"https?://[^\s<>]+", str(raw)):
        parsed = urlparse(text.rstrip(".,;)"))
        if not parsed.hostname or not parsed.hostname.endswith("facebook.com"):
            continue
        query = parse_qs(parsed.query)
        for value in query.get("set", []):
            match = re.fullmatch(r"(?:pcb|gm)\.(\d+)", value)
            if match:
                keys.add("facebook_post:" + match[1])
        for value in query.get("story_fbid", []):
            if value.isdigit():
                keys.add("facebook_post:" + value)
        match = re.search(r"/(?:posts|permalink)/(\d+)(?:/|$)", parsed.path)
        if match:
            keys.add("facebook_post:" + match[1])
        match = re.search(r"/photos/(?:pcb|gm)\.(\d+)/", parsed.path)
        if match:
            keys.add("facebook_post:" + match[1])
    return keys


def species(label: str):
    label = label.strip(" ")
    return MAPPING[label][0] if label in MAPPING else label


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def load_additions(v2: pd.DataFrame, corrections: dict) -> pd.DataFrame:
    rows = []
    # Photos recovered from multi-snake posts (one copy per distinct photo).
    recovered = pd.read_csv(RECOVERED, dtype={"row_id": str})
    for item in recovered.itertuples(index=False):
        label = corrections.get(item.row_id, item.label)
        rows.append({"source_file": item.example_path, "row_id": item.row_id, "source_label": label,
                     "content_sha256": item.content_sha256, "source_type": item.source_type,
                     "seed_group": "post_" + item.group, "video_id": "",
                     "provenance": "recovered_" + item.basis.replace(" + ", "_and_")})
    # Set-aside images whose only conflict was a name variant or a genus-only label.
    quarantine = pd.read_json(V2_ROOT / "quarantined_conflicting_rows.jsonl", lines=True, dtype={"row_id": str})
    v1 = pd.read_csv(V1_ROOT / "manifest.csv", dtype=str, keep_default_na=False).set_index("relative_path")
    for component, part in quarantine.groupby("audit_component_id"):
        mapped = {species(label) for label in part.class_label}
        kept = {m for m in mapped if m is not REMOVE}
        if len(kept) != 1:
            continue
        for item in part.itertuples(index=False):
            if species(item.class_label) is REMOVE:
                continue
            source = v1.loc[item.source_relative_path]
            rows.append({"source_file": str(V1_ROOT / item.source_relative_path), "row_id": item.row_id,
                         "source_label": corrections.get(item.row_id, item.class_label),
                         "content_sha256": source.content_sha256, "source_type": source.source_type,
                         "seed_group": "component_" + component, "video_id": source.video_id,
                         "provenance": "reinstated_name_variant"})
    added = pd.DataFrame(rows)
    assert not added.content_sha256.duplicated().any(), "An added photo appears twice"
    already = added.content_sha256.isin(set(v2.content_sha256))
    return added[~already].reset_index(drop=True), int(already.sum())


def main():
    assert sha256(V2_ROOT / "manifest.csv") == V2_SHA, "v2 manifest changed"
    assert sha256(WORKBOOK) == WORKBOOK_SHA, "Workbook changed"
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists; refusing to overwrite a built dataset")
    v2 = pd.read_csv(V2_ROOT / "manifest.csv", dtype=str, keep_default_na=False)
    corrections = dict(pd.read_csv(CORRECTIONS, dtype=str)[["row_id", "corrected_label"]].values)
    added, duplicate_additions = load_additions(v2, corrections)

    base = pd.DataFrame({
        "source_file": [str(V2_ROOT / p) for p in v2.relative_path], "row_id": v2.row_id,
        "source_label": [corrections.get(r, l) for r, l in zip(v2.row_id, v2.class_label)],
        "content_sha256": v2.content_sha256, "source_type": v2.source_type,
        "seed_group": "v2_" + v2.leakage_group_id, "video_id": v2.video_id,
        "provenance": "v2_manifest", "v2_split": v2.split, "v2_relative_path": v2.relative_path,
    })
    added["v2_split"] = ""
    added["v2_relative_path"] = ""
    frame = pd.concat([base, added], ignore_index=True)
    frame["class_label"] = frame.source_label.map(species)
    removed = frame[frame.class_label.isna()]
    frame = frame[frame.class_label.notna()].reset_index(drop=True)

    # Post-aware components over v2 groups, rows, and explicit post keys.
    workbook = pd.read_excel(WORKBOOK, sheet_name="Snake list", usecols=["Figure Link"]).fillna("")
    graph = Components()
    for item in frame.itertuples(index=False):
        graph.union("row:" + item.row_id, item.seed_group)
        for key in post_keys(workbook.iloc[int(item.row_id)]["Figure Link"]):
            graph.union("row:" + item.row_id, key)
        graph.union("row:" + item.row_id, "hash:" + item.content_sha256)
    frame["component"] = ["post_aware_" + hashlib.sha256(graph.find("row:" + r).encode()).hexdigest()[:16] for r in frame.row_id]

    # Partitions: test is protected; added images go to development only.
    has_test = frame[frame.v2_split == "test"].groupby("component").size()
    frame["split"] = frame.v2_split.replace("", "train")
    linked = frame.component.isin(has_test.index) & frame.split.ne("test")
    excluded = frame[linked].assign(reason=np.where(frame.loc[linked, "provenance"].eq("v2_manifest"),
                                                    "development image linked to a test group by row/post",
                                                    "added image linked to a test group; test kept unchanged"))
    frame = frame[~linked].reset_index(drop=True)
    frame["split"] = frame.split.where(frame.split.eq("test"), "development")

    # Five grouped folds over development.
    dev = frame[frame.split == "development"]
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    frame["cv_fold"] = "test"
    for number, (_, val_idx) in enumerate(splitter.split(dev, dev.class_label, dev.component), start=1):
        frame.loc[dev.index[val_idx], "cv_fold"] = str(number)

    # Integrity: rehash every file, then check disjointness.
    actual = [sha256(Path(p)) for p in frame.source_file]
    bad = [p for p, a, e in zip(frame.source_file, actual, frame.content_sha256) if a != e]
    assert not bad, f"{len(bad)} files do not match their recorded hash; first {bad[:1]}"
    for key in ("component", "row_id", "content_sha256"):
        assert frame.groupby(key).split.nunique().max() == 1, f"{key} crosses development/test"
        assert frame[frame.split == "development"].groupby(key).cv_fold.nunique().max() == 1, f"{key} crosses CV folds"
    # v2 already holds repeated bytes within one group (shared videos/photos across
    # same-label rows); hash nodes join the graph, so a repeat never spans groups.
    assert not frame[frame.provenance != "v2_manifest"].content_sha256.isin(
        frame[frame.provenance == "v2_manifest"].content_sha256).any(), "An added image duplicates v2 content"

    # Hard-link media into the new root.
    OUTPUT.mkdir(parents=True)
    relative = []
    for item in frame.itertuples(index=False):
        rel = (f"media/v2/{item.v2_relative_path}" if item.provenance == "v2_manifest"
               else f"media/added/row_{item.row_id}/{item.content_sha256[:16]}{Path(item.source_file).suffix.lower()}")
        target = OUTPUT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(item.source_file, target)
        relative.append(rel)
    frame["relative_path"] = relative
    columns = ["relative_path", "split", "cv_fold", "class_label", "source_label", "row_id", "component",
               "seed_group", "content_sha256", "source_type", "video_id", "provenance", "v2_split", "v2_relative_path"]
    frame[columns].to_csv(OUTPUT / "manifest.csv", index=False, encoding="utf-8")
    excluded.drop(columns=["source_file"]).to_csv(OUTPUT / "excluded_test_linked.csv", index=False, encoding="utf-8")
    removed.drop(columns=["source_file"]).to_csv(OUTPUT / "removed_genus_only.csv", index=False, encoding="utf-8")

    counts = frame.groupby("class_label").agg(images=("relative_path", "size"), groups=("component", "nunique"))
    test_species = set(frame.loc[frame.split == "test", "class_label"])
    summary = {
        "built_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "inputs": {"v2_manifest_sha256": V2_SHA, "workbook_sha256": WORKBOOK_SHA,
                   "species_mapping_sha256": sha256(ROOT / "scripts" / "species_mapping.py"),
                   "corrections_sha256": sha256(CORRECTIONS), "recovered_sha256": sha256(RECOVERED)},
        "manifest_sha256": sha256(OUTPUT / "manifest.csv"),
        "images": len(frame), "species": int(frame.class_label.nunique()),
        "groups": int(frame.component.nunique()),
        "split_images": frame.split.value_counts().to_dict(),
        "split_groups": frame.groupby("split").component.nunique().to_dict(),
        "fold_images": frame[frame.split == "development"].cv_fold.value_counts().sort_index().to_dict(),
        "species_with_test_images": len(test_species),
        "species_development_only": int(frame.class_label.nunique() - len(test_species)),
        "species_by_groups": {">=5": int((counts.groups >= 5).sum()), "3-4": int(counts.groups.between(3, 4).sum()),
                              "1-2": int((counts.groups <= 2).sum())},
        "added_images": frame.provenance.ne("v2_manifest").sum().item(),
        "added_by_provenance": frame[frame.provenance != "v2_manifest"].provenance.value_counts().to_dict(),
        "added_already_present_skipped": duplicate_additions,
        "removed_genus_only_images": len(removed),
        "repeated_content_records_inherited_from_v2": int(frame.content_sha256.duplicated(keep=False).sum()),
        "excluded_test_linked": excluded.reason.value_counts().to_dict(),
        "test_unchanged_from_v2": bool(set(frame.loc[frame.split == "test", "v2_relative_path"]) <= set(v2.loc[v2.split == "test", "relative_path"])),
        "v2_test_images_retained": int((frame.split == "test").sum()),
        "v2_test_images_removed_genus_only": int((removed.v2_split == "test").sum()),
        "fold_assignment": f"StratifiedGroupKFold(n_splits={FOLDS}, shuffle=True, random_state={SEED}) on components, "
                           f"stratified by species; scikit-learn {__import__('sklearn').__version__}; stored in manifest cv_fold",
        "checks": "every file rehashed; no component, row, or content hash crosses development/test or CV folds",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
