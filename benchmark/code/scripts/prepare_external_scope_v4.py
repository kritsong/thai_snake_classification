"""External SnakeCLEF Thailand scope for the 139-species models (dataset v4).

Uses the frozen Thailand extraction of the public SnakeCLEF 2023 validation set
(409 images, 223 observations, 99 names; 240 px small-size variant). A SnakeCLEF
name is in scope only when it equals one of the 139 class names after applying
the same approved local name mapping and its expert revision of 20 October 2026
(scripts/species_mapping.py), i.e. exact string identity or an approved
spelling/synonym/subspecies merge. No new taxonomic equivalence is introduced
beyond two author-approved SnakeCLEF name mappings (EXTERNAL_NAME_MAP); since the
revision, SnakeCLEF's Fowlea piscator matches the local class of that name. Observation 96180737, a confirmed
copy of a local photograph, is excluded entirely.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from species_mapping import final_class  # noqa: E402

EXTERNAL = ROOT / "paper_experiments" / "external_snakeclef2024"
V4 = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
OUT = EXTERNAL / "full_species_v4_taxonomy_scope"
EXCLUDED = {"96180737"}
# Author decision, 8 October 2026 (Fowlea piscator removed after the revision of 20 October 2026):
# these SnakeCLEF names denote the same model classes.
EXTERNAL_NAME_MAP = {
    "Fowlea flavipunctata": "Fowlea flavipunctatus",   # Latin gender ending of the same name
    "Bungarus multicinctus": "Bungarus wanghaotingi",  # GBIF synonym; local display name kept
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapped(name: str):
    name = name.strip()
    if name in EXTERNAL_NAME_MAP:
        return EXTERNAL_NAME_MAP[name]
    return final_class(name)


def main():
    classes = set(pd.read_csv(V4 / "manifest.csv", usecols=["class_label"]).class_label)
    records = [json.loads(l) for l in (EXTERNAL / "thailand_validation_manifest.jsonl").read_text(encoding="utf-8").splitlines() if l]
    kept, reasons = [], defaultdict(set)
    for r in records:
        target = mapped(r["snakeclef_species"])
        if r["observation_id"] in EXCLUDED:
            reasons["excluded_known_overlap_observation"].add(r["snakeclef_species"])
            continue
        if target is None or target not in classes:
            reasons["no_matching_class"].add(r["snakeclef_species"])
            continue
        path = ROOT / r["relative_extracted_path"]
        assert sha256(path) == r["sha256"], f"External image changed: {path}"
        kept.append({**r, "model_class": target,
                     "match": "exact" if target == r["snakeclef_species"] else "approved_name_mapping"})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scope.jsonl").write_text("\n".join(json.dumps(k, ensure_ascii=False) for k in kept) + "\n", encoding="utf-8")
    frame = pd.DataFrame(kept)
    summary = {
        "source_manifest_sha256": sha256(EXTERNAL / "thailand_validation_manifest.jsonl"),
        "scope_sha256": sha256(OUT / "scope.jsonl"),
        "images": len(frame), "observations": int(frame.observation_id.nunique()),
        "species": int(frame.model_class.nunique()),
        "matched_by_approved_mapping": sorted(set(frame.loc[frame.match != "exact", "snakeclef_species"] + " -> " + frame.loc[frame.match != "exact", "model_class"])),
        "unmatched_snakeclef_names": sorted(reasons["no_matching_class"]),
        "excluded_observations": sorted(EXCLUDED),
        "rule": "SnakeCLEF name equals a model class after the approved local name mapping; no other equivalence; observation-level probability averaging at evaluation",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "unmatched_snakeclef_names"}, indent=2, ensure_ascii=False))
    print("unmatched names:", len(summary["unmatched_snakeclef_names"]))
    print(summary["unmatched_snakeclef_names"])


if __name__ == "__main__":
    main()
