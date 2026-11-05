"""Extract the public SnakeCLEF validation images tagged Thailand.

The archive is kept intact; only metadata-listed Thailand images are copied.
The resulting manifest is an external evaluation input, not training data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_experiments" / "external_snakeclef2024"
META = OUT / "SnakeCLEF2023-ValMetadata.csv"
ARCHIVE = OUT / "SnakeCLEF2023-val-small_size.tar.gz"
IMAGES = OUT / "thailand_images"
MANIFEST = OUT / "thailand_validation_manifest.jsonl"
SUMMARY = OUT / "thailand_validation_summary.json"
LOCAL = ROOT / "1 คัดแล้ว" / "combined_training_top10_leakage_safe" / "manifest.csv"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    with META.open(encoding="utf-8-sig", newline="") as file:
        all_external = list(csv.DictReader(file))
    thailand = [row for row in all_external if row["code"] == "TH"]
    if not thailand:
        raise ValueError("No Thailand records found")
    with LOCAL.open(encoding="utf-8-sig", newline="") as file:
        local_rows = list(csv.DictReader(file))
    label_counts = Counter(row["class_label"] for row in local_rows)
    local_labels = set(label_counts)
    primary_labels = {name for name, count in label_counts.items() if count >= 100}

    desired: dict[str, dict] = {}
    for row in thailand:
        parts = PurePosixPath(row["image_path"]).parts
        if not parts or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"Unsafe image path: {row['image_path']}")
        name = "SnakeCLEF2023-small_size/" + "/".join(parts)
        if name in desired:
            raise ValueError(f"Duplicate metadata image path: {name}")
        desired[name] = row

    IMAGES.mkdir(parents=True, exist_ok=True)
    found = set()
    with tarfile.open(ARCHIVE, "r:gz") as source:
        for member in source:
            row = desired.get(member.name)
            if row is None:
                continue
            if not member.isfile() or member.issym() or member.islnk():
                raise ValueError(f"Expected regular file: {member.name}")
            relative = PurePosixPath(row["image_path"])
            destination = IMAGES.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            input_file = source.extractfile(member)
            if input_file is None:
                raise ValueError(f"Cannot extract: {member.name}")
            with input_file, destination.open("wb") as output_file:
                for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                    output_file.write(chunk)
            found.add(member.name)
    missing = sorted(set(desired) - found)
    if missing:
        raise ValueError(f"Missing {len(missing)} Thailand images, first: {missing[:3]}")

    records = []
    for row in thailand:
        path = IMAGES.joinpath(*PurePosixPath(row["image_path"]).parts)
        records.append({
            "observation_id": row["observation_id"],
            "snakeclef_species": row["binomial_name"],
            "snakeclef_class_id": row["class_id"],
            "country_code": row["code"],
            "captive": row["captive"],
            "endemic": row["endemic"],
            "subset": row["subset"],
            "source_image_path": row["image_path"],
            "relative_extracted_path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path),
            "exact_local_label_match": row["binomial_name"] in local_labels,
            "exact_primary_label_match": row["binomial_name"] in primary_labels,
        })
    with MANIFEST.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "source": "https://www.imageclef.org/node/319",
        "metadata_url": "https://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-ValMetadata.csv",
        "archive_url": "https://ptak.felk.cvut.cz/plants/plants/SnakeCLEF2023/SnakeCLEF2023-val-small_size.tar.gz",
        "metadata_sha256": digest(META),
        "archive_sha256": digest(ARCHIVE),
        "all_validation_images": len(all_external),
        "thailand_images": len(records),
        "thailand_observations": len({row["observation_id"] for row in records}),
        "thailand_species": len({row["snakeclef_species"] for row in records}),
        "exact_local_label_images": sum(row["exact_local_label_match"] for row in records),
        "exact_primary_label_images": sum(row["exact_primary_label_match"] for row in records),
        "captive_values": dict(Counter(row["captive"] for row in records)),
        "image_variant": "maximum side 240 pixels",
        "note": "The public validation set is external to our dataset, but is not the SnakeCLEF private competition test set.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
