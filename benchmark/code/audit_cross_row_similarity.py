"""Find visually similar images from different rows across dataset splits.

This is a candidate-generation audit, not an automatic duplicate adjudication.
The original manifest and partition assignments are never changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "1 คัดแล้ว" / "combined_training_top10_leakage_safe"
OUTPUT = ROOT / "paper_experiments" / "cross_row_similarity"
VERSION = "phash64-dct32-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_phash(path: Path) -> int:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))[:8, :8].copy()
    values = dct.ravel()
    threshold = float(np.median(values[1:]))
    bits = values > threshold
    return int("".join("1" if bit else "0" for bit in bits), 2)


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


class BKTree:
    def __init__(self) -> None:
        self.root: tuple[int, list[int], dict] | None = None

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = (value, [index], {})
            return
        node = self.root
        while True:
            distance = hamming(value, node[0])
            if distance == 0:
                node[1].append(index)
                return
            next_node = node[2].get(distance)
            if next_node is None:
                node[2][distance] = (value, [index], {})
                return
            node = next_node

    def within(self, value: int, radius: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        matches = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            distance = hamming(value, node[0])
            if distance <= radius:
                matches.extend((index, distance) for index in node[1])
            stack.extend(child for edge, child in node[2].items()
                         if distance - radius <= edge <= distance + radius)
        return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--max-distance", type=int, default=6)
    args = parser.parse_args()
    if not 0 <= args.max_distance <= 16:
        parser.error("--max-distance must be between 0 and 16")

    manifest = args.dataset_root / "manifest.csv"
    manifest_hash = sha256(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    cache_path = args.output / "image_phashes.jsonl"
    pairs_path = args.output / "cross_split_candidates.jsonl"
    all_pairs_path = args.output / "all_cross_row_candidates.jsonl"
    summary_path = args.output / "summary.json"
    with manifest.open("r", encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"relative_path", "split", "class_label", "row_id",
                "leakage_group_id", "content_sha256", "source_type"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest is empty or lacks required audit columns")

    by_hash: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_hash[row["content_sha256"]].append(row)
    cached: dict[str, int] = {}
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("algorithm") == VERSION:
                    cached[record["content_sha256"]] = int(record["phash64"], 16)

    start = time.monotonic()
    errors = []
    with cache_path.open("a", encoding="utf-8") as cache:
        for index, (content_hash, associated_rows) in enumerate(by_hash.items(), 1):
            if content_hash in cached:
                continue
            path = args.dataset_root / associated_rows[0]["relative_path"]
            try:
                value = image_phash(path)
            except Exception as exc:
                errors.append({"content_sha256": content_hash,
                               "relative_path": associated_rows[0]["relative_path"],
                               "error": str(exc)})
                continue
            cached[content_hash] = value
            cache.write(json.dumps({"algorithm": VERSION, "content_sha256": content_hash,
                                    "phash64": f"{value:016x}"}) + "\n")
            if index % 500 == 0:
                cache.flush()
                print(f"Hashed {index}/{len(by_hash)} unique files", flush=True)

    tree = BKTree()
    unique = [(content_hash, associated_rows, cached[content_hash])
              for content_hash, associated_rows in by_hash.items() if content_hash in cached]
    candidates = []
    all_candidates = []
    with pairs_path.open("w", encoding="utf-8") as destination, all_pairs_path.open("w", encoding="utf-8") as all_destination:
        for index, (content_hash, associated_rows, value) in enumerate(unique):
            current_splits = {row["split"] for row in associated_rows}
            current_groups = {row["leakage_group_id"] for row in associated_rows}
            current_rows = {row["row_id"] for row in associated_rows}
            for previous_index, distance in tree.within(value, args.max_distance):
                previous_hash, previous_rows, _ = unique[previous_index]
                previous_splits = {row["split"] for row in previous_rows}
                if current_groups & {row["leakage_group_id"] for row in previous_rows}:
                    continue
                if current_rows & {row["row_id"] for row in previous_rows}:
                    continue
                left, right = previous_rows[0], associated_rows[0]
                candidate = {
                    "phash_distance": distance,
                    "same_class_label": left["class_label"] == right["class_label"],
                    "left": {key: left[key] for key in ("relative_path", "split", "class_label",
                                                        "row_id", "leakage_group_id", "source_type")},
                    "right": {key: right[key] for key in ("relative_path", "split", "class_label",
                                                          "row_id", "leakage_group_id", "source_type")},
                    "left_content_sha256": previous_hash,
                    "right_content_sha256": content_hash,
                }
                all_destination.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                all_candidates.append(candidate)
                if current_splits.isdisjoint(previous_splits):
                    destination.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    candidates.append(candidate)
            tree.add(value, index)

    by_distance = {str(distance): sum(row["phash_distance"] == distance for row in candidates)
                   for distance in range(args.max_distance + 1)}
    summary = {
        "manifest_sha256": manifest_hash,
        "algorithm": VERSION,
        "max_phash_hamming_distance": args.max_distance,
        "manifest_images": len(rows),
        "unique_content_hashes": len(by_hash),
        "hashed_unique_images": len(unique),
        "decode_errors": errors,
        "cross_split_cross_row_candidate_pairs": len(candidates),
        "all_cross_row_candidate_pairs": len(all_candidates),
        "within_split_cross_row_candidate_pairs": len(all_candidates) - len(candidates),
        "candidate_pairs_by_distance": by_distance,
        "same_label_candidate_pairs": sum(row["same_class_label"] for row in candidates),
        "different_label_candidate_pairs": sum(not row["same_class_label"] for row in candidates),
        "elapsed_seconds": round(time.monotonic() - start, 1),
        "interpretation": "Candidates require visual review; pHash distance alone does not prove the same encounter or individual.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
