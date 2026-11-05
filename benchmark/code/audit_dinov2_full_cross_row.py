"""Extend cached DINOv2 embeddings to all original images and mine neighbors."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset

from run_leakage_safe_dinov2_benchmark import build_model_and_transform


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "1 คัดแล้ว" / "combined_training_top10_leakage_safe"
OLD = ROOT / "paper_experiments" / "dinov2_linear_common32"
OUT = ROOT / "paper_experiments" / "cross_row_similarity"
FEATURES = OUT / "dinov2_all_original_features.npy"
SIGNATURE = OUT / "dinov2_all_original_feature_metadata.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Paths(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(DATASET / self.rows[index]["relative_path"]) as file:
            return self.transform(file.convert("RGB"))


def main() -> None:
    manifest = DATASET / "manifest.csv"
    manifest_hash = digest(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if FEATURES.exists() and SIGNATURE.exists():
        meta = json.loads(SIGNATURE.read_text(encoding="utf-8"))
        if meta.get("manifest_sha256") != manifest_hash:
            raise ValueError("Cached full features belong to another manifest")
        features = np.load(FEATURES)
    else:
        old_meta = json.loads((OLD / "feature_metadata.json").read_text(encoding="utf-8"))
        if old_meta["manifest_sha256"] != manifest_hash:
            raise ValueError("Cached 32-label features belong to another manifest")
        with (OLD / "selected_manifest.csv").open(encoding="utf-8-sig", newline="") as file:
            old_rows = list(csv.DictReader(file))
        old_features = np.load(OLD / "features.npy")
        if len(old_rows) != len(old_features):
            raise ValueError("Old feature row count mismatch")
        by_hash = {row["content_sha256"]: old_features[i]
                   for i, row in enumerate(old_rows)}
        missing_by_hash = {}
        for row in rows:
            if row["content_sha256"] not in by_hash:
                missing_by_hash.setdefault(row["content_sha256"], row)
        missing = list(missing_by_hash.items())
        print(f"Extracting {len(missing)} additional unique images", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, transform = build_model_and_transform(SimpleNamespace(
            model_name="vit_base_patch14_reg4_dinov2.lvd142m", image_size=224), device)
        loader = DataLoader(Paths([row for _, row in missing], transform),
                            batch_size=32, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
        blocks = []
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader, 1):
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=device.type == "cuda"):
                    block = model(batch.to(device, non_blocking=True))
                blocks.append(block.float().cpu().numpy())
                if batch_index % 25 == 0:
                    print(f"Extracted {min(batch_index * 32, len(missing))}/{len(missing)}", flush=True)
        computed = np.concatenate(blocks)
        for (content_hash, _), vector in zip(missing, computed):
            by_hash[content_hash] = vector
        features = np.stack([by_hash[row["content_sha256"]] for row in rows]).astype(np.float32)
        np.save(FEATURES, features)
        SIGNATURE.write_text(json.dumps({"manifest_sha256": manifest_hash,
                                         "images": len(rows), "feature_shape": list(features.shape),
                                         "additional_unique_images": len(missing),
                                         "model_name": "vit_base_patch14_reg4_dinov2.lvd142m",
                                         "image_size": 224}, indent=2), encoding="utf-8")
    if len(features) != len(rows):
        raise ValueError("Full feature row count mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vectors = normalize(torch.from_numpy(features).to(device), dim=1)
    pairs = []
    seen = set()
    for start in range(0, len(rows), 512):
        end = min(start + 512, len(rows))
        values, indices = (vectors[start:end] @ vectors.T).topk(30, dim=1)
        values, indices = values.cpu().numpy(), indices.cpu().numpy()
        for offset in range(end - start):
            i = start + offset
            left = rows[i]
            for cosine, raw_j in zip(values[offset], indices[offset]):
                j = int(raw_j)
                if j == i or cosine < .95:
                    continue
                right = rows[j]
                if (left["row_id"] == right["row_id"] or
                        left["leakage_group_id"] == right["leakage_group_id"]):
                    continue
                key = tuple(sorted((i, j)))
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({"cosine_similarity": round(float(cosine), 6),
                              "cross_split": left["split"] != right["split"],
                              "same_label": left["class_label"] == right["class_label"],
                              "left": {k: left[k] for k in ("relative_path", "split", "class_label", "row_id", "leakage_group_id")},
                              "right": {k: right[k] for k in ("relative_path", "split", "class_label", "row_id", "leakage_group_id")}})
    pairs.sort(key=lambda row: -row["cosine_similarity"])
    (OUT / "dinov2_all_cross_row_candidates.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in pairs) + "\n", encoding="utf-8")
    report = {"scope": "all original manifest labels", "images": len(rows),
              "candidate_pairs_cosine_ge_0_95": len(pairs),
              "cross_split_pairs": sum(row["cross_split"] for row in pairs),
              "at_least_0_99": sum(row["cosine_similarity"] >= .99 for row in pairs),
              "at_least_0_995": sum(row["cosine_similarity"] >= .995 for row in pairs),
              "same_label_pairs": dict(Counter(row["same_label"] for row in pairs)),
              "caveat": "Embedding neighbors are candidates, not proof of same encounter."}
    (OUT / "dinov2_all_cross_row_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
