"""Find possible SnakeCLEF/local shared photographs using DINOv2 features."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import normalize
from torch.utils.data import DataLoader, Dataset

from run_leakage_safe_dinov2_benchmark import build_model_and_transform


ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / "1 คัดแล้ว" / "combined_training_top10_leakage_safe"
AUDIT = ROOT / "paper_experiments" / "cross_row_similarity"
EXT = ROOT / "paper_experiments" / "external_snakeclef2024"


class Images(Dataset):
    def __init__(self, rows: list[dict], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        with Image.open(ROOT / self.rows[index]["relative_extracted_path"]) as file:
            return self.transform(file.convert("RGB"))


def main() -> None:
    with (LOCAL / "manifest.csv").open(encoding="utf-8-sig", newline="") as file:
        local = list(csv.DictReader(file))
    external = [json.loads(line) for line in (EXT / "thailand_validation_manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    local_features = np.load(AUDIT / "dinov2_all_original_features.npy")
    if len(local_features) != len(local):
        raise ValueError("Local feature count mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_model_and_transform(SimpleNamespace(
        model_name="vit_base_patch14_reg4_dinov2.lvd142m", image_size=224), device)
    loader = DataLoader(Images(external, transform), batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    blocks = []
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                output = model(batch.to(device, non_blocking=True))
            blocks.append(output.float().cpu().numpy())
    external_features = np.concatenate(blocks)
    a = normalize(torch.from_numpy(external_features).to(device), dim=1)
    b = normalize(torch.from_numpy(local_features).to(device), dim=1)
    values, indices = (a @ b.T).topk(5, dim=1)
    values, indices = values.cpu().numpy(), indices.cpu().numpy()
    candidates = []
    for i, ext_row in enumerate(external):
        for cosine, j in zip(values[i], indices[i]):
            if cosine < .93:
                continue
            local_row = local[int(j)]
            candidates.append({"cosine_similarity": round(float(cosine), 6),
                               "external_observation_id": ext_row["observation_id"],
                               "external_species": ext_row["snakeclef_species"],
                               "external_image": ext_row["relative_extracted_path"],
                               "local_row_id": local_row["row_id"],
                               "local_label": local_row["class_label"],
                               "local_split": local_row["split"],
                               "local_image": local_row["relative_path"]})
    candidates.sort(key=lambda row: -row["cosine_similarity"])
    (EXT / "cross_dataset_dinov2_candidates.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in candidates) + "\n", encoding="utf-8")
    report = {"external_images": len(external), "local_images": len(local),
              "cosine_threshold": .93, "candidate_pairs": len(candidates),
              "external_images_with_candidates": len({row["external_image"] for row in candidates}),
              "observations_with_candidates": len({row["external_observation_id"] for row in candidates}),
              "caveat": "These are visual-neighbor candidates; inspect pairs before excluding or claiming overlap."}
    (EXT / "cross_dataset_dinov2_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
