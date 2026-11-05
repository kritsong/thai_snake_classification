"""Which of the 139 v4 classes appear in SnakeCLEF 2023 and in BioCLIP 2's training taxa.

SnakeCLEF names are matched with the external-evaluation rule
(scripts/prepare_external_scope_v4.py: local mapping and its revision plus two
approved SnakeCLEF names). TreeOfLife-200M (BioCLIP 2's training set) is matched by exact
binomial against its species list (embeddings/txt_emb_species.json, dataset
revision 5f2dc49), with Bungarus wanghaotingi looked up under its GBIF synonym
B. multicinctus and Asthenodipsas malaccana under A. malaccanus. Frequency bands use development (training) image counts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from prepare_external_scope_v4 import mapped  # noqa: E402
from run_bioclip2_v4 import TEXT_SYNONYMS, TREEOFLIFE_SPECIES  # noqa: E402

EXT = ROOT / "paper_experiments" / "external_snakeclef2024"
V4 = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
OUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "analysis"


def main():
    man = pd.read_csv(V4 / "manifest.csv", usecols=["class_label", "split"])
    table = pd.DataFrame({
        "our_images": man.groupby("class_label").size(),
        "development_images": man[man.split == "development"].groupby("class_label").size(),
    }).fillna(0).astype(int)
    frames = [pd.read_csv(EXT / name, usecols=["binomial_name", "code"], dtype=str)
              for name in ("SnakeCLEF2023-TrainMetadata-iNat.csv", "SnakeCLEF2023-TrainMetadata-HM.csv",
                           "SnakeCLEF2023-ValMetadata.csv")]
    snakeclef = pd.concat(frames)
    snakeclef["model_class"] = [mapped(n) for n in snakeclef.binomial_name]
    snakeclef = snakeclef[snakeclef.model_class.isin(table.index)]
    table["snakeclef_images"] = snakeclef.groupby("model_class").size().reindex(table.index).fillna(0).astype(int)
    table["snakeclef_thailand_images"] = (snakeclef[snakeclef.code == "TH"].groupby("model_class").size()
                                          .reindex(table.index).fillna(0).astype(int))
    binomials = set()
    for ranks, _common in json.loads(TREEOFLIFE_SPECIES.read_text(encoding="utf-8")):
        if ranks[5].strip() and ranks[6].strip():
            binomials.add(f"{ranks[5].strip()} {ranks[6].strip().split()[0]}")
    table["in_treeoflife"] = [TEXT_SYNONYMS.get(n, n) in binomials for n in table.index]
    table["band"] = pd.cut(table.development_images, [-1, 19, 99, 10**9], labels=["<20", "20-99", ">=100"])
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "species_overlap_external_sources.csv", index_label="class_label")
    summary = table.groupby("band", observed=True).agg(
        species=("our_images", "size"),
        in_snakeclef=("snakeclef_images", lambda s: int((s > 0).sum())),
        in_treeoflife=("in_treeoflife", "sum"),
        median_development_images=("development_images", "median"),
        median_snakeclef_images=("snakeclef_images", "median"))
    print(summary.to_string())
    print("total in SnakeCLEF:", int((table.snakeclef_images > 0).sum()), "| in TreeOfLife:", int(table.in_treeoflife.sum()),
          "| SnakeCLEF has more images than we do:", int((table.snakeclef_images > table.our_images).sum()))


if __name__ == "__main__":
    main()
