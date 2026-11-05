"""Venom status for the 139 species, from SnakeCLEF's medically important venomous species list.

MIVS flags come from SnakeCLEF 2024 (venomous_status_list.csv, class_id -> MIVS),
named through the SnakeCLEF 2023 training and validation metadata. A local species
receives the flag of the SnakeCLEF name that equals it after the approved local
name mapping and its expert revision of 20 October 2026, or after the
author-approved SnakeCLEF name map; for the venom flag
only, Bungarus wanghaotingi uses Bungarus multicinctus (author decision,
8 October 2026). Species without a match are 'unavailable'.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_external_scope_v4 import EXTERNAL_NAME_MAP  # noqa: E402
from species_mapping import final_class  # noqa: E402

EXT = ROOT / "paper_experiments" / "external_snakeclef2024"
V4 = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
OUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "species_venom_status.csv"
FILES = {"venomous_status_list_snakeclef2024.csv": "e51138ae1dbb72faacf07b68a464b3696a88ec19345876bd6edc86abb83533fc",
         "SnakeCLEF2023-TrainMetadata-HM.csv": "83a93d4972ab4bb9b667fe86dccb66836c2695b67f3df809c777157fd6b9a4c7",
         "SnakeCLEF2023-TrainMetadata-iNat.csv": "0748b85225610459038d5899312e7ca73e89eaa469ffeb56134c4793c5b653a6",
         "SnakeCLEF2023-ValMetadata.csv": "dd1ce7c96cb36225675bd7b73b69d083dee9b011e06425f5c064fe6822ff2a68"}


def main():
    for name, digest in FILES.items():
        assert hashlib.sha256((EXT / name).read_bytes()).hexdigest() == digest, f"{name} changed"
    names = pd.concat([pd.read_csv(EXT / f, usecols=["class_id", "binomial_name"]) for f in list(FILES)[1:]]).drop_duplicates()
    venom = pd.read_csv(EXT / "venomous_status_list_snakeclef2024.csv").merge(names, on="class_id")

    def to_class(n):
        if n in EXTERNAL_NAME_MAP:
            return EXTERNAL_NAME_MAP[n], "author-approved SnakeCLEF name map"
        if final_class(n) != n:
            return final_class(n), "approved local name mapping"
        return n, "exact name"

    venom[["model_class", "basis"]] = venom.binomial_name.apply(lambda n: pd.Series(to_class(n)))
    venom = venom.sort_values("basis").drop_duplicates("model_class")
    species = sorted(pd.read_csv(V4 / "manifest.csv", usecols=["class_label"]).class_label.unique())
    rows = []
    for s in species:
        hit = venom[venom.model_class == s]
        if len(hit):
            h = hit.iloc[0]
            rows.append({"species": s, "mivs": int(h.MIVS), "snakeclef_name": h.binomial_name,
                         "snakeclef_class_id": int(h.class_id), "basis": h.basis})
        else:
            rows.append({"species": s, "mivs": "", "snakeclef_name": "", "snakeclef_class_id": "", "basis": "unavailable"})
    frame = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False)
    print(frame.basis.value_counts().to_dict(), "| MIVS species:", int((frame.mivs == 1).sum()))
    print("unavailable:", frame.loc[frame.basis == "unavailable", "species"].tolist())


if __name__ == "__main__":
    main()
