"""Split venomous-to-harmless (MIVS -> non-MIVS) errors into predictions of non-venomous species and of venomous
species that are not on the SnakeCLEF MIVS list (front-fanged elapids and vipers, and Rhabdophis keelbacks).

Writes paper_experiments/full_species_v4_taxonomy/analysis/venom_detail.csv for every benchmark system.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402
from benchmark_subset_v4 import SYSTEMS  # noqa: E402

# Non-MIVS species that are nonetheless dangerously venomous (front-fanged elapids and vipers; Rhabdophis).
VENOMOUS_NON_MIVS = {
    "Calliophis maculiceps", "Sinomicrurus macclellandi", "Hydrophis curtus", "Hydrophis cyanocinctus", "Hydrophis platurus",
    "Laticauda colubrina", "Laticauda laticaudata", "Ovophis monticola", "Trimeresurus cardamomensis",
    "Trimeresurus kanburiensis", "Trimeresurus phuketensis", "Trimeresurus popeiorum", "Trimeresurus sabahi",
    "Rhabdophis chrysargos", "Rhabdophis nigrocinctus", "Rhabdophis subminiatus",
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    venom = pd.read_csv(full.RUNS / "species_venom_status.csv", dtype=str, keep_default_na=False)
    status = dict(zip(venom.species, venom.mivs))
    assert VENOMOUS_NON_MIVS <= {s for s, m in status.items() if m == "0"}
    manifest = pd.read_csv(full.V4 / "manifest.csv", dtype=str, keep_default_na=False)
    labels = sorted(manifest.class_label.unique())
    found = full.systems()
    reference, _ = full.load(found["ConvNeXt-V2-B 384, cross-entropy"])
    truth = reference.label_id.to_numpy(int)
    true_status = np.array([status.get(labels[t], "") for t in truth])
    venomous = true_status == "1"
    rows = []
    for name, label in SYSTEMS.items():
        pred = full.load(found[name], reference)[1].argmax(1)
        pred_label = np.array([labels[p] for p in pred])
        pred_status = np.array([status.get(l, "") for l in pred_label])
        to_harmless = venomous & (pred_status == "0")
        to_venomous_non_mivs = to_harmless & np.isin(pred_label, list(VENOMOUS_NON_MIVS))
        rows.append({"system": name, "label": label, "venomous_images": int(venomous.sum()),
                     "mivs_to_non_mivs": int(to_harmless.sum()),
                     "mivs_to_venomous_non_mivs": int(to_venomous_non_mivs.sum()),
                     "mivs_to_non_venomous": int((to_harmless & ~to_venomous_non_mivs).sum()),
                     "mivs_to_unflagged": int((venomous & (pred_status == "")).sum())})
    frame = pd.DataFrame(rows)
    frame["non_venomous_share"] = frame.mivs_to_non_venomous / frame.venomous_images
    frame["unflagged_share"] = frame.mivs_to_unflagged / frame.venomous_images
    frame.to_csv(full.OUT / "venom_detail.csv", index=False)
    print(frame.round(4).to_string(index=False))
    ce = found["ConvNeXt-V2-B 384, cross-entropy"]
    pred = full.load(ce, reference)[1].argmax(1)
    pairs = pd.Series([f"{labels[t]} -> {labels[p]}" for t, p in zip(truth, pred) if t != p]).value_counts()
    print("\nTrimeresurus albolabris/macrops confusions (CE 384):",
          {k: int(v) for k, v in pairs.items() if "albolabris" in k and "macrops" in k})
    print("top 3 confusions:", pairs.head(3).to_dict())


if __name__ == "__main__":
    main()
