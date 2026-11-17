"""Top-5 accuracy for every reported system (16 Nov 2026; the manuscript switches from top-3 to top-5).

Reads saved probabilities only (no training, no inference):
- locked test: the twelve benchmark systems (benchmark_subset_v4.SYSTEMS, loaded as in analyze_full_species_v4);
- the two 384-px models fused with BioCLIP 2 zero-shot (p proportional to p_t * p_z**lambda, lambda from
  post_hoc_decisions.json, as in venom_alert_and_tail_fusion_v4.py);
- external SnakeCLEF Thailand: observation-level probabilities saved by evaluate_external_v4.py.
Each top-1 and top-3 value is first checked against the saved analysis outputs, so the top-5 values come from the
same probabilities as every number already in the paper. Group-bootstrap intervals for test top-5 use the same
resampling as accuracy (2,000 resamples of test groups, seed 20260925).
Writes paper_experiments/full_species_v4_taxonomy/analysis/top5_accuracy.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_full_species_v4 as full  # noqa: E402
from benchmark_subset_v4 import SYSTEMS  # noqa: E402

A = full.OUT
EXTERNAL = full.RUNS / "external_snakeclef"
FUSED = {"ConvNeXt-V2 384, cross-entropy": "ConvNeXt-V2-B 384, cross-entropy",
         "ConvNeXt-V2 384, Seesaw": "ConvNeXt-V2-B 384, seesaw"}


def topk(probs, truth, k):
    return (np.argsort(-probs, axis=1)[:, :k] == truth[:, None]).any(1)


def fuse(p_trained, p_zero, lam):
    logp = np.log(np.clip(p_trained, 1e-12, None)) + lam * np.log(np.clip(p_zero, 1e-12, None))
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def group_ci(correct, groups, rng):
    codes, inverse = np.unique(groups, return_inverse=True)
    sums = np.bincount(inverse, weights=correct.astype(float))
    counts = np.bincount(inverse)
    values = []
    for _ in range(full.REPS):
        pick = rng.integers(0, len(codes), len(codes))
        values.append(sums[pick].sum() / counts[pick].sum())
    return np.percentile(values, [2.5, 97.5])


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


def main():
    rows = []
    paths = full.systems()
    reference, _ = full.load(paths["BioCLIP 2 zero-shot"])
    labels = [c for c in reference.columns if c.startswith("probability_")]
    truth = reference.label_id.astype(int).to_numpy()
    groups = reference.leakage_group_id.astype(str).to_numpy()
    saved = pd.read_csv(A / "benchmark_systems.csv").set_index("system")
    probs = {}
    for name, label in SYSTEMS.items():
        frame, p = full.load(paths[name], reference)
        probs[name] = p
        top1, top3, top5 = (topk(p, truth, k) for k in (1, 3, 5))
        assert close(top1.mean(), saved.loc[name, "accuracy"]) and close(top3.mean(), saved.loc[name, "top3_accuracy"]), name
        low, high = group_ci(top5, groups, np.random.default_rng(full.SEED))
        rows.append({"scope": "test", "system": label, "n": len(truth), "top1": top1.mean(), "top3": top3.mean(),
                     "top5": top5.mean(), "top5_ci_low": low, "top5_ci_high": high})

    decisions = json.loads((A / "post_hoc_decisions.json").read_text(encoding="utf-8"))
    p_zero = probs["BioCLIP 2 zero-shot"]
    for item in decisions["fusion"]:
        p = fuse(probs[FUSED[item["system"]]], p_zero, item["lambda"])
        top1, top3, top5 = (topk(p, truth, k) for k in (1, 3, 5))
        assert close(top1.mean(), item["test_accuracy_fused"]) and close(top3.mean(), item["test_top3_accuracy_fused"]), item["system"]
        low, high = group_ci(top5, groups, np.random.default_rng(full.SEED))
        rows.append({"scope": "test fused", "system": item["system"] + " + zero-shot", "n": len(truth),
                     "top1": top1.mean(), "top3": top3.mean(), "top5": top5.mean(), "top5_ci_low": low, "top5_ci_high": high})

    # Model output order: sorted class names, checked against the (class_label, label_id) pairs of the test file.
    classes = sorted(pd.read_csv(full.V4 / "manifest.csv", usecols=["class_label"]).class_label.unique())
    assert len(classes) == len(labels)
    index = {n: i for i, n in enumerate(classes)}
    assert all(index[c] == int(i) for c, i in zip(reference.class_label, reference.label_id))
    external = pd.read_csv(EXTERNAL / "external_systems.csv").set_index("system")
    for name, label in SYSTEMS.items():
        slug = "".join(ch if ch.isalnum() else "_" for ch in name)
        obs = pd.read_csv(EXTERNAL / f"observations_{slug}.csv", keep_default_na=False)
        cols = [c for c in obs.columns if c.startswith("probability_")]
        assert cols == labels
        p = obs[cols].to_numpy(float)
        t = np.array([index[c] for c in obs.true])
        top1, top3, top5 = (topk(p, t, k) for k in (1, 3, 5))
        assert close(top1.mean(), external.loc[name, "accuracy"]) and close(top3.mean(), external.loc[name, "top3_accuracy"]), name
        rows.append({"scope": "external", "system": label, "n": len(t), "top1": top1.mean(), "top3": top3.mean(),
                     "top5": top5.mean(), "top5_ci_low": np.nan, "top5_ci_high": np.nan})

    out = pd.DataFrame(rows)
    out.to_csv(A / "top5_accuracy.csv", index=False)
    show = out.copy()
    for c in ("top1", "top3", "top5", "top5_ci_low", "top5_ci_high"):
        show[c] = (100 * show[c]).round(1)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
