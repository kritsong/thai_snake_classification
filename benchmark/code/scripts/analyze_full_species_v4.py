"""Analysis of the full-species v4 study (reads saved predictions only; no fitting).

Systems: the nine CV-selected architectures (the 12-configuration grid, the
4-configuration grid that added ConvNeXt-V2-Base and DINOv2-Base/14, and the
2-configuration BioCLIP 2 grid), the
frozen DINOv2 and BioCLIP 2 logistic-regression probes, BioCLIP 2 zero-shot, and
the SnakeCLEF-style ConvNeXt-V2 pair (seesaw, CE).
Every system is scored on the same locked test images. Outputs:
  * cv_configurations.csv: all grid configurations, five-fold mean and SD;
  * test_systems.csv: point estimates with 95% species-stratified group-bootstrap intervals,
    including macro-F1 over species present in the test set;
  * frequency_bands.csv: accuracy and mean species recall by training frequency;
  * paired_comparisons.csv: paired component-bootstrap differences (same
    resamples for both systems), McNemar tests for clustered matched pairs
    (Durkalski et al. 2003; primary) and image-level exact McNemar tests
    (reference), each Holm-adjusted;
  * poster_overlap_sensitivity.csv: test accuracy after removing test images
    whose reported poster also appears in development (poster != photographer);
  * genus_level.csv: accuracy when only the predicted genus must be correct;
  * per_species_test.csv: support, precision, recall, F1 for every system.
Missing systems are skipped, so this can run on partial results.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
# Results after the taxonomy revision of 20 Oct 2026 (v4 split; scripts/build_taxonomy_revision_v4.py).
RUNS = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
ADDED = RUNS / "grid_added"
BIOCLIP_GRID = RUNS / "grid_bioclip2"
ABLATION = RUNS / "grid_ablation"
GRIDS = (RUNS, ADDED, BIOCLIP_GRID, ABLATION)
LR_ABLATION = "ConvNeXt-V2-Base-lr2e-5 (fine_tune+low)"
STRONG = RUNS / "strong"
ARCHITECTURES = ("MobileNetV3-Small", "EfficientNet-B0", "ResNet-50", "ViT-Base/16", "DeiT-Base/16", "Swin-Base",
                 "ConvNeXt-V2-Base", "DINOv2-Base/14", "BioCLIP-2")
CONVNEXT_224 = "ConvNeXt-V2-B 224, cross-entropy"  # ConvNeXt-V2-Base under the SnakeCLEF-style recipe at 224 px


def strong_name(model):
    """System name of an architecture under the third (SnakeCLEF-style, 224 px) recipe."""
    return CONVNEXT_224 if model == "ConvNeXt-V2-Base" else f"{model} (strong)"
OUT = RUNS / "analysis"
POST = ROOT / "paper_experiments" / "source_post_metadata_audit" / "row_metadata_private.jsonl"
WORKBOOK = ROOT / "snake_found_new.xlsx"
SEED, REPS = 20260925, 2000
BANDS = [(500, np.inf, ">=500"), (100, 500, "100-499"), (20, 100, "20-99"), (0, 20, "<20")]


def systems():
    found = {}
    for grid in GRIDS:
        selection = grid / "selected_final_configurations.json"
        if selection.exists():
            for model, choice in json.loads(selection.read_text(encoding="utf-8"))["choices"].items():
                path = grid / "full_species" / model.replace("/", "-") / choice["regime"] / choice["augmentation"] / "test_predictions.csv"
                if path.exists():
                    found[f"{model} ({choice['regime']}+{choice['augmentation']})"] = path
    for name, path in (("DINOv2 probe (frozen)", RUNS / "dinov2_probe" / "test_predictions.csv"),
                       ("BioCLIP 2 probe (frozen)", RUNS / "bioclip2" / "test_predictions.csv"),
                       ("BioCLIP 2 zero-shot", RUNS / "bioclip2" / "zero_shot_test_predictions.csv"),
                       ("CLIP ViT-L/14 probe (frozen)", RUNS / "clip_vitl14_laion2b" / "test_predictions.csv"),
                       ("CLIP ViT-L/14 zero-shot", RUNS / "clip_vitl14_laion2b" / "zero_shot_test_predictions.csv"),
                       (LR_ABLATION, ABLATION / "full_species" / "ConvNeXt-V2-Base-lr2e-5" / "fine_tune" / "low" / "test_predictions.csv"),
                       ("ConvNeXt-V2-B 384, seesaw", RUNS / "snakeclef_style" / "convnextv2_base_384_seesaw" / "test_predictions.csv"),
                       ("ConvNeXt-V2-B 384, cross-entropy", RUNS / "snakeclef_style" / "convnextv2_base_384_ce" / "test_predictions.csv"),
                       ("ConvNeXt-V2-B 224, cross-entropy", RUNS / "snakeclef_style" / "convnextv2_base_224_ce" / "test_predictions.csv")):
        if path.exists():
            found[name] = path
    for model in ARCHITECTURES:
        path = STRONG / model.replace("/", "-") / "test_predictions.csv"
        if model != "ConvNeXt-V2-Base" and path.exists():
            found[strong_name(model)] = path
    return found


def load(path, reference=None):
    frame = pd.read_csv(path, keep_default_na=False)
    prob_cols = [c for c in frame.columns if c.startswith("probability_")]
    if reference is not None:
        assert frame.relative_path.tolist() == reference.relative_path.tolist(), f"Test order differs: {path}"
    return frame, frame[prob_cols].to_numpy(float)


def metrics(truth, probs, tested):
    pred = probs.argmax(1)
    top3 = np.argsort(-probs, axis=1)[:, :3]
    return {"accuracy": accuracy_score(truth, pred),
            "balanced_accuracy": balanced_accuracy_score(truth, pred),
            "macro_f1_tested_species": f1_score(truth, pred, labels=tested, average="macro", zero_division=0),
            "top3_accuracy": float((top3 == truth[:, None]).any(1).mean())}


def bootstrap_indices(groups, species, rng, stratified):
    """Group bootstrap of test images; identical resamples are shared by all systems.

    Unstratified: groups are resampled with replacement (used for image-level means:
    accuracy, top-3, frequency bands, accuracy differences). Stratified: each group
    is assigned to the species of most of its images and resampled within it, so
    every tested species is present in every resample (used for species-averaged
    metrics: macro-F1 and balanced accuracy). Species with a single test group
    contribute no resampling variability, so stratified intervals can be narrow.
    """
    frame = pd.DataFrame({"group": groups, "species": species})
    members = frame.groupby("group").indices
    if stratified:
        stratum = frame.groupby("group").species.agg(lambda s: s.value_counts().index[0])
        pools = [list(g) for _, g in stratum.groupby(stratum).groups.items()]
    else:
        pools = [list(members)]
    for _ in range(REPS):
        picks = [members[gs[i]] for gs in pools for i in rng.integers(0, len(gs), len(gs))]
        yield np.concatenate(picks)


def clustered_mcnemar(correct_a, correct_b, groups):
    """McNemar test for clustered matched pairs (Durkalski et al., 2003).

    Each test group contributes d_k = b_k - c_k, its count of images only system a
    got right minus images only b got right; X2 = (sum d_k)^2 / sum d_k^2 with
    one degree of freedom. Images of one group are not treated as independent.
    """
    d = pd.Series((correct_a & ~correct_b).astype(int) - (~correct_a & correct_b).astype(int)).groupby(groups).sum()
    denominator = float((d ** 2).sum())
    if denominator == 0:
        return 0.0, 1.0
    statistic = float(d.sum()) ** 2 / denominator
    return statistic, float(chi2.sf(statistic, 1))


def holm(p_values):
    """Holm step-down adjusted p-values, in the input order."""
    p = pd.Series(p_values, dtype=float)
    adjusted, running = pd.Series(index=p.index, dtype=float), 0.0
    for rank, idx in enumerate(p.sort_values(kind="stable").index):
        running = max(running, min(1.0, (len(p) - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


IMAGE_LEVEL = ("accuracy", "top3_accuracy")
SPECIES_LEVEL = ("balanced_accuracy", "macro_f1_tested_species")


def poster_key(raw_url, poster):
    parsed = urlparse(str(raw_url).strip())
    match = re.search(r"/user/(\d+)(?:/|$)", parsed.path)
    if match:
        return "profile_id:" + match[1]
    query = parse_qs(parsed.query)
    if parsed.path.rstrip("/").endswith("profile.php") and query.get("id", [""])[0].isdigit():
        return "profile_id:" + query["id"][0]
    name = " ".join(str(poster).split()).casefold()
    return "reported_poster:" + name if name else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(V4 / "manifest.csv", dtype=str, keep_default_na=False)
    found = systems()
    if not found:
        raise SystemExit("No system predictions yet")
    reference, _ = load(next(iter(found.values())))
    labels = sorted(manifest.class_label.unique())
    test = reference[["relative_path", "class_label", "row_id", "leakage_group_id"]].copy()
    truth = reference.label_id.to_numpy(int)
    tested = sorted(set(truth))
    dev_counts = manifest[manifest.split == "development"].class_label.value_counts()

    # Grid CV table.
    rows = []
    for summary in sorted(s for grid in GRIDS for s in (grid / "full_species").glob("*/*/*/cv_summary.json")):
        data = json.loads(summary.read_text(encoding="utf-8"))
        p = data["protocol"]
        rows.append({"model": p["model"], "regime": p["regime"], "augmentation": p["augmentation"],
                     **{f"cv_{k}_mean": v for k, v in data["mean"].items()}, **{f"cv_{k}_sd": v for k, v in data["sd"].items()},
                     "selected_final_epochs": data["selected_final_epochs"]})
    for size, name in ((384, "seesaw"), (384, "ce"), (224, "ce")):
        summary = RUNS / "snakeclef_style" / f"convnextv2_base_{size}_{name}" / "cv_summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            rows.append({"model": f"ConvNeXt-V2-B {size}", "regime": "fine_tune", "augmentation": f"snakeclef_style_{name}",
                         **{f"cv_{k}_mean": v for k, v in data["mean"].items()}, **{f"cv_{k}_sd": v for k, v in data["sd"].items()}})
    dino = RUNS / "dinov2_probe" / "summary.json"
    if dino.exists():
        data = json.loads(dino.read_text(encoding="utf-8"))["cv"]
        rows.append({"model": "DINOv2 ViT-B/14 probe", "regime": "frozen", "augmentation": "none",
                     **{f"cv_{k}_mean": v for k, v in data["mean"].items()}, **{f"cv_{k}_sd": v for k, v in data["sd"].items()}})
    for model in ARCHITECTURES:
        summary = STRONG / model.replace("/", "-") / "cv_summary.json"
        if model != "ConvNeXt-V2-Base" and summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))
            rows.append({"model": model, "regime": "fine_tune", "augmentation": "strong",
                         **{f"cv_{k}_mean": v for k, v in data["mean"].items()}, **{f"cv_{k}_sd": v for k, v in data["sd"].items()}})
    for folder, model in (("bioclip2", "BioCLIP 2 ViT-L/14 probe"), ("clip_vitl14_laion2b", "CLIP ViT-L/14 probe")):
        summary = RUNS / folder / "summary.json"
        if summary.exists():
            data = json.loads(summary.read_text(encoding="utf-8"))["probe"]["cv"]
            rows.append({"model": model, "regime": "frozen", "augmentation": "none",
                         **{f"cv_{k}_mean": v for k, v in data["mean"].items()}, **{f"cv_{k}_sd": v for k, v in data["sd"].items()}})
    cv_table = pd.DataFrame(rows)
    cv_table.to_csv(OUT / "cv_configurations.csv", index=False)

    # Recipe selection over three recipes per architecture (frozen+low, fine_tune+low,
    # SnakeCLEF-style at 224 px), by mean CV accuracy, then macro-F1, then lower SD.
    selection = []
    for model in ARCHITECTURES:
        options = cv_table[cv_table.model == model].copy()
        if options.empty:  # partial results: no configuration of this architecture yet
            continue
        options.loc[options.augmentation == "strong", "recipe"] = "strong"
        options.loc[options.augmentation == "low", "recipe"] = options.regime + "+low"
        if model == "ConvNeXt-V2-Base":
            strong_row = cv_table[(cv_table.model == "ConvNeXt-V2-B 224") & (cv_table.augmentation == "snakeclef_style_ce")]
            options = pd.concat([options, strong_row.assign(model=model, recipe="strong")])
        options = options.dropna(subset=["recipe"])
        if len(options) < 3:
            continue
        best = options.sort_values(["cv_accuracy_mean", "cv_macro_f1_mean", "cv_accuracy_sd"], ascending=[False, False, True]).iloc[0]
        low_regime = "frozen" if model == "BioCLIP-2" else "fine_tune"
        selection.append({"model": model, "selected_recipe": best.recipe, "cv_accuracy": best.cv_accuracy_mean,
                          "system": strong_name(model) if best.recipe == "strong" else f"{model} ({best.recipe})",
                          "low_recipe_system": f"{model} ({low_regime}+low)", "strong_system": strong_name(model)})
    selection = pd.DataFrame(selection)
    selection.to_csv(OUT / "recipe_selection.csv", index=False)

    # Test metrics: image-level metrics use the unstratified group bootstrap,
    # species-averaged metrics the species-stratified one; resamples are shared.
    probs = {name: load(path, reference)[1] for name, path in found.items()}
    groups = test.leakage_group_id.to_numpy()
    samples = list(bootstrap_indices(groups, truth, np.random.default_rng(SEED), stratified=False))
    strat_samples = list(bootstrap_indices(groups, truth, np.random.default_rng(SEED + 1), stratified=True))
    table, draws = [], {}
    for name, p in probs.items():
        point = metrics(truth, p, tested)
        plain = pd.DataFrame([metrics(truth[i], p[i], tested) for i in samples])
        strat = pd.DataFrame([metrics(truth[i], p[i], tested) for i in strat_samples])
        boot = pd.concat([plain[list(IMAGE_LEVEL)], strat[list(SPECIES_LEVEL)]], axis=1)
        draws[name] = boot
        table.append({"system": name, **point, **{f"{k}_ci_low": boot[k].quantile(.025) for k in point},
                      **{f"{k}_ci_high": boot[k].quantile(.975) for k in point}})
    pd.DataFrame(table).sort_values("accuracy", ascending=False).to_csv(OUT / "test_systems.csv", index=False)

    # Frequency bands by number of development images of the true species.
    band_rows = []
    species_dev = test.class_label.map(dev_counts).fillna(0).to_numpy()
    for low, high, band in BANDS:
        mask = (species_dev >= low) & (species_dev < high)
        for name, p in probs.items():
            pred = p.argmax(1)
            recalls = [np.mean(pred[mask & (truth == c)] == c) for c in sorted(set(truth[mask]))]
            band_rows.append({"band_dev_images": band, "system": name, "test_images": int(mask.sum()),
                              "species": len(recalls), "accuracy": float(np.mean(pred[mask] == truth[mask])) if mask.any() else np.nan,
                              "mean_species_recall": float(np.mean(recalls)) if recalls else np.nan})
    pd.DataFrame(band_rows).to_csv(OUT / "frequency_bands.csv", index=False)

    # Paired comparisons: best system against each other, plus the planned
    # contrasts (loss: seesaw vs CE; architecture under the shared grid recipe:
    # ConvNeXt-V2 vs Swin; recipe at equal 224 px: SnakeCLEF-style 224 CE vs grid
    # ConvNeXt-V2; resolution at equal recipe: SnakeCLEF-style 384 CE vs 224 CE;
    # frozen foundation features: BioCLIP 2 vs DINOv2 probe; learning rate alone:
    # grid ConvNeXt-V2 at the transformer rate vs the CNN rate; rest of the recipe:
    # SnakeCLEF-style 224 CE vs that ablation; biological pretraining at fixed
    # architecture and size: BioCLIP 2 vs its LAION-2B CLIP initialization, as probes).
    best = max(probs, key=lambda n: metrics(truth, probs[n], tested)["accuracy"])
    pairs = [(best, other) for other in probs if other != best]
    planned = [("ConvNeXt-V2-B 384, seesaw", "ConvNeXt-V2-B 384, cross-entropy")]
    grid_names = {n.split(" (")[0]: n for n in probs if n.endswith("+low)")}
    if {"ConvNeXt-V2-Base", "Swin-Base"} <= set(grid_names):
        planned.append((grid_names["ConvNeXt-V2-Base"], grid_names["Swin-Base"]))
    if "ConvNeXt-V2-Base" in grid_names:
        planned.append(("ConvNeXt-V2-B 224, cross-entropy", grid_names["ConvNeXt-V2-Base"]))
    planned.append(("ConvNeXt-V2-B 384, cross-entropy", "ConvNeXt-V2-B 224, cross-entropy"))
    planned.append(("BioCLIP 2 probe (frozen)", "DINOv2 probe (frozen)"))
    if "ConvNeXt-V2-Base" in grid_names:
        planned.append((LR_ABLATION, grid_names["ConvNeXt-V2-Base"]))
    planned.append(("ConvNeXt-V2-B 224, cross-entropy", LR_ABLATION))
    planned.append(("BioCLIP 2 probe (frozen)", "CLIP ViT-L/14 probe (frozen)"))
    # Third recipe: SnakeCLEF-style at 224 px against each architecture's low-augmentation
    # final (fine-tuned, or frozen for BioCLIP 2), and the closest pair it produces.
    for row in selection.itertuples():
        planned.append((row.strong_system, row.low_recipe_system))
    planned.append((strong_name("BioCLIP-2"), strong_name("DINOv2-Base/14")))
    pairs += [pair for pair in planned if set(pair) <= set(probs)]
    pair_rows = []
    unique = {}
    for pair in pairs:
        unique.setdefault(frozenset(pair), pair)
    for a, b in unique.values():
        correct_a, correct_b = probs[a].argmax(1) == truth, probs[b].argmax(1) == truth
        only_a, only_b = int((correct_a & ~correct_b).sum()), int((~correct_a & correct_b).sum())
        p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        cluster_chi2, cluster_p = clustered_mcnemar(correct_a, correct_b, groups)
        diff = draws[a] - draws[b]
        point = {k: metrics(truth, probs[a], tested)[k] - metrics(truth, probs[b], tested)[k] for k in ("accuracy", "macro_f1_tested_species", "balanced_accuracy")}
        pair_rows.append({"system_a": a, "system_b": b, **{f"diff_{k}": v for k, v in point.items()},
                          **{f"diff_{k}_ci_low": diff[k].quantile(.025) for k in point},
                          **{f"diff_{k}_ci_high": diff[k].quantile(.975) for k in point},
                          "only_a_correct": only_a, "only_b_correct": only_b, "mcnemar_exact_p": p_value,
                          "clustered_mcnemar_chi2": cluster_chi2, "clustered_mcnemar_p": cluster_p})
    pairs_frame = pd.DataFrame(pair_rows)
    if len(pairs_frame):
        # The clustered test is primary; the image-level exact test is kept for reference.
        pairs_frame["clustered_mcnemar_holm_p"] = holm(pairs_frame.clustered_mcnemar_p)
        pairs_frame["mcnemar_holm_p"] = holm(pairs_frame.mcnemar_exact_p)
    pairs_frame.to_csv(OUT / "paired_comparisons.csv", index=False)

    # Loss ablation within frequency bands: paired component bootstrap of seesaw minus CE.
    a_name, b_name = "ConvNeXt-V2-B 384, seesaw", "ConvNeXt-V2-B 384, cross-entropy"
    if {a_name, b_name} <= set(probs):
        correct_a, correct_b = probs[a_name].argmax(1) == truth, probs[b_name].argmax(1) == truth
        ablation = []
        for low, high, band in BANDS:
            mask = (species_dev >= low) & (species_dev < high)
            diffs = [np.mean(correct_a[i][mask[i]]) - np.mean(correct_b[i][mask[i]]) for i in samples if mask[i].any()]
            only_a, only_b = int((correct_a & ~correct_b & mask).sum()), int((~correct_a & correct_b & mask).sum())
            ablation.append({"band_dev_images": band, "test_images": int(mask.sum()),
                             "seesaw_accuracy": float(correct_a[mask].mean()), "ce_accuracy": float(correct_b[mask].mean()),
                             "diff": float(correct_a[mask].mean() - correct_b[mask].mean()),
                             "diff_ci_low": float(np.quantile(diffs, .025)), "diff_ci_high": float(np.quantile(diffs, .975)),
                             "only_seesaw_correct": only_a, "only_ce_correct": only_b,
                             "mcnemar_exact_p": binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0,
                             "clustered_mcnemar_p": clustered_mcnemar(correct_a[mask], correct_b[mask], groups[mask])[1]})
        pd.DataFrame(ablation).to_csv(OUT / "seesaw_vs_ce_by_band.csv", index=False)

    # Poster-overlap sensitivity (reported poster/profile, not verified photographer).
    sheet = pd.read_excel(WORKBOOK, sheet_name="Snake list", usecols=["URL", "Poster"]).fillna("")
    keys = {r: poster_key(sheet.at[int(r), "URL"], sheet.at[int(r), "Poster"]) for r in manifest.row_id.unique()}
    dev_keys = {keys[r] for r in manifest.loc[manifest.split == "development", "row_id"]} - {None}
    overlap = test.row_id.astype(str).map(keys).isin(dev_keys).to_numpy()
    sens = [{"system": n, "all_test_images": len(truth), "poster_disjoint_images": int((~overlap).sum()),
             "accuracy_all": float(np.mean(p.argmax(1) == truth)),
             "accuracy_poster_disjoint": float(np.mean(p.argmax(1)[~overlap] == truth[~overlap]))} for n, p in probs.items()]
    pd.DataFrame(sens).to_csv(OUT / "poster_overlap_sensitivity.csv", index=False)

    # Genus level.
    genus = np.array([l.split()[0] for l in labels])
    pd.DataFrame([{"system": n, "species_accuracy": float(np.mean(p.argmax(1) == truth)),
                   "genus_accuracy": float(np.mean(genus[p.argmax(1)] == genus[truth]))} for n, p in probs.items()]
                 ).to_csv(OUT / "genus_level.csv", index=False)

    # Medically important venomous (MIVS) confusions, SnakeCLEF definition (scripts/build_venom_map_v4.py).
    venom_path = RUNS / "species_venom_status.csv"
    if venom_path.exists():
        venom = pd.read_csv(venom_path, dtype=str, keep_default_na=False)
        status = np.array([dict(zip(venom.species, venom.mivs)).get(l, "") for l in labels])
        true_status = status[truth]
        venomous, harmless = true_status == "1", true_status == "0"
        venom_rows = []
        for name, p in probs.items():
            pred = p.argmax(1)
            pred_status = status[pred]
            venom_rows.append({
                "system": name, "venomous_test_images": int(venomous.sum()), "harmless_test_images": int(harmless.sum()),
                "status_unavailable_test_images": int((true_status == "").sum()),
                "venomous_correct_species": float(np.mean(pred[venomous] == truth[venomous])),
                "venomous_predicted_harmless": float(np.mean(pred_status[venomous] == "0")),
                "venomous_predicted_other_venomous": float(np.mean((pred_status[venomous] == "1") & (pred[venomous] != truth[venomous]))),
                "harmless_predicted_venomous": float(np.mean(pred_status[harmless] == "1")),
                "predicted_species_status_unavailable": float(np.mean(pred_status[venomous | harmless] == "")),
                "venom_status_accuracy": float(np.mean(pred_status[venomous | harmless] == true_status[venomous | harmless])),
            })
        pd.DataFrame(venom_rows).to_csv(OUT / "venom_confusions.csv", index=False)

    # Per-species table.
    per = []
    for name, p in probs.items():
        prec, rec, f1, sup = precision_recall_fscore_support(truth, p.argmax(1), labels=tested, zero_division=0)
        per += [{"system": name, "species": labels[c], "dev_images": int(dev_counts.get(labels[c], 0)), "test_images": int(s),
                 "precision": a, "recall": b, "f1": c_} for c, a, b, c_, s in zip(tested, prec, rec, f1, sup)]
    pd.DataFrame(per).to_csv(OUT / "per_species_test.csv", index=False)
    print(pd.DataFrame(table)[["system", "accuracy", "accuracy_ci_low", "accuracy_ci_high", "macro_f1_tested_species", "top3_accuracy"]].to_string(index=False))
    print(f"poster-overlapping test images: {int(overlap.sum())} of {len(truth)}")


if __name__ == "__main__":
    sys.exit(main())
