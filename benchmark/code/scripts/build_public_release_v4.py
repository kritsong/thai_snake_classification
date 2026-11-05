"""Public, media-free release of the benchmark (v4 split, taxonomy revision of 20 Oct 2026).

Writes snake_paper/public_release_v4_taxonomy/, copied to the GitHub repository as benchmark/:
  data/         manifest (split, CV fold, label, original label, hashes), label changes, class order,
                species crosswalk, venom flags, external SnakeCLEF scope;
  predictions/  every reported system, seed repeat, zero-shot and preliminary run: test and development
                (out-of-fold) prediction tables, gzip, probabilities to 6 significant digits (the
                predicted_label column comes from full precision), with CV and test metrics and protocols;
  analysis/     the analysis tables behind the manuscript;
  code/         the dataset, training, evaluation and analysis scripts.
No photographs, videos, URLs, poster fields or local paths are included. Spreadsheet-row, group and video
identifiers become keyed hashes (HMAC-SHA256, key kept outside the release in snake_paper/release_private/),
so they link records within the release but cannot be traced to source rows by enumeration.
"""
from __future__ import annotations

import gzip
import hashlib
import hmac
import json
import re
import secrets
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020"
RESULTS = ROOT / "paper_experiments" / "full_species_v4_taxonomy"
SCOPE = ROOT / "paper_experiments" / "external_snakeclef2024" / "full_species_v4_taxonomy_scope"
OUT = ROOT / "snake_paper" / "public_release_v4_taxonomy"
KEY_FILE = ROOT / "snake_paper" / "release_private" / "hmac_key.txt"
MANIFEST_SHA = "542e09c4bd9eec06dd8fed38e114b2c13c9b50d6da7662bd279e9d4a114c764b"
KEYED = {"row_id": "row_key", "leakage_group_id": "group_key", "component": "group_key", "video_id": "video_key"}
CODE = [
    "run_original_six_grouped.py", "run_original_six_grid.py", "run_full_species_grouped.py",
    "run_full_species_grouped_added.py", "run_full_species_grouped_bioclip2.py", "run_full_species_grouped_ablation.py",
    "run_full_species_grid.py", "run_full_species_grid_added.py", "run_full_species_grid_bioclip2.py",
    "run_strong_recipe_v4.py", "run_snakeclef_style_convnext.py", "run_seed_repeats_v4.py", "run_bioclip2_v4.py",
    "run_dinov2_probe_v4.py", "run_clip_control_v4.py", "run_leakage_safe_dinov2_benchmark.py", "evaluate_external_v4.py",
    "extract_video_frames_1sec.py", "detect_snakes_sam3.py", "deduplicate_high_score_frames.py",
    "audit_cross_row_similarity.py", "audit_dinov2_full_cross_row.py", "audit_external_dinov2_overlap.py",
    "prepare_external_snakeclef.py",
    "scripts/build_full_species_dataset.py", "scripts/build_fresh_split_v4.py", "scripts/build_taxonomy_revision_v4.py",
    "scripts/species_mapping.py", "scripts/prepare_external_scope_v4.py", "scripts/build_venom_map_v4.py",
    "scripts/analyze_full_species_v4.py", "scripts/benchmark_subset_v4.py", "scripts/seed_repeats_summary_v4.py",
    "scripts/venom_detail_v4.py", "scripts/confusions_full_species_v4.py", "scripts/review_checks_v4.py",
    "scripts/species_overlap_external_sources.py", "scripts/plot_frequency_bands_v4.py", "scripts/plot_confusions_v4.py",
    "scripts/generate_gradcam_v4.py", "scripts/compose_gradcam_figure_v4.py", "scripts/sample_predictions_v4.py",
    "scripts/locate_sample_images_v4.py", "scripts/count_macs_v4.py", "scripts/benchmark_full_species_v4.py",
    "scripts/benchmark_selected_six.py", "scripts/exif_orientation_audit_v4.py", "scripts/build_public_release_v4.py",
    "scripts/run_taxonomy_revision_queue.sh", "scripts/run_taxonomy_revision_resume.sh",
    "scripts/run_taxonomy_revision_grids.sh", "scripts/run_taxonomy_revision_preliminary.sh",
]
PRIVATE = re.compile(r"facebook\.com|fbcdn|story_fbid|profile\.php|C:\\|C:/|/Users/|ADMIN", re.I)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key() -> bytes:
    if not KEY_FILE.exists():
        KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
        KEY_FILE.write_text(secrets.token_hex(32), encoding="utf-8")
    return bytes.fromhex(KEY_FILE.read_text(encoding="utf-8").strip())


def keyed(frame: pd.DataFrame, secret: bytes) -> pd.DataFrame:
    """Replace identifier columns by keyed hashes (16 hex digits); empty values stay empty."""
    for column, name in KEYED.items():
        if column in frame.columns:
            values = frame.pop(column).astype(str)
            frame[name] = [hmac.new(secret, f"{name}\0{v}".encode(), "sha256").hexdigest()[:16] if v else ""
                           for v in values]
    return frame


def write_table(source: Path, target: Path, secret: bytes, compress: bool) -> None:
    frame = keyed(pd.read_csv(source, dtype={c: str for c in KEYED}, keep_default_na=False), secret)
    probs = [c for c in frame.columns if c.startswith("probability_")]
    target.parent.mkdir(parents=True, exist_ok=True)
    text = frame.to_csv(index=False, float_format="%.6g" if probs else None)
    if compress:
        with gzip.open(target.with_name(target.name + ".gz"), "wt", encoding="utf-8", compresslevel=9) as handle:
            handle.write(text)
    else:
        target.write_text(text, encoding="utf-8")


def copy_run(folder: Path, target: Path, secret: bytes) -> None:
    for name in ("protocol.json", "cv_summary.json", "final_metrics.json", "summary.json", "locked_test_started.json"):
        if (folder / name).exists():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(folder / name, target / name)
    for name in ("cross_validation_metrics.csv", "zero_shot_class_texts.csv"):
        if (folder / name).exists():
            write_table(folder / name, target / name, secret, compress=False)
    for csv in sorted(folder.glob("*predictions.csv")):
        write_table(csv, target / csv.name, secret, compress=True)


def main():
    assert sha256(DATASET / "manifest.csv") == MANIFEST_SHA, "manifest changed"
    if OUT.exists():
        shutil.rmtree(OUT)
    secret = key()

    # Data records.
    data = OUT / "data"
    data.mkdir(parents=True)
    manifest = pd.read_csv(DATASET / "manifest.csv", dtype=str, keep_default_na=False).drop(columns=["v3_or_v2_source"])
    keyed(manifest, secret).to_csv(data / "manifest.csv", index=False)
    write_table(DATASET / "label_changes.csv", data / "label_changes.csv", secret, compress=False)
    shutil.copyfile(DATASET / "summary.json", data / "dataset_summary.json")
    write_table(DATASET.parent / "full_species_fresh_split_v4_20261007" / "split_adjustments.csv",
                data / "split_adjustments.csv", secret, compress=False)
    (data / "classes.json").write_text(json.dumps(sorted(manifest.class_label.unique()), indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from species_mapping import MAPPING, REVISION, final_class
    rows = [{"source_label": label, "class": cls if cls else "", "action": action, "reason": reason,
             "revision_20261020": REVISION.get(label, ("", ""))[1], "final_class": final_class(label) or ""}
            for label, (cls, action, reason, _) in MAPPING.items()]
    rows += [{"source_label": label, "class": "", "action": "", "reason": "", "revision_20261020": reason,
              "final_class": cls} for label, (cls, reason) in REVISION.items() if label not in MAPPING]
    pd.DataFrame(rows).to_csv(data / "species_crosswalk_changes.csv", index=False)
    shutil.copyfile(RESULTS / "species_venom_status.csv", data / "species_venom_status.csv")
    shutil.copyfile(SCOPE / "scope.jsonl", data / "external_scope.jsonl")
    shutil.copyfile(SCOPE / "summary.json", data / "external_scope_summary.json")

    # Predictions: reported systems, seeds, zero-shot and probes, external, preliminary recipes.
    pred = OUT / "predictions"
    for folder in sorted((RESULTS / "strong").iterdir()):
        copy_run(folder, pred / "reported" / f"{folder.name}_224", secret)
    for folder in sorted((RESULTS / "snakeclef_style").iterdir()):
        copy_run(folder, pred / "reported" / folder.name, secret)
    copy_run(RESULTS / "bioclip2", pred / "bioclip2_zero_shot_and_probe", secret)
    for folder in sorted((RESULTS / "seed_repeats").iterdir()):
        for repeat in sorted(folder.glob("repeat_*")):
            copy_run(repeat, pred / "seed_repeats" / folder.name / repeat.name, secret)
    ext = RESULTS / "external_snakeclef"
    for csv in sorted(ext.glob("*.csv")):
        write_table(csv, pred / "external_snakeclef" / csv.name, secret, compress=csv.name.startswith("observations_"))
    shutil.copyfile(ext / "summary.json", pred / "external_snakeclef" / "summary.json")
    for grid, name in ((RESULTS, "grid_original_six"), (RESULTS / "grid_added", "grid_added"),
                       (RESULTS / "grid_bioclip2", "grid_bioclip2"), (RESULTS / "grid_ablation", "convnextv2_lr_ablation")):
        for folder in sorted((grid / "full_species").glob("*/*/*")):
            copy_run(folder, pred / "preliminary" / name / folder.relative_to(grid / "full_species"), secret)
        if (grid / "selected_final_configurations.json").exists():
            shutil.copyfile(grid / "selected_final_configurations.json", pred / "preliminary" / name / "selected_final_configurations.json")
    for name in ("dinov2_probe", "clip_vitl14_laion2b"):
        copy_run(RESULTS / name, pred / "preliminary" / name, secret)

    # Analysis tables (sample_image_locations links samples to benchmark groups and is left out).
    for path in sorted((RESULTS / "analysis").iterdir()):
        if path.suffix in (".csv", ".json", ".txt") and not path.name.startswith(("run_", "sample_image_locations")):
            if path.suffix == ".csv":
                write_table(path, OUT / "analysis" / path.name, secret, compress=False)
            else:
                (OUT / "analysis").mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, OUT / "analysis" / path.name)

    # Code.
    for rel in CODE:
        target = OUT / "code" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / rel, target)

    # Privacy scan of every text file, then checksums.
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.parent.name != "code" and path.parent.parent.name != "code":
            text = gzip.open(path, "rt", encoding="utf-8").read() if path.suffix == ".gz" else path.read_text(encoding="utf-8")
            assert not PRIVATE.search(text), f"private pattern in {path}"
            for column in KEYED:
                assert f"{column}," not in text.split("\n", 1)[0] + ",", f"raw identifier column {column} in {path}"
    checksums = {str(p.relative_to(OUT)).replace("\\", "/"): sha256(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    (OUT / "checksums.json").write_text(json.dumps(checksums, indent=1), encoding="utf-8")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(checksums)} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
