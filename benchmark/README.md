# Thai snake identification benchmark: records, predictions and code

Media-free records for the regional benchmark of Thai snake identification from citizen-contributed
photographs and video frames: 26,368 images of 139 species from 12,169 source records, grouped into 12,022
independent groups (all media from one report, one Facebook post or a duplicated photograph form one
group), with a grouped development/test split, five grouped cross-validation folds, and the predictions of
every model reported in the paper.

Trained checkpoints: <https://huggingface.co/kritaphatson/thai_snake_image_classifier> (folder `benchmark_v4/`).

## Contents

| Folder | What it holds |
|---|---|
| `data/manifest.csv` | One row per image: split (`development`/`test`), CV fold, species label (`class_label`), original source label, keyed record/group/video identifiers, SHA-256 of the source file and of the resized copy, image size, source type (still or video frame) |
| `data/label_changes.csv` | The 54 images whose label was revised after expert review (20 Oct 2026): *Xenochrophis piscator* records to *Fowlea piscator*; *Pareas malaccanus* and *Asthenodipsas malaccanus* to *Asthenodipsas malaccana* |
| `data/classes.json` | The 139 species in model output order (`probability_000` ... `probability_138`) |
| `data/species_crosswalk_changes.csv` | Every source label that was renamed, merged or removed, with the reason |
| `data/species_venom_status.csv` | SnakeCLEF 2024 medically important venomous species (MIVS) flag per species |
| `data/external_scope.jsonl` | The 368 SnakeCLEF Thailand validation images (201 observations, 82 species) used for external validation |
| `predictions/reported/` | The nine architectures at 224 px and ConvNeXt-V2-Base at 384 px (cross-entropy and Seesaw): CV summary, per-fold metrics, out-of-fold development predictions, locked-test predictions, protocol |
| `predictions/seed_repeats/` | Test predictions of the two additional training seeds of the five most accurate systems |
| `predictions/bioclip2_zero_shot_and_probe/` | BioCLIP 2 without training on benchmark images (zero-shot): class texts, development and test predictions; and the frozen-feature probe |
| `predictions/external_snakeclef/` | Observation-level external predictions and scores |
| `predictions/preliminary/` | The two lighter recipes, the learning-rate ablation and the feature probes (Appendix A): metrics, protocols and test predictions. Their out-of-fold development predictions are on Hugging Face (`benchmark_v4/predictions_preliminary_oof/`) |
| `analysis/` | The tables behind the paper: `benchmark_systems.csv` (Table 5), `benchmark_seed_repeats.csv` (Table 6), `benchmark_frequency_bands.csv` (Table 7), `benchmark_venom_confusions.csv` (Table 8), `venom_alert.csv` (Table 9) with the full threshold curve `venom_alert_curve.csv`, `benchmark_external_systems.csv` (Table 10), `tail_fusion.csv` and `tail_fusion_bands.csv` (zero-shot fusion, Section 3.5 and the fused rows of Table 7), `post_hoc_decisions.json` (alert thresholds and fusion weights chosen on development predictions), paired tests, poster-overlap sensitivity, group-weighted accuracy, confusions |
| `code/` | Dataset building, training, evaluation and analysis scripts |
| `checksums.json` | SHA-256 of every file |

Prediction tables are gzip-compressed CSV files, one row per image, with the true and predicted species and all
139 probabilities (6 significant digits; `predicted_label` was taken from full precision).

## Privacy and media

The photographs and videos come from public social-media posts and are not redistributed: public
accessibility does not establish permission to republish them. The release contains no media, URLs,
poster or profile fields, or locations. Source-record, group and video identifiers are keyed hashes
(HMAC-SHA256 with a private key), so they link records within this release but cannot be traced back to
the source spreadsheet. The SHA-256 content hashes allow anyone who holds an image to locate it in the manifest.

## Quick check

```python
import pandas as pd
p = pd.read_csv("predictions/reported/convnextv2_base_384_ce/test_predictions.csv.gz")
print((p.predicted_label == p.class_label).mean())  # 0.708, Table 5
```

Fusion with the zero-shot predictions (weight 0.6, chosen on development predictions):

```python
import numpy as np
cols = [f"probability_{i:03d}" for i in range(139)]
z = pd.read_csv("predictions/bioclip2_zero_shot_and_probe/zero_shot_test_predictions.csv.gz")
z = z.set_index("relative_path").loc[p.relative_path]
logp = np.log(p[cols].clip(lower=1e-12).to_numpy()) + 0.6 * np.log(z[cols].clip(lower=1e-12).to_numpy())
print((logp.argmax(1) == p.label_id).mean())  # 0.731, Section 3.5
```

## Licence

Code: see the repository licence. The ConvNeXt-V2 pretrained weights used by three systems are distributed
under CC-BY-NC-4.0.
