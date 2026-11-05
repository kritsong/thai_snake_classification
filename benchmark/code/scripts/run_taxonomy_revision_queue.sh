#!/usr/bin/env bash
# Retraining of every reported system after the taxonomy revision of 20 Oct 2026
# (dataset full_species_fresh_split_v4_taxonomy_20261020; same split and folds), as two GPU lanes.
#   lane a: ConvNeXt-V2-Base 384 px with cross-entropy and Seesaw, BioCLIP 2 (224 px),
#           then the seed repeats of these three. The two 384-px runs and BioCLIP 2 share
#           a lane so that no two of the largest jobs hold GPU memory at the same time.
#   lane b: the other eight 224-px systems, then the seed repeats of DINOv2 and ConvNeXt-V2-Base.
# Each system: 5-fold CV, then one locked-test refit; a failed CV skips that system's refit.
# Outputs: paper_experiments/full_species_v4_taxonomy/. Usage: bash scripts/run_taxonomy_revision_queue.sh a|b
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
ROOT="C:/Users/ADMIN/PycharmProjects/snake_research"
PY="/c/Users/ADMIN/anaconda3/envs/snake_research_310/python.exe"
LOGS="$ROOT/paper_experiments/full_species_v4_taxonomy/logs"
mkdir -p "$LOGS"
cd "$ROOT"
LANE="$1"
QUEUE="$LOGS/lane_$LANE.log"

step() {  # step <log-name> <command...>
  local name="$1"; shift
  echo "$(date -u +%FT%TZ) START $name" >> "$QUEUE"
  "$@" >> "$LOGS/$name.log" 2>&1
  local code=$?
  echo "$(date -u +%FT%TZ) END $name exit=$code" >> "$QUEUE"
  return $code
}

slug() { echo "$1" | tr -c '[:alnum:]\n' '_'; }

strong() {  # strong <model>: SnakeCLEF-style recipe at 224 px
  step "strong_$(slug "$1")_cv" "$PY" -u run_strong_recipe_v4.py --model "$1" --mode cv &&
    step "strong_$(slug "$1")_final" "$PY" -u run_strong_recipe_v4.py --model "$1" --mode final
}

style() {  # style <size> <loss>: ConvNeXt-V2-Base
  step "style_${1}_${2}_cv" "$PY" -u run_snakeclef_style_convnext.py --image-size "$1" --loss "$2" --mode cv &&
    step "style_${1}_${2}_final" "$PY" -u run_snakeclef_style_convnext.py --image-size "$1" --loss "$2" --mode final
}

repeats() {  # repeats <system>: seeds 2 and 3
  for k in 1 2; do step "seed_$(slug "$1")_r$k" "$PY" -u run_seed_repeats_v4.py --system "$1" --repeat "$k"; done
}

if [ "$LANE" = "a" ]; then
  style 384 ce
  style 384 seesaw
  strong "BioCLIP-2"
  repeats "ConvNeXt-V2-B 384, cross-entropy"
  repeats "ConvNeXt-V2-B 384, seesaw"
  repeats "BioCLIP-2 (strong recipe)"
elif [ "$LANE" = "b" ]; then
  strong "DINOv2-Base/14"
  style 224 ce
  for m in "Swin-Base" "ViT-Base/16" "DeiT-Base/16" "ResNet-50" "EfficientNet-B0" "MobileNetV3-Small"; do strong "$m"; done
  repeats "DINOv2-Base/14 (strong recipe)"
  repeats "ConvNeXt-V2-B 224, cross-entropy"
fi
echo "$(date -u +%FT%TZ) LANE $LANE COMPLETE" >> "$QUEUE"
