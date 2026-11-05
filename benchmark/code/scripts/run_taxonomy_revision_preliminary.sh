#!/usr/bin/env bash
# Preliminary recipes of Appendix Table A.1 and the learning-rate ablation, rerun on the taxonomy
# revision of 20 Oct 2026, after the reported systems (scripts/run_taxonomy_revision_queue.sh):
#   1. when lane b has finished: the ConvNeXt-V2-Base learning-rate ablation (CV, then the refit);
#   2. when lane a has also finished: the three grid controllers (frozen+low and fine-tune+low for the
#      original six, for ConvNeXt-V2-Base and DINOv2, and for BioCLIP 2), each with two CV jobs at a time,
#      CV-based recipe selection and one locked-test refit per architecture.
# Outputs: paper_experiments/full_species_v4_taxonomy/{full_species,grid_added,grid_bioclip2,grid_ablation}.
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
ROOT="C:/Users/ADMIN/PycharmProjects/snake_research"
PY="/c/Users/ADMIN/anaconda3/envs/snake_research_310/python.exe"
RESULTS="$ROOT/paper_experiments/full_species_v4_taxonomy"
LOGS="$RESULTS/logs"
mkdir -p "$LOGS"
cd "$ROOT"
QUEUE="$LOGS/preliminary.log"

step() {  # step <log-name> <command...>
  local name="$1"; shift
  echo "$(date -u +%FT%TZ) START $name" >> "$QUEUE"
  "$@" >> "$LOGS/$name.log" 2>&1
  local code=$?
  echo "$(date -u +%FT%TZ) END $name exit=$code" >> "$QUEUE"
  return $code
}

wait_lane() {  # wait_lane a|b
  echo "$(date -u +%FT%TZ) waiting for lane $1" >> "$QUEUE"
  until grep -q "LANE $1 COMPLETE" "$LOGS/lane_$1.log" 2>/dev/null; do sleep 300; done
}

ABLATION_ARGS=(--output-root "$RESULTS/grid_ablation" --model ConvNeXt-V2-Base-lr2e-5
  --regime fine_tune --augmentation low --threshold-unit images --min-images-per-label 1 --folds 5 --epochs 50
  --early-stopping-patience 10 --early-stopping-min-epochs 10 --batch-size 16 --workers 4 --seed 20260925
  --bootstrap-replicates 2000)

wait_lane b
step ablation_lr2e-5_cv "$PY" -u run_full_species_grouped_ablation.py --mode cv "${ABLATION_ARGS[@]}" &&
  step ablation_lr2e-5_final "$PY" -u run_full_species_grouped_ablation.py --mode final "${ABLATION_ARGS[@]}"
wait_lane a
step grid_original_six "$PY" -u run_full_species_grid.py --mode run
step grid_added "$PY" -u run_full_species_grid_added.py --mode run
step grid_bioclip2 "$PY" -u run_full_species_grid_bioclip2.py --mode run
echo "$(date -u +%FT%TZ) PRELIMINARY COMPLETE" >> "$QUEUE"
