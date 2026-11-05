#!/usr/bin/env bash
# Resume of the taxonomy-revision queues after both were stopped on 22 Oct 2026 (~08:16 UTC) when the
# Claude Code session that launched them exited. Only the unfinished steps are run; finished CV folds
# are verified and skipped by the runners.
#   lane a: BioCLIP 2 locked-test refit (its CV is complete), then the seed repeats of BioCLIP 2 and of
#           the two 384-px ConvNeXt-V2 models; writes "LANE a COMPLETE" to lane_a.log as before.
#   lane p: the learning-rate ablation (fold 5, then the refit), then, once lane a is complete, the three
#           preliminary grid controllers.
# Usage: bash scripts/run_taxonomy_revision_resume.sh a|p
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
ROOT="C:/Users/ADMIN/PycharmProjects/snake_research"
PY="/c/Users/ADMIN/anaconda3/envs/snake_research_310/python.exe"
RESULTS="$ROOT/paper_experiments/full_species_v4_taxonomy"
LOGS="$RESULTS/logs"
cd "$ROOT"
LANE="$1"
QUEUE="$LOGS/$([ "$LANE" = a ] && echo lane_a || echo preliminary).log"

step() {  # step <log-name> <command...>
  local name="$1"; shift
  echo "$(date -u +%FT%TZ) START $name" >> "$QUEUE"
  "$@" >> "$LOGS/$name.log" 2>&1
  local code=$?
  echo "$(date -u +%FT%TZ) END $name exit=$code" >> "$QUEUE"
  return $code
}

slug() { echo "$1" | tr -c '[:alnum:]\n' '_'; }

repeats() {  # repeats <system>: seeds 2 and 3
  for k in 1 2; do step "seed_$(slug "$1")_r$k" "$PY" -u run_seed_repeats_v4.py --system "$1" --repeat "$k"; done
}

ABLATION_ARGS=(--output-root "$RESULTS/grid_ablation" --model ConvNeXt-V2-Base-lr2e-5
  --regime fine_tune --augmentation low --threshold-unit images --min-images-per-label 1 --folds 5 --epochs 50
  --early-stopping-patience 10 --early-stopping-min-epochs 10 --batch-size 16 --workers 4 --seed 20260925
  --bootstrap-replicates 2000)

echo "$(date -u +%FT%TZ) RESUME lane $LANE" >> "$QUEUE"
if [ "$LANE" = "a" ]; then
  step strong_BioCLIP_2_final "$PY" -u run_strong_recipe_v4.py --model "BioCLIP-2" --mode final
  repeats "BioCLIP-2 (strong recipe)"
  repeats "ConvNeXt-V2-B 384, cross-entropy"
  repeats "ConvNeXt-V2-B 384, seesaw"
  echo "$(date -u +%FT%TZ) LANE a COMPLETE" >> "$QUEUE"
elif [ "$LANE" = "p" ]; then
  step ablation_lr2e-5_cv "$PY" -u run_full_species_grouped_ablation.py --mode cv "${ABLATION_ARGS[@]}" &&
    step ablation_lr2e-5_final "$PY" -u run_full_species_grouped_ablation.py --mode final "${ABLATION_ARGS[@]}"
  echo "$(date -u +%FT%TZ) waiting for lane a" >> "$QUEUE"
  until grep -q "LANE a COMPLETE" "$LOGS/lane_a.log" 2>/dev/null; do sleep 300; done
  step grid_original_six "$PY" -u run_full_species_grid.py --mode run
  step grid_added "$PY" -u run_full_species_grid_added.py --mode run
  step grid_bioclip2 "$PY" -u run_full_species_grid_bioclip2.py --mode run
  echo "$(date -u +%FT%TZ) PRELIMINARY COMPLETE" >> "$QUEUE"
fi
