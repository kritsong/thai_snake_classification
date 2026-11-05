#!/usr/bin/env bash
# Preliminary-recipe grids of the taxonomy revision only (the learning-rate ablation and all reported systems
# are complete). Written after a Windows-update reboot stopped the first attempt on 22 Oct 2026 (23:29 UTC);
# the grid controllers verify and skip finished configurations and folds, so this can be rerun after any stop.
# Usage: bash scripts/run_taxonomy_revision_grids.sh
set -u
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONWARNINGS=ignore
ROOT="C:/Users/ADMIN/PycharmProjects/snake_research"
PY="/c/Users/ADMIN/anaconda3/envs/snake_research_310/python.exe"
LOGS="$ROOT/paper_experiments/full_species_v4_taxonomy/logs"
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

echo "$(date -u +%FT%TZ) RESUME grids" >> "$QUEUE"
step grid_original_six "$PY" -u run_full_species_grid.py --mode run
step grid_added "$PY" -u run_full_species_grid_added.py --mode run
step grid_bioclip2 "$PY" -u run_full_species_grid_bioclip2.py --mode run
echo "$(date -u +%FT%TZ) PRELIMINARY COMPLETE" >> "$QUEUE"
