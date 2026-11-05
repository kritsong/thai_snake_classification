"""Grid controller for the two models added to the full-species v4 comparison.

Same controller, recipes, folds, selection and refit rule as
run_full_species_grid.py; only the model list, runner wrapper and output folder
differ. A separate output folder keeps the finished 12-configuration grid's
progress file and lock untouched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import run_full_species_grid as full_grid

grid = full_grid.grid
ROOT = Path(__file__).resolve().parent
grid.MODELS = ("ConvNeXt-V2-Base", "DINOv2-Base/14")
grid.OUTPUT = ROOT / "paper_experiments" / "full_species_v4_taxonomy" / "grid_added"
grid.PLAN = grid.OUTPUT / "full_species_added_plan.csv"
grid.RUNNER = ROOT / "run_full_species_grouped_added.py"
full_grid.RUNNER_WRAPPER = "run_full_species_grouped_added.py"
full_grid._LONGEST_FIRST = grid.MODELS

if __name__ == "__main__":
    full_grid.write_plan()
    sys.argv += ["--dataset-root", str(grid.DATASET), "--output-root", str(grid.OUTPUT), "--plan-path", str(grid.PLAN)]
    grid.main()
