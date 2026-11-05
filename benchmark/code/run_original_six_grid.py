"""Run the focused six-model study from the audited fixed manifest.

The 18 focused architecture/regime/augmentation configurations are evaluated by
five-fold group cross-validation. For each architecture, the best recipe is
chosen by mean development-fold accuracy, then refitted and evaluated once on
the locked test set. This controller is restartable: finished configurations
are checked and skipped, and the underlying runner resumes completed folds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import psutil


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "1 คัดแล้ว" / "combined_training_top10_leakage_safe"
OUTPUT = ROOT / "paper_experiments" / "original_six_grouped"
PLAN = OUTPUT / "images100_plan.csv"
RUNNER = ROOT / "run_original_six_grouped.py"
MODELS = (
    "MobileNetV3-Small",
    "EfficientNet-B0",
    "ResNet-50",
    "ViT-Base/16",
    "DeiT-Base/16",
    "Swin-Base",
)
REGIMES = ("frozen", "fine_tune")
AUGMENTATIONS = ("none", "low", "medium", "high")
RECIPES = (
    ("frozen", "none"),
    ("fine_tune", "none"),
    ("fine_tune", "medium"),
)
FOLDS = 5
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_EPOCHS = 10
BATCH_SIZE = 16
WORKERS = 4
SEED = 20260925
BOOTSTRAP = 2000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def config_dir(model: str, regime: str, augmentation: str) -> Path:
    return OUTPUT / "images100" / model.replace("/", "-") / regime / augmentation


def config_id(model: str, regime: str, augmentation: str) -> str:
    return f"{model} | {regime} | {augmentation}"


def load_plan() -> list[tuple[str, str, str]]:
    with PLAN.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    configs = [(row["model"], row["regime"], row["augmentation"]) for row in rows]
    expected = {
        (model, regime, augmentation)
        for model in MODELS
        for regime, augmentation in RECIPES
    }
    if len(configs) != 18 or set(configs) != expected:
        raise ValueError("The saved plan differs from the 18 focused configurations")
    return sorted(
        configs,
        key=lambda item: (
            AUGMENTATIONS.index(item[2]),
            REGIMES.index(item[1]),
            MODELS.index(item[0]),
        ),
    )


def check_summary(
    path: Path, model: str, regime: str, augmentation: str, manifest_hash: str
) -> dict | None:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data["protocol"]
    expected = {
        "manifest_sha256": manifest_hash,
        "model": model,
        "regime": regime,
        "augmentation": augmentation,
        "threshold_unit": "images",
        "min_images_per_label": 100,
        "folds": FOLDS,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
    }
    bad = {key: (protocol.get(key), value) for key, value in expected.items() if protocol.get(key) != value}
    if bad:
        raise ValueError(f"Existing output has mismatched protocol: {path}: {bad}")
    if len(data.get("folds", [])) != FOLDS:
        raise ValueError(f"Existing CV output does not contain {FOLDS} folds: {path}")
    for required in ("cross_validation_metrics.csv", "development_oof_predictions.csv"):
        if not (path.parent / required).is_file():
            raise FileNotFoundError(f"Completed CV summary is missing {required}: {path}")
    return data


def command(mode: str, model: str, regime: str, augmentation: str) -> list[str]:
    return [
        sys.executable, "-u", str(RUNNER),
        "--mode", mode,
        "--dataset-root", str(DATASET),
        "--output-root", str(OUTPUT),
        "--model", model,
        "--regime", regime,
        "--augmentation", augmentation,
        "--threshold-unit", "images",
        "--min-images-per-label", "100",
        "--folds", str(FOLDS),
        "--epochs", str(EPOCHS),
        "--early-stopping-patience", str(EARLY_STOPPING_PATIENCE),
        "--early-stopping-min-epochs", str(EARLY_STOPPING_MIN_EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--workers", str(WORKERS),
        "--seed", str(SEED),
        "--bootstrap-replicates", str(BOOTSTRAP),
    ]


def run_one(mode: str, model: str, regime: str, augmentation: str) -> None:
    folder = OUTPUT / "grid_logs"
    folder.mkdir(parents=True, exist_ok=True)
    safe_model = model.replace("/", "_")
    log = folder / f"{safe_model}_{regime}_{augmentation}_{mode}.log"
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{now()}] START {mode} {config_id(model, regime, augmentation)}\n")
        stream.flush()
        completed = subprocess.run(
            command(mode, model, regime, augmentation),
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"[{now()}] EXIT {completed.returncode}\n")
    if completed.returncode:
        raise RuntimeError(f"{config_id(model, regime, augmentation)} {mode} failed; see {log}")


@contextmanager
def controller_lock():
    """Allow only one grid controller to schedule work in this output tree."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "grid_controller.lock").open("a+b") as lock_file:
        if lock_file.seek(0, os.SEEK_END) == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another grid controller already holds the output lock") from exc
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another grid controller already holds the output lock") from exc
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@dataclass
class CVJob:
    config: tuple[str, str, str]
    process: subprocess.Popen | psutil.Process
    external: bool
    log_stream: TextIO | None = None

    @property
    def pid(self) -> int:
        return self.process.pid

    def poll(self) -> int | None:
        if self.external:
            # We are not the parent of an adopted process, so its exit status
            # is unavailable. Its verified CV summary is the completion proof.
            return None if self.process.is_running() else 0
        return self.process.poll()

    def finish(self, return_code: int) -> None:
        if self.log_stream is not None:
            self.log_stream.write(f"[{now()}] EXIT {return_code}\n")
            self.log_stream.close()


def start_cv(config: tuple[str, str, str]) -> CVJob:
    model, regime, augmentation = config
    log_dir = OUTPUT / "grid_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = log_dir / f"{model.replace('/', '_')}_{regime}_{augmentation}_cv.log"
    stream = log.open("a", encoding="utf-8")
    try:
        stream.write(f"\n[{now()}] START cv {config_id(*config)}\n")
        stream.flush()
        process = subprocess.Popen(
            command("cv", *config), cwd=ROOT,
            stdout=stream, stderr=subprocess.STDOUT,
        )
    except BaseException:
        stream.close()
        raise
    return CVJob(config, process, external=False, log_stream=stream)


def adopt_cv(config: tuple[str, str, str], pid: int) -> CVJob:
    """Reserve a slot for an already-running CV child of an older controller."""
    try:
        process = psutil.Process(pid)
        if not process.is_running():
            raise ValueError(f"External CV process {pid} is no longer running")
        actual = process.cmdline()
    except (psutil.NoSuchProcess, psutil.ZombieProcess) as exc:
        raise ValueError(f"External CV process {pid} is no longer running") from exc
    expected = command("cv", *config)
    if actual[1:] != expected[1:]:
        raise ValueError(
            f"PID {pid} is not the exact CV command for {config_id(*config)}; "
            "refusing to adopt a different process"
        )
    return CVJob(config, process, external=True)


def discover_running_cv(configs: list[tuple[str, str, str]]) -> dict[tuple[str, str, str], int]:
    """Find orphaned runner children so a restarted controller never duplicates them."""
    by_command = {tuple(command("cv", *config)[1:]): config for config in configs}
    found: dict[tuple[str, str, str], int] = {}
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            actual = process.info["cmdline"] or []
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
        config = by_command.get(tuple(actual[1:]))
        if config is None:
            continue
        if config in found:
            raise RuntimeError(f"Multiple CV processes are writing {config_id(*config)}")
        found[config] = process.pid
    return found


def completed_cv(configs: list[tuple[str, str, str]], manifest_hash: str) -> set[tuple[str, str, str]]:
    return {
        config for config in configs
        if check_summary(config_dir(*config) / "cv_summary.json", *config, manifest_hash) is not None
    }


def write_cv_state(state_path: Path, state: dict, complete: set, active: dict) -> None:
    state["phase"] = "cross_validation"
    state["current"] = config_id(*next(iter(active))) if active else None
    state["active_cv_configurations"] = [
        {
            "configuration": config_id(*config),
            "pid": job.pid,
            "adopted": job.external,
        }
        for config, job in active.items()
    ]
    state["completed_cv_configurations"] = len(complete)
    state["updated_utc"] = now()
    write_json(state_path, state)


def run_cv_grid(
    configs: list[tuple[str, str, str]], manifest_hash: str,
    state_path: Path, state: dict, parallel_cv: int,
    external: tuple[tuple[str, str, str], int] | None,
) -> None:
    complete = completed_cv(configs, manifest_hash)
    pending = [config for config in configs if config not in complete]
    active: dict[tuple[str, str, str], CVJob] = {}
    discovered = discover_running_cv(configs)
    if external is not None:
        external_config, pid = external
        if external_config not in configs:
            raise ValueError(f"External configuration is not in the fixed plan: {external_config}")
        if external_config in discovered and discovered[external_config] != pid:
            raise RuntimeError(f"Another CV PID is already running {config_id(*external_config)}")
        discovered[external_config] = pid
    for config, pid in discovered.items():
        if config in complete:
            continue
        active[config] = adopt_cv(config, pid)
        pending.remove(config)
    write_cv_state(state_path, state, complete, active)

    first_error: Exception | None = None
    while pending or active:
        while pending and len(active) < parallel_cv and first_error is None:
            if digest(PLAN) != state["plan_sha256"]:
                first_error = RuntimeError("The fixed configuration plan changed during the grid run")
                break
            if digest(DATASET / "manifest.csv") != manifest_hash:
                first_error = RuntimeError("The fixed manifest changed during the grid run")
                break
            config = pending.pop(0)
            try:
                active[config] = start_cv(config)
            except Exception as exc:
                first_error = exc
                break
            write_cv_state(state_path, state, complete, active)

        if not active:
            break
        time.sleep(5)
        for config, job in list(active.items()):
            return_code = job.poll()
            if return_code is None:
                continue
            job.finish(return_code)
            del active[config]
            try:
                if return_code:
                    raise RuntimeError(
                        f"{config_id(*config)} cv exited with code {return_code}; "
                        f"see grid_logs/{config[0].replace('/', '_')}_{config[1]}_{config[2]}_cv.log"
                    )
                summary = check_summary(
                    config_dir(*config) / "cv_summary.json", *config, manifest_hash
                )
                if summary is None:
                    raise RuntimeError(f"{config_id(*config)} cv ended without a verified summary")
                complete.add(config)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                    state["last_error"] = repr(exc)
            write_cv_state(state_path, state, complete, active)
        if first_error is not None:
            state["phase"] = "finishing_active_after_error" if active else "failed"
            state["last_error"] = repr(first_error)
            state["updated_utc"] = now()
            write_json(state_path, state)

    if first_error is not None:
        raise first_error
    if len(complete) != len(configs):
        raise AssertionError("CV grid exited before all configurations were verified")


def choose_final(configs: list[tuple[str, str, str]], manifest_hash: str) -> dict:
    choices = {}
    for model in MODELS:
        candidates = []
        for candidate_model, regime, augmentation in configs:
            if candidate_model != model:
                continue
            summary = check_summary(
                config_dir(model, regime, augmentation) / "cv_summary.json",
                model, regime, augmentation, manifest_hash,
            )
            if summary is None:
                raise FileNotFoundError(f"CV result missing for {config_id(model, regime, augmentation)}")
            candidates.append({
                "model": model,
                "regime": regime,
                "augmentation": augmentation,
                "cv_accuracy": summary["mean"]["accuracy"],
                "cv_macro_f1": summary["mean"]["macro_f1"],
                "cv_accuracy_sd": summary["sd"]["accuracy"],
                "selected_final_epochs": summary["selected_final_epochs"],
            })
        choices[model] = max(
            candidates,
            key=lambda row: (row["cv_accuracy"], row["cv_macro_f1"], -row["cv_accuracy_sd"]),
        )
    return {
        "selection_rule": "highest five-fold mean accuracy within each architecture; macro-F1 then lower accuracy SD break ties",
        "manifest_sha256": manifest_hash,
        "plan_sha256": digest(PLAN),
        "choices": choices,
    }


def main() -> None:
    global DATASET, OUTPUT, PLAN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "run"), default="plan")
    parser.add_argument("--dataset-root", type=Path, default=DATASET)
    parser.add_argument("--output-root", type=Path, default=OUTPUT)
    parser.add_argument("--plan-path", type=Path,
                        help="Path to the focused 18-configuration plan; defaults to output-root/images100_plan.csv")
    parser.add_argument("--parallel-cv", type=int, default=2,
                        help="Maximum simultaneous CV configurations (default: 2)")
    parser.add_argument("--external-active",
                        help="Already-running CV configuration as model|regime|augmentation")
    parser.add_argument("--external-pid", type=int,
                        help="PID of the already-running CV child to adopt")
    args = parser.parse_args()
    DATASET = args.dataset_root.resolve()
    OUTPUT = args.output_root.resolve()
    PLAN = (args.plan_path.resolve() if args.plan_path else OUTPUT / "images100_plan.csv")
    if args.parallel_cv < 1:
        parser.error("--parallel-cv must be positive")
    if (args.external_active is None) != (args.external_pid is None):
        parser.error("--external-active and --external-pid must be supplied together")
    if args.external_pid is not None and args.external_pid < 1:
        parser.error("--external-pid must be positive")
    external = None
    if args.external_active is not None:
        pieces = tuple(part.strip() for part in args.external_active.split("|"))
        if len(pieces) != 3:
            parser.error("--external-active must be model|regime|augmentation")
        external = (pieces, args.external_pid)

    configs = load_plan()
    manifest_hash = digest(DATASET / "manifest.csv")
    state_path = OUTPUT / "grid_progress.json"
    state = {
        "started_utc": now(),
        "updated_utc": now(),
        "manifest_sha256": manifest_hash,
        "plan_sha256": digest(PLAN),
        "total_cv_configurations": len(configs),
        "completed_cv_configurations": 0,
        "completed_final_models": 0,
        "phase": "planned",
        "current": None,
        "parallel_cv": args.parallel_cv,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "active_cv_configurations": [],
        "last_error": None,
    }
    if args.mode == "plan":
        state["first_configuration"] = config_id(*configs[0])
        state["last_configuration"] = config_id(*configs[-1])
        print(json.dumps(state, indent=2))
        return

    with controller_lock():
        try:
            run_cv_grid(configs, manifest_hash, state_path, state, args.parallel_cv, external)
            selection = choose_final(configs, manifest_hash)
            write_json(OUTPUT / "selected_final_configurations.json", selection)
            for model in MODELS:
                chosen = selection["choices"][model]
                regime, augmentation = chosen["regime"], chosen["augmentation"]
                state["phase"] = "locked_test"
                state["current"] = config_id(model, regime, augmentation)
                state["updated_utc"] = now()
                write_json(state_path, state)
                folder = config_dir(model, regime, augmentation)
                if (folder / "locked_test_started.json").exists() and not (folder / "final_metrics.json").exists():
                    raise RuntimeError(f"Locked test started without completed metrics: {folder}")
                if not (folder / "final_metrics.json").exists():
                    run_one("final", model, regime, augmentation)
                final = json.loads((folder / "final_metrics.json").read_text(encoding="utf-8"))
                if final.get("protocol") != json.loads((folder / "cv_summary.json").read_text(encoding="utf-8"))["protocol"]:
                    raise ValueError(f"Final test protocol differs from CV: {folder}")
                state["completed_final_models"] += 1
                state["updated_utc"] = now()
                write_json(state_path, state)
            state["phase"] = "complete"
            state["current"] = None
            state["updated_utc"] = now()
            write_json(state_path, state)
        except Exception as exc:
            state["phase"] = "failed"
            state["last_error"] = repr(exc)
            state["updated_utc"] = now()
            write_json(state_path, state)
            raise


if __name__ == "__main__":
    main()
