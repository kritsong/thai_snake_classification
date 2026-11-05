"""Keep only high-score snake frames and remove visually similar consecutive frames.

The source extraction is never modified. The script creates a separate output tree,
hard-linking retained images where possible and copying only when hard links fail.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity


PROJECT = Path(__file__).resolve().parent
FRAME_ROOT = PROJECT / "1 คัดแล้ว" / "video_frames_1sec"
SCORES_PATH = FRAME_ROOT / "sam3_presence" / "scores.jsonl"
DEFAULT_OUTPUT = FRAME_ROOT / "high_score_deduplicated"


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def visual_signature(path: Path, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    # cv2.imread cannot reliably open non-ASCII paths on Windows. Reading the
    # bytes first keeps Thai and other Unicode directory names intact.
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    dct_input = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequency = cv2.dct(dct_input)[:8, :8]
    values = low_frequency.flatten()[1:]
    phash = values > np.median(values)
    return resized, phash


def compare_signatures(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[float, int]:
    ssim = float(structural_similarity(first[0], second[0], data_range=255))
    phash_distance = int(np.count_nonzero(first[1] != second[1]))
    return ssim, phash_distance


def choose_best(group: list[dict]) -> dict:
    return max(group, key=lambda row: (float(row["sam3_score"]), -int(row["requested_second"])))


def deduplicate(
    rows: list[dict],
    threshold: float,
    ssim_threshold: float,
    phash_distance: int,
    include_videos: set[str] | None = None,
) -> tuple[list[dict], list[dict], dict]:
    include_videos = include_videos or set()
    eligible = []
    threshold_eligible = 0
    manual_override_eligible = 0
    for row in rows:
        if row.get("error") is not None or row.get("sam3_score") is None:
            continue
        meets_threshold = float(row["sam3_score"]) >= threshold
        is_manual_override = row["video"] in include_videos and not meets_threshold
        if not meets_threshold and not is_manual_override:
            continue
        selected = dict(row)
        if is_manual_override:
            selected["selection_reason"] = "manual_video_override"
            selected["manual_override_note"] = "Visually confirmed snake video included for training despite score threshold."
            manual_override_eligible += 1
        else:
            selected["selection_reason"] = "sam3_score_threshold"
            threshold_eligible += 1
        eligible.append(selected)
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_video[row["video"]].append(row)

    kept: list[dict] = []
    discarded: list[dict] = []
    comparisons = 0
    similar_comparisons = 0

    def finish_group(group: list[dict], metrics: dict[str, tuple[float, int]]) -> None:
        nonlocal similar_comparisons
        winner = choose_best(group)
        group_id = f"{Path(winner['video']).stem}__{group[0]['requested_second']:06d}-{group[-1]['requested_second']:06d}"
        kept_row = dict(winner)
        kept_row.update(
            {
                "dedup_group_id": group_id,
                "dedup_group_size": len(group),
                "dedup_group_seconds": [int(item["requested_second"]) for item in group],
            }
        )
        kept.append(kept_row)
        if len(group) > 1:
            similar_comparisons += len(group) - 1
        for item in group:
            if item is winner:
                continue
            # The first frame is the comparison anchor and therefore has no
            # separately recorded metric; it is identical to itself.
            ssim, distance = metrics.get(item["file"], (1.0, 0))
            discarded.append(
                {
                    "file": item["file"],
                    "video": item["video"],
                    "requested_second": item["requested_second"],
                    "sam3_score": item["sam3_score"],
                    "reason": "visually_similar_consecutive_frame",
                    "kept_file": winner["file"],
                    "kept_second": winner["requested_second"],
                    "kept_sam3_score": winner["sam3_score"],
                    "anchor_ssim": round(ssim, 6),
                    "anchor_phash_distance": distance,
                    "dedup_group_id": group_id,
                }
            )

    for video in sorted(by_video, key=str.casefold):
        video_rows = sorted(by_video[video], key=lambda row: int(row["requested_second"]))
        group: list[dict] = []
        metrics: dict[str, tuple[float, int]] = {}
        anchor_signature: tuple[np.ndarray, np.ndarray] | None = None
        previous_second: int | None = None

        for row in video_rows:
            second = int(row["requested_second"])
            signature = visual_signature(FRAME_ROOT / row["file"])
            if not group:
                group = [row]
                anchor_signature = signature
                previous_second = second
                continue

            is_consecutive = second == previous_second + 1
            if is_consecutive:
                comparisons += 1
                ssim, distance = compare_signatures(anchor_signature, signature)
            else:
                ssim, distance = 0.0, 64

            if is_consecutive and ssim >= ssim_threshold and distance <= phash_distance:
                group.append(row)
                metrics[row["file"]] = (ssim, distance)
            else:
                finish_group(group, metrics)
                group = [row]
                metrics = {}
                anchor_signature = signature
            previous_second = second

        if group:
            finish_group(group, metrics)

    kept.sort(key=lambda row: (row["video"].casefold(), int(row["requested_second"])))
    discarded.sort(key=lambda row: (row["video"].casefold(), int(row["requested_second"])))
    summary = {
        "source_scores": str(SCORES_PATH),
        "score_threshold": threshold,
        "ssim_threshold": ssim_threshold,
        "max_phash_distance": phash_distance,
        "manually_included_videos": sorted(include_videos, key=str.casefold),
        "total_scored_frames": len(rows),
        "high_score_input_frames": threshold_eligible,
        "manual_override_input_frames": manual_override_eligible,
        "eligible_input_frames_after_overrides": len(eligible),
        "low_score_frames_ignored": len(rows) - len(eligible),
        "consecutive_pairs_compared": comparisons,
        "similar_frames_removed": len(discarded),
        "retained_frames": len(kept),
        "retention_percent": round(100 * len(kept) / len(eligible), 2) if eligible else 0,
        "videos_with_retained_frames": len(by_video),
    }
    return kept, discarded, summary


def materialize(kept: list[dict], output: Path) -> dict[str, int]:
    frames_output = output / "frames"
    hard_links = 0
    copies = 0
    already_present = 0
    for row in kept:
        source = FRAME_ROOT / row["file"]
        relative = Path(row["file"])
        if relative.parts and relative.parts[0].casefold() == "frames":
            relative = Path(*relative.parts[1:])
        destination = frames_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            already_present += 1
            continue
        try:
            os.link(source, destination)
            hard_links += 1
        except OSError:
            shutil.copy2(source, destination)
            copies += 1
    return {
        "hard_links_created_this_run": hard_links,
        "files_copied_this_run": copies,
        "files_already_present": already_present,
        "materialized_files": hard_links + copies + already_present,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--ssim-threshold", type=float, default=0.92)
    parser.add_argument("--max-phash-distance", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-video",
        action="append",
        default=[],
        help="Include every frame from this visually confirmed video regardless of score; may be repeated.",
    )
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(SCORES_PATH)
    kept, discarded, summary = deduplicate(
        rows,
        threshold=args.score_threshold,
        ssim_threshold=args.ssim_threshold,
        phash_distance=args.max_phash_distance,
        include_videos=set(args.include_video),
    )
    if not args.analyze_only:
        args.output.mkdir(parents=True, exist_ok=True)
        summary.update(materialize(kept, args.output))
        write_jsonl(args.output / "kept_frames.jsonl", kept)
        write_jsonl(args.output / "removed_similar_frames.jsonl", discarded)
        (args.output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        override_text = (
            "Visually confirmed manual video overrides: " + ", ".join(args.include_video) + ".\n"
            if args.include_video
            else "No manual video overrides were applied.\n"
        )
        (args.output / "README.txt").write_text(
            "SAM3 scores at or above the configured score threshold are included.\n"
            + override_text
            + "Other low-score frames are ignored. Similar consecutive frames from the same video are grouped,\n"
            "and the highest-scoring frame in each group is retained. Original extracted frames are unchanged.\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
