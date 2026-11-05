"""Score extracted video frames for snake presence with the locally cached SAM 3 model.

The scan is resumable. A score below --threshold is routed to manual review; images
are never moved or modified.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


PROJECT = Path(__file__).resolve().parent
FRAME_ROOT = PROJECT / "1 คัดแล้ว" / "video_frames_1sec"
MANIFEST = FRAME_ROOT / "manifest.jsonl"
OUTPUT = FRAME_ROOT / "sam3_presence"
DEFAULT_MODEL = Path(
    r"C:\Users\ADMIN\.cache\huggingface\hub\models--facebook--sam3"
    r"\snapshots\3c879f39826c281e95690f02c7821c4de09afae7"
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_frame_records() -> list[dict]:
    records = []
    for video in read_jsonl(MANIFEST):
        for frame in video.get("frames", []):
            records.append(
                {
                    "file": frame["file"],
                    "video": video["video"],
                    "requested_second": frame["requested_second"],
                    "decoded_timestamp_ms": frame["decoded_timestamp_ms"],
                }
            )
    return records


def load_existing_manual_labels() -> dict[str, dict]:
    label_path = FRAME_ROOT / "manual_review" / "review_labels.json"
    if not label_path.exists():
        return {}
    return {row["file"]: row for row in json.loads(label_path.read_text(encoding="utf-8"))}


def load_images(rows: list[dict]) -> list[Image.Image]:
    images = []
    try:
        for row in rows:
            with Image.open(FRAME_ROOT / row["file"]) as source:
                images.append(source.convert("RGB"))
    except Exception:
        for image in images:
            image.close()
        raise
    return images


def score_batch(
    rows: list[dict],
    processor: Sam3Processor,
    model: Sam3Model,
    prompt: str,
    threshold: float,
) -> list[dict]:
    images = load_images(rows)
    try:
        inputs = processor(images=images, text=[prompt] * len(images), return_tensors="pt")
        inputs = {key: (value.cuda() if hasattr(value, "cuda") else value) for key, value in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_sizes = [image.size[::-1] for image in images]
        detections = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)
        presence = torch.sigmoid(outputs.presence_logits).detach().float().cpu()
        scored = []
        for index, (row, result) in enumerate(zip(rows, detections)):
            scores = result["scores"].detach().float().cpu()
            boxes = result["boxes"].detach().float().cpu()
            best_index = int(scores.argmax())
            score = float(scores[best_index])
            box = [round(float(value), 2) for value in boxes[best_index].tolist()]
            scored.append(
                {
                    **row,
                    "prompt": prompt,
                    "sam3_score": score,
                    "sam3_presence_probability": float(presence[index].max()),
                    "best_box_xyxy": box,
                    "threshold": threshold,
                    "automatic_decision": "snake_detected" if score >= threshold else "manual_review",
                    "error": None,
                }
            )
        return scored
    finally:
        for image in images:
            image.close()


def score_with_recovery(rows, processor, model, prompt, threshold):
    try:
        return score_batch(rows, processor, model, prompt, threshold)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        middle = len(rows) // 2
        return score_with_recovery(rows[:middle], processor, model, prompt, threshold) + score_with_recovery(
            rows[middle:], processor, model, prompt, threshold
        )
    except Exception as exc:
        if len(rows) > 1:
            middle = len(rows) // 2
            return score_with_recovery(rows[:middle], processor, model, prompt, threshold) + score_with_recovery(
                rows[middle:], processor, model, prompt, threshold
            )
        return [
            {
                **rows[0],
                "prompt": prompt,
                "sam3_score": None,
                "sam3_presence_probability": None,
                "best_box_xyxy": None,
                "threshold": threshold,
                "automatic_decision": "manual_review",
                "error": f"{type(exc).__name__}: {exc}",
            }
        ]


def build_outputs(scores: list[dict], threshold: float, elapsed_seconds: float) -> dict:
    manual_labels = load_existing_manual_labels()
    for row in scores:
        previous = manual_labels.get(row["file"])
        row["manual_review_status"] = previous.get("review_status") if previous else "not_reviewed"
        row["manual_review_note"] = previous.get("review_note") if previous else None
        row["final_label"] = (
            previous.get("review_status")
            if previous
            else "snake_visible"
            if row["automatic_decision"] == "snake_detected"
            else "pending_manual_review"
        )

    scores.sort(key=lambda row: (row["video"].casefold(), row["requested_second"]))
    review_queue = [row for row in scores if row["automatic_decision"] == "manual_review"]
    pending_review = [row for row in review_queue if row["manual_review_status"] == "not_reviewed"]
    high_confidence = [
        row
        for row in scores
        if row["automatic_decision"] == "snake_detected"
        and row["final_label"] not in {"no_snake_visible", "uncertain"}
    ]
    write_jsonl_atomic(OUTPUT / "scores.jsonl", scores)
    write_jsonl_atomic(OUTPUT / "review_queue_all.jsonl", review_queue)
    write_jsonl_atomic(OUTPUT / "review_queue_pending.jsonl", pending_review)
    write_jsonl_atomic(OUTPUT / "automatic_snake_detected.jsonl", high_confidence)

    valid_scores = [row["sam3_score"] for row in scores if row["sam3_score"] is not None]
    summary = {
        "model": "facebook/sam3 (local cached checkpoint)",
        "prompt": "snake",
        "threshold": threshold,
        "total_frames": len(scores),
        "automatic_snake_detected": len(high_confidence),
        "manual_review_total": len(review_queue),
        "manual_review_previously_completed": len(review_queue) - len(pending_review),
        "manual_review_pending": len(pending_review),
        "inference_errors": sum(row["error"] is not None for row in scores),
        "score_min": min(valid_scores) if valid_scores else None,
        "score_max": max(valid_scores) if valid_scores else None,
        "score_mean": sum(valid_scores) / len(valid_scores) if valid_scores else None,
        "score_bands": dict(
            Counter(
                "<0.01"
                if score < 0.01
                else "0.01-0.10"
                if score < 0.10
                else "0.10-0.30"
                if score < 0.30
                else "0.30-0.50"
                if score < 0.50
                else ">=0.50"
                for score in valid_scores
            )
        ),
        "elapsed_seconds_this_run": round(elapsed_seconds, 2),
    }
    (OUTPUT / "scan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT / "README.txt").write_text(
        "SAM 3 SNAKE-PRESENCE SCREEN\n\n"
        "Every frame is scored using the text prompt 'snake'. Images are referenced in place and are not moved or edited.\n"
        f"Scores >= {threshold:.2f}: automatic_snake_detected.jsonl\n"
        f"Scores < {threshold:.2f} or errors: review_queue_all.jsonl\n"
        "Already reviewed frames retain their earlier manual labels; unreviewed low-score frames are in review_queue_pending.jsonl.\n"
        "A low detector score does not mean no snake. It means a person must inspect the image.\n"
        "The threshold was selected from a 45-frame pilot: every no-snake/uncertain frame and three difficult snake frames were routed to review.\n"
        "High-score results should receive a random quality-control audit because the pilot sample is small.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--prompt", default="snake")
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.model.exists():
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {args.model}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frames = load_frame_records()
    existing_rows = read_jsonl(OUTPUT / "scores.jsonl")
    existing = {row["file"]: row for row in existing_rows}
    pending = [row for row in frames if row["file"] not in existing]
    if not pending:
        print(json.dumps(build_outputs(list(existing.values()), args.threshold, 0), ensure_ascii=False), flush=True)
        return

    print(
        json.dumps(
            {"total": len(frames), "already_scored": len(existing), "pending": len(pending), "batch_size": args.batch_size}
        ),
        flush=True,
    )
    processor = Sam3Processor.from_pretrained(args.model, local_files_only=True)
    model = Sam3Model.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.bfloat16
    ).cuda().eval()
    start = time.time()
    score_path = OUTPUT / "scores.jsonl"
    with score_path.open("a", encoding="utf-8", newline="\n") as stream:
        for begin in range(0, len(pending), args.batch_size):
            batch = pending[begin : begin + args.batch_size]
            results = score_with_recovery(batch, processor, model, args.prompt, args.threshold)
            for row in results:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[row["file"]] = row
            stream.flush()
            completed = begin + len(batch)
            elapsed = time.time() - start
            if completed % (args.batch_size * 10) == 0 or completed == len(pending):
                rate = completed / elapsed
                eta = (len(pending) - completed) / rate if rate else math.inf
                print(
                    json.dumps(
                        {
                            "completed_this_run": completed,
                            "pending_this_run": len(pending),
                            "frames_per_second": round(rate, 2),
                            "eta_seconds": round(eta),
                        }
                    ),
                    flush=True,
                )
    summary = build_outputs(list(existing.values()), args.threshold, time.time() - start)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
