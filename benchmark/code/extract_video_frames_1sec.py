"""Extract original-resolution JPEGs at 0, 1, 2, ... seconds; keep provenance."""
import concurrent.futures
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'venv' / 'Lib' / 'site-packages'))
import cv2
cv2.setNumThreads(1)

SOURCE = ROOT / '1 คัดแล้ว'
OUTPUT = SOURCE / 'video_frames_1sec'
EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v', '.wmv'}


def extract(source):
    result = {'video': source.name, 'frames': [], 'errors': []}
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    try:
        if not capture.isOpened():
            raise ValueError('Cannot open video')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f'Invalid metadata: fps={fps}, count={count}')
        duration = count / fps
        result.update(fps=fps, source_frame_count=count, duration_seconds=duration)
        folder = OUTPUT / 'frames' / source.stem
        folder.mkdir(parents=True, exist_ok=True)
        second = 0
        result['sampling_method'] = 'first decoded frame at or after each integer second, using presentation timestamp'
        for target in range(count):
            if not capture.grab():
                result['errors'].append(f'Cannot decode source frame {target}')
                break
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if timestamp_ms / 1000 + 0.000001 < second:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                result['errors'].append(f'Cannot retrieve at {second}s (frame {target})')
                break
            filename = folder / f'{source.stem}__t{second:06d}s.jpg'
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                result['errors'].append(f'Cannot encode at {second}s')
                continue
            encoded.tofile(str(filename))
            result['frames'].append({
                'file': filename.relative_to(OUTPUT).as_posix(),
                'requested_second': second,
                'source_frame_index': target,
                'decoded_timestamp_ms': timestamp_ms,
                'width': frame.shape[1], 'height': frame.shape[0],
                'review_status': 'not_reviewed',
            })
            second += 1
        result['expected_frames'] = second
        expected_files = {Path(f['file']).name for f in result['frames']}
        for old_file in folder.glob(f'{source.stem}__t*s.jpg'):
            if old_file.name not in expected_files:
                old_file.unlink()
    except Exception as exc:
        result['errors'].append(str(exc))
    finally:
        capture.release()
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    videos = sorted(p for p in SOURCE.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    (OUTPUT / 'README.txt').write_text(
        'Original-resolution JPEG frames sampled at 0, 1, 2, ... seconds.\n'
        'Sources are unchanged. Each video has its own frames subfolder.\n'
        'Sampling uses integer-second targets; recorded decoded timestamps allow verification.\n'
        'Clips with variable-frame-rate drift are corrected using presentation timestamps.\n'
        'manifest.jsonl contains source provenance and extraction errors.\n'
        'Frames default to not_reviewed; manual labels are recorded separately.\n'
        'Species names in filenames are inherited, not independently verified.\n', encoding='utf-8')
    start = time.time()
    prior = []
    manifest_path = OUTPUT / 'manifest.jsonl'
    if manifest_path.exists():
        prior = [json.loads(line) for line in manifest_path.read_text(encoding='utf-8').splitlines()]
    finished = {r['video'] for r in prior}
    total = sum(len(r['frames']) for r in prior)
    errors = sum(len(r['errors']) for r in prior)
    completed = len(prior)
    with manifest_path.open('a', encoding='utf-8') as manifest:
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(extract, video) for video in videos if video.name not in finished]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                manifest.write(json.dumps(result, ensure_ascii=False) + '\n')
                manifest.flush()
                completed += 1
                total += len(result['frames'])
                errors += len(result['errors'])
                if completed % 25 == 0 or completed == len(videos):
                    print(json.dumps({'completed_videos': completed, 'videos': len(videos), 'frames': total, 'errors': errors, 'elapsed_seconds': round(time.time()-start)}), flush=True)
    summary = {'source_videos': len(videos), 'extracted_frames': total, 'extraction_errors': errors, 'elapsed_seconds': round(time.time()-start)}
    (OUTPUT / 'extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
