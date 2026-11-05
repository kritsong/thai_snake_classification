"""Count source still photographs whose EXIF orientation tag is not 1 (normal).

The benchmark copies were resized without applying EXIF orientation; this audit shows how many source
files carried a rotation tag that a viewer would have applied. It covers every still file in the curated
media directory (all candidates, including those later recovered, reinstated or excluded) and, separately,
the distinct source files behind the v4 benchmark still records. Writes analysis/exif_orientation_audit.json.
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "1 คัดแล้ว"
V4 = MEDIA / "full_species_fresh_split_v4_20261007" / "manifest.csv"
OUT = ROOT / "paper_experiments" / "full_species_fresh_v4" / "analysis" / "exif_orientation_audit.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    v4 = pd.read_csv(V4, dtype=str, keep_default_na=False)
    wanted = set(v4[v4.source_type == "still_image"].content_sha256)
    counts, in_benchmark, seen = Counter(), Counter(), set()
    files = [p for p in MEDIA.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    for p in files:
        try:
            with Image.open(p) as im:
                orientation = im.getexif().get(0x0112, 1)
        except Exception:
            orientation = "unreadable"
        counts[orientation] += 1
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        if digest in wanted and digest not in seen:
            seen.add(digest)
            in_benchmark[orientation] += 1
    result = {"source_still_files": len(files), "orientation_counts": {str(k): v for k, v in counts.items()},
              "distinct_benchmark_still_files_found": len(seen), "distinct_benchmark_still_files_expected": len(wanted),
              "benchmark_orientation_counts": {str(k): v for k, v in in_benchmark.items()}}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
