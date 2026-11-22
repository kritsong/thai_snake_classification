"""Province of every benchmark report, cleaned to the 77 Thai provinces (20 Nov 2026), for the post-hoc location
prior (scripts/location_prior_v4.py).

Reads the Thai `Province` column of the source spreadsheet (snake_found_new.xlsx, sheet "Snake list"; row_id is the
zero-based data index). Cleaning: Unicode NFC, removal of spaces and zero-width characters, the decomposed sara am
(U+0E4D U+0E32) joined to U+0E33, repeated vowel or tone marks collapsed, five spelling errors corrected, and
entries that are not one Thai province (a Malaysian state, a two-province entry, "CAPTIVE") set to missing.
Also stores the SHA-256 of each report's poster key (same rule as analyze_full_species_v4.py), so the location
gain can be measured on test images whose poster contributed no development images.
Writes paper_experiments/source_post_metadata_audit/row_province_private.csv (row_id, province,
poster_key_sha256). The file stays
private: the public release contains no locations. Needs pandas with openpyxl (the snake_research environment).
"""
import hashlib
import re
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "snake_found_new.xlsx"
MANIFEST = ROOT / "1 คัดแล้ว" / "full_species_fresh_split_v4_taxonomy_20261020" / "manifest.csv"
OUT = ROOT / "paper_experiments" / "source_post_metadata_audit" / "row_province_private.csv"
TYPOS = {"นราธิวาศ": "นราธิวาส", "กรุะบี่": "กระบี่", "สมุทปราการ": "สมุทรปราการ", "นนทุบรี": "นนทบุรี",
         "จันทรบุรี": "จันทบุรี"}
NOT_A_PROVINCE = {"ไทรบุรี", "ตาก-อุทัยธานี", "CAPTIVE"}


def clean(value):
    if not isinstance(value, str):
        return None
    value = unicodedata.normalize("NFC", value)
    value = re.sub(r"[\s​‌‍﻿]", "", value)
    value = value.replace("ํา", "ำ")
    value = re.sub(r"([ั-ฺ็-๎])\1+", r"\1", value)
    value = TYPOS.get(value, value)
    return None if not value or value in NOT_A_PROVINCE else value


def poster_key(raw_url, poster):
    """Same rule as analyze_full_species_v4.poster_key; stored only as a SHA-256 hash."""
    parsed = urlparse(str(raw_url).strip())
    match = re.search(r"/user/(\d+)(?:/|$)", parsed.path)
    if match:
        key = "profile_id:" + match[1]
    else:
        query = parse_qs(parsed.query)
        if parsed.path.rstrip("/").endswith("profile.php") and query.get("id", [""])[0].isdigit():
            key = "profile_id:" + query["id"][0]
        else:
            name = " ".join(str(poster).split()).casefold()
            key = "reported_poster:" + name if name else None
    return hashlib.sha256(key.encode("utf-8")).hexdigest() if key else ""


def main():
    workbook = pd.read_excel(WORKBOOK, sheet_name="Snake list", usecols=["Province", "URL", "Poster"])
    rows = sorted(pd.read_csv(MANIFEST, usecols=["row_id"], dtype=str).row_id.astype(int).unique())
    url, poster = workbook.URL.fillna(""), workbook.Poster.fillna("")
    frame = pd.DataFrame({"row_id": rows, "province": [clean(workbook.Province.iloc[r]) for r in rows],
                          "poster_key_sha256": [poster_key(url.iloc[r], poster.iloc[r]) for r in rows]})
    provinces = frame.province.dropna().unique()
    assert len(provinces) == 77, (len(provinces), sorted(provinces))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, encoding="utf-8")
    print(len(frame), "rows,", frame.province.notna().mean().round(3), "with a province,", len(provinces), "provinces")


if __name__ == "__main__":
    main()
