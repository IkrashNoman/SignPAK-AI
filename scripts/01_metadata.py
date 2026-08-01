"""
01_metadata.py — SignPAK-AI
==============================
Scans data/Signer_1 ... Signer_5 AND data/raw/, builds master manifests.

Run from: SIGNPAK-AI root → python scripts/01_metadata.py
"""

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

print(f"Project root : {PROJECT_ROOT}")
print(f"Data dir     : {DATA_DIR}")

# ── Include all Signer directories PLUS raw ──────────────────────────────────
SIGNER_DIRS = sorted(
    p for p in DATA_DIR.iterdir()
    if p.is_dir() and (re.match(r"Signer_\d+$", p.name, re.IGNORECASE) or p.name.lower() == "raw")
)

CATEGORY_MAP = {
    "greetings & politeness":      "greetings",
    "yes_no & common expressions": "common_expressions",
}

LABEL_MERGE = {
    "hello": "hello_sorry",
    "sorry": "hello_sorry",
}

SPLIT_MAP = {
    "Signer_1": "train",
    "Signer_2": "train",
    "Signer_3": "train",
    "raw":      "train",
    "Signer_4": "val",
    "Signer_5": "test",
}

OUT_META = DATA_DIR / "metadata"
OUT_CSV  = DATA_DIR / "csv"

for d in [OUT_META / "master", OUT_META / "categories", OUT_META / "reports", OUT_CSV]:
    d.mkdir(parents=True, exist_ok=True)

def detect_category(folder_name: str) -> str:
    fl = folder_name.lower()
    for key, val in CATEGORY_MAP.items():
        if key in fl:
            return val
    return "common"

def normalize_label(stem: str) -> str:
    s = stem.lower()
    s = re.sub(r"[^a-z0-9\s_]", " ", s)
    s = re.sub(r"[\s_]+", "_", s.strip())
    return s.strip("_")

def scan_dataset() -> list[dict]:
    records = []
    for signer_dir in SIGNER_DIRS:
        signer_id = signer_dir.name

        for cat_folder in sorted(signer_dir.iterdir()):
            if not cat_folder.is_dir():
                continue
            category = detect_category(cat_folder.name)

            for mp4 in sorted(cat_folder.rglob("*.mp4")):
                if mp4.name.startswith("._") or "duplicate" in mp4.name.lower():
                    continue

                raw_label   = normalize_label(mp4.stem)
                final_label = LABEL_MERGE.get(raw_label, raw_label)
                rel_parts   = mp4.relative_to(cat_folder).parts
                is_extra    = "extra" in [p.lower() for p in rel_parts[:-1]]

                records.append({
                    "signer_id": signer_id,
                    "split":     SPLIT_MAP.get(signer_id, "train"),
                    "category":  category,
                    "is_extra":  is_extra,
                    "raw_label": raw_label,
                    "label":     final_label,
                    "file_path": str(mp4.relative_to(PROJECT_ROOT)),
                })
    return records

def main():
    records = scan_dataset()
    if not records:
        print("❌ No videos found! Check data/ subfolder structure.")
        return

    all_labels = sorted({r["label"] for r in records})
    label_to_idx = {lbl: i for i, lbl in enumerate(all_labels)}

    fieldnames = ["signer_id", "split", "label", "label_idx", "category", "is_extra", "file_path", "landmark_path", "augmented", "aug_id"]

    for split in ["train", "val", "test"]:
        rows = []
        for r in records:
            if r["split"] == split:
                rows.append({
                    "signer_id":     r["signer_id"],
                    "split":         split,
                    "label":         r["label"],
                    "label_idx":     label_to_idx[r["label"]],
                    "category":      r["category"],
                    "is_extra":      r["is_extra"],
                    "file_path":     r["file_path"],
                    "landmark_path": "",
                    "augmented":     False,
                    "aug_id":        "",
                })
        with open(OUT_CSV / f"{split}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  ✅ Saved {split}.csv ({len(rows)} video entries)")

    with open(OUT_META / "master" / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)

    print(f"\n✅ Metadata Generation Complete: Found {len(records)} videos across {len(set(r['signer_id'] for r in records))} sources.")

if __name__ == "__main__":
    main()