"""
01_metadata.py  —  SignPAK-AI
==============================
Scans data/Signer_1 … Signer_5, builds master metadata and all CSVs.

Place in:  scripts/01_metadata.py
Run from:  SIGNPAK-AI root  →  python scripts/01_metadata.py
           OR from scripts/  →  python 01_metadata.py

Outputs
-------
data/metadata/master/master_dataset.json
data/metadata/master/categories.json
data/metadata/categories/greetings.json
data/metadata/categories/common_expressions.json
data/csv/labels.csv
data/csv/train.csv   (Signer_1, 2, 3)
data/csv/val.csv     (Signer_4  — signer-independent)
data/csv/test.csv    (Signer_5  — signer-independent)
data/metadata/reports/validation_report.csv
"""

import os
import re
import csv
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── Resolve project root regardless of where script is run from ───────────────
_THIS = Path(__file__).resolve()
# scripts/01_metadata.py  →  parent = scripts/  →  parent = SIGNPAK-AI/
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

print(f"Project root : {PROJECT_ROOT}")
print(f"Data dir     : {DATA_DIR}")

# ── Signer directories (folders only, skip .zip and other files) ──────────────
SIGNER_DIRS = sorted(
    p for p in DATA_DIR.iterdir()
    if p.is_dir() and re.match(r"Signer_\d+$", p.name)
)

# ── Category folder name fragments → canonical key ────────────────────────────
CATEGORY_MAP = {
    "greetings & politeness":      "greetings",
    "yes_no & common expressions": "common_expressions",
}

# ── Merge Hello + Sorry into one class ───────────────────────────────────────
LABEL_MERGE = {
    "hello": "hello_sorry",
    "sorry": "hello_sorry",
}

# ── Signer → split assignment ─────────────────────────────────────────────────
SPLIT_MAP = {
    "Signer_1": "train",
    "Signer_2": "train",
    "Signer_3": "train",
    "Signer_4": "val",
    "Signer_5": "test",
}

# ── Output directories ────────────────────────────────────────────────────────
OUT_META        = DATA_DIR / "metadata"
OUT_MASTER      = OUT_META / "master"
OUT_CATEGORIES  = OUT_META / "categories"
OUT_REPORTS     = OUT_META / "reports"
OUT_CSV         = DATA_DIR / "csv"

for d in [OUT_MASTER, OUT_CATEGORIES, OUT_REPORTS, OUT_CSV]:
    d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def detect_category(folder_name: str) -> str | None:
    fl = folder_name.lower()
    for key, val in CATEGORY_MAP.items():
        if key in fl:
            return val
    return None


def is_duplicate_file(filename: str) -> bool:
    """
    Skip files that start with a number prefix like '18_Have_A_Good_Day__clip1.mp4'
    or '09_Thank_You_For_Dining_With_Us__clip2.mp4'.
    Keep clean-named files like 'Have_A_Good_Day!.mp4'.
    """
    return bool(re.match(r"^\d+_", filename))


def normalize_label(stem: str) -> str:
    """
    'Good morning'                           → 'good_morning'
    'No__U__Turn'                            → 'no_u_turn'
    'There_Is_No_Sign_For_That;_You_Have_To_Fingerspell_It.' → label
    Lowercases, collapses spaces/underscores, strips specials.
    """
    s = stem.lower()
    s = re.sub(r"[^a-z0-9\s_]", " ", s)   # replace specials with space
    s = re.sub(r"[\s_]+", "_", s.strip())  # collapse spaces/underscores
    s = s.strip("_")
    return s


def apply_merge(label: str) -> str:
    return LABEL_MERGE.get(label, label)


# ═════════════════════════════════════════════════════════════════════════════
# Scanner
# ═════════════════════════════════════════════════════════════════════════════

def scan_signers() -> list[dict]:
    records = []

    for signer_dir in SIGNER_DIRS:
        signer_id = signer_dir.name

        for cat_folder in sorted(signer_dir.iterdir()):
            if not cat_folder.is_dir():
                continue

            category = detect_category(cat_folder.name)
            if category is None:
                continue

            # Walk ALL mp4 files recursively (including Extra/)
            for mp4 in sorted(cat_folder.rglob("*.mp4")):

                # Skip _duplicates folders at any depth
                rel_parts = mp4.relative_to(cat_folder).parts
                if any(p.lower().startswith("_duplicate") for p in rel_parts):
                    continue

                # Skip numbered-prefix duplicate files (Signer_2 artefact)
                if is_duplicate_file(mp4.name):
                    continue

                # Determine if this is an Extra (phrase-level) sign
                is_extra = "extra" in [p.lower() for p in rel_parts[:-1]]

                raw_label   = normalize_label(mp4.stem)
                final_label = apply_merge(raw_label)

                stat = mp4.stat()
                records.append({
                    "signer_id":    signer_id,
                    "split":        SPLIT_MAP.get(signer_id, "unknown"),
                    "category":     category,
                    "is_extra":     is_extra,
                    "raw_label":    raw_label,
                    "label":        final_label,
                    "file_path":    str(mp4.relative_to(PROJECT_ROOT)),
                    "abs_path":     str(mp4),
                    "file_exists":  True,
                    "file_size_kb": round(stat.st_size / 1024, 1),
                    "scanned_at":   datetime.now().isoformat(timespec="seconds"),
                })

    return records


# ═════════════════════════════════════════════════════════════════════════════
# Writers
# ═════════════════════════════════════════════════════════════════════════════

def write_master_json(records: list[dict]) -> None:
    out = OUT_MASTER / "master_dataset.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"  ✅  master_dataset.json     ({len(records)} records)")


def write_categories_json(records: list[dict]) -> None:
    by_cat: dict[str, list] = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)

    summary = {}
    for cat, items in sorted(by_cat.items()):
        labels  = sorted({i["label"] for i in items})
        signers = sorted({i["signer_id"] for i in items})
        summary[cat] = {
            "total_videos": len(items),
            "unique_labels": labels,
            "label_count":  len(labels),
            "signers":      signers,
        }

    with open(OUT_MASTER / "categories.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅  categories.json")

    for cat, items in sorted(by_cat.items()):
        out = OUT_CATEGORIES / f"{cat}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        print(f"  ✅  categories/{cat}.json  ({len(items)} videos)")


def build_label_index(records: list[dict]) -> dict[str, int]:
    all_labels = sorted({r["label"] for r in records})
    return {lbl: i for i, lbl in enumerate(all_labels)}


def write_labels_csv(records: list[dict], label_to_idx: dict[str, int]) -> None:
    out = OUT_CSV / "labels.csv"
    fieldnames = [
        "signer_id", "split", "category", "is_extra",
        "raw_label", "label", "label_idx",
        "file_path", "file_size_kb",
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow({
                "signer_id":    r["signer_id"],
                "split":        r["split"],
                "category":     r["category"],
                "is_extra":     r["is_extra"],
                "raw_label":    r["raw_label"],
                "label":        r["label"],
                "label_idx":    label_to_idx[r["label"]],
                "file_path":    r["file_path"],
                "file_size_kb": r["file_size_kb"],
            })
    print(f"  ✅  labels.csv              ({len(records)} rows, "
          f"{len(label_to_idx)} unique labels)")


def write_split_csvs(records: list[dict], label_to_idx: dict[str, int]) -> None:
    """
    Signer-independent splits:
      train → Signer_1, 2, 3
      val   → Signer_4
      test  → Signer_5

    Columns include 'augmented' and 'aug_id' so 02_augment.py can append
    augmented landmark rows later without rewriting this file.
    """
    fieldnames = [
        "signer_id", "split", "label", "label_idx",
        "category", "is_extra", "file_path",
        "landmark_path",   # filled by 02_augment.py
        "augmented",       # False here; True when added by augmentation
        "aug_id",          # empty here; e.g. "aug_003" when augmented
    ]

    buckets: dict[str, list] = {"train": [], "val": [], "test": []}

    for r in records:
        split = r["split"]
        if split not in buckets:
            continue
        buckets[split].append({
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

    for split, rows in buckets.items():
        out = OUT_CSV / f"{split}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"  ✅  {split}.csv              ({len(rows)} videos)")


def write_validation_report(
    records: list[dict],
    label_to_idx: dict[str, int],
) -> None:
    """Coverage matrix: label × signer, flagging missing cells."""
    all_signers = sorted({r["signer_id"] for r in records})
    all_labels  = sorted(label_to_idx.keys())

    coverage: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in all_signers
    }
    for r in records:
        coverage[r["signer_id"]][r["label"]] += 1

    out = OUT_REPORTS / "validation_report.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "label_idx"] + all_signers + ["total", "missing_from"])
        for lbl in all_labels:
            counts  = [coverage[s][lbl] for s in all_signers]
            total   = sum(counts)
            missing = [s for s, c in zip(all_signers, counts) if c == 0]
            w.writerow([lbl, label_to_idx[lbl]] + counts
                       + [total, "; ".join(missing) if missing else ""])
    print(f"  ✅  validation_report.csv   "
          f"({len(all_labels)} labels × {len(all_signers)} signers)")


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(records: list[dict], label_to_idx: dict[str, int]) -> None:
    by_signer: dict[str, int] = defaultdict(int)
    for r in records:
        by_signer[r["signer_id"]] += 1

    all_labels = sorted(label_to_idx.keys())

    print("\n" + "=" * 62)
    print("  DATASET SUMMARY")
    print("=" * 62)
    print(f"  Total videos      : {len(records)}")
    print(f"  Unique labels     : {len(all_labels)}")
    print(f"  Signers found     : {', '.join(sorted(by_signer))}")
    print()
    print("  Videos per signer:")
    for s in sorted(by_signer):
        split = SPLIT_MAP.get(s, "?")
        print(f"    {s}  ({split:5s})  {by_signer[s]} videos")
    print()
    print(f"  All labels ({len(all_labels)}):")
    for idx, lbl in enumerate(all_labels):
        print(f"    {idx:3d}  {lbl}")
    print("=" * 62)

    # Warn about missing signs
    all_signers = sorted(by_signer.keys())
    coverage: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in all_signers
    }
    for r in records:
        coverage[r["signer_id"]][r["label"]] += 1

    warnings = []
    for lbl in all_labels:
        missing = [s for s in all_signers if coverage[s][lbl] == 0]
        if missing:
            warnings.append((lbl, missing))

    if warnings:
        print("\n  ⚠️  Missing signs (signer has no video for this label):")
        for lbl, signers in warnings:
            print(f"    {lbl:45s}  missing in: {', '.join(signers)}")
    print()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n📂  Scanning signer folders …")

    if not SIGNER_DIRS:
        print(f"❌  No Signer_* directories found under {DATA_DIR}")
        print("    Make sure you are running from SIGNPAK-AI/ root or scripts/")
        return

    print(f"    Found signers: {[d.name for d in SIGNER_DIRS]}\n")

    records = scan_signers()

    if not records:
        print("❌  No .mp4 files found. Check your data/ folder structure.")
        return

    print(f"💾  Writing outputs …\n")
    write_master_json(records)
    write_categories_json(records)
    label_to_idx = build_label_index(records)
    write_labels_csv(records, label_to_idx)
    write_split_csvs(records, label_to_idx)
    write_validation_report(records, label_to_idx)

    print_summary(records, label_to_idx)
    print("✅  Done.  Check data/csv/ and data/metadata/ for outputs.\n")


if __name__ == "__main__":
    main()