"""
check_landmarks.py — SignPAK-AI Integrity Verification Tool
==============================================================
Scans data/landmarks/ and data/csv/*.csv to verify file health,
detect 0-byte or corrupt arrays, and count samples per signer.

Run from: SIGNPAK-AI root → python scripts/check_landmarks.py
"""

import csv
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
LANDMARK_DIR = DATA_DIR / "landmarks"
CSV_DIR      = DATA_DIR / "csv"

def inspect_landmark_files():
    print("\n" + "=" * 65)
    print(" 🔍 SignPAK-AI Landmark Disk Audit")
    print("=" * 65)

    if not LANDMARK_DIR.exists():
        print(f"❌ Directory missing: {LANDMARK_DIR}")
        return

    npy_files = list(LANDMARK_DIR.rglob("*.npy"))
    print(f"📁 Total .npy files found on disk: {len(npy_files)}")

    signers = set()
    valid_count = 0
    corrupt_files = []

    for p in npy_files:
        rel_parts = p.relative_to(LANDMARK_DIR).parts
        if len(rel_parts) >= 1:
            signers.add(rel_parts[0])

        # Check integrity
        if p.stat().st_size < 1024:
            corrupt_files.append((p, "File size under 1 KB"))
            continue

        try:
            arr = np.load(str(p))
            if arr.ndim != 2 or arr.shape[1] != 225:
                corrupt_files.append((p, f"Invalid shape: {arr.shape}"))
            elif np.all(arr == 0):
                corrupt_files.append((p, "Array is entirely zeros"))
            else:
                valid_count += 1
        except Exception as e:
            corrupt_files.append((p, f"Corrupted load: {e}"))

    print(f"  ✅ Valid Landmark Files   : {valid_count}")
    print(f"  ❌ Corrupt/Empty Files    : {len(corrupt_files)}")
    print(f"  👤 Signers Detected on Disk: {sorted(signers)}\n")

    if corrupt_files:
        print("⚠️ Corrupt files detail (First 5):")
        for cf, reason in corrupt_files[:5]:
            print(f"   - {cf.relative_to(PROJECT_ROOT)}: {reason}")
        print()

def inspect_csv_manifests():
    print("=" * 65)
    print(" 📊 CSV Manifest Verification (data/csv/)")
    print("=" * 65)

    for split in ["train", "val", "test"]:
        csv_path = CSV_DIR / f"{split}.csv"
        if not csv_path.exists():
            print(f"  ❌ {split}.csv is missing!")
            continue

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))

        orig_rows = [r for r in reader if r.get("augmented", "").lower() != "true"]
        aug_rows  = [r for r in reader if r.get("augmented", "").lower() == "true"]
        signers   = sorted(set(r.get("signer_id", "") for r in reader))

        print(f"  📄 {split.upper()}.csv | Total Rows: {len(reader)} | Original: {len(orig_rows)} | Augmented: {len(aug_rows)}")
        print(f"     Signers included: {signers}\n")

if __name__ == "__main__":
    inspect_landmark_files()
    inspect_csv_manifests()