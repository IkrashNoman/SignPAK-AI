"""
02_segment_videos.py  —  SignPAK-AI
=====================================
Problem: Raw and Signer videos contain non-signing actions (e.g., reaching 
         to press record/stop buttons) and/or multiple signing attempts.
Solution: Detect valid hand presence in the signing space using MediaPipe Tasks API,
          trim lead-in/lead-out button presses, and extract clean signing clips.

Input  : data/raw/**/*.mp4 AND data/Signer_*/**/*.mp4
Output : data/processed/<source_type>/<category>/<label>.mp4

Place in:  scripts/02_segment_videos.py
Run from:  SIGNPAK-AI root  →  python scripts/02_segment_videos.py
"""

import re
import cv2
import json
import shutil
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUT_DIR      = DATA_DIR / "processed"
LOG_DIR      = DATA_DIR / "logs"
MODEL_PATH   = PROJECT_ROOT / "models" / "hand_landmarker.task"

LOG_DIR.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    "greetings & politeness":      "greetings",
    "yes_no & common expressions": "common_expressions",
}

# ── Segmentation parameters ───────────────────────────────────────────────────
MIN_SIGN_FRAMES  = 8     # Min consecutive frames required for a valid sign
SILENCE_THRESH   = 10    # Consecutive no-hand frames to end a window
PAD_FRAMES       = 3     # Frame padding before/after sign start/end
HAND_CONFIDENCE  = 0.5   # Min detection confidence

# Y-axis cutoff: Ignore hands in the bottom 15% of the frame (reaching for screen/keyboard)
BOTTOM_CUTOFF_RATIO = 0.85 

# ── MediaPipe Tasks API Setup ─────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand landmarker model not found at: {MODEL_PATH}\n"
        f"Ensure 'hand_landmarker.task' exists in 'models/' directory."
    )

base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=HAND_CONFIDENCE,
    min_tracking_confidence=0.4
)
landmarker = vision.HandLandmarker.create_from_options(options)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def detect_category(path: Path) -> str:
    path_str = str(path).lower()
    for key, val in CATEGORY_MAP.items():
        if key in path_str:
            return val
    return "uncategorized"


def normalize_label(stem: str) -> str:
    label = stem.lower()
    label = re.sub(r"[^a-z0-9\s_]", "_", label)
    label = re.sub(r"_+", "_", label).strip("_")
    return label


def get_hand_presence(video_path: Path) -> list[bool]:
    """
    Returns boolean list indicating active sign presence per frame.
    Filters out hand positions that correspond to pressing start/stop buttons.
    """
    cap = cv2.VideoCapture(str(video_path))
    presence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        result = landmarker.detect(mp_image)
        
        valid_hand_in_frame = False
        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                # Check normalized Y coordinate of wrist landmark (index 0)
                wrist_y = hand[0].y 
                # If wrist is above bottom cutoff ratio, it's inside active signing area
                if wrist_y < BOTTOM_CUTOFF_RATIO:
                    valid_hand_in_frame = True
                    break

        presence.append(valid_hand_in_frame)

    cap.release()
    return presence


def find_signing_windows(presence: list[bool]) -> list[tuple[int, int]]:
    windows = []
    n = len(presence)
    i = 0
    while i < n:
        if not presence[i]:
            i += 1
            continue
        start = i
        silent_count = 0
        j = i
        while j < n:
            if presence[j]:
                silent_count = 0
            else:
                silent_count += 1
                if silent_count >= SILENCE_THRESH:
                    break
            j += 1
        end = j - silent_count
        if (end - start) >= MIN_SIGN_FRAMES:
            windows.append((start, end))
        i = j + 1

    return windows


def extract_clip(video_path: Path, start: int, end: int, out_path: Path) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    s = max(0, start - PAD_FRAMES)
    e = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), end + PAD_FRAMES)

    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    for _ in range(s, e):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return out_path.exists() and out_path.stat().st_size > 1000


# ═════════════════════════════════════════════════════════════════════════════
# Main processing
# ═════════════════════════════════════════════════════════════════════════════

def process_video_sources() -> None:
    # Discover both raw/ and Signer_*/ targets
    target_dirs = []
    if (DATA_DIR / "raw").exists():
        target_dirs.append(DATA_DIR / "raw")
    
    signer_dirs = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and re.match(r"Signer_\d+$", p.name, re.IGNORECASE)
    )
    target_dirs.extend(signer_dirs)

    if not target_dirs:
        print("No valid 'raw' or 'Signer_*' directories found in data/")
        return

    log_entries = []
    stats = defaultdict(int)

    for target_dir in target_dirs:
        source_name = target_dir.name
        print(f"\n── Processing Directory: {source_name} ────────────────────")
        
        videos = sorted(target_dir.rglob("*.mp4"))

        for mp4 in videos:
            # Ignore duplicates if present
            if any(part.lower().startswith("_duplicate") for part in mp4.parts):
                continue

            category = detect_category(mp4)
            label = normalize_label(mp4.stem)
            
            # Save structure to data/processed/<source_name>/<category>/<label>.mp4
            out_path = OUT_DIR / source_name / category / f"{label}.mp4"

            entry = {
                "source_dir":   source_name,
                "category":     category,
                "label":        label,
                "source":       str(mp4.relative_to(PROJECT_ROOT)),
                "output":       str(out_path.relative_to(PROJECT_ROOT)),
                "status":       "",
                "windows_found": 0,
                "window_used":   "",
                "processed_at":  datetime.now().isoformat(timespec="seconds"),
            }

            print(f"  {source_name}/{category}/{label} ...", end=" ", flush=True)

            try:
                presence = get_hand_presence(mp4)
            except Exception as exc:
                print(f"❌ (hand detection failed: {exc})")
                entry["status"] = f"error: {exc}"
                log_entries.append(entry)
                stats["error"] += 1
                continue

            windows = find_signing_windows(presence)
            entry["windows_found"] = len(windows)

            if not windows:
                print(f"⚠️  no sign window detected, copying full video")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(mp4, out_path)
                entry["status"] = "fallback_full_copy"
                entry["window_used"] = f"0-{len(presence)}"
                stats["fallback"] += 1
            else:
                # Select FIRST complete signing window
                start, end = windows[0]
                entry["window_used"] = f"{start}-{end}"

                ok = extract_clip(mp4, start, end, out_path)
                if ok:
                    fps = cv2.VideoCapture(str(mp4)).get(cv2.CAP_PROP_FPS) or 25.0
                    dur_s = round((end - start) / max(fps, 1.0), 2)
                    print(f"✅  frames {start}–{end}  ({dur_s}s)  [{len(windows)} window(s) found]")
                    entry["status"] = "ok"
                    stats["ok"] += 1
                else:
                    print(f"❌  write failed")
                    entry["status"] = "write_failed"
                    stats["error"] += 1

            log_entries.append(entry)

    # ── Write log ─────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "02_segmentation_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entries, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  PROCESSING COMPLETE")
    print("=" * 60)
    print(f"  ✅  Trimmed & Cleaned : {stats['ok']}")
    print(f"  ⚠️   Fallback Copied   : {stats['fallback']}")
    print(f"  ❌  Errors            : {stats['error']}")
    print(f"  Log File Path        : {log_path}")
    print("=" * 60)


if __name__ == "__main__":
    process_video_sources()