"""
04_extract_augment_2.py — SignPAK-AI (V2 Pipeline - Fast Extraction with Integrity Checks)
========================================================================================
Extracts clean MediaPipe landmarks ONLY for ALL signers.
Validates existing V2 .npy files for corruption before skipping.

Run from: SIGNPAK-AI root → python scripts/04_extract_augment_2.py
"""

import os
# ── Silence C++ & MediaPipe Telemetry Logs (MUST BE BEFORE IMPORTS) ───────────
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import csv
import gc
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Paths ─────────────────────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CROPPED_DIR  = DATA_DIR / "cropped"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR      = DATA_DIR / "raw"
LANDMARK_V2  = DATA_DIR / "landmarks_v2"
CSV_V2_DIR   = DATA_DIR / "csv" / "v2"
MODEL_DIR    = PROJECT_ROOT / "models"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

POSE_LM = 33
HAND_LM = 21
COORD   = 3
LM_DIM  = POSE_LM * COORD + HAND_LM * COORD * 2  # 225

CSV_FIELDNAMES = [
    "signer_id", "split", "label", "label_idx",
    "category", "is_extra", "file_path",
    "landmark_path", "augmented", "aug_id",
]

def ensure_models_exist():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LANDMARK_V2.mkdir(parents=True, exist_ok=True)
    CSV_V2_DIR.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL_PATH.exists():
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    if not HAND_MODEL_PATH.exists():
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)


def is_valid_landmark_file(file_path: Path) -> bool:
    """Validates file integrity before skipping."""
    if not file_path.exists() or file_path.stat().st_size < 1024:
        return False
    try:
        arr = np.load(str(file_path))
        if arr.ndim != 2 or arr.shape[1] != LM_DIM or arr.shape[0] < 4:
            return False
        if np.all(arr == 0):
            return False
        return True
    except Exception:
        if file_path.exists():
            try: file_path.unlink()
            except Exception: pass
        return False


def resolve_video_path(file_path_str: str) -> Path | None:
    p = PROJECT_ROOT / file_path_str
    if p.exists(): return p

    rel_data = p.relative_to(DATA_DIR) if DATA_DIR in p.parents else Path(file_path_str)
    for base in [CROPPED_DIR, PROCESSED_DIR, RAW_DIR]:
        candidate = base / rel_data
        if candidate.exists(): return candidate
        candidate_flat = base / p.name
        if candidate_flat.exists(): return candidate_flat
    return None


def process_video_v2_worker(row: dict) -> dict:
    video_path = resolve_video_path(row["file_path"])
    if not video_path or not video_path.exists():
        row["landmark_path"] = ""
        return row

    label     = row["label"]
    signer_id = row["signer_id"]

    lm_base = (LANDMARK_V2 / signer_id / label)
    lm_base.mkdir(parents=True, exist_ok=True)
    clean_lm_path = lm_base / "clean_original.npy"

    # Integrity Check before skipping
    if is_valid_landmark_file(clean_lm_path):
        row["landmark_path"] = str(clean_lm_path.relative_to(PROJECT_ROOT))
        row["augmented"]     = False
        row["aug_id"]        = ""
        return row

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame.shape[0] > 384 or frame.shape[1] > 384:
            frame = cv2.resize(frame, (384, 384), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()

    if not frames:
        row["landmark_path"] = ""
        return row

    base_pose_opts = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH), delegate=python.BaseOptions.Delegate.CPU)
    pose_opts = vision.PoseLandmarkerOptions(base_options=base_pose_opts, running_mode=vision.RunningMode.IMAGE, min_pose_detection_confidence=0.5)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    base_hand_opts = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH), delegate=python.BaseOptions.Delegate.CPU)
    hand_opts = vision.HandLandmarkerOptions(base_options=base_hand_opts, running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=0.4)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    rows = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        row_vec = np.zeros(LM_DIM, dtype=np.float32)

        pose_res = pose_landmarker.detect(mp_image)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks[0]) >= POSE_LM:
            for idx, lm in enumerate(pose_res.pose_landmarks[0][:POSE_LM]):
                row_vec[idx * 3 : idx * 3 + 3] = [lm.x, lm.y, lm.z]

        hand_res = hand_landmarker.detect(mp_image)
        if hand_res.hand_landmarks and hand_res.handedness:
            for hand_lms, handedness_cat in zip(hand_res.hand_landmarks, hand_res.handedness):
                lbl = handedness_cat[0].category_name.lower()
                offset = 99 if lbl == "left" else 162
                for idx, lm in enumerate(hand_lms[:HAND_LM]):
                    row_vec[offset + idx * 3 : offset + idx * 3 + 3] = [lm.x, lm.y, lm.z]

        rows.append(row_vec)

    arr = np.stack(rows)
    for col in range(LM_DIM):
        zero_mask = (arr[:, col] == 0)
        if zero_mask.any() and not zero_mask.all():
            indices = np.arange(len(arr))
            arr[:, col] = np.interp(indices, indices[~zero_mask], arr[~zero_mask, col])

    np.save(str(clean_lm_path), arr)

    pose_landmarker.close()
    hand_landmarker.close()
    del frames
    gc.collect()

    row["landmark_path"] = str(clean_lm_path.relative_to(PROJECT_ROOT))
    row["augmented"]     = False
    row["aug_id"]        = ""
    return row


def process_v2_split(split: str) -> None:
    src_csv = DATA_DIR / "csv" / f"{split}.csv"
    if not src_csv.exists(): return

    with open(src_csv, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    base_rows = [r for r in reader if r.get("augmented", "").lower() != "true"]

    print(f"\n{'='*60}\n  Extracting Clean V2 Landmarks: {split.upper()}\n{'='*60}")
    num_workers = min(4, max(1, os.cpu_count() - 2))
    processed_rows = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video_v2_worker, r): r for r in base_rows}
        for future in as_completed(futures):
            res_row = future.result()
            if res_row.get("landmark_path"):
                processed_rows.append(res_row)
                print(f"  ✅ Saved V2 Clean: {res_row['signer_id']}/{res_row['label']}")

    dst_csv = CSV_V2_DIR / f"{split}.csv"
    with open(dst_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(processed_rows)

    print(f"  ✅ V2 Manifest Saved: {dst_csv} ({len(processed_rows)} rows)")


def main():
    ensure_models_exist()
    print(f"🚀 V2 Pipeline Active — Landmarks saving to {LANDMARK_V2}\n")
    process_v2_split("train")
    process_v2_split("val")
    process_v2_split("test")
    print("\n✅ Clean V2 Extractions Complete!\n")

if __name__ == "__main__":
    main()