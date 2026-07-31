"""
04_augment_extract.py — SignPAK-AI
=======================================
Multi-Core Parallelized Feature Extraction & Data Augmentation.

For TRAIN and TEST videos:
  - Generate 30 unique, compound augmentations per video in memory.
  - Extract 225-dim MediaPipe landmark vectors (33 Pose + 21 Left + 21 Right).
  - Save arrays to data/landmarks/ and append rows to CSV manifests.

For VAL videos:
  - Extract landmarks from originals only (no augmentation).

Place in:  scripts/04_augment_extract.py
Run from:  SIGNPAK-AI root → python scripts/04_augment_extract.py
"""

import re
import csv
import json
import random
import urllib.request
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import cv2
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
CROPPED_DIR  = DATA_DIR / "cropped"
LANDMARK_DIR = DATA_DIR / "landmarks"
CSV_DIR      = DATA_DIR / "csv"
LOG_DIR      = DATA_DIR / "logs"
MODEL_DIR    = PROJECT_ROOT / "models"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

# ── Feature Dimensions ────────────────────────────────────────────────────────
POSE_LM = 33
HAND_LM = 21
COORD   = 3
LM_DIM  = POSE_LM * COORD + HAND_LM * COORD * 2  # 225

N_AUGMENTS  = 30
RANDOM_SEED = 42

CSV_FIELDNAMES = [
    "signer_id", "split", "label", "label_idx",
    "category", "is_extra", "file_path",
    "landmark_path", "augmented", "aug_id",
]

def ensure_models_exist():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not POSE_MODEL_PATH.exists():
        print(f"📦 Downloading pose_landmarker model to {POSE_MODEL_PATH} ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    if not HAND_MODEL_PATH.exists():
        print(f"📦 Downloading hand_landmarker model to {HAND_MODEL_PATH} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)


# ═════════════════════════════════════════════════════════════════════════════
# Augmentation Functions
# ═════════════════════════════════════════════════════════════════════════════

def aug_speed(frames: list, factor: float) -> list:
    n = len(frames)
    new_n = max(4, int(n / factor))
    indices = np.linspace(0, n - 1, new_n).astype(int)
    return [frames[i] for i in indices]

def aug_brightness(frames: list, delta: float) -> list:
    out = []
    for f in frames:
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + delta), 0, 255)
        out.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return out

def aug_contrast(frames: list, factor: float) -> list:
    out = []
    for f in frames:
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_l = lab[:, :, 0].mean()
        lab[:, :, 0] = np.clip((lab[:, :, 0] - mean_l) * factor + mean_l, 0, 255)
        out.append(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))
    return out

def aug_gaussian_noise(frames: list, sigma: float) -> list:
    out = []
    for f in frames:
        noise = np.random.normal(0, sigma, f.shape).astype(np.float32)
        out.append(np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    return out

def aug_gaussian_blur(frames: list, ksize: int) -> list:
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return [cv2.GaussianBlur(f, (ksize, ksize), 0) for f in frames]

def aug_zoom(frames: list, factor: float) -> list:
    out = []
    h, w = frames[0].shape[:2]
    for f in frames:
        if factor > 1:
            crop_h = int(h / factor)
            crop_w = int(w / factor)
            y0 = (h - crop_h) // 2
            x0 = (w - crop_w) // 2
            cropped = f[y0:y0 + crop_h, x0:x0 + crop_w]
            out.append(cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR))
        else:
            pad_h = int(h * (1 - factor) / 2)
            pad_w = int(w * (1 - factor) / 2)
            padded = cv2.copyMakeBorder(f, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_REPLICATE)
            out.append(cv2.resize(padded, (w, h), interpolation=cv2.INTER_LINEAR))
    return out

def aug_rotation(frames: list, angle_deg: float) -> list:
    h, w = frames[0].shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return [cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for f in frames]

def aug_translate(frames: list, tx_frac: float, ty_frac: float) -> list:
    h, w = frames[0].shape[:2]
    tx = int(w * tx_frac)
    ty = int(h * ty_frac)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return [cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for f in frames]

def aug_drop_frames(frames: list, n_drop: int) -> list:
    if len(frames) <= n_drop + 4:
        return frames
    keep = sorted(random.sample(range(len(frames)), len(frames) - n_drop))
    return [frames[i] for i in keep]

def aug_duplicate_frames(frames: list, n_dup: int) -> list:
    result = list(frames)
    for _ in range(n_dup):
        idx = random.randint(0, len(result) - 1)
        result.insert(idx, result[idx])
    return result

def aug_salt_pepper(frames: list, prob: float = 0.02) -> list:
    out = []
    for f in frames:
        noisy = f.copy()
        mask  = np.random.random(f.shape[:2])
        noisy[mask < prob / 2]       = 0
        noisy[mask > 1 - prob / 2]   = 255
        out.append(noisy)
    return out


AUG_POOL = [
    ("speed_0.60",   aug_speed,           {"factor": 0.60}),
    ("speed_0.75",   aug_speed,           {"factor": 0.75}),
    ("speed_0.85",   aug_speed,           {"factor": 0.85}),
    ("speed_1.15",   aug_speed,           {"factor": 1.15}),
    ("speed_1.25",   aug_speed,           {"factor": 1.25}),
    ("speed_1.40",   aug_speed,           {"factor": 1.40}),
    ("bright+0.20", aug_brightness,      {"delta":  0.20}),
    ("bright+0.40", aug_brightness,      {"delta":  0.40}),
    ("bright-0.20", aug_brightness,      {"delta": -0.20}),
    ("bright-0.40", aug_brightness,      {"delta": -0.40}),
    ("contrast+0.20", aug_contrast,      {"factor": 1.20}),
    ("contrast+0.40", aug_contrast,      {"factor": 1.40}),
    ("contrast-0.20", aug_contrast,      {"factor": 0.80}),
    ("contrast-0.40", aug_contrast,      {"factor": 0.60}),
    ("noise_light",   aug_gaussian_noise, {"sigma": 8}),
    ("noise_medium",  aug_gaussian_noise, {"sigma": 18}),
    ("blur_slight",   aug_gaussian_blur,  {"ksize": 3}),
    ("blur_motion",   aug_gaussian_blur,  {"ksize": 5}),
    ("zoom_0.90",   aug_zoom,            {"factor": 0.90}),
    ("zoom_1.10",   aug_zoom,            {"factor": 1.10}),
    ("zoom_1.20",   aug_zoom,            {"factor": 1.20}),
    ("rot_-10",     aug_rotation,        {"angle_deg": -10}),
    ("rot_-5",      aug_rotation,        {"angle_deg": -5}),
    ("rot_+5",      aug_rotation,        {"angle_deg":  5}),
    ("rot_+10",     aug_rotation,        {"angle_deg":  10}),
    ("trans_left",  aug_translate,       {"tx_frac": -0.05, "ty_frac": 0.0}),
    ("trans_right", aug_translate,       {"tx_frac":  0.05, "ty_frac": 0.0}),
    ("trans_up",    aug_translate,       {"tx_frac":  0.0,  "ty_frac": -0.05}),
    ("trans_down",  aug_translate,       {"tx_frac":  0.0,  "ty_frac":  0.05}),
    ("drop_2",      aug_drop_frames,     {"n_drop": 2}),
    ("drop_3",      aug_drop_frames,     {"n_drop": 3}),
    ("dup_2",       aug_duplicate_frames, {"n_dup": 2}),
    ("dup_3",       aug_duplicate_frames, {"n_dup": 3}),
    ("salt_pepper", aug_salt_pepper,     {"prob": 0.02}),
]

AUG_NAMES = [a[0] for a in AUG_POOL]

def generate_aug_sets(n: int = N_AUGMENTS, seed: int = RANDOM_SEED) -> list[list[int]]:
    rng = random.Random(seed)
    used_sets: set[frozenset] = set()
    result = []
    attempts = 0

    while len(result) < n and attempts < 5000:
        attempts += 1
        k = rng.choice([2, 3, 4, 5])
        chosen = rng.sample(range(len(AUG_POOL)), k)
        fs = frozenset(chosen)
        if fs not in used_sets:
            used_sets.add(fs)
            result.append(chosen)

    if len(result) < n:
        raise RuntimeError(f"Could not generate {n} unique augmentation sets.")
    return result

def apply_aug_set(frames: list, aug_indices: list[int]) -> list:
    result = frames
    for idx in aug_indices:
        _, fn, kwargs = AUG_POOL[idx]
        result = fn(result, **kwargs)
    return result


# ═════════════════════════════════════════════════════════════════════════════
# MediaPipe Extraction Worker (Runs inside independent process)
# ═════════════════════════════════════════════════════════════════════════════

def process_single_video_worker(row: dict, augment: bool) -> tuple[list[dict], list[dict]]:
    """Worker function for ProcessPoolExecutor"""
    raw_path = PROJECT_ROOT / row["file_path"]
    rel_in_data = (PROJECT_ROOT / row["file_path"]).relative_to(DATA_DIR)
    cropped_path = CROPPED_DIR / rel_in_data
    video_path   = cropped_path if cropped_path.exists() else raw_path

    if not video_path.exists():
        row["landmark_path"] = ""
        return [row], []

    label     = row["label"]
    signer_id = row["signer_id"]

    lm_base = (LANDMARK_DIR / signer_id / label)
    lm_base.mkdir(parents=True, exist_ok=True)
    orig_lm_path = lm_base / "original.npy"

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        row["landmark_path"] = ""
        return [row], []

    # Initialize task engines per process
    base_pose_opts = python.BaseOptions(
        model_asset_path=str(POSE_MODEL_PATH),
        delegate=python.BaseOptions.Delegate.CPU
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=base_pose_opts,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.5
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    base_hand_opts = python.BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH),
        delegate=python.BaseOptions.Delegate.CPU
    )
    hand_opts = vision.HandLandmarkerOptions(
        base_options=base_hand_opts,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4
    )
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    def extract_landmarks(frame_seq: list) -> np.ndarray:
        rows = []
        for frame in frame_seq:
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
        return arr

    # Extract original
    orig_lm = extract_landmarks(frames)
    np.save(str(orig_lm_path), orig_lm)

    row["landmark_path"] = str(orig_lm_path.relative_to(PROJECT_ROOT))
    row["augmented"]     = False
    row["aug_id"]        = ""
    
    out_rows = [row]
    meta_rows = []

    if augment:
        vid_seed = RANDOM_SEED + abs(hash(row["file_path"])) % 100000
        aug_sets = generate_aug_sets(n=N_AUGMENTS, seed=vid_seed)

        for aug_idx, aug_combo in enumerate(aug_sets):
            aug_names = [AUG_NAMES[i] for i in aug_combo]
            aug_id    = f"aug_{aug_idx:03d}"

            try:
                aug_frames = apply_aug_set(frames, aug_combo)
                aug_lm     = extract_landmarks(aug_frames)
            except Exception:
                continue

            aug_lm_path = lm_base / f"{aug_id}.npy"
            np.save(str(aug_lm_path), aug_lm)

            aug_row = dict(row)
            aug_row["landmark_path"] = str(aug_lm_path.relative_to(PROJECT_ROOT))
            aug_row["augmented"]     = True
            aug_row["aug_id"]        = aug_id
            out_rows.append(aug_row)

            meta_rows.append({
                "signer_id":     signer_id,
                "label":         label,
                "aug_id":        aug_id,
                "aug_combo":     "+".join(aug_names),
                "landmark_path": str(aug_lm_path.relative_to(PROJECT_ROOT)),
                "shape":         str(aug_lm.shape),
            })

    pose_landmarker.close()
    hand_landmarker.close()

    return out_rows, meta_rows


# ═════════════════════════════════════════════════════════════════════════════
# Main Split Coordinator
# ═════════════════════════════════════════════════════════════════════════════

def process_split_parallel(split: str, augment: bool) -> None:
    csv_path = CSV_DIR / f"{split}.csv"
    if not csv_path.exists():
        print(f"  ⚠️  {csv_path} not found — skipping split.")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"\n{'='*60}")
    print(f"  Processing Split: {split.upper()} (Parallel Execution)")
    print(f"{'='*60}")

    num_workers = max(1, os.cpu_count() - 2)
    print(f"🚀 Spawning {num_workers} parallel CPU process workers ...")

    all_new_rows = []
    all_meta_rows = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_video_worker, row, augment): row for row in rows}
        
        for future in as_completed(futures):
            row_ref = futures[future]
            try:
                out_rows, meta_rows = future.result()
                all_new_rows.extend(out_rows)
                all_meta_rows.extend(meta_rows)
                print(f"  ✅ Finished: {row_ref['signer_id']}/{row_ref['label']} (+{len(out_rows)-1} augments)")
            except Exception as exc:
                print(f"  ❌ Error processing {row_ref['signer_id']}/{row_ref['label']}: {exc}")

    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(all_new_rows)

    print(f"  ✅ Updated {split}.csv ({len(all_new_rows)} rows total)")


def main() -> None:
    ensure_models_exist()
    print(f"\nProject Root : {PROJECT_ROOT}")
    print(f"Landmark Dir : {LANDMARK_DIR}\n")

    process_split_parallel("train", augment=True)
    process_split_parallel("test",  augment=True)
    process_split_parallel("val",   augment=False)

    print("\n✅ All landmark extractions complete. Outputs saved to data/landmarks/\n")


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()