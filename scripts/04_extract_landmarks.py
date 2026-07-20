# scripts/04_extract_landmarks.py
import sys
from pathlib import Path
import cv2
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# --- SETUP HOLISTIC TASK REQUIREMENT ---
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

# Optimized Face landmark indices (Lips, Eyes, Eyebrows)
FACIAL_EXPRESSION_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 418, 312, 311, 310, 13, 14, 82, 81, 80,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300
]

def ensure_model_exists(project_root: Path) -> Path:
    """Downloads the raw holistic bundle file automatically if missing."""
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "holistic_landmarker.task"
    if not model_path.exists():
        print("📥 Model binary bundle missing. Downloading from Google servers...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("✅ Download finished.")
    return model_path

def extract_landmarks_from_video(video_path: Path, landmarker) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_ms = 0.0
    sequence_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Calculate strict progressive timestamp tracking context
        timestamp_ms = int(frame_ms)
        frame_ms += (1000.0 / fps)

        # Run modern Tasks processing context
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # 1. Upper Pose Only (0 to 22: Face details + Arms/Wrists) -> 23 * 4 = 92 values
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(results.pose_landmarks) if i <= 22]).flatten()
        else:
            pose = np.zeros(23 * 4)

        # 2. Filtered Expressive Face Mesh -> 50 * 3 = 150 values
        if results.face_landmarks:
            face_all = results.face_landmarks
            face = np.array([[face_all[idx].x, face_all[idx].y, face_all[idx].z] for idx in FACIAL_EXPRESSION_INDICES]).flatten()
        else:
            face = np.zeros(len(FACIAL_EXPRESSION_INDICES) * 3)

        # 3. Left Hand -> 21 * 3 = 63 values
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten()
        else:
            lh = np.zeros(21 * 3)

        # 4. Right Hand -> 21 * 3 = 63 values
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten()
        else:
            rh = np.zeros(21 * 3)

        # Combined optimized shape signature matrix profile
        frame_features = np.concatenate([pose, face, lh, rh])
        sequence_data.append(frame_features)

    cap.release()
    return np.array(sequence_data)

def main():
    print("=" * 60)
    print("🧬 TASKS API: EXTRACTING LANDMARK COORDINATES")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed_videos"
    features_base_dir = project_root / "data" / "features"

    model_asset_path = ensure_model_exists(project_root)

    video_files = list(processed_dir.glob("**/*.mp4"))
    if not video_files:
        print("❌ No segmented videos found. Run 03_preprocess_videos.py first.")
        return

    # Setup modern task options configuration rules
    base_options = python.BaseOptions(model_asset_path=str(model_asset_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_face_blendshapes=False
    )

    # Re-instantiate a fresh landmarker for every single video file to prevent timestamp collision
    for video_path in video_files:
        relative_folder = video_path.parent.relative_to(processed_dir)
        output_folder = features_base_dir / relative_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        output_npy_path = output_folder / f"{video_path.stem}.npy"

        print(f"Extracting skeletons: {relative_folder / video_path.name}")
        
        with vision.HolisticLandmarker.create_from_options(options) as landmarker:
            landmarks_matrix = extract_landmarks_from_video(video_path, landmarker)
            np.save(str(output_npy_path), landmarks_matrix)

    print("\n\n✅ Extraction complete! Optimized features saved under 'data/features/'.")

if __name__ == "__main__":
    main()