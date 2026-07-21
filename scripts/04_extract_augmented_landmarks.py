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

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

FACIAL_EXPRESSION_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 418, 312, 311, 310, 13, 14, 82, 81, 80,
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300
]

NOSE_INDEX = 0
CHIN_INDEX = 17
TARGET_VARIANTS = 30
SEQUENCE_LENGTH = 30
TOTAL_FEATURE_DIM = 746  # 373 spatial + 373 velocity

def ensure_model_exists(project_root: Path) -> Path:
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "holistic_landmarker.task"
    if not model_path.exists():
        print("📥 Model binary bundle missing. Downloading from Google servers...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        print("✅ Download finished.")
    return model_path

def extract_base_landmarks(frames, landmarker) -> np.ndarray:
    sequence_data = []

    for frame in frames:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # Using detect() for IMAGE mode instead of detect_for_video()
        results = landmarker.detect(mp_image)

        # 1. Pose (23 keypoints x 4 values = 92)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for i, lm in enumerate(results.pose_landmarks) if i <= 22]).flatten()
        else:
            pose = np.zeros(23 * 4)

        # 2. Face (50 keypoints x 3 values = 150)
        if results.face_landmarks:
            face_all = results.face_landmarks
            face = np.array([[face_all[idx].x, face_all[idx].y, face_all[idx].z] for idx in FACIAL_EXPRESSION_INDICES]).flatten()
            global_nose = np.array([face_all[NOSE_INDEX].x, face_all[NOSE_INDEX].y, face_all[NOSE_INDEX].z])
            global_chin = np.array([face_all[CHIN_INDEX].x, face_all[CHIN_INDEX].y, face_all[CHIN_INDEX].z])
        else:
            face = np.zeros(len(FACIAL_EXPRESSION_INDICES) * 3)
            global_nose = np.zeros(3)
            global_chin = np.zeros(3)

        # 3. Left Hand (21 keypoints x 3 values = 63, zero-centered)
        if results.left_hand_landmarks:
            lh_raw = results.left_hand_landmarks
            global_lh_wrist = np.array([lh_raw[0].x, lh_raw[0].y, lh_raw[0].z])
            lh = np.array([[lm.x - global_lh_wrist[0], lm.y - global_lh_wrist[1], lm.z - global_lh_wrist[2]] for lm in lh_raw]).flatten()
        else:
            lh = np.zeros(21 * 3)
            global_lh_wrist = np.zeros(3)

        # 4. Right Hand (21 keypoints x 3 values = 63, zero-centered)
        if results.right_hand_landmarks:
            rh_raw = results.right_hand_landmarks
            global_rh_wrist = np.array([rh_raw[0].x, rh_raw[0].y, rh_raw[0].z])
            rh = np.array([[lm.x - global_rh_wrist[0], lm.y - global_rh_wrist[1], lm.z - global_rh_wrist[2]] for lm in rh_raw]).flatten()
        else:
            rh = np.zeros(21 * 3)
            global_rh_wrist = np.zeros(3)

        # 5. Relative Distances (5 values)
        d1 = np.linalg.norm(global_lh_wrist - global_nose) if np.any(global_lh_wrist) and np.any(global_nose) else 0.0
        d2 = np.linalg.norm(global_lh_wrist - global_chin) if np.any(global_lh_wrist) and np.any(global_chin) else 0.0
        d3 = np.linalg.norm(global_rh_wrist - global_nose) if np.any(global_rh_wrist) and np.any(global_nose) else 0.0
        d4 = np.linalg.norm(global_rh_wrist - global_chin) if np.any(global_rh_wrist) and np.any(global_chin) else 0.0
        d5 = np.linalg.norm(global_lh_wrist - global_rh_wrist) if np.any(global_lh_wrist) and np.any(global_rh_wrist) else 0.0
        distances = np.array([d1, d2, d3, d4, d5])

        spatial_frame = np.concatenate([pose, face, lh, rh, distances])
        sequence_data.append(spatial_frame)

    if len(sequence_data) == 0:
        return np.zeros((SEQUENCE_LENGTH, 373))

    spatial_matrix = np.array(sequence_data)
    
    # Resample linearly to fixed sequence length
    indices = np.linspace(0, len(spatial_matrix) - 1, SEQUENCE_LENGTH).astype(int)
    return spatial_matrix[indices]

def compute_velocities_and_combine(spatial_matrix: np.ndarray) -> np.ndarray:
    """Computes velocities safely and returns (30, 746) feature tensor."""
    num_frames = spatial_matrix.shape[0]
    velocity_data = np.zeros_like(spatial_matrix)
    
    for t in range(1, num_frames):
        delta = spatial_matrix[t] - spatial_matrix[t-1]
        delta[np.abs(delta) > 0.5] = 0.0  # Cap tracking jump glitches
        velocity_data[t] = delta

    return np.concatenate([spatial_matrix, velocity_data], axis=1)

def augment_spatial_matrix(spatial_matrix: np.ndarray) -> np.ndarray:
    """Applies fast 3D matrix-level geometric transforms directly to landmarks."""
    aug = spatial_matrix.copy()
    
    # 1. Random Scale (0.9x to 1.1x)
    scale = np.random.uniform(0.9, 1.1)
    aug[:, :305] *= scale  # Scale spatial coordinates, skip distances
    
    # 2. Random Shift (-0.05 to +0.05)
    shift = np.random.uniform(-0.05, 0.05, size=(1, aug.shape[1]))
    aug += shift
    
    # 3. Gaussian Noise Jitter
    noise = np.random.normal(0, 0.005, size=aug.shape)
    aug += noise
    
    return aug

def main():
    print("=" * 60)
    print("🧬 VECTORIZED LANDMARK EXTRACTION & MATRIX AUGMENTATION")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed_videos"
    features_base_dir = project_root / "data" / "features"

    model_asset_path = ensure_model_exists(project_root)
    video_files = list(processed_dir.glob("**/*.mp4"))
    
    if not video_files:
        print("❌ No processed videos found. Run step 03 first.")
        return

    base_options = python.BaseOptions(model_asset_path=str(model_asset_path))
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Switching to IMAGE mode prevents timestamp errors
        output_face_blendshapes=False
    )

    # Initialize MediaPipe landmarker ONCE for all processed video clips
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for video_path in video_files:
            cap = cv2.VideoCapture(str(video_path))
            frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            if not frames:
                continue

            relative_folder = video_path.parent.relative_to(processed_dir)
            output_folder = features_base_dir / relative_folder
            output_folder.mkdir(parents=True, exist_ok=True)

            print(f"Extracting & augmenting features for: {video_path.name}")

            # Step 1: Run MediaPipe ONCE on base frames
            base_spatial = extract_base_landmarks(frames, landmarker)
            
            # Save Base Variant
            base_full = compute_velocities_and_combine(base_spatial)
            np.save(str(output_folder / f"{video_path.stem}_aug_1.npy"), base_full)

            # Step 2: Generate 29 augmented variations instantly via matrix operations
            for v_idx in range(2, TARGET_VARIANTS + 1):
                aug_spatial = augment_spatial_matrix(base_spatial)
                aug_full = compute_velocities_and_combine(aug_spatial)
                np.save(str(output_folder / f"{video_path.stem}_aug_{v_idx}.npy"), aug_full)

    print("\n✅ High-Speed Extraction & Augmentation Complete! Saved under 'data/features/'.")

if __name__ == "__main__":
    main()