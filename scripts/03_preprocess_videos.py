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

TARGET_FPS = 30
TARGET_WIDTH = 640
TARGET_HEIGHT = 640
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

def ensure_hand_model_exists(project_root: Path) -> Path:
    model_dir = project_root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "hand_landmarker.task"
    if not model_path.exists():
        print("📥 Hand Landmarker model missing. Downloading...")
        urllib.request.urlretrieve(HAND_MODEL_URL, model_path)
        print("✅ Download finished.")
    return model_path

def segment_video_by_hand_presence(video_path: Path, output_dir: Path, hand_landmarker):
    """
    Segments videos by tracking hand presence/activity using MediaPipe Tasks API in IMAGE mode.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"⚠️ Could not open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or TARGET_FPS
    frames = []
    hand_activity = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        frames.append(resized)

        # Process frame with HandLandmarker in IMAGE mode
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = hand_landmarker.detect(mp_image)

        if results.hand_landmarks:
            hand_activity.append(1)
        else:
            hand_activity.append(0)

    cap.release()

    if not frames or sum(hand_activity) == 0:
        return

    # Find continuous blocks where hands are active
    clips = []
    in_sign = False
    start_frame = 0
    padding = int(fps * 0.2)  # 200ms buffer before/after
    max_gap = int(fps * 0.5)  # Allow up to 0.5s pause mid-sign
    gap_counter = 0

    for idx, active in enumerate(hand_activity):
        if active == 1:
            if not in_sign:
                in_sign = True
                start_frame = idx
            gap_counter = 0
        else:
            if in_sign:
                gap_counter += 1
                if gap_counter > max_gap:
                    end_frame = idx - gap_counter
                    clips.append((max(0, start_frame - padding), min(len(frames), end_frame + padding)))
                    in_sign = False
                    gap_counter = 0

    if in_sign:
        clips.append((max(0, start_frame - padding), len(frames)))

    # Save individual continuous sub-clips
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_index = 1
    
    for start, end in clips:
        # Drop tiny noise fragments shorter than 0.4s
        if (end - start) < (fps * 0.4):
            continue

        out_name = f"{video_path.stem}_clip_{clip_index}.mp4"
        out_path = output_dir / out_name

        writer = cv2.VideoWriter(str(out_path), fourcc, TARGET_FPS, (TARGET_WIDTH, TARGET_HEIGHT))
        for f_idx in range(start, end):
            writer.write(frames[f_idx])
        writer.release()
        
        clip_index += 1

def main():
    print("=" * 60)
    print("🎬 ISOLATING SIGN REPETITIONS VIA HAND DETECTION")
    print("=" * 60)
    
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = Path(config.RAW_VIDEOS_DIR)
    processed_base = raw_dir.parent / "processed_videos"
    
    model_path = ensure_hand_model_exists(project_root)

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # IMAGE mode avoids timestamp reset crashes
        num_hands=2
    )

    videos = list(raw_dir.glob("**/*.mp4"))
    
    with vision.HandLandmarker.create_from_options(options) as hand_landmarker:
        for video in videos:
            relative_folder = video.parent.relative_to(raw_dir)
            output_folder = processed_base / relative_folder
            output_folder.mkdir(parents=True, exist_ok=True)
            
            print(f"Segmenting sign instances for: {video.name}")
            segment_video_by_hand_presence(video, output_folder, hand_landmarker)
        
    print("\n✅ Preprocessing complete. Individual sign clips extracted cleanly.")

if __name__ == "__main__":
    main()