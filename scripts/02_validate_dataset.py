# scripts/02_validate_dataset.py
import sys
from pathlib import Path
import cv2

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def validate_dataset():
    print("=" * 60)
    print("🔍 VALIDATING DOWNLOADED DATASET")
    print("=" * 60)
    
    target_dir = Path(config.RAW_VIDEOS_DIR)
    video_files = list(target_dir.glob("**/*.mp4"))
    
    if not video_files:
        print("❌ No videos found! Please run your download script first.")
        return

    corrupt_count = 0
    valid_count = 0

    for video_path in video_files:
        # Check 1: Physical file size
        if video_path.stat().st_size == 0:
            print(f"❌ Corrupt (0 Bytes): {video_path.relative_to(target_dir)}")
            corrupt_count += 1
            continue

        # Check 2: OpenCV Frame Integrity
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ Corrupt (Cannot Open): {video_path.relative_to(target_dir)}")
            corrupt_count += 1
            cap.release()
            continue

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"❌ Corrupt (No Frame Data): {video_path.relative_to(target_dir)}")
            corrupt_count += 1
        else:
            valid_count += 1
        cap.release()

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Videos Checked : {len(video_files)}")
    print(f"Valid Videos         : {valid_count}")
    print(f"Corrupt Videos       : {corrupt_count}")
    print("=" * 60)

if __name__ == "__main__":
    validate_dataset()