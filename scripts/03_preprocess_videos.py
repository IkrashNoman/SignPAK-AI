# scripts/03_preprocess_videos.py
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def motion_based_trimmer(video_path: Path, output_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames = []
    motion_scores = []
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        
        # Determine movement profile via frame differences
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is None:
            prev_frame = gray
            motion_scores.append(0)
            continue
            
        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        score = np.sum(thresh)
        motion_scores.append(score)
        prev_frame = gray
    cap.release()

    if not motion_scores:
        return

    # Establish dynamic threshold based on variance of video motion
    median_motion = np.median(motion_scores)
    threshold = median_motion * 1.5
    
    # Track spans of activity
    active_frames = [idx for idx, score in enumerate(motion_scores) if score > threshold]
    
    if not active_frames:
        return

    # Group continuous blocks of motion
    clips = []
    start_f = active_frames[0]
    
    for i in range(1, len(active_frames)):
        if active_frames[i] - active_frames[i-1] > (fps * 1.5): # Gap boundary longer than 1.5 sec
            clips.append((max(0, start_f - 5), min(len(frames), active_frames[i-1] + 5)))
            start_f = active_frames[i]
    clips.append((max(0, start_f - 5), min(len(frames), active_frames[-1] + 5)))

    # Export clean continuous sub-clips
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    clip_index = 1
    for start, end in clips:
        if (end - start) < (fps * 0.5): # Drop fragments under 0.5 seconds
            continue
            
        out_name = f"{video_path.stem}_clip_{clip_index}.mp4"
        out_path = output_dir / out_name
        
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        for f_idx in range(start, end):
            writer.write(frames[f_idx])
        writer.release()
        clip_index += 1

def main():
    print("=" * 60)
    print("🎬 SPLITTING AND PREPROCESSING VIDEO DATASET")
    print("=" * 60)
    
    raw_dir = Path(config.RAW_VIDEOS_DIR)
    # Output segmented videos cleanly into their respective matching dynamic categories
    processed_base = raw_dir.parent / "processed_videos"
    
    videos = list(raw_dir.glob("**/*.mp4"))
    for video in videos:
        # Replicates your folder structures perfectly inside processed_videos/
        relative_folder = video.parent.relative_to(raw_dir)
        output_folder = processed_base / relative_folder
        output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing segments for: {video.name}")
        motion_based_trimmer(video, output_folder)
        
    print("\n✅ Preprocessing complete. Ready for MediaPipe feature landmark extraction.")

if __name__ == "__main__":
    main()