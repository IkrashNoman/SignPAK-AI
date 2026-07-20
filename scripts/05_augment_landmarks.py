# scripts/05_augment_landmarks.py
import sys
from pathlib import Path
import numpy as np
import random

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

TARGET_FRAMES = 30  # Standard fixed macro-picture size
DESIRED_SAMPLES_PER_CLIP = 30  # Quota calculation target

def resize_sequence(sequence: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    """Standardizes sequence lengths using linear frame interpolation."""
    current_frames = sequence.shape[0]
    if current_frames == 0:
        return np.zeros((target_frames, sequence.shape[1]))
        
    if current_frames == target_frames:
        return sequence

    # Calculate distributed indices across the timeline
    indices = np.linspace(0, current_frames - 1, target_frames)
    indices_floor = np.floor(indices).astype(int)
    indices_ceil = np.ceil(indices).astype(int)
    weight = indices - indices_floor

    resized = []
    for i in range(target_frames):
        f_low = sequence[indices_floor[i]]
        f_high = sequence[indices_ceil[i]]
        # Linearly blend adjacent frame values
        blended_frame = f_low + (f_high - f_low) * weight[i]
        resized.append(blended_frame)

    return np.array(resized)

def apply_coordinate_jitter(sequence: np.ndarray, noise_level: float = 0.003) -> np.ndarray:
    """Injects micro-muscle tremors/noise safely into the coordinate tracking fields."""
    jitter = np.random.normal(0, noise_level, sequence.shape)
    # Don't add noise to zeroed out/untracked padding coordinates
    mask = (sequence != 0.0).astype(float)
    return sequence + (jitter * mask)

def apply_spatial_scaling(sequence: np.ndarray, lower_bound: float = 0.95, upper_bound: float = 1.05) -> np.ndarray:
    """Rescales structural dimensions to simulate varying body shapes/camera distances."""
    factor = random.uniform(lower_bound, upper_bound)
    return sequence * factor

def main():
    print("=" * 60)
    print("🚀 EXECUTING LANDMARK COORDINATE MATHEMATICAL AUGMENTATION")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    features_dir = project_root / "data" / "features"
    augmented_dir = project_root / "data" / "augmented_features"

    if not features_dir.exists():
        print("❌ 'features' directory missing. Please complete file 04 first.")
        return

    feature_files = list(features_dir.glob("**/*.npy"))
    print(f"Found {len(feature_files)} standard extracted feature matrices.")

    for npy_path in feature_files:
        relative_folder = npy_path.parent.relative_to(features_dir)
        output_folder = augmented_dir / relative_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        # Load raw baseline array coordinates
        raw_sequence = np.load(str(npy_path))
        
        # Step A: Enforce standardized timeline constraint (30 macro-frames)
        standardized_base = resize_sequence(raw_sequence, TARGET_FRAMES)
        
        # Save the clean standardized master copy as Sample #1
        np.save(str(output_folder / f"{npy_path.stem}_aug_1.npy"), standardized_base)

        # Step B: Generate 29 distinct alternative mutations using our matrix parameters
        for idx in range(2, DESIRED_SAMPLES_PER_CLIP + 1):
            augmented = standardized_base.copy()
            
            # Apply our spatial tracking mutations alternately
            augmented = apply_spatial_scaling(augmented)
            augmented = apply_coordinate_jitter(augmented)
            
            output_path = output_folder / f"{npy_path.stem}_aug_{idx}.npy"
            np.save(str(output_path), augmented)

        print(f"Generated 30 augmented samples for: {relative_folder / npy_path.name}")

    print("\n" + "=" * 60)
    print("✅ Augmentation complete! Expanded arrays saved under 'data/augmented_features/'.")
    print("=" * 60)

if __name__ == "__main__":
    main()