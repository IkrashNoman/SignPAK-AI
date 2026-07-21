# scripts/05_create_csv_splits.py
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def get_label_and_clip(file_path: Path) -> tuple:
    stem_name = file_path.stem  # e.g., "Wrong_clip_1_aug_14"
    if "_aug_" in stem_name:
        base_name = stem_name.split("_aug_")[0]
    else:
        base_name = stem_name
        
    if "_clip_" in base_name:
        label_part, clip_part = base_name.split("_clip_", 1)
        clip_id = f"{label_part}_clip_{clip_part}"
    else:
        label_part = base_name
        clip_id = base_name
        
    clean_label = label_part.replace("_", " ").strip()
    return clean_label, clip_id

def main():
    print("=" * 60)
    print("📋 GENERATING LEAK-FREE SPLIT MANIFESTS")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    features_dir = project_root / "data" / "features"
    csv_output_dir = project_root / "data" / "csv"
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    all_npy_files = list(features_dir.glob("**/*.npy"))
    if not all_npy_files:
        print("❌ No feature matrices found. Run step 04 first.")
        return

    records = []
    for npy_path in all_npy_files:
        label, clip_id = get_label_and_clip(npy_path)
        records.append({
            "feature_path": str(npy_path.relative_to(project_root)).replace("\\", "/"),
            "label": label,
            "clip_id": clip_id
        })

    df = pd.DataFrame(records)
    train_records, val_records = [], []

    for label, group in df.groupby("label"):
        clips = group["clip_id"].unique()
        clips_series = pd.Series(clips).sample(frac=1, random_state=42).values
        
        split_idx = int(len(clips_series) * 0.8)
        if split_idx == 0 and len(clips_series) > 1:
            split_idx = 1
            
        train_clips = clips_series[:split_idx]
        val_clips = clips_series[split_idx:] if len(clips_series) > 1 else clips_series

        train_records.append(group[group["clip_id"].isin(train_clips)])
        val_records.append(group[group["clip_id"].isin(val_clips)])

    df_train = pd.concat(train_records).drop(columns=["clip_id"]).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.concat(val_records).drop(columns=["clip_id"]).sample(frac=1, random_state=42).reset_index(drop=True)

    df_train.to_csv(csv_output_dir / "train.csv", index=False)
    df_val.to_csv(csv_output_dir / "val.csv", index=False)

    print(f"✅ Splits Generated: Train ({len(df_train)} rows), Validation ({len(df_val)} rows)")

if __name__ == "__main__":
    main()