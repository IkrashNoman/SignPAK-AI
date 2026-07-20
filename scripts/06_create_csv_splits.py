# scripts/06_create_csv_splits.py
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def get_label_from_path(file_path: Path, base_features_dir: Path) -> str:
    """
    Extracts the clean sign class name from the folder path.
    Handles 'Category/Sign_Name_clip_1_aug_1.npy' and 'Category/Extra/Sign_Name_clip_1_aug_1.npy'
    """
    stem_name = file_path.stem
    if "_clip_" in stem_name:
        clean_name = stem_name.split("_clip_")[0]
    else:
        clean_name = stem_name.split("_aug_")[0]
        
    return clean_name.replace("_", " ").strip()

def main():
    print("=" * 60)
    print("📋 GENERATING DATASET INDEX AND TRAIN/VAL SPLITS")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    augmented_dir = project_root / "data" / "augmented_features"
    csv_output_dir = project_root / "data" / "csv"
    
    csv_output_dir.mkdir(parents=True, exist_ok=True)

    if not augmented_dir.exists():
        print(f"❌ 'augmented_features' folder missing. Run step 05 first.")
        return

    all_npy_files = list(augmented_dir.glob("**/*.npy"))
    if not all_npy_files:
        print("❌ No augmented numerical matrices found.")
        return

    dataset_records = []
    
    print("Parsing augmented file system structures...")
    for npy_path in all_npy_files:
        label = get_label_from_path(npy_path, augmented_dir)
        relative_path = npy_path.relative_to(project_root)
        
        dataset_records.append({
            "feature_path": str(relative_path).replace("\\", "/"), 
            "label": label
        })

    df = pd.DataFrame(dataset_records)
    
    print(f"\nSuccessfully indexed {len(df)} total files.")
    print(f"Total Unique Sign Vocabulary Classes discovered: {df['label'].nunique()}")

    # --- STRATIFIED TRAIN / VALIDATION SPLIT (80% / 20%) ---
    train_data = []
    val_data = []

    for label_name, group in df.groupby("label"):
        shuffled_group = group.sample(frac=1, random_state=42).reset_index(drop=True)
        split_point = int(len(shuffled_group) * 0.8)
        
        train_data.append(shuffled_group.iloc[:split_point])
        val_data.append(shuffled_group.iloc[split_point:])

    df_train = pd.concat(train_data).sample(frac=1, random_state=42).reset_index(drop=True)
    df_val = pd.concat(val_data).sample(frac=1, random_state=42).reset_index(drop=True)

    train_csv_path = csv_output_dir / "train.csv"
    val_csv_path = csv_output_dir / "val.csv"

    df_train.to_csv(train_csv_path, index=False)
    df_val.to_csv(val_csv_path, index=False)

    print("\n" + "=" * 60)
    print("✅ INDEX SPLITTING COMPLETE")
    print("=" * 60)
    print(f"Saved: {train_csv_path.relative_to(project_root)} ({len(df_train)} samples)")
    print(f"Saved: {val_csv_path.relative_to(project_root)} ({len(df_val)} samples)")
    print("=" * 60)

if __name__ == "__main__":
    main()