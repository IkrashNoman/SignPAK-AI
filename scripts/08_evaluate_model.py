# scripts/08_evaluate_model.py
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# --- Re-declared Dataset Mapping for Self-Containment ---
class SignLanguageDataset(Dataset):
    def __init__(self, csv_file: Path, project_root: Path, label_to_idx=None):
        self.df = pd.read_csv(csv_file)
        self.project_root = project_root
        
        if label_to_idx is None:
            unique_labels = sorted(self.df["label"].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = self.project_root / row["feature_path"]
        features = np.load(str(file_path)).astype(np.float32)
        label_str = row["label"]
        label_idx = self.label_to_idx[label_str]
        return torch.tensor(features), torch.tensor(label_idx, dtype=torch.long)

# --- Re-declared Architecture for Self-Containment ---
class SignPAKBiLSTM(nn.Module):
    def __init__(self, input_size=368, hidden_size=128, num_layers=2, num_classes=29):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.5 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :] 
        return self.fc(out)

def main():
    print("=" * 60)
    print("📊 EVALUATING TRAINED MODEL & GENERATING METRICS")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    csv_dir = project_root / "data" / "csv"
    models_dir = project_root / "models"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Running evaluations on: {device.type.upper()}")

    # 1. Load class mapping dictionaries
    mapping_path = models_dir / "sign_classes.json"
    if not mapping_path.exists():
        print("❌ Map file missing under models/ directory.")
        return
        
    with open(mapping_path, "r") as f:
        label_to_idx = json.load(f)
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    # 2. Setup Validation Data Loader
    val_csv = csv_dir / "val.csv"
    val_dataset = SignLanguageDataset(val_csv, project_root, label_to_idx=label_to_idx)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 3. Reconstruct Model and Apply Weights Profile
    model = SignPAKBiLSTM(input_size=368, num_classes=len(class_names)).to(device)
    weights_path = models_dir / "signpak_lstm_model.pth"
    
    if not weights_path.exists():
        print("❌ Model weights profile file not found.")
        return
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 4. Predict validation set arrays
    all_preds = []
    all_targets = []

    print("Processing evaluation forward passes...")
    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_labels.numpy())

    # 5. Build textual classification report metrics
    print("\n📝 GENERATING CLASSIFICATION REPORT:\n")
    report = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    print(report)
    
    # Save text report for your thesis copy-pasting
    with open(reports_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # 6. Generate and save the Confusion Matrix Heatmap Chart
    print("🎨 Rendering confusion matrix plot...")
    cm = confusion_matrix(all_targets, all_preds)
    
    plt.figure(figsize=(16, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        xticklabels=class_names, 
        yticklabels=class_names,
        cbar=True
    )
    plt.title("SignPAK-AI: Confusion Matrix Heatmap Validation Profile", fontsize=14, fontweight="bold", pad=20)
    plt.ylabel("Actual Sign Class Label", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Sign Class Label", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    matrix_img_path = reports_dir / "confusion_matrix.png"
    plt.savefig(matrix_img_path, dpi=300)
    plt.close()

    print("\n" + "=" * 60)
    print("✅ EVALUATION REPORT GENERATION COMPLETE")
    print(f"Text breakdown saved: reports/classification_report.txt")
    print(f"Heatmap chart saved:   reports/confusion_matrix.png")
    print("=" * 60)

if __name__ == "__main__":
    main()