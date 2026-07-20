# scripts/07_train_model.py
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

# --- Pytorch Dataset Mapping Configuration ---
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

# --- Deep Neural Network Architecture Configuration ---
class SignPAKBiLSTM(nn.Module):
    def __init__(self, input_size=368, hidden_size=128, num_layers=2, num_classes=29):
        super().__init__()  # Fixed syntax error here
        
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
    print("🧠 SIGNPAK-AI DEEP LEARNING ENGINE: TRAINING CORE")
    print("=" * 60)

    project_root = Path(__file__).resolve().parents[1]
    csv_dir = project_root / "data" / "csv"
    models_out_dir = project_root / "models"
    models_out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Execution hardware selected: {device.type.upper()}")
    if device.type == "cuda":
        print(f"🔥 GPU Device Detected: {torch.cuda.get_device_name(0)}")

    train_csv = csv_dir / "train.csv"
    val_csv = csv_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        print("❌ Missing spreadsheet maps. Run 06_create_csv_splits.py first.")
        return

    train_dataset = SignLanguageDataset(train_csv, project_root)
    val_dataset = SignLanguageDataset(val_csv, project_root, label_to_idx=train_dataset.label_to_idx)

    with open(models_out_dir / "sign_classes.json", "w") as f:
        json.dump(train_dataset.label_to_idx, f, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    num_classes = len(train_dataset.label_to_idx)
    model = SignPAKBiLSTM(input_size=368, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 40
    best_val_loss = float("inf")

    print(f"\nBeginning Network Optimization Profile over {epochs} Epoch iterations...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                
                val_loss += loss.item() * batch_features.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()

        epoch_train_loss = train_loss / total_train
        epoch_train_acc = (correct_train / total_train) * 100
        epoch_val_loss = val_loss / total_val
        epoch_val_acc = (correct_val / total_val) * 100

        print(f"Epoch [{epoch:02d}/{epochs}] -> Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% || Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), models_out_dir / "signpak_lstm_model.pth")

    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING PIPELINE COMPLETE")
    print(f"Best Weights Profile Saved: models/signpak_lstm_model.pth")
    print(f"Class Mapping Config Saved: models/sign_classes.json")
    print("=" * 60)

if __name__ == "__main__":
    main()