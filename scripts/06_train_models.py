# scripts/06_train_models.py
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
        features = np.load(str(self.project_root / row["feature_path"])).astype(np.float32)
        label_idx = self.label_to_idx[row["label"]]
        return torch.tensor(features), torch.tensor(label_idx, dtype=torch.long)

class SignPAKBiGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 256, num_layers: int = 2, num_classes: int = 29):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 4, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        avg_pool = torch.mean(gru_out, dim=1)
        max_pool, _ = torch.max(gru_out, dim=1)
        return self.classifier(torch.cat([avg_pool, max_pool], dim=1))

def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_dir = project_root / "data" / "csv"
    models_out_dir = project_root / "models"
    models_out_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Device: {device.type.upper()}")

    train_dataset = SignLanguageDataset(csv_dir / "train.csv", project_root)
    val_dataset = SignLanguageDataset(csv_dir / "val.csv", project_root, label_to_idx=train_dataset.label_to_idx)

    with open(models_out_dir / "sign_classes.json", "w") as f:
        json.dump(train_dataset.label_to_idx, f, indent=4)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = SignPAKBiGRU(input_size=30*2 if False else train_dataset[0][0].shape[1], hidden_size=256, num_classes=len(train_dataset.label_to_idx)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)

    best_val_acc = 0.0
    for epoch in range(1, 81):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                _, predicted = torch.max(model(batch_features), 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        val_acc = (correct / total) * 100
        scheduler.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/80] -> Validation Accuracy: {val_acc:.2f}%")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), models_out_dir / "signpak_lstm_model.pth")

    print(f"\n✅ Training Complete! Peak Val Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()