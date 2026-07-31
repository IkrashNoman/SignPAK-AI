"""
05_train_scratch.py — SignPAK-AI
==================================
Trains a 1D-CNN + BiLSTM + Attention network from scratch on SignPAK-AI 
landmark tensors (shape: T, 225).

Input  : data/csv/train.csv, val.csv, test.csv
Output : models/checkpoints/best_scratch_model.pth

Run from: SIGNPAK-AI root → python scripts/05_train_scratch.py
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ── Configuration & Paths ─────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CSV_DIR      = DATA_DIR / "csv"
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

MAX_SEQ_LEN = 60      # Standardized frame length
FEATURE_DIM = 225     # 33 pose + 21 left_hand + 21 right_hand x 3 (x,y,z)
BATCH_SIZE  = 16      # Optimized for 4GB VRAM (Quadro T1000)
EPOCHS      = 50
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── PyTorch Dataset Class ─────────────────────────────────────────────────────
class LandmarkDataset(Dataset):
    def __init__(self, csv_path: Path, max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        self.samples = []
        
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lm_path = row.get("landmark_path", "")
                if lm_path and (PROJECT_ROOT / lm_path).exists():
                    self.samples.append({
                        "file": PROJECT_ROOT / lm_path,
                        "label": int(row["label_idx"])
                    })
                    
        # Extract unique label count
        self.num_classes = len(set(s["label"] for s in self.samples)) if self.samples else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        data = np.load(str(item["file"]))  # Shape: (T, 225)
        
        # Resample sequence length T to max_len
        t = data.shape[0]
        if t != self.max_len:
            indices = np.linspace(0, t - 1, self.max_len).astype(int)
            data = data[indices]
            
        tensor_x = torch.tensor(data, dtype=torch.float32)
        tensor_y = torch.tensor(item["label"], dtype=torch.long)
        return tensor_x, tensor_y


# ── Model Architecture: 1D-CNN + BiLSTM + Self-Attention ──────────────────────
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return torch.sum(x * weights, dim=1)           # (B, hidden_dim)


class SignPAKModel(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, num_classes=30):
        super().__init__()
        # 1D Conv Spatial Feature Extraction
        self.conv1 = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(128)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(256)
        
        # BiLSTM Sequence Modeling
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        self.attn = TemporalAttention(256)  # 128 * 2 = 256
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, 225)
        x = x.transpose(1, 2)            # (B, 225, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)            # (B, T, 256)
        
        out, _ = self.lstm(x)            # (B, T, 256)
        context = self.attn(out)         # (B, 256)
        logits = self.classifier(context)
        return logits


# ── Training Loop ─────────────────────────────────────────────────────────────
def train_scratch():
    print(f"🚀 Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    train_dataset = LandmarkDataset(CSV_DIR / "train.csv")
    val_dataset   = LandmarkDataset(CSV_DIR / "val.csv")
    
    if len(train_dataset) == 0:
        raise FileNotFoundError("train.csv is empty or landmark files are missing.")
        
    num_classes = train_dataset.num_classes
    print(f"Dataset Loaded: Train samples={len(train_dataset)}, Val samples={len(val_dataset)}, Classes={num_classes}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    model = SignPAKModel(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_val_acc = 0.0
    save_path = CKPT_DIR / "best_scratch_model.pth"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_acc = train_correct / total_train

        # Validation
        model.eval()
        val_loss, val_correct, total_val = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = val_correct / total_val if total_val > 0 else 0.0

        print(f"Epoch [{epoch:02d}/{EPOCHS}] - Train Loss: {train_loss/total_train:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/total_val:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Best Model Saved (Val Acc: {best_val_acc*100:.2f}%)")

    print(f"\n✅ Scratch Training Complete. Saved to: {save_path}")

if __name__ == "__main__":
    train_scratch()