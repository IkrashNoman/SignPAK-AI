"""
05_finetune_pretrained.py — SignPAK-AI
=======================================
INDEPENDENT Fine-Tuning Script.
Contains its own Model, Dataset, and Training logic. Does NOT depend on 05_train_scratch.py.

Usage:
  1. If 'models/checkpoints/include_pretrained.pth' exists, it loads backbone weights and fine-tunes.
  2. If 'include_pretrained.pth' is missing, it automatically falls back to clean initialization 
     with differential learning rates.

Input  : data/csv/train.csv, val.csv
Output : models/checkpoints/best_finetuned_model.pth

Run from: SIGNPAK-AI root → python scripts/05_finetune_pretrained.py
"""

import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

# ── Paths & Parameters ────────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CSV_DIR      = DATA_DIR / "csv"
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

PRETRAINED_PATH = CKPT_DIR / "include_pretrained.pth"
SAVE_PATH       = CKPT_DIR / "best_finetuned_model.pth"

MAX_SEQ_LEN   = 60      # Sequence frame count
FEATURE_DIM   = 225     # 33 pose + 21 left_hand + 21 right_hand x 3 (x,y,z)
BATCH_SIZE    = 16      # Fits Quadro T1000 4GB VRAM
WARMUP_EPOCHS = 10     # Backbone frozen
TOTAL_EPOCHS  = 50     # Total epochs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Independent Dataset
# ═════════════════════════════════════════════════════════════════════════════

class IndependentLandmarkDataset(Dataset):
    def __init__(self, csv_path: Path, max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        self.samples = []
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lm_path = row.get("landmark_path", "")
                if lm_path and (PROJECT_ROOT / lm_path).exists():
                    self.samples.append({
                        "file": PROJECT_ROOT / lm_path,
                        "label": int(row["label_idx"])
                    })
                    
        self.num_classes = len(set(s["label"] for s in self.samples)) if self.samples else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        data = np.load(str(item["file"]))  # Shape: (T, 225)
        
        # Resample temporal length T to max_len
        t = data.shape[0]
        if t != self.max_len:
            indices = np.linspace(0, t - 1, self.max_len).astype(int)
            data = data[indices]
            
        tensor_x = torch.tensor(data, dtype=torch.float32)
        tensor_y = torch.tensor(item["label"], dtype=torch.long)
        return tensor_x, tensor_y


# ═════════════════════════════════════════════════════════════════════════════
# Independent Model Architecture (1D-CNN + BiLSTM + Attention)
# ═════════════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return torch.sum(x * weights, dim=1)           # (B, hidden_dim)


class FineTuneSignPAKModel(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, num_classes: int = 30):
        super().__init__()
        # 1D-CNN Feature Extractor
        self.conv1 = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(128)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(256)
        
        # BiLSTM Sequence Encoder
        self.lstm = nn.LSTM(
            input_size=256, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )
        
        self.attn = TemporalAttention(256)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: (B, T, 225)
        x = x.transpose(1, 2)            # (B, 225, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)            # (B, T, 256)
        
        out, _ = self.lstm(x)            # (B, T, 256)
        context = self.attn(out)         # (B, 256)
        logits = self.classifier(context)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# Helper: Pretrained Weights Loader
# ═════════════════════════════════════════════════════════════════════════════

def load_pretrained_weights(model: nn.Module, pretrained_path: Path):
    if not pretrained_path.exists():
        print(f"⚠️  Pretrained model file '{pretrained_path.name}' not found.")
        print("   Proceeding with standard layer initialization + differential learning rates...\n")
        return model

    print(f"📦 Loading pretrained backbone weights from: {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location=DEVICE)
    model_dict = model.state_dict()

    # Exclude classifier layer weights due to label count mismatch
    filtered_dict = {
        k: v for k, v in state_dict.items() 
        if k in model_dict and "classifier" not in k and v.shape == model_dict[k].shape
    }

    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)
    print(f"✅ Loaded {len(filtered_dict)} transferred backbone weight tensors!\n")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Main Training Loop
# ═════════════════════════════════════════════════════════════════════════════

def run_finetune():
    print(f"🚀 Fine-Tuning Execution Engine")
    print(f"   Target Device : {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    train_dataset = IndependentLandmarkDataset(CSV_DIR / "train.csv")
    val_dataset   = IndependentLandmarkDataset(CSV_DIR / "val.csv")
    num_classes   = train_dataset.num_classes

    if len(train_dataset) == 0:
        raise RuntimeError("Train dataset empty. Ensure 04_augment_extract.py has generated data/csv/train.csv.")

    print(f"   Dataset Info  : Train={len(train_dataset)} | Val={len(val_dataset)} | Classes={num_classes}\n")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Initialize model
    model = FineTuneSignPAKModel(num_classes=num_classes)
    model = load_pretrained_weights(model, PRETRAINED_PATH).to(DEVICE)

    # ── STAGE 1: Freeze Backbone ─────────────────────────────────────────────
    for param in model.conv1.parameters(): param.requires_grad = False
    for param in model.conv2.parameters(): param.requires_grad = False
    for param in model.lstm.parameters():  param.requires_grad = False

    print("🔒 Stage 1: Backbone frozen. Training classification head only...")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    best_val_acc = 0.0

    for epoch in range(1, TOTAL_EPOCHS + 1):
        # ── STAGE 2: Unfreeze Backbone with Differential LRs ─────────────────
        if epoch == WARMUP_EPOCHS + 1:
            print("\n🔓 Stage 2: Unfreezing backbone with differential learning rates...")
            for param in model.parameters():
                param.requires_grad = True

            # Pretrained backbone receives 10x smaller learning rate
            optimizer = optim.AdamW([
                {'params': model.conv1.parameters(), 'lr': 1e-4},
                {'params': model.conv2.parameters(), 'lr': 1e-4},
                {'params': model.lstm.parameters(),  'lr': 1e-4},
                {'params': model.attn.parameters(),  'lr': 1e-3},
                {'params': model.classifier.parameters(), 'lr': 1e-3},
            ], weight_decay=1e-4)

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

        train_acc = train_correct / total_train

        # Validation Pass
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
        stage_tag = "Warmup" if epoch <= WARMUP_EPOCHS else "FineTune"

        print(f"[{stage_tag}] Epoch [{epoch:02d}/{TOTAL_EPOCHS}] - Train Loss: {train_loss/total_train:.4f} Acc: {train_acc*100:.2f}% | Val Loss: {val_loss/total_val:.4f} Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  ⭐ Best Model Saved (Val Acc: {best_val_acc*100:.2f}%)")

    print(f"\n✅ Fine-tuning complete. Output checkpoint: {SAVE_PATH}\n")


if __name__ == "__main__":
    run_finetune()