"""
06_evaluation.py — SignPAK-AI
================================
Evaluates saved model checkpoints on the holdout test set (test.csv / Signer_5).
Calculates Accuracy, Precision, Recall, F1-Score, and displays the Confusion Matrix.

Outputs evaluation for both:
  - models/checkpoints/best_scratch_model.pth
  - models/checkpoints/best_finetuned_model.pth

Run from: SIGNPAK-AI root → python scripts/06_evaluation.py
"""

import csv
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ── Paths & Configuration ─────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
CSV_DIR      = DATA_DIR / "csv"
CKPT_DIR     = PROJECT_ROOT / "models" / "checkpoints"

MAX_SEQ_LEN = 60
FEATURE_DIM = 225
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Independent Dataset Definition
# ═════════════════════════════════════════════════════════════════════════════

class IndependentTestDataset(Dataset):
    def __init__(self, csv_path: Path, label_to_idx: dict, max_len: int = MAX_SEQ_LEN):
        self.max_len = max_len
        self.samples = []
        self.label_to_idx = label_to_idx

        if not csv_path.exists():
            raise FileNotFoundError(f"Test manifest CSV not found at: {csv_path}")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lm_path = row.get("landmark_path", "")
                if lm_path and (PROJECT_ROOT / lm_path).exists():
                    label_str = row["label"]
                    if label_str in self.label_to_idx:
                        self.samples.append({
                            "file": PROJECT_ROOT / lm_path,
                            "label": self.label_to_idx[label_str]
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        data = np.load(str(item["file"]))  # Shape: (T, 225)

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


class IndependentEvalModel(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, num_classes: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, 128, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(128)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(256)

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
        x = x.transpose(1, 2)            # (B, 225, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)            # (B, T, 256)

        out, _ = self.lstm(x)            # (B, T, 256)
        context = self.attn(out)         # (B, 256)
        logits = self.classifier(context)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation Engine
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(ckpt_name: str, label_to_idx: dict):
    ckpt_path = CKPT_DIR / ckpt_name
    if not ckpt_path.exists():
        print(f"⚠️ Checkpoint file '{ckpt_name}' not found at {ckpt_path}. Skipping...")
        return

    num_classes = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    print(f"\n{'='*65}")
    print(f" 📊 EVALUATING CHECKPOINT: {ckpt_name}")
    print(f"{'='*65}")

    test_path = CSV_DIR / "test.csv"
    test_dataset = IndependentTestDataset(test_path, label_to_idx=label_to_idx)

    if len(test_dataset) == 0:
        print("⚠️ No valid test samples found matching label_map.json.")
        return

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = IndependentEvalModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    target_names = [idx_to_label[i] for i in range(num_classes)]

    print("\n📋 Classification Report (Signer_5 Holdout Test Set):")
    print(classification_report(all_targets, all_preds, target_names=target_names, zero_division=0))

    cm = confusion_matrix(all_targets, all_preds)
    print("🧩 Confusion Matrix:")
    print(cm)
    print("=" * 65)


def main():
    print(f"🚀 Independent Evaluation Engine | Target Device: {DEVICE}")

    label_map_path = CKPT_DIR / "label_map.json"
    if not label_map_path.exists():
        raise FileNotFoundError(f"Label map not found at {label_map_path}. Run training script 05 first.")

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    # 1. Evaluate Scratch Model
    evaluate_checkpoint("best_scratch_model.pth", label_to_idx)

    # 2. Evaluate Fine-Tuned Model
    evaluate_checkpoint("best_finetuned_model.pth", label_to_idx)


if __name__ == "__main__":
    main()