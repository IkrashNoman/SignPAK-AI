# scripts/08_evaluate_model.py
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from scripts.06_train_models import SignPAKBiGRU, SignLanguageDataset

def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_dir = project_root / "data" / "csv"
    models_dir = project_root / "models"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = SignLanguageDataset(csv_dir / "val.csv", project_root)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SignPAKBiGRU(input_size=val_dataset[0][0].shape[1], hidden_size=256, num_classes=len(val_dataset.label_to_idx)).to(device)
    model.load_state_dict(torch.load(models_dir / "signpak_lstm_model.pth", map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    print(f"\n========================================")
    print(f"🎯 FINAL EVALUATION ACCURACY: {acc:.2f}%")
    print(f"========================================")
    
    # Reverse index mapping for printing labels
    idx_to_label = {v: k for k, v in val_dataset.label_to_idx.items()}
    target_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    print(classification_report(all_labels, all_preds, target_names=target_names))

if __name__ == "__main__":
    main()