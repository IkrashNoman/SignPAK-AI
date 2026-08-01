"""
07_realtime_webcam.py — SignPAK-AI
==================================
Live Webcam Sign Language Recognition.
Completely standalone and independent script.

Captures webcam frames, extracts MediaPipe landmarks in real time,
runs inference using PyTorch model checkpoint, and displays predictions on screen.

Run from: SIGNPAK-AI root → python scripts/07_realtime_webcam.py
"""

import cv2
import json
import torch
import torch.nn as nn
import collections
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Paths & Config ────────────────────────────────────────────────────────────
_THIS        = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
CKPT_DIR     = MODEL_DIR / "checkpoints"

POSE_MODEL_PATH = MODEL_DIR / "pose_landmarker.task"
HAND_MODEL_PATH = MODEL_DIR / "hand_landmarker.task"
LABEL_MAP_PATH  = CKPT_DIR / "label_map.json"

FEATURE_DIM = 225
MAX_SEQ_LEN = 60
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# Independent Model Definition
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


class StandaloneInferenceModel(nn.Module):
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
# Live Webcam Engine
# ═════════════════════════════════════════════════════════════════════════════

def run_live_webcam():
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}. Run script 05 first.")

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_to_idx = json.load(f)

    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes  = len(label_to_idx)

    # Select best available model checkpoint
    ckpt_path = CKPT_DIR / "best_finetuned_model.pth"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "best_scratch_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError("No trained model checkpoint found in models/checkpoints/. Run script 05 first.")

    print(f"📦 Loading Model Weights: {ckpt_path.name}")
    model = StandaloneInferenceModel(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
    model.eval()

    # Initialize MediaPipe Task Engines
    base_pose_opts = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
    pose_opts = vision.PoseLandmarkerOptions(base_options=base_pose_opts, running_mode=vision.RunningMode.IMAGE)
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_opts)

    base_hand_opts = python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH))
    hand_opts = vision.HandLandmarkerOptions(base_options=base_hand_opts, running_mode=vision.RunningMode.IMAGE, num_hands=2)
    hand_landmarker = vision.HandLandmarker.create_from_options(hand_opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Web camera could not be opened.")
        return

    print("\n🎥 Real-Time Sign Language Recognition Active!")
    print("   Press 'q' or 'ESC' on the camera window to quit.\n")

    frame_buffer = collections.deque(maxlen=MAX_SEQ_LEN)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror view
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        row_vec = np.zeros(225, dtype=np.float32)

        # Pose Extraction
        pose_res = pose_landmarker.detect(mp_image)
        if pose_res.pose_landmarks and len(pose_res.pose_landmarks[0]) >= 33:
            for idx, lm in enumerate(pose_res.pose_landmarks[0][:33]):
                row_vec[idx * 3 : idx * 3 + 3] = [lm.x, lm.y, lm.z]

        # Hand Extraction
        hand_res = hand_landmarker.detect(mp_image)
        if hand_res.hand_landmarks and hand_res.handedness:
            for hand_lms, handedness_cat in zip(hand_res.hand_landmarks, hand_res.handedness):
                lbl = handedness_cat[0].category_name.lower()
                offset = 99 if lbl == "left" else 162
                for idx, lm in enumerate(hand_lms[:21]):
                    row_vec[offset + idx * 3 : offset + idx * 3 + 3] = [lm.x, lm.y, lm.z]

        frame_buffer.append(row_vec)

        predicted_sign = "Buffering..."
        confidence = 0.0

        if len(frame_buffer) == MAX_SEQ_LEN:
            buffer_array = np.array(frame_buffer)
            tensor_in = torch.tensor(buffer_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor_in)
                probs  = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)

                confidence = conf.item()
                if confidence > 0.40:
                    predicted_sign = idx_to_label[pred_idx.item()]
                else:
                    predicted_sign = "Listening..."

        # UI Overlay
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)
        display_str = f"SIGN: {predicted_sign.upper()} ({confidence*100:.1f}%)"
        cv2.putText(frame, display_str, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow("SignPAK-AI Live Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_landmarker.close()
    hand_landmarker.close()


if __name__ == "__main__":
    run_live_webcam()