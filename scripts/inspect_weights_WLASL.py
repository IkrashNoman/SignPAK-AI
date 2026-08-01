"""
inspect_weights.py — Pretrained Weight Inspector
=================================================
Prints the first 15 layer names and tensor shapes stored inside models/tgcn_wlasl_pretrained.pth
"""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "tgcn_wlasl_pretrained.pth"

def main():
    if not MODEL_PATH.exists():
        print(f"❌ Weight file not found at: {MODEL_PATH}")
        return

    print(f"📦 Loading weight file: {MODEL_PATH}")
    sd = torch.load(MODEL_PATH, map_location="cpu")
    
    # Handle nested state dictionaries if wrapped in dict keys
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

    if not isinstance(sd, dict):
        print(f"⚠️ State dict is type {type(sd)}, trying state_dict() method...")
        sd = getattr(sd, "state_dict", lambda: {})()

    keys = list(sd.keys())
    print(f"\n✅ Total layers found in checkpoint: {len(keys)}")
    print("=" * 60)
    print("First 15 Tensor Keys & Shapes in Checkpoint:")
    print("=" * 60)

    for k in keys[:15]:
        val = sd[k]
        shape_str = str(val.shape) if hasattr(val, "shape") else type(val)
        print(f" - {k:<45} | Shape: {shape_str}")

if __name__ == "__main__":
    main()