"""
inspect_weights_INCLUDE.py — INCLUDE Pretrained Weight Inspector
==================================================================
Inspects layer names, tensor shapes, and total parameters stored inside:
models/checkpoints/include_pretrained.pth
"""

import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH   = PROJECT_ROOT / "models" / "checkpoints" / "include_pretrained.pth"

def main():
    if not MODEL_PATH.exists():
        print(f"❌ File not found at: {MODEL_PATH}")
        print("   Please make sure the file exists in that directory.")
        return

    print(f"📦 Inspecting INCLUDE Checkpoint: {MODEL_PATH}")
    
    # Load PyTorch state dict safely
    try:
        sd = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except Exception:
        sd = torch.load(MODEL_PATH, map_location="cpu")

    # Handle wrapped dictionaries
    if isinstance(sd, dict):
        if "state_dict" in sd:
            sd = sd["state_dict"]
        elif "model" in sd:
            sd = sd["model"]

    if not isinstance(sd, dict):
        print(f"⚠️ State dict object is of type {type(sd)}, trying state_dict()...")
        sd = getattr(sd, "state_dict", lambda: {})()

    keys = list(sd.keys())
    print(f"\n✅ Total Tensors Found in INCLUDE Checkpoint: {len(keys)}")
    print("=" * 70)
    print(f"{'Layer / Parameter Name':<45} | {'Tensor Shape'}")
    print("=" * 70)

    total_params = 0
    for k in keys:
        val = sd[k]
        if hasattr(val, "shape"):
            shape_str = str(tuple(val.shape))
            total_params += val.numel()
        else:
            shape_str = str(type(val))
        print(f" - {k:<43} | {shape_str}")

    print("=" * 70)
    print(f"📊 Total Parameters in Backbone Checkpoint: {total_params:,}")
    print("=" * 70)

if __name__ == "__main__":
    main()