"""
download_pretrained.py — Download Pretrained WLASL Weights via HuggingFace Hub
==============================================================================
"""

import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_PATH = MODEL_DIR / "tgcn_wlasl_pretrained.pth"

def download():
    print("📦 Installing/verifying huggingface_hub helper ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    
    from huggingface_hub import hf_hub_download
    import torch

    print("📥 Downloading official pre-trained Pose-TGCN weights ...")
    try:
        # Download from HuggingFace mirror repository
        downloaded_file = hf_hub_download(
            repo_id="sharonn18/tgcn-wlasl",
            filename="checkpoints/asl2000/pytorch_model.bin"
        )
        
        # Load and verify PyTorch state dict
        state_dict = torch.load(downloaded_file, map_location="cpu")
        torch.save(state_dict, WEIGHTS_PATH)
        
        print(f"✅ Pretrained weights successfully saved to: {WEIGHTS_PATH}")
        print(f"   File Size: {WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ Download error: {e}")

if __name__ == "__main__":
    download()