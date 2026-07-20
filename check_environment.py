import importlib
import platform
import subprocess
import shutil
import sys

print("=" * 80)
print("SignPAK-AI Environment Check")
print("=" * 80)

print(f"Python : {platform.python_version()}")
print(f"Platform: {platform.system()} {platform.release()}")
print()


# -------------------------------------------------------
# NVIDIA GPU
# -------------------------------------------------------

def check_nvidia():
    print("=" * 80)
    print("NVIDIA GPU")
    print("=" * 80)

    if shutil.which("nvidia-smi") is None:
        print("❌ nvidia-smi not found")
        return

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        for line in result.stdout.strip().splitlines():
            idx, name, mem, driver = [x.strip() for x in line.split(",")]

            print(f"GPU {idx}")
            print(f"Name   : {name}")
            print(f"Memory : {mem}")
            print(f"Driver : {driver}")
            print()

    except Exception as e:
        print(e)


check_nvidia()


# -------------------------------------------------------
# Package versions
# -------------------------------------------------------

packages = {
    "torch": "torch",
    "torchvision": "torchvision",
    "torchaudio": "torchaudio",
    "opencv-python": "cv2",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "Pillow": "PIL",
    "tqdm": "tqdm",
    "jupyter": "jupyter",
    "ipykernel": "ipykernel",
    "PyYAML": "yaml",
    "psutil": "psutil",
    "tensorboard": "tensorboard",
    "albumentations": "albumentations",
    "torchinfo": "torchinfo",
}

print("=" * 80)
print("Installed Packages")
print("=" * 80)

loaded = {}

for display_name, module_name in packages.items():
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "Unknown")
        loaded[module_name] = module
        print(f"✅ {display_name:<20} {version}")
    except Exception as e:
        print(f"❌ {display_name:<20} {e}")

print()


# -------------------------------------------------------
# PyTorch CUDA
# -------------------------------------------------------

print("=" * 80)
print("PyTorch CUDA Test")
print("=" * 80)

try:
    import torch
    import torchvision
    import torchaudio

    print("Torch       :", torch.__version__)
    print("TorchVision :", torchvision.__version__)
    print("TorchAudio  :", torchaudio.__version__)

    print()

    print("CUDA Available :", torch.cuda.is_available())
    print("CUDA Version   :", torch.version.cuda)
    print("cuDNN Enabled  :", torch.backends.cudnn.enabled)

    if torch.cuda.is_available():

        print("GPU Count      :", torch.cuda.device_count())

        for i in range(torch.cuda.device_count()):

            print()
            print(f"GPU {i}")

            print("Name :", torch.cuda.get_device_name(i))

            props = torch.cuda.get_device_properties(i)

            print(f"VRAM : {props.total_memory / (1024**3):.2f} GB")
            print(f"Capability : {props.major}.{props.minor}")

        print()

        print("Running tensor test...")

        x = torch.randn((5000, 5000), device="cuda")
        y = torch.randn((5000, 5000), device="cuda")

        z = torch.matmul(x, y)

        print("✅ GPU tensor multiplication successful")
        print("Tensor device:", z.device)

    else:
        print("⚠ CUDA NOT AVAILABLE")
        print("Torch will use CPU.")

except Exception as e:
    print("PyTorch Error:")
    print(e)


print()

# -------------------------------------------------------
# MediaPipe
# -------------------------------------------------------

print("=" * 80)
print("MediaPipe Test")
print("=" * 80)

try:
    import mediapipe as mp

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2
    )

    print("✅ MediaPipe Hands initialized")

    hands.close()

except Exception as e:
    print(e)

print()

# -------------------------------------------------------
# OpenCV
# -------------------------------------------------------

print("=" * 80)
print("OpenCV Test")
print("=" * 80)

try:
    import cv2
    import numpy as np

    img = np.zeros((100, 100, 3), dtype=np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("✅ OpenCV image conversion successful")

except Exception as e:
    print(e)

print()

print("=" * 80)
print("Environment check completed.")
print("=" * 80)