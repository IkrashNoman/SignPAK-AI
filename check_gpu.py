import platform
import subprocess
import shutil


def check_nvidia_smi():
    """Check GPUs using nvidia-smi."""
    if shutil.which("nvidia-smi") is None:
        print("❌ nvidia-smi not found.")
        return

    print("=" * 60)
    print("NVIDIA GPUs (nvidia-smi)")
    print("=" * 60)

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

        gpus = result.stdout.strip().splitlines()

        if not gpus:
            print("No NVIDIA GPUs detected.")
            return

        print(f"Found {len(gpus)} GPU(s):\n")
        for gpu in gpus:
            idx, name, memory, driver = [x.strip() for x in gpu.split(",")]
            print(f"GPU {idx}")
            print(f"  Name    : {name}")
            print(f"  Memory  : {memory}")
            print(f"  Driver  : {driver}")
            print()

    except Exception as e:
        print("Error:", e)


def check_torch():
    """Check GPUs using PyTorch if installed."""
    try:
        import torch

        print("=" * 60)
        print("PyTorch")
        print("=" * 60)

        print("PyTorch version:", torch.__version__)
        print("CUDA available :", torch.cuda.is_available())
        print("GPU count      :", torch.cuda.device_count())

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}")
                print(f"  Name      : {props.name}")
                print(f"  VRAM      : {props.total_memory / 1024**3:.2f} GB")
                print(f"  Capability: {props.major}.{props.minor}")

    except ImportError:
        print("PyTorch is not installed.")


def check_tensorflow():
    """Check GPUs using TensorFlow if installed."""
    try:
        import tensorflow as tf

        print("=" * 60)
        print("TensorFlow")
        print("=" * 60)

        gpus = tf.config.list_physical_devices("GPU")
        print("Detected GPUs:", len(gpus))

        for gpu in gpus:
            print(gpu)

    except ImportError:
        print("TensorFlow is not installed.")


if __name__ == "__main__":
    print(f"Python : {platform.python_version()}")
    print(f"System : {platform.system()} {platform.release()}\n")

    check_nvidia_smi()
    print()

    check_torch()
    print()

    check_tensorflow()