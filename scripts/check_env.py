"""Diagnostic script to inspect WSL2 environment, Python 3.14, and GPU status."""

import os
import shutil
import subprocess
import sys


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
        return res.returncode, res.stdout.strip()
    except Exception as e:
        return -1, str(e)


def main():
    print("=== ENVIRONMENT DIAGNOSTIC REPORT ===")
    print(f"Python Executable: {sys.executable}")
    print(f"Python Version: {sys.version}")
    in_venv = sys.prefix != sys.base_prefix
    print(f"Inside Virtual Environment: {in_venv} (prefix: {sys.prefix})")

    print("\n--- Package Manager & System Info ---")
    ret, pip_ver = run_cmd([sys.executable, "-m", "pip", "--version"])
    print(f"pip version: {pip_ver if ret == 0 else f'pip check failed ({pip_ver})'}")

    # Check for installed packages
    print("\n--- Key Package Inspections ---")
    for pkg in ["torch", "torchvision", "torchaudio", "ultralytics", "cv2", "yaml", "numpy"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            file_loc = getattr(mod, "__file__", "unknown")
            print(f"[{pkg}] Installed: version {ver} at {file_loc}")
            if pkg == "torch":
                cuda_avail = getattr(mod.cuda, "is_available")()
                device_count = getattr(mod.cuda, "device_count")() if cuda_avail else 0
                device_name = getattr(mod.cuda, "get_device_name")(0) if cuda_avail and device_count > 0 else "None"
                print(f"  torch.cuda.is_available(): {cuda_avail}")
                print(f"  torch.cuda.device_count(): {device_count}")
                print(f"  torch.cuda.get_device_name(0): {device_name}")
        except ImportError:
            print(f"[{pkg}] Not installed")
        except Exception as e:
            print(f"[{pkg}] Error during check: {e}")

    # Check GPU via nvidia-smi
    print("\n--- WSL2 GPU Passthrough (nvidia-smi) ---")
    smi_path = shutil.which("nvidia-smi")
    print(f"nvidia-smi path: {smi_path}")
    if smi_path:
        ret, smi_out = run_cmd(["nvidia-smi"])
        if ret == 0:
            print("nvidia-smi output:")
            print(smi_out)
        else:
            print(f"nvidia-smi returned non-zero code {ret}: {smi_out}")
    else:
        # Check standard WSL lib location
        wsl_lib = "/usr/lib/wsl/lib"
        print(f"Checking {wsl_lib}: exists = {os.path.exists(wsl_lib)}")
        if os.path.exists(wsl_lib):
            try:
                print(f"Contents of {wsl_lib}: {os.listdir(wsl_lib)[:10]}")
            except Exception as e:
                print(f"Could not list {wsl_lib}: {e}")

    print("\n=== END OF REPORT ===")


if __name__ == "__main__":
    main()
