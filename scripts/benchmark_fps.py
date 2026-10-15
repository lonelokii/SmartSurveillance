"""Benchmark processing FPS for Smart Surveillance System."""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import PersonDetector
from src.runtime import RuntimeDiagnostics


def benchmark(device: str, num_frames: int = 150):
    print(f"\n--- Starting FPS Benchmark on device: {device} ({num_frames} frames) ---")
    detector = PersonDetector(weights_path="yolov8n.pt", device=device, confidence_threshold=0.5)

    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(5):
        detector.detect(dummy_frame)

    start = time.perf_counter()
    for _ in range(num_frames):
        detector.detect(dummy_frame)
    elapsed = time.perf_counter() - start

    fps = num_frames / elapsed
    print(f"Results for {device}: {num_frames} frames in {elapsed:.2f}s => {fps:.2f} FPS")
    return fps


def main():
    parser = argparse.ArgumentParser(description="FPS Benchmark")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()

    cuda_avail, _, gpu_name, functional, _ = RuntimeDiagnostics.get_cuda_status()

    # Benchmark CPU
    cpu_fps = benchmark("cpu", num_frames=args.frames)

    # Benchmark GPU if available and functional
    gpu_fps = None
    if cuda_avail and functional:
        gpu_fps = benchmark("cuda:0", num_frames=args.frames)

    # Benchmark Apple Silicon MPS if available
    mps_fps = None
    try:
        import torch
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_fps = benchmark("mps", num_frames=args.frames)
    except Exception:
        pass

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"CPU Performance: {cpu_fps:.2f} FPS (Target: >= 5.0 FPS) - {'PASS' if cpu_fps >= 5.0 else 'FAIL'}")
    if gpu_fps is not None:
        print(f"GPU ({gpu_name}) Performance: {gpu_fps:.2f} FPS (Target: >= 5.0 FPS) - {'PASS' if gpu_fps >= 5.0 else 'FAIL'}")
    if mps_fps is not None:
        print(f"Apple Silicon (MPS) Performance: {mps_fps:.2f} FPS (Target: >= 5.0 FPS) - {'PASS' if mps_fps >= 5.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
