"""10-minute stability test runner monitoring runtime stability and resource usage."""

import argparse
import os
import resource
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detector import PersonDetector
from src.runtime import RuntimeDiagnostics


def main():
    parser = argparse.ArgumentParser(description="Surveillance Stability Runner")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds (default: 600)")
    args = parser.parse_args()

    dev = RuntimeDiagnostics.resolve_device("cuda:0", auto_fallback=True)
    print(f"Starting stability run for {args.duration} seconds on {dev}...")

    detector = PersonDetector(weights_path="yolov8n.pt", device=dev, confidence_threshold=0.5)
    dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)

    start_time = time.time()
    frames_processed = 0

    while time.time() - start_time < args.duration:
        detector.detect(dummy_frame)
        frames_processed += 1

        elapsed = time.time() - start_time
        if frames_processed % 200 == 0:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            print(f"[{elapsed:.1f}s / {args.duration}s] Processed {frames_processed} frames. RSS Memory: {usage.ru_maxrss} KB")

    total_time = time.time() - start_time
    avg_fps = frames_processed / total_time
    print(f"\nStability run completed successfully! Total frames: {frames_processed}, Avg FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
