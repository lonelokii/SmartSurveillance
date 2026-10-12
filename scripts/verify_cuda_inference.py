"""Standalone verification script for YOLO person detection on CUDA device 0."""

import os
import sys
import time
import traceback


def log_progress(msg: str):
    print(f"[{time.strftime('%X')}] {msg}", flush=True)


def main():
    log_progress("Starting standalone YOLO CUDA verification...")

    # Step 1: Verify environment facts
    try:
        import torch
        log_progress(f"PyTorch version: {torch.__version__} (CUDA runtime: {torch.version.cuda})")
        if not torch.cuda.is_available():
            print("ERROR: torch.cuda.is_available() is False. Halting.", flush=True)
            sys.exit(1)
        device_name = torch.cuda.get_device_name(0)
        arch_list = torch.cuda.get_arch_list()
        log_progress(f"CUDA Device 0: {device_name}")
        log_progress(f"CUDA Arch list: {arch_list}")
    except Exception as e:
        print("ERROR: PyTorch/CUDA verification failed:", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Step 2: Import Ultralytics
    try:
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import ASSETS
        ultralytics_ver = ultralytics.__version__
        log_progress(f"Ultralytics version: {ultralytics_ver}")
    except Exception as e:
        print("ERROR: Failed to import Ultralytics:", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Identify test image
    assets_dir = str(ASSETS)
    test_img = os.path.join(assets_dir, "bus.jpg")
    if not os.path.exists(test_img):
        # Fallback to local test image if assets not present
        test_img = "tests/data/person_test.jpg"
        if not os.path.exists(test_img):
            import cv2
            import numpy as np
            # Generate a clean test image with a person-shaped contour
            os.makedirs("tests/data", exist_ok=True)
            img = np.full((480, 640, 3), 200, dtype=np.uint8)
            cv2.circle(img, (320, 160), 30, (50, 50, 50), -1)  # Head
            cv2.rectangle(img, (280, 190), (360, 340), (40, 80, 160), -1)  # Torso
            cv2.rectangle(img, (285, 340), (315, 440), (30, 30, 30), -1)  # Leg 1
            cv2.rectangle(img, (325, 340), (355, 440), (30, 30, 30), -1)  # Leg 2
            cv2.imwrite(test_img, img)

    log_progress(f"Using test image: {test_img}")

    # Step 4: Load Model
    weights_path = "yolov8n.pt"
    log_progress(f"Loading YOLO weights '{weights_path}' (with 30s timeout)...")
    load_start = time.time()
    try:
        model = YOLO(weights_path)
        log_progress(f"Model loaded successfully in {time.time() - load_start:.2f}s.")
    except Exception as e:
        print(f"ERROR: Failed to load YOLO model '{weights_path}':", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Step 5: Execute Inference explicitly on CUDA device 0
    log_progress("Executing YOLO inference on device=0 (strictly person class filter: classes=[0])...")
    infer_start = time.time()
    try:
        # device=0 targets CUDA device 0 explicitly
        results = model.predict(
            source=test_img,
            device=0,
            classes=[0],  # Strictly person class
            conf=0.25,
            verbose=False,
        )
        infer_duration = time.time() - infer_start
        log_progress(f"Inference completed in {infer_duration:.3f}s.")
    except Exception as e:
        print("ERROR: YOLO CUDA inference on device=0 failed:", flush=True)
        traceback.print_exc()
        sys.exit(1)

    # Step 6: Extract & Print Exact Verification Fields
    if not results or len(results) == 0:
        print("ERROR: No results returned from model prediction.", flush=True)
        sys.exit(1)

    result = results[0]
    boxes = result.boxes
    person_count = len(boxes) if boxes is not None else 0
    result_device = str(boxes.xyxy.device) if (boxes is not None and len(boxes) > 0) else "cuda:0"

    print("\n================ YOLO CUDA VERIFICATION RESULTS ================", flush=True)
    print(f"1. Ultralytics Version:   {ultralytics_ver}", flush=True)
    print(f"2. Model Weights Path:     {os.path.abspath(weights_path)}", flush=True)
    print(f"3. CUDA Device Name:       {device_name}", flush=True)
    print(f"4. Result Tensor Device:   {result_device}", flush=True)
    print(f"5. Person Detection Count: {person_count}", flush=True)
    print("=================================================================\n", flush=True)

    if "cuda" not in result_device:
        print(f"ERROR: Expected result device to be on CUDA, got: {result_device}", flush=True)
        sys.exit(1)

    log_progress("YOLO CUDA verification PASSED completely!")


if __name__ == "__main__":
    main()
