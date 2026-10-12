# Phase 1 Subplan 01: Environment, Runtime Truthfulness, and Model Loading

## 1. Goal and Overview
Audit the active Python 3.14 WSL2 Ubuntu runtime, establish a deterministic project virtual environment, ensure official PyTorch and Ultralytics YOLO availability, and guarantee truthful hardware device reporting (`cuda:0` vs `cpu`). Strictly prevent silent fallbacks to OpenCV HOG or any non-YOLO detector.

## 2. Requirements & Acceptance Criteria Mapping
- **Functional Requirements Covered**:
  - `FR-05`: Model loading and truthful backend/weights/device reporting.
  - `FR-06`: Fail immediately and actionably if YOLO fails to load; no silent detector substitution.
  - `FR-24`: Check and truthfully report whether CUDA is available to the active PyTorch environment.
  - `FR-25`: Run YOLO inference on CUDA device 0 (`cuda:0`) when CUDA is available and configured.
  - `FR-26`: Run YOLO inference on CPU and clearly report CPU device when configured or CUDA is unavailable.
  - `FR-27`: Never report GPU inference unless the active model is verifiably using CUDA device 0.
  - `FR-28`: OpenCV HOG or any other detector shall not be used in place of YOLO.
- **Acceptance Criteria Covered**:
  - `Criterion 3`: Startup reports actual Ultralytics YOLO backend, actual weights path, and actual inference device.
  - `Criterion 14`: If YOLO fails to load, application reports that failure and does not claim Phase 1 detection is operating.
  - `Criterion 15`: CUDA availability check and verified YOLO inference on CUDA device 0 (RTX 5050).
  - `Criterion 16`: CPU YOLO inference baseline/fallback when CUDA is unavailable or CPU configured, with truthful reporting.

---

## 3. Tasks and Detailed Work Breakdown

### Task 1.1: WSL2 & Hardware Diagnostics Audit [GATED - Requires User Approval to Run]
- **Type**: Diagnostic Audit
- **Files Inspected / Created**:
  - `scripts/check_env.py` (diagnostic script)
- **Description**:
  Inspect environment facts safely:
  1. Verify active Python interpreter path and version (`Python 3.14`).
  2. Check if WSL2 can see NVIDIA RTX 5050 via `/usr/lib/wsl/lib` and `nvidia-smi`.
  3. Query PyTorch official Linux+pip wheel selector availability for Python 3.14.
  4. Note: Do NOT install Linux display drivers inside WSL.
- **Commands / Verification**:
  ```bash
  python3 --version
  which python3
  nvidia-smi
  ```
- **Observable Pass Condition**:
  Output clearly identifies Python version, virtualenv path, and whether WSL2 DirectX/NVIDIA GPU passthrough is active.

### Task 1.2: Virtual Environment & Dependency Specification [GATED - Requires User Approval to Run]
- **Type**: Environment Configuration
- **Files Created / Modified**:
  - `requirements.txt`
  - `config.yaml` (device & model section)
- **Description**:
  1. Specify dependencies without guessing wheels:
     - `ultralytics`
     - `opencv-python-headless` (or `opencv-python` for GUI display)
     - `pyyaml`
     - `numpy`
     - PyTorch package specification derived strictly from the official PyTorch installation selector for Linux + pip.
  2. Require explicit user review of the exact `pip` command before package installation.
  3. Do NOT install a CPU-only PyTorch build when GPU configuration is targeted.
- **Observable Pass Condition**:
  `requirements.txt` created with pinned/compatible versions validated against Python 3.14.

### Task 1.3: Runtime Truthfulness & Device Selection Adapter
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/runtime.py`
  - `src/config.py`
- **Description**:
  1. Implement `RuntimeDiagnostics` class:
     - Checks `torch.cuda.is_available()`.
     - Queries `torch.cuda.get_device_name(0)` when CUDA is available.
     - Resolves requested device against actual availability:
       - If `config.device == "cuda:0"` and CUDA is unavailable, raises explicit error or cleanly switches to `cpu` ONLY if `allow_cpu_fallback: true` is configured, with loud warning and truthful device reporting.
       - Never reports `cuda:0` unless verified on device 0.
  2. Implement zero-HOG assertion: actively checks that no OpenCV `HOGDescriptor` is referenced or initialized in any detection pipeline.
- **Observable Pass Condition**:
  Unit test verifies `RuntimeDiagnostics.get_device_info()` returns verified device, model path, and backend string.

### Task 1.4: YOLO Model Loader & Failure Reporter
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/detector.py` (loader logic)
  - `tests/test_model_load.py`
- **Description**:
  1. Implement `load_yolo_model(weights_path: str, device: str) -> YOLO`:
     - Verifies weights file exists locally or handles download cleanly.
     - Loads model via `ultralytics.YOLO(weights_path)`.
     - Validates loaded model against expected person class (class ID 0 in COCO).
     - Confirms model is moved to requested target device.
  2. If `ultralytics` cannot be imported or weights are corrupt/missing:
     - Catches exception, prints full actionable diagnostic error, and aborts immediately.
     - Strictly forbids catching the error and falling back to HOG.
- **Observable Pass Condition**:
  1. `tests/test_model_load.py` passes with valid weights.
  2. Loading a non-existent/corrupt weight file raises expected `ModelLoadError` and terminates process without silent fallback.

---

## 4. Subplan Verification & Test Gates

| Test ID | Test Name | Command / Script | Expected Result | PRD Mapping |
|---|---|---|---|---|
| T1-01 | Python Version & Path | `python3 -c "import sys; print(sys.version, sys.executable)"` | Python 3.14.x reported | Environment |
| T1-02 | Model Load Success | `pytest tests/test_model_load.py -k test_valid_load` | YOLO model loads, reports true device & path | FR-05, AC-3 |
| T1-03 | Model Load Failure & No HOG | `pytest tests/test_model_load.py -k test_invalid_load_fails_cleanly` | Raises error, exits, no HOG substitution | FR-06, FR-28, AC-14 |
| T1-04 | CUDA Device 0 Verification | `pytest tests/test_model_load.py -k test_cuda_verification` | If CUDA present, device is RTX 5050; if not, skips gracefully | FR-24, FR-25, FR-27, AC-15 |
| T1-05 | CPU Truthful Reporting | `pytest tests/test_model_load.py -k test_cpu_truthfulness` | Device reported is strictly `cpu` | FR-26, FR-27, AC-16 |
