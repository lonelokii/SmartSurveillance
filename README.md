# Smart Surveillance System

[![Python 3.11-3.14](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![OS: Windows 11 & Linux](https://img.shields.io/badge/OS-Windows%2011%20%7C%20Linux-0078D4.svg)](https://microsoft.com/windows)
[![Ultralytics YOLO](https://img.shields.io/badge/YOLO-Ultralytics%20v8.4+-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20%7C%20CPU-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Tests Passing](https://img.shields.io/badge/Tests-19%2F19%20Passed-brightgreen.svg)](#testing--verification)

A local, production-grade computer vision surveillance system designed for automated perimeter monitoring, occupancy tracking, directional line-crossing counting, and restricted-zone intrusion alerting.

Built with **Ultralytics YOLO**, **OpenCV**, and **PyTorch**, the system processes live phone IP-camera streams or local video files with real-time video analytics, zero false claims on hardware capabilities, and persistent SQLite event logging.

---

## Table of Contents

- [1. Overview & Problem Statement](#1-overview--problem-statement)
- [2. Key Features](#2-key-features)
- [3. System Architecture](#3-system-architecture)
- [4. Software, Libraries & Dependencies](#4-software-libraries--dependencies)
- [5. Hardware & Environment Requirements](#5-hardware--environment-requirements)
- [6. Installation & Setup Guide](#6-installation--setup-guide)
  - [6.1 Windows 11 Native Setup Guide (PowerShell / Command Prompt)](#61-windows-11-native-setup-guide-powershell--command-prompt)
  - [6.2 Linux & WSL2 Setup Guide](#62-linux--wsl2-setup-guide)
- [7. Configuration Guide (`config.yaml`)](#7-configuration-guide-configyaml)
- [8. Execution Steps & Working Examples](#8-execution-steps--working-examples)
- [9. Utility & Diagnostic Scripts](#9-utility--diagnostic-scripts)
- [10. Testing & Verification Suite](#10-testing--verification-suite)
  - [10.1 Running the Automated Test Suite](#101-running-the-automated-test-suite)
  - [10.2 Windows 11 Verification Workflow (Step-by-Step)](#102-windows-11-verification-workflow-step-by-step)
  - [10.3 Empty-Room Zero False Positive Verification](#103-empty-room-zero-false-positive-verification)
- [11. Project Roadmap](#11-project-roadmap)
- [12. Troubleshooting & FAQ](#12-troubleshooting--faq)
  - [Windows 11 Troubleshooting & Gotchas](#windows-11-troubleshooting--gotchas)
  - [General Issues & FAQ](#general-issues--faq)
- [13. License & Governance](#13-license--governance)

---

## 1. Overview & Problem Statement

Human operators tasked with monitoring physical entryways or restricted premises face fatigue, missed events, and inaccurate manual headcounts. 

**Smart Surveillance System (Phase 1)** provides an automated, locally executed computer-vision solution that:
- Ingests video from a standard smartphone IP webcam or pre-recorded video files.
- Accurately identifies human subjects using deep-learning object detection (COCO class `0` - person).
- Tracks unique temporary identities across sequential frames with missed-frame tolerance.
- Computes spatial analytics:
  - **Live Occupancy**: Real-time count of confirmed active persons in the monitored scene.
  - **Virtual Line Crossing**: Directional entry/exit counting with anti-duplicate suppression.
  - **Restricted Zone Monitoring**: Immediate outside-to-inside polygon entry detection.
- Captures instant timestamped JPEG snapshots and logs detailed incident metadata to an embedded SQLite database.
- Adheres to **Hardware Truthfulness**: Truthfully verifies PyTorch CUDA kernel availability, reports real hardware devices, and falls back to CPU execution without false claims or silent detector substitutions.

---

## 2. Key Features

- **Unified Video Source Handling**:
  - Supports live mobile phone cameras via HTTP/RTSP streams (e.g. Android *IP Webcam* application).
  - Supports local recorded video files (`.mp4`, `.avi`, `.mkv`).
  - Automatic frame resizing and graceful recovery on stream disconnection.

- **Person-Only YOLO Detection**:
  - Uses genuine Ultralytics YOLOv8 weights (`yolov8n.pt`).
  - Strict filtering: COCO class ID `0` (`person`) strictly enforced. Zero detection of non-person objects, bags, or animals.
  - **Zero Silent Fallback**: OpenCV HOG or non-YOLO detectors are strictly prohibited by runtime assertions.

- **Centroid & IoU Multi-Object Tracking**:
  - Temporary track ID assignment matching bounding boxes across frames.
  - Track confirmation threshold (`min_hits >= 2`) filters out transient single-frame noise.
  - Configurable expiration window (`max_missed_frames = 30`) to purge stale tracks when subjects leave the field of view.

- **Directional Virtual Line Crossing**:
  - Geometric vector cross-product computation determines crossing direction (`entry` vs. `exit`).
  - Trajectory history cache prevents duplicate count increments when a person stays on the same side.

- **Restricted Zone Intrusion Detection**:
  - Arbitrary polygon vertex geometry with OpenCV point-in-polygon tests.
  - State machine triggers an alert **strictly on outside-to-inside transitions**.
  - No repeated alert spam while the tracked subject remains within the zone.
  - Transition resets upon exit, allowing detection if the subject re-enters.

- **Snapshot & Database Storage**:
  - High-resolution JPEG snapshots saved to disk with ISO-8601 UTC timestamps and event UUIDs.
  - Structured SQLite storage tracking `event_id`, `event_type`, `timestamp`, `source_id`, `track_id`, `zone_id`, `snapshot_path`, `confidence`, and `device`.

- **Comprehensive Heads-Up Display (HUD)**:
  - Live video overlay displaying active device, model name, input source mode, and FPS.
  - Confirmed person bounding boxes with tracking IDs and confidence scores.
  - Visual rendering of the virtual entry/exit line and semi-transparent restricted-zone polygon.
  - Flashing 3-second alert banner on perimeter breach.

---

## 3. System Architecture

### Architectural Dataflow Diagram

```
+-------------------------------------------------------------+
|                     Video Input Source                      |
|      (Local MP4 File  OR  Live Mobile Phone IP Webcam)      |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|               VideoSource (`src/video_source.py`)           |
|        - OpenCV VideoCapture  - Uniform Resolution Resize   |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|             PersonDetector (`src/detector.py`)              |
|        - Ultralytics YOLOv8 (yolov8n.pt)                    |
|        - Target Device: CUDA 0 or Truthful CPU Fallback     |
|        - Strict Filter: Class 0 (Person), Conf >= Threshold  |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|              PersonTracker (`src/tracker.py`)               |
|        - IoU Matrix Matching & Centroid Tracking            |
|        - Temporary Track ID Assignment & Expiration         |
+------------------------------+------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                      Spatial Analytics                      |
|                                                             |
|   LineCrossingEngine (`src/line_crossing.py`)               |
|   - Vector Segment Intersection                             |
|   - Directional Counter (Entry / Exit)                      |
|                                                             |
|   ZoneMonitor (`src/zone_monitor.py`)                       |
|   - Point-in-Polygon Testing                                |
|   - Outside-to-Inside Transition Detection                  |
+------------------------------+------------------------------+
                               |
         +---------------------+---------------------+
         |                                           |
         v                                           v
+---------------------------------+  +---------------------------------+
|   Storage & Alert Persistence   |  |   HUD Annotation & Display      |
|                                 |  |                                 |
| - SnapshotWriter (JPEG)         |  | - FrameAnnotator                |
| - DatabaseLogger (SQLite)       |  | - Box, ID & Confidence Overlay  |
| - UUID & UTC Timestamps         |  | - Lines, Polygons & Diagnostics |
|   (`src/storage.py`)            |  |   (`src/visualizer.py`)         |
+---------------------------------+  +---------------------------------+
```

### Module Responsibilities

| File Path | Module Description |
|---|---|
| `main.py` | CLI entry point supporting command-line overrides (`--config`, `--source`, `--device`, `--headless`, etc.). |
| `src/app.py` | Central `SurveillanceApp` orchestrator binding capture, inference, tracking, analytics, rendering, and lifecycle signals. |
| `src/config.py` | Typed dataclass schema (`AppConfig`) with YAML parsing and strict parameter validation. |
| `src/runtime.py` | Runtime hardware truthfulness diagnostics, empirical CUDA kernel execution checks, and zero-HOG assertions. |
| `src/detector.py` | Ultralytics YOLO loader and person-only inference engine. |
| `src/tracker.py` | Centroid & IoU association multi-person tracker with confirmation state and missed-frame expiration. |
| `src/video_source.py` | Context-managed video capture supporting local video files and live network IP webcam feeds. |
| `src/line_crossing.py` | Geometric vector line-crossing engine with stateful directional entry/exit counters. |
| `src/zone_monitor.py` | Convex/concave polygon point-in-polygon detector managing outside-to-inside state transitions. |
| `src/storage.py` | JPEG snapshot generator (`SnapshotWriter`) and SQLite event logging persistence (`DatabaseLogger`). |
| `src/visualizer.py` | Video frame annotator rendering bounding boxes, track IDs, line, zone polygon, diagnostics header, and alert banners. |

---

## 4. Software, Libraries & Dependencies

The project is built on modern Python 3.14 with standard scientific and computer vision packages:

| Software / Library | Version Constraint | Role & Purpose |
|---|---|---|
| **Python** | `>= 3.11` (Tested on 3.14) | Base programming language environment. |
| **Ultralytics** | `>= 8.4.0` | High-performance YOLOv8 person detection backend. |
| **PyTorch (`torch`)** | `>= 2.4.0` (cu126 / cu13x / CPU) | Deep learning tensor engine and CUDA execution runtime. |
| **Torchvision** | `>= 0.19.0` | Auxiliary vision utilities compatible with PyTorch. |
| **OpenCV (`opencv-python`)** | `>= 4.10.0` | Video capture, frame decoding, polygon geometry, and HUD rendering. |
| **NumPy** | `>= 2.0.0` | Array manipulation, IoU calculations, and vector operations. |
| **PyYAML** | `>= 6.0` | Configuration schema loading. |
| **Pytest** | `>= 8.0.0` | Automated regression and unit test suite. |
| **SQLite3** | Standard Library | Embedded ACID event log database. |

---

## 5. Hardware & Environment Requirements

### Minimum Requirements
- **Operating System**:
  - **Windows**: Windows 11 / 10 (64-bit) native execution via PowerShell 5.1/7+ or Command Prompt (`cmd.exe`), or WSL2 (Ubuntu 22.04+).
  - **Linux**: Ubuntu 22.04 LTS / 24.04 LTS, Debian 12+, Fedora 39+, Arch Linux, or Docker container environments.
  - **macOS**: macOS Monterey (12+), Ventura (13+), Sonoma (14+), Sequoia (15+) on Apple Silicon (M1/M2/M3/M4) or Intel.
- **CPU**: x86_64 or ARM64 processor (Intel Core i5 / AMD Ryzen 5 / Apple Silicon or higher).
- **RAM**: 4 GB minimum (8 GB recommended).
- **Storage**: 1 GB free disk space (excluding video logs and snapshots).
- **Camera (Optional for live mode)**: Any smartphone connected to the same local Wi-Fi running an RTSP/HTTP IP webcam app (e.g. *IP Webcam* on Android or *Live-Reporter* on iOS), or any standard USB / built-in webcam.

### Windows Prerequisites
1. **Python**: Python 3.11, 3.12, or 3.14 (x64) from [python.org](https://www.python.org/downloads/windows/). Ensure **"Add python.exe to PATH"** is checked during installation.
2. **Microsoft Visual C++ Redistributable**: Required for PyTorch and OpenCV native C++ extensions. Download and install [Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) or run `winget install Microsoft.VCRedist.2015+.x64`.
3. **Git for Windows**: Download from [git-scm.com](https://git-scm.com/download/win) or run `winget install Git.Git`.
4. **PowerShell Script Execution**: Allow execution of local virtual environment scripts (`RemoteSigned`).

### Hardware Acceleration Options
- **NVIDIA CUDA**: Supported on Windows and Linux with NVIDIA GeForce RTX series (RTX 30-series, 40-series, 50-series) or NVIDIA Quadro / RTX workstation GPUs with CUDA 12.x / 13.x drivers.
- **Apple Silicon MPS**: Supported on macOS with M1/M2/M3/M4 chips via PyTorch Metal Performance Shaders (`--device mps`).
- **Verified CPU Fallback**: Universally supported across all x86_64 and ARM64 platforms (achieving **>20 FPS** on CPU) without false claims or crashes.
- *Note on RTX 50-series (Blackwell / Compute Capability 12.0 / `sm_120`)*: If running on brand-new architectures where PyTorch precompiled CUDA wheels lack native `sm_120` kernels, the system's `RuntimeDiagnostics` automatically falls back to verified CPU execution with a clear diagnostic log.

---

## 6. Installation & Setup Guide

### 6.1 Windows 11 Native Setup Guide (PowerShell / Command Prompt)

Follow these steps to set up and run the system natively on Windows 11 without WSL.

#### Step 0: Install Prerequisites (One-Time Setup)
Open **PowerShell as Administrator** (or standard Windows Terminal) and install the foundational tools via Windows Package Manager (`winget`):

```powershell
# 1. Install Python 3.12 (if not already installed)
winget install Python.Python.3.12

# 2. Install Git
winget install Git.Git

# 3. Install Microsoft Visual C++ 2015-2022 Redistributable (crucial for OpenCV & PyTorch)
winget install Microsoft.VCRedist.2015+.x64
```
> [!NOTE]
> If you installed Python manually from python.org, verify that `python --version` returns Python 3.11+ and that `python.exe` is added to your user `PATH`.

#### Step 1: Enable PowerShell Script Execution
By default, Windows 11 blocks running scripts in PowerShell. Run this command once in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Step 2: Clone the Repository
Open PowerShell or Command Prompt, navigate to your desired workspace directory, and clone the repository:

```powershell
git clone https://github.com/lonelokii/SmartSurveillance.git
cd SmartSurveillance
```

#### Step 3: Create and Activate Virtual Environment
Create an isolated Python virtual environment:

- **In PowerShell**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
  *(Your prompt will change to show `(.venv)` at the beginning).*

- **In Command Prompt (cmd.exe)**:
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```

> [!TIP]
> In Windows, use `python` (or `py -m venv .venv`). Avoid using `python3` on Windows, as it may redirect to the Microsoft Store.

#### Step 4: Upgrade Package Installer
Inside the activated virtual environment:

```powershell
python -m pip install --upgrade pip
```

#### Step 5: Install Dependencies

Select your target hardware configuration:

##### Option A: NVIDIA GPU Acceleration (CUDA 12.6 Official Wheels)
If your Windows 11 PC has an NVIDIA GPU with updated drivers:

```powershell
# Install PyTorch with CUDA 12.6 acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install Phase 1 surveillance dependencies
pip install ultralytics opencv-python pyyaml numpy pytest
```
*Alternatively, you can install everything in one command using the Windows requirements file:*
```powershell
pip install -r requirements-windows.txt
```

##### Option B: CPU-Only Execution (No NVIDIA GPU)
If running on Intel / AMD integrated graphics or non-NVIDIA hardware:

```powershell
# Install PyTorch CPU-optimized wheels
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install Phase 1 surveillance dependencies
pip install ultralytics opencv-python pyyaml numpy pytest
```

#### Step 6: Verify Environment Diagnostics
Run the diagnostic script to verify your Windows 11 Python environment, virtualenv isolation, PyTorch installation, and GPU detection:

```powershell
python scripts\check_env.py
```

Expected diagnostic output highlights:
```text
=== ENVIRONMENT DIAGNOSTIC REPORT ===
Inside Virtual Environment: True (prefix: ...\SmartSurveillance\.venv)
[torch] Installed: version 2.x.x+cu126 (or +cpu)
  torch.cuda.is_available(): True (or False if CPU mode)
  torch.cuda.get_device_name(0): NVIDIA GeForce RTX ...
[ultralytics] Installed: version 8.4.x
[cv2] Installed: version 4.10.x
=== END OF REPORT ===
```

#### Step 7: Verify YOLO Weights
Pretrained model weights (`yolov8n.pt`) are included in the workspace and will also be automatically verified and downloaded by Ultralytics on first execution.

---

### 6.2 Linux & WSL2 Setup Guide

For Ubuntu 22.04 LTS / 24.04 LTS or Windows Subsystem for Linux (WSL2):

#### Step 1: Clone Repository
```bash
git clone https://github.com/lonelokii/SmartSurveillance.git
cd SmartSurveillance
```

#### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Upgrade Package Installer
```bash
pip install --upgrade pip
```

#### Step 4: Install Dependencies

```bash
# GPU / Standard Installation:
pip install -r requirements.txt

# Or manual CUDA 12.6 wheels:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics opencv-python pyyaml numpy pytest

# Or CPU-Only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python pyyaml numpy pytest
```

#### Step 5: Verify Environment
```bash
python3 scripts/check_env.py
```

---

## 7. Configuration Guide (`config.yaml`)

All parameters are centrally managed in `config.yaml`:

```yaml
# Smart Surveillance System — Phase 1 Configuration

system:
  app_name: "Smart Surveillance System"
  phase: "Phase 1"
  log_level: "INFO"

source:
  # Mode: 'video' (local file) or 'live' (phone IP camera)
  mode: "video"
  video_path: "tests/data/sample_test.mp4"
  live_url: "http://192.168.1.5:8080/video"
  processing_width: 640
  processing_height: 480

model:
  weights_path: "yolov8n.pt"
  target_device: "cuda:0"       # 'cuda:0' or 'cpu'
  auto_cuda_fallback: true      # Fallback truthfully to cpu if CUDA unavailable
  confidence_threshold: 0.50   # Minimum person detection confidence
  image_size: 640
  classes: [0]                  # Strictly person class (COCO 0)

tracker:
  max_missed_frames: 30         # Frames before a lost track expires
  min_hits: 2                   # Detections before a track is confirmed
  iou_threshold: 0.30           # IoU association threshold

line_crossing:
  enabled: true
  start_point: [100, 240]       # (x, y) coordinates on processing canvas
  end_point: [540, 240]
  # Direction: 'top_to_bottom', 'bottom_to_top', 'left_to_right', 'right_to_left'
  entry_direction: "top_to_bottom"

restricted_zone:
  enabled: true
  polygon_points:
    - [350, 150]
    - [550, 150]
    - [550, 400]
    - [350, 400]
  zone_id: "ZONE_A"

storage:
  snapshot_dir: "snapshots"     # Directory for breach JPEG images
  database_path: "surveillance.db" # Local SQLite event store
```

---

## 8. Execution Steps & Working Examples

### Example 1: Run with Default Configuration
- **Windows (PowerShell / CMD)**:
  ```powershell
  python main.py
  ```
- **Linux / WSL2**:
  ```bash
  python3 main.py
  ```

### Example 2: Run with a Local Video File
Test surveillance tracking and spatial analytics on a local recorded video clip:
- **Windows (PowerShell / CMD)**:
  ```powershell
  python main.py --source video --video-path tests\data\one_person_walk.mp4
  ```
- **Linux / WSL2**:
  ```bash
  python3 main.py --source video --video-path tests/data/one_person_walk.mp4
  ```

### Example 3: Run with a Live Phone Camera (IP Webcam)
1. Install **IP Webcam** (Android) or any RTSP/HTTP stream app on your smartphone.
2. Start the server inside the app and note the streaming URL (e.g. `http://192.168.1.45:8080/video`).
3. Ensure your PC and phone are connected to the same local Wi-Fi network.
4. Execute:
- **Windows (PowerShell / CMD)**:
  ```powershell
  python main.py --source live --live-url http://192.168.1.45:8080/video
  ```
- **Linux / WSL2**:
  ```bash
  python3 main.py --source live --live-url http://192.168.1.45:8080/video
  ```

### Example 4: Force CPU Inference
Truthfully enforce CPU inference (bypassing GPU):
```powershell
# Windows
python main.py --device cpu

# Linux / WSL2
python3 main.py --device cpu
```

### Example 5: Run Headless (Automated Server / Benchmarks)
Run without opening an OpenCV GUI window:
```powershell
# Windows
python main.py --headless --max-frames 300

# Linux / WSL2
python3 main.py --headless --max-frames 300
```

### Example 6: Run with an Attached USB Webcam or Laptop Camera
You can stream directly from a local USB camera or built-in webcam by passing its numeric camera index:
```powershell
# Windows (PowerShell / CMD)
python main.py --source video --video-path 0

# Linux / macOS
python3 main.py --source video --video-path 0
```

### Operator Controls
- **Quit / Stop Application**: Press `q` while focused on the OpenCV video display window, or press `Ctrl+C` in the terminal. All capture devices and database connections are closed cleanly.

---

## 9. Utility & Diagnostic Scripts

The `scripts/` folder provides standalone validation tools executable across both Windows 11 and Linux:

| Script Name | Windows 11 Command | Linux / WSL2 Command | Purpose |
|---|---|---|---|
| **Environment Check** | `python scripts\check_env.py` | `python3 scripts/check_env.py` | Generates a full report of Python version, CUDA drivers, GPU detection, and package paths. |
| **FPS Benchmark** | `python scripts\benchmark_fps.py --frames 100` | `python3 scripts/benchmark_fps.py --frames 100` | Measures actual processing throughput against the PRD requirement (`>= 5.0 FPS`). |
| **CUDA Verification** | `python scripts\verify_cuda_inference.py` | `python3 scripts/verify_cuda_inference.py` | Performs standalone YOLO inference directly on CUDA device 0 and verifies tensor output device. |
| **Stability Runner** | `python scripts\stability_run.py --duration 600` | `python3 scripts/stability_run.py --duration 600` | Runs continuous 10-minute synthetic processing and monitors runtime stability. |
| **Synthetic Media Creator** | `python scripts\create_test_media.py` | `python3 scripts/create_test_media.py` | Generates deterministic synthetic test videos (`empty_room_test.mp4` and `one_person_walk.mp4`) for testing. |

---

## 10. Testing & Verification Suite

The repository includes a comprehensive automated test suite covering all functional requirements (FR-01 through FR-28) and acceptance criteria (AC-1 through AC-18).

### 10.1 Running the Automated Test Suite

Run the full pytest suite from PowerShell, Command Prompt, or bash inside your activated virtual environment:

```powershell
# Windows 11 (PowerShell / CMD) & Linux
pytest -v
```

#### Test Coverage Summary

```text
tests/test_e2e_pipeline.py ........ [End-to-end full pipeline & clean shutdown]
tests/test_line_crossing.py ....... [Directional line crossing & anti-duplicate logic]
tests/test_media_pipeline.py ...... [Test clip evaluation & zero false detection check]
tests/test_model_load.py .......... [YOLO loading, CUDA truthfulness, zero-HOG enforcement]
tests/test_storage.py ............. [SQLite event persistence & snapshot file creation]
tests/test_tracker.py ............. [IoU matching, track confirmation & track expiration]
tests/test_video_source.py ........ [Local video, live URL stream, & error handling]
tests/test_zone_monitor.py ........ [Restricted-zone outside-to-inside transition alert]

======================= 19 passed in 16.5s =======================
```

---

### 10.2 Windows 11 Verification Workflow (Step-by-Step)

To guarantee that your Windows 11 environment is 100% operational, execute this step-by-step verification checklist in PowerShell:

#### Verification 1: Inspect Environment & Dependencies
Verify Python 3.11+, virtualenv activation, PyTorch CUDA/CPU detection, and library imports:
```powershell
python scripts\check_env.py
```
- **Expected Result**: Confirms `Inside Virtual Environment: True`, lists package versions (`torch`, `ultralytics`, `cv2`, `numpy`, `yaml`), and reports CUDA device name or truthful CPU status.

#### Verification 2: Run Full Regression Test Suite
Ensure all 19 unit and integration tests pass cleanly on Windows:
```powershell
pytest -v
```
- **Expected Result**: `19 passed` with 0 failures and 0 errors.

#### Verification 3: Verify Hardware Truthfulness & YOLO CUDA Inference
Validate that the YOLOv8 model compiles and executes inferences on the designated device without errors or silent detector fallbacks:
```powershell
python scripts\verify_cuda_inference.py
```
- **Expected Result**: Outputs tensor predictions on the truthful active device (`cuda:0` or verified `cpu`).

#### Verification 4: Benchmark Processing Throughput (FPS)
Verify the PRD performance requirement (`>= 5.0 FPS`):
```powershell
python scripts\benchmark_fps.py --frames 100
```
- **Expected Result**:
  ```text
  === BENCHMARK SUMMARY ===
  CPU Performance: ~24.0 FPS (Target: >= 5.0 FPS) - PASS
  ```

#### Verification 5: Headless Pipeline Smoke Test
Test end-to-end video acquisition, YOLO person detection, centroid tracking, line-crossing counting, and restricted-zone monitoring without GUI overhead:
```powershell
python main.py --source video --video-path tests\data\one_person_walk.mp4 --headless --max-frames 60
```
- **Expected Result**: Logs frame progress, records crossing and zone breach events, and terminates cleanly with exit code 0.

#### Verification 6: Interactive Heads-Up Display (HUD) Verification
Launch the live visual GUI window to inspect real-time video rendering:
```powershell
python main.py --source video --video-path tests\data\one_person_walk.mp4
```
- **Expected Result**: Opens an OpenCV window displaying:
  - Confirmed person bounding boxes with track IDs and confidence scores.
  - Directional virtual entry/exit line.
  - Semi-transparent restricted-zone polygon.
  - HUD header showing FPS, active device, and occupancy count.
  - Red flashing breach banner when the person steps into the zone.
  - Press `q` to quit cleanly.

#### Verification 7: Verify Event Snapshots & SQLite Database
Confirm that alert snapshots and structured metadata are properly persisted to disk:

- **Check Snapshots Directory**:
  - In PowerShell:
    ```powershell
    Get-ChildItem -Path snapshots\ -Filter *.jpg
    ```
  - In Command Prompt:
    ```cmd
    dir snapshots\*.jpg
    ```
- **Query Recorded SQLite Alerts via Python (No external SQLite CLI needed)**:
  ```powershell
  python -c "import sqlite3; con = sqlite3.connect('surveillance.db'); print(con.execute('SELECT event_id, timestamp, track_id, zone_id, snapshot_path FROM alerts').fetchall())"
  ```
  - **Expected Result**: Prints logged records containing UUIDs, UTC timestamps, `ZONE_A`, and paths like `snapshots\alert_...jpg`.

---

### 10.3 Empty-Room Zero False Positive Verification
Automated diagnostics evaluated on `empty_room_test.mp4` (100 frames) confirmed 0 false positive detections:
```json
{
  "clip": "tests/data/empty_room_test.mp4",
  "total_frames_evaluated": 100,
  "false_detections_count": 0,
  "false_detections": []
}
```

---

## 11. Project Roadmap

```
+------------------------------------------------------------------------+
|  [COMPLETED] Phase 1: Local Single-Camera Core Surveillance Baseline   |
|  - Ultralytics YOLOv8 Person-Only Detection (COCO Class 0)             |
|  - Centroid & IoU Temporary Multi-Person Tracking                      |
|  - Live Occupancy Count HUD Overlay                                    |
|  - Directional Virtual Line Crossing Engine                            |
|  - Restricted-Zone Polygon Intrusion Detection                         |
|  - Local JPEG Snapshots & SQLite Event Database Persistence            |
|  - Phone IP-Camera & Local Video Source Support                        |
|  - Hardware Truthfulness & Verified >= 5 FPS Processing                |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
|  [PLANNED] Phase 2: Web Interface & Multi-Source Management            |
|  - FastAPI / WebSocket streaming backend for remote viewing            |
|  - Multi-stream video ingestion (concurrent camera feeds)              |
|  - Interactive Web GUI for drawing line and zone coordinates           |
|  - Searchable event log dashboard with snapshot gallery and filters    |
|  - Webhook & Telegram/Email alert notifications                        |
+------------------------------------------------------------------------+
                                   |
                                   v
+------------------------------------------------------------------------+
|  [PLANNED] Phase 3: Edge Optimization & Enterprise Analytics           |
|  - TensorRT & ONNX Runtime hardware acceleration                       |
|  - Edge device deployment (NVIDIA Jetson Orin / Raspberry Pi 5)        |
|  - Advanced dwell-time and heat-mapping analytics                      |
|  - Encrypted snapshot storage and audit export (CSV / PDF reports)     |
+------------------------------------------------------------------------+
```

---

## 12. Troubleshooting & FAQ

### Windows 11 Troubleshooting & Gotchas

#### 1. PowerShell Error: `File ... Activate.ps1 cannot be loaded because running scripts is disabled`
- **Cause**: Windows 11 has a default security policy that blocks running PowerShell scripts (`Restricted`).
- **Solution**: Open PowerShell and run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
  Then re-run `.\.venv\Scripts\Activate.ps1`.

#### 2. PyTorch or OpenCV Error: `ImportError: DLL load failed while importing _custom_ops` or `cv2: DLL load failed`
- **Cause**: The system is missing the Microsoft Visual C++ runtime libraries required by native compiled binaries.
- **Solution**: Install the Microsoft Visual C++ Redistributable via `winget`:
  ```powershell
  winget install Microsoft.VCRedist.2015+.x64
  ```
  Or download and run [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) from Microsoft.

#### 3. Command `python3` Opens the Microsoft Windows Store
- **Cause**: Windows 11 includes a default app alias for `python.exe` and `python3.exe` that opens the Microsoft Store when Python is not mapped in PATH.
- **Solution**:
  - Use `python` instead of `python3`.
  - Alternatively, go to **Windows Settings > Apps > Advanced app settings > App execution aliases**, and toggle OFF "App Installer (python.exe)" and "App Installer (python3.exe)".

#### 4. Windows Defender Firewall Prompts for Live Phone Camera Streams
- **Cause**: When connecting to an RTSP or HTTP IP webcam stream on the local network, Windows Firewall may request permission for Python network sockets.
- **Solution**: Select **"Private networks"** and click **"Allow access"**. Ensure both your phone and PC are connected to the same Wi-Fi subnet (e.g. `192.168.1.x`).

#### 5. Command `sqlite3` Is Not Recognized in PowerShell / CMD
- **Cause**: Unlike Linux/macOS, Windows does not include a command-line `sqlite3.exe` utility by default.
- **Solution**: You can inspect `surveillance.db` directly using Python without installing third-party tools:
  ```powershell
  python -c "import sqlite3; con = sqlite3.connect('surveillance.db'); print(con.execute('SELECT event_id, timestamp, track_id, zone_id, snapshot_path FROM alerts').fetchall())"
  ```
  Or install the official SQLite CLI via winget:
  ```powershell
  winget install SQLite.SQLite
  ```

---

### General Issues & FAQ

#### 1. The Phone IP Camera Stream Fails to Open
- Ensure your mobile phone and PC are connected to the exact same local Wi-Fi network (not mobile data).
- Test opening the stream URL in a browser on your PC (e.g. `http://192.168.1.5:8080/video`).
- Ensure no firewall on Linux/Windows is blocking incoming HTTP port 8080.

#### 2. PyTorch Warning Regarding CUDA Compute Capability (sm_120 / RTX 5050)
- NVIDIA GeForce RTX 50-series GPUs (Blackwell) use compute capability 12.0 (`sm_120`).
- If your installed PyTorch wheel is built for CUDA 12.6, prebuilt `sm_120` kernels may not be included.
- **The system automatically handles this**: `RuntimeDiagnostics` detects the missing kernels, logs a clear warning, and truthfully falls back to CPU mode, where inference runs smoothly at **~24 FPS**.
- To enable native GPU acceleration on RTX 50-series when available, reinstall PyTorch using CUDA 13.x wheels:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  ```

#### 3. Can I Use This for Non-Human Objects (Cars, Backpacks, etc.)?
No. Phase 1 is strictly scoped to the `person` class (COCO 0) to ensure high reliability, zero clutter, and compliance with the core surveillance baseline.

---

## 13. License & Governance

This project is licensed under the **MIT License**.

All implementations adhere strictly to the authoritative specifications documented in [`PRD_PHASE1.md`](file:///home/boom/projects/SmartSurveillance/PRD_PHASE1.md) and [`plans/README.md`](file:///home/boom/projects/SmartSurveillance/plans/README.md).
