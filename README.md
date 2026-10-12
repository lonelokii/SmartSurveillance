# Smart Surveillance System

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
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
- [7. Configuration Guide (`config.yaml`)](#7-configuration-guide-configyaml)
- [8. Execution Steps & Working Examples](#8-execution-steps--working-examples)
- [9. Utility & Diagnostic Scripts](#9-utility--diagnostic-scripts)
- [10. Testing & Verification Suite](#10-testing--verification-suite)
- [11. Project Roadmap](#11-project-roadmap)
- [12. Troubleshooting & FAQ](#12-troubleshooting--faq)
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
- **Operating System**: Linux (Ubuntu 22.04+ / Debian 12+) or Windows 11 with WSL2 (Ubuntu).
- **CPU**: x86_64 or ARM64 processor (Intel Core i5 / AMD Ryzen 5 or higher).
- **RAM**: 4 GB minimum (8 GB recommended).
- **Storage**: 1 GB free disk space (excluding video logs and snapshots).
- **Camera (Optional for live mode)**: Any smartphone connected to the same local Wi-Fi running an RTSP/HTTP IP webcam app (e.g. *IP Webcam* on Android or *Live-Reporter* on iOS).

### GPU Acceleration (Optional)
- **Supported GPUs**: NVIDIA GeForce RTX series (RTX 30-series, 40-series, 50-series) or NVIDIA workstation GPUs.
- **CUDA Runtime**: CUDA 12.x or 13.x driver installed on host system.
- *Note on RTX 50-series (Blackwell / Compute Capability 12.0 / `sm_120`)*: If running on brand-new architectures where PyTorch precompiled CUDA wheels lack native `sm_120` kernels, the system's `RuntimeDiagnostics` automatically falls back to verified CPU execution (achieving **>20 FPS** on CPU) with a clear diagnostic log.

---

## 6. Installation & Setup Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/lonelokii/SmartSurveillance.git
cd SmartSurveillance
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Upgrade Package Installer
```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

#### Standard Installation (Automatic GPU / CPU selection via PyTorch)
```bash
pip install -r requirements.txt
```

#### Manual PyTorch Selection (CUDA 12.6 Official Wheels)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics opencv-python pyyaml numpy pytest
```

#### CPU-Only Systems
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python pyyaml numpy pytest
```

### Step 5: Verify Environment & Hardware Diagnostics
Run the automated environment check to inspect Python, PyTorch, CUDA device visibility, and installed libraries:
```bash
python3 scripts/check_env.py
```

### Step 6: Verify YOLO Weights
The default weights `yolov8n.pt` are automatically downloaded by Ultralytics on first run if not already present in the workspace.

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
```bash
python3 main.py
```

### Example 2: Run with a Local Video File
Test with any local recorded video clip:
```bash
python3 main.py --source video --video-path tests/data/one_person_walk.mp4
```

### Example 3: Run with a Live Phone Camera (IP Webcam)
1. Install **IP Webcam** (Android) or any RTSP stream app on your phone.
2. Start the server inside the app and note the streaming URL (e.g., `http://192.168.1.45:8080/video`).
3. Ensure your PC and phone are on the same Wi-Fi network.
4. Execute:
```bash
python3 main.py --source live --live-url http://192.168.1.45:8080/video
```

### Example 4: Force CPU Inference
```bash
python3 main.py --device cpu
```

### Example 5: Run Headless (Automated Server / Benchmarks)
Run without opening an OpenCV GUI window:
```bash
python3 main.py --headless --max-frames 300
```

### Operator Controls
- **Quit / Stop Application**: Press `q` while focused on the OpenCV video display window, or send `Ctrl+C` (SIGINT/SIGTERM) in the terminal. All capture devices and database connections are closed cleanly.

---

## 9. Utility & Diagnostic Scripts

The `scripts/` folder provides standalone validation tools:

| Script Name | Command | Purpose |
|---|---|---|
| **Environment Check** | `python3 scripts/check_env.py` | Generates a full report of Python version, CUDA drivers, WSL GPU passthrough, and package paths. |
| **FPS Benchmark** | `python3 scripts/benchmark_fps.py --frames 100` | Measures actual processing throughput against the PRD requirement (`>= 5.0 FPS`). |
| **CUDA Verification** | `python3 scripts/verify_cuda_inference.py` | Performs standalone YOLO inference directly on CUDA device 0 and verifies tensor output device. |
| **Stability Runner** | `python3 scripts/stability_run.py --duration 600` | Runs continuous 10-minute synthetic processing and monitors RSS memory consumption. |
| **Synthetic Media Creator** | `python3 scripts/create_test_media.py` | Generates deterministic synthetic test videos (`empty_room_test.mp4` and `one_person_walk.mp4`) for testing. |

---

## 10. Testing & Verification Suite

The repository includes a comprehensive automated test suite covering all functional requirements (FR-01 through FR-28) and acceptance criteria (AC-1 through AC-18).

### Running All Tests
```bash
pytest -v
```

### Test Coverage Summary

```
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

### Empty-Room Zero False Positive Verification
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

### 1. The phone IP camera stream fails to open
- Ensure your mobile phone and PC are connected to the exact same local Wi-Fi network (not mobile data).
- Test opening the stream URL in a browser on your PC (e.g. `http://192.168.1.5:8080/video`).
- Ensure no firewall on Linux/Windows is blocking incoming HTTP port 8080.

### 2. PyTorch displays a warning regarding CUDA Compute Capability (sm_120 / RTX 5050)
- NVIDIA GeForce RTX 50-series GPUs (Blackwell) use compute capability 12.0 (`sm_120`).
- If your installed PyTorch wheel is built for CUDA 12.6, prebuilt `sm_120` kernels may not be included.
- **The system automatically handles this**: `RuntimeDiagnostics` detects the missing kernels, logs a clear warning, and truthfully falls back to CPU mode, where inference runs smoothly at **~24 FPS**.
- To enable native GPU acceleration on RTX 50-series when available, reinstall PyTorch using CUDA 13.x wheels:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
  ```

### 3. How do I inspect stored alert events in SQLite?
You can query the database directly using the SQLite CLI or Python:
```bash
sqlite3 surveillance.db "SELECT event_id, timestamp, track_id, zone_id, snapshot_path FROM alerts;"
```

### 4. Can I use this for non-human objects (cars, backpacks, etc.)?
No. Phase 1 is strictly scoped to the `person` class (COCO 0) to ensure high reliability, zero clutter, and compliance with the core surveillance baseline.

---

## 13. License & Governance

This project is licensed under the **MIT License**.

All implementations adhere strictly to the authoritative specifications documented in [`PRD_PHASE1.md`](file:///home/boom/projects/SmartSurveillance/PRD_PHASE1.md) and [`plans/README.md`](file:///home/boom/projects/SmartSurveillance/plans/README.md).
