# Phase 1 Subplan 03: Analytics, Alerts, Storage, and Lifecycle

## 1. Goal and Overview
Implement spatial analytics (line-crossing directional counters and restricted-zone polygon transition detection), snapshot generation, SQLite event persistence, configuration management, and the central runtime orchestrator. Validate end-to-end operation against acceptance criteria, 10-minute stability, and performance FPS benchmarks.

## 2. Requirements & Acceptance Criteria Mapping
- **Functional Requirements Covered**:
  - `FR-12`: Display configured entry/exit line on video.
  - `FR-13`: Increment entry/exit counter exactly once when a person crosses in configured direction.
  - `FR-14`: Do not increment counter again for repeated frames on the same side.
  - `FR-15`: Display configured restricted-zone polygon on video.
  - `FR-16`: Generate zone alert only when a confirmed track transitions from outside to inside.
  - `FR-17`: Remaining inside zone produces no repeated alerts.
  - `FR-18`: Alert record contains event ID, timestamp, camera/source ID, track ID, zone ID, snapshot path, confidence, inference device.
  - `FR-19`: Save one snapshot image per zone-entry event.
  - `FR-20`: Store each alert in a local SQLite database.
  - `FR-21`: Display entry count, exit count, visible count, and recent alert status on video.
  - `FR-22`: Clean stop on operator request.
  - `FR-23`: Clear error and clean resource release on source interruption.
- **Non-Functional Requirements Covered**:
  - Target >= 5 FPS processing speed on test setup; separate measurement for CPU and GPU.
  - 10-minute continuous stability without crashing or memory leaks.
- **Acceptance Criteria Covered**:
  - `Criterion 7`: A person crossing the configured line once increments exactly one direction counter once.
  - `Criterion 8`: A person entering the restricted zone creates exactly one alert and one snapshot.
  - `Criterion 9`: Remaining inside the zone does not repeatedly create alerts.
  - `Criterion 10`: Leaving and later re-entering the zone creates a new alert.
  - `Criterion 12`: The event is present in SQLite and its snapshot file exists.
  - `Criterion 13`: The application can run for 10 minutes on the test source without crashing.
  - `Criterion 17`: Target processing speed (>= 5 FPS) measured and reported for CPU and GPU.
  - `Criterion 18`: Clean shutdown and resource release.

---

## 3. Tasks and Detailed Work Breakdown

### Task 3.1: Virtual Line Crossing Engine (`src/line_crossing.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/line_crossing.py`
  - `tests/test_line_crossing.py`
- **Description**:
  1. Define line segment `((x1, y1), (x2, y2))` and normal orientation (entry vs exit vector).
  2. For each active track:
     - Track previous centroid `P_{t-1}` and current centroid `P_t`.
     - Detect line intersection between segment `[P_{t-1}, P_t]` and line `L`.
     - Compute cross product / dot product to determine crossing direction (`entry` vs `exit`).
  3. Anti-duplicate counting logic:
     - Record `last_crossing_frame` and `side_state` per track ID.
     - Increment counter exactly once per geometric transition.
     - Staying on the crossed side across subsequent frames produces zero additional increments.
- **Observable Pass Condition**:
  Unit test simulates trajectory crossing from left to right: entry counter increments by 1. Subsequent frames on the right side do not increment. Reverse crossing increments exit counter by 1.

### Task 3.2: Restricted-Zone Polygon Monitor (`src/zone_monitor.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/zone_monitor.py`
  - `tests/test_zone_monitor.py`
- **Description**:
  1. Define polygon vertices `[(x1, y1), (x2, y2), ...]`.
  2. Implement point-in-polygon test (using ray-casting algorithm or `cv2.pointPolygonTest`).
  3. Track per-track zone state: `is_inside: bool`.
  4. Transition detection rule:
     - Trigger alert ONLY on state transition `False -> True` (outside to inside).
     - State `True -> True` (person staying inside) produces NO alert.
     - State `True -> False` (person exits zone) resets state to `False` without triggering entry alert.
     - Subsequent `False -> True` (re-entering zone) triggers exactly one new alert.
- **Observable Pass Condition**:
  Unit test simulates entry -> triggers alert. Simulates 10 frames staying inside -> triggers 0 alerts. Simulates exit then re-entry -> triggers exactly 1 new alert.

### Task 3.3: Snapshot Generator & SQLite Event Store (`src/storage.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/storage.py`
  - `tests/test_storage.py`
- **Description**:
  1. Implement `SnapshotWriter`:
     - Creates configured snapshot directory (`snapshots/`) if missing.
     - Saves JPEG snapshot named `alert_{event_id}_{timestamp}.jpg` with full frame and detection overlay.
  2. Implement `DatabaseLogger`:
     - Initializes SQLite database schema (`surveillance.db`) with `alerts` table:
       - `event_id` (TEXT PRIMARY KEY)
       - `event_type` (TEXT, e.g. "ZONE_ENTRY")
       - `timestamp` (TEXT, ISO 8601)
       - `source_id` (TEXT)
       - `track_id` (INTEGER)
       - `zone_id` (TEXT)
       - `snapshot_path` (TEXT)
       - `confidence` (REAL)
       - `device` (TEXT)
     - Implements safe, parameterized insert queries (`INSERT INTO alerts VALUES (...)`).
     - Includes clean connection closing on application shutdown.
- **Observable Pass Condition**:
  `test_storage.py` logs an alert, validates file existence of snapshot on disk, and verifies exact row values via `SELECT * FROM alerts`.

### Task 3.4: Main Pipeline Orchestrator & Configuration
- **Type**: Implementation & Orchestration
- **Files Created / Modified**:
  - `config.yaml`
  - `src/config.py`
  - `src/app.py`
  - `main.py`
- **Description**:
  1. Create `config.yaml` loading:
     - Source mode (`video` / `live`), paths/URLs.
     - Model weights, device selection (`cuda:0` / `cpu`), confidence threshold (e.g. 0.5), input resolution.
     - Tracker parameters (`max_missed_frames`, `min_hits`).
     - Line coordinates and direction.
     - Restricted-zone polygon coordinates.
     - Storage paths (`snapshots/`, `surveillance.db`).
  2. Create `SurveillanceApp` (`src/app.py`):
     - Pipeline loop: Frame read -> Detection -> Tracking -> Line analytics -> Zone analytics -> Snapshot & DB log -> Visualizer render -> OpenCV window display (`cv2.imshow`) or headless frame emission.
     - Graceful shutdown handlers: intercepts `SIGINT` (Ctrl+C) and `'q'` key press, closes video stream, flushes and closes SQLite database, releases OpenCV windows.
- **Observable Pass Condition**:
  `main.py --help` prints usage; pipeline initializes, processes sample video, and exits cleanly with exit code 0.

### Task 3.5: End-to-End Acceptance Tests, 10-Minute Stability & FPS Benchmark
- **Type**: Verification
- **Files Created / Modified**:
  - `tests/test_e2e_pipeline.py`
  - `scripts/benchmark_fps.py`
  - `scripts/stability_run.py`
- **Description**:
  1. E2E Scenario Test (`test_e2e_pipeline.py`):
     - Plays `one_person_walk.mp4`.
     - Validates:
       - Initial visible count = 0.
       - Person detected, count = 1.
       - Line crossed -> entry count increments from 0 to 1.
       - Restricted zone entered -> snapshot written, SQLite row inserted with correct fields.
       - Inside zone -> no duplicate snapshot/DB entries.
       - Exits frame -> visible count returns to 0.
  2. FPS Benchmark (`benchmark_fps.py`):
     - Runs 300 frames on test resolution.
     - Computes average FPS separately for CPU and GPU (when CUDA available).
     - Asserts FPS >= 5.0.
  3. 10-Minute Stability Run (`stability_run.py`):
     - Loops test source for 10 minutes (600 seconds).
     - Monitors memory RSS and confirms no crashes, leaks, or unhandled exceptions.
- **Observable Pass Condition**:
  All tests in `test_e2e_pipeline.py` pass; benchmark confirms >= 5 FPS; 10-minute stability run finishes with exit code 0.

---

## 4. Subplan Verification & Test Gates

| Test ID | Test Name | Command / Script | Expected Result | PRD Mapping |
|---|---|---|---|---|
| T3-01 | Line Crossing Increment | `pytest tests/test_line_crossing.py` | Exactly 1 increment per crossing, 0 on repeated frames | FR-12, FR-13, FR-14, AC-7 |
| T3-02 | Zone Single Alert & Snapshot | `pytest tests/test_zone_monitor.py -k test_entry_alert` | Exactly 1 alert on outside->inside transition | FR-15, FR-16, FR-19, AC-8 |
| T3-03 | Zone Persistence No Duplicate | `pytest tests/test_zone_monitor.py -k test_stay_inside` | 0 duplicate alerts while remaining inside | FR-17, AC-9 |
| T3-04 | Zone Re-entry New Alert | `pytest tests/test_zone_monitor.py -k test_reentry` | Exactly 1 new alert upon exit and re-entry | FR-16, AC-10 |
| T3-05 | SQLite Persistence & Snapshot | `pytest tests/test_storage.py` | Row present with all 9 fields; snapshot file exists | FR-18, FR-20, AC-12 |
| T3-06 | 10-Minute Stability Run | `python3 scripts/stability_run.py --duration 600` | Runs 600s with zero crash or memory leak | NFR, AC-13 |
| T3-07 | Processing Speed Benchmark | `python3 scripts/benchmark_fps.py` | >= 5 FPS measured and reported for CPU and GPU | NFR, AC-17 |
| T3-08 | Graceful Shutdown | `pytest tests/test_e2e_pipeline.py -k test_clean_shutdown` | Resources closed, zero resource leaks | FR-22, FR-23, AC-18 |
