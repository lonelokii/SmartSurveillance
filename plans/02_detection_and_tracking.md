# Phase 1 Subplan 02: YOLO Person Detection, Tracking, and Annotated Display

## 1. Goal and Overview
Implement the unified video input handler (local video and live IP-camera stream), person-only detection using real Ultralytics YOLO, temporary multi-object person tracking with track IDs and expiration, and live annotated frame rendering via OpenCV. Strictly exclude non-person classes, loitering/behavior analysis, face recognition, and carried objects.

## 2. Requirements & Acceptance Criteria Mapping
- **Functional Requirements Covered**:
  - `FR-01`: Accept one configured video source at a time.
  - `FR-02`: Support local video file as source.
  - `FR-03`: Support phone IP-camera video URL (RTSP/HTTP MJPEG) as source.
  - `FR-04`: Report clear actionable error if selected source cannot be opened.
  - `FR-07`: Detect the person class only (COCO class ID 0).
  - `FR-08`: Filter detections using configurable confidence threshold (`confidence_threshold`).
  - `FR-09`: Display bounding box, class label (`person`), confidence, and tracking ID for each confirmed person track.
  - `FR-10`: Count only currently confirmed, recently observed person tracks as visible people.
  - `FR-11`: Expire track after configurable consecutive missed frames (`max_missed_frames`).
  - `FR-21`: Display source mode, model status, actual inference device, visible-person count, and counts on frame overlay.
  - `FR-22`: Operator clean stop (e.g. keyboard 'q' or interrupt).
  - `FR-23`: Clear error and resource release on source interruption.
- **Acceptance Criteria Covered**:
  - `Criterion 1`: A local video file opens and displays.
  - `Criterion 2`: A live phone-camera stream opens and displays.
  - `Criterion 4`: Known person test clip displays at least one correctly labeled person detection.
  - `Criterion 5`: Known empty-room test clip produces no false detections during test interval; any false positives are documented.
  - `Criterion 6`: One person entering frame creates confirmed track and updates visible-person count to 1.
  - `Criterion 11`: Person track disappears from visible-person count after configured expiration period.
  - `Criterion 18`: Operator clean stop and source interruption releases capture and DB resources cleanly.

---

## 3. Tasks and Detailed Work Breakdown

### Task 2.1: Video Source Handler (`src/video_source.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/video_source.py`
  - `tests/test_video_source.py`
- **Description**:
  1. Implement `VideoSource` class supporting:
     - Mode `video`: opens local file path with `cv2.VideoCapture`.
     - Mode `live`: opens IP webcam URL (e.g. `http://<ip>:<port>/video` or `rtsp://...`).
  2. Implement robust connection validation:
     - Checks `cap.isOpened()` immediately.
     - Performs probe read `cap.read()`. If frame is None, raises `SourceOpenError` with specific troubleshooting advice (network unreachable, file not found, bad codec).
  3. Implement clean context manager `__enter__` / `__exit__` and `release()` ensuring `cap.release()` is called in all failure scenarios.
  4. Implement stream reconnect/interruption detection: raises `SourceInterruptedError` if stream terminates unexpectedly.
- **Observable Pass Condition**:
  Unit tests pass opening a test video file and failing cleanly on a non-existent URL or file path.

### Task 2.2: Genuine YOLO Person-Only Detector (`src/detector.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/detector.py`
  - `tests/test_detector.py`
- **Description**:
  1. Implement `PersonDetector` wrapping the loaded `ultralytics.YOLO` model.
  2. In `detect(frame) -> list[Detection]`:
     - Run inference: `model.predict(source=frame, classes=[0], conf=conf_threshold, device=device, verbose=False)`.
     - Note: `classes=[0]` restricts detections strictly to `person` at the model inference level.
     - Defensive filter: assert every returned box has `cls == 0` (`person`). Reject any non-person classes.
     - Extract normalized or pixel bounding boxes `(x1, y1, x2, y2)`, confidence `conf`, and class label `"person"`.
  3. Prohibited item safeguard: Zero detection logic for backpacks, bags, phones, faces, or postures.
- **Observable Pass Condition**:
  `test_detector.py` verifies only detections with class label `"person"` and confidence `>= conf_threshold` are returned.

### Task 2.3: Temporary Person Tracker & Confirmed Occupancy (`src/tracker.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/tracker.py`
  - `tests/test_tracker.py`
- **Description**:
  1. Implement lightweight Centroid / IoU-based `PersonTracker`:
     - Assigns monotonic integer tracking IDs (`id=1, 2, 3...`).
     - Tracks bounding box history, centroid `(cx, cy)`, consecutive hit count, and missed frames count.
  2. Implement track confirmation & expiration:
     - Track marked confirmed after `min_hits` consecutive detections (e.g., 2 frames to avoid single-frame noise).
     - Track marked expired and pruned when `missed_frames > max_missed_frames` (configurable, default 15-30 frames).
  3. Implement `get_active_tracks() -> list[Track]`:
     - Returns only active, confirmed tracks.
  4. Visible-person count: `len(active_tracks)` represents the exact live visible-person occupancy.
- **Observable Pass Condition**:
  Unit test simulates 5 frames of detections, validates ID persistence, simulates 30 empty frames, validates track expiration and occupancy count returning to 0.

### Task 2.4: Video Frame Overlay & Annotator (`src/visualizer.py`)
- **Type**: Implementation
- **Files Created / Modified**:
  - `src/visualizer.py`
  - `tests/test_visualizer.py`
- **Description**:
  1. Implement `FrameAnnotator` drawing:
     - Person bounding box (e.g. green rectangle).
     - Text label: `person #{track_id} ({conf:.2f})`.
     - Centroid marker (small circle).
     - HUD / Status Header overlay displaying:
       - Source mode (`video` / `live`).
       - Active Model (`yolov8n.pt`).
       - Actual Inference Device (`cuda:0 (NVIDIA RTX 5050)` or `cpu`).
       - Visible Person Count (`Occupancy: X`).
       - Entry / Exit counts and latest alert banner.
  2. Clean rendering: Ensure text is readable with background contrast box.
- **Observable Pass Condition**:
  `test_visualizer.py` validates that output frame dimensions and data types match input frame.

### Task 2.5: Synthetic & Sample Media Verification
- **Type**: Verification & Test Data Setup
- **Files Created / Modified**:
  - `scripts/create_test_media.py`
  - `tests/test_media_pipeline.py`
- **Description**:
  1. Create synthetic / test clips:
     - `tests/data/empty_room_test.mp4`: Known 100-frame video without any people.
     - `tests/data/one_person_walk.mp4`: Controlled video with one person walking across frame.
  2. Implement empty-room verification test adhering to the **Empty-Room Test Rule**:
     - Evaluate `empty_room_test.mp4`.
     - Measure detections at configured confidence threshold (e.g., 0.5).
     - Expected result: 0 detections across the clip.
     - If any false positive is detected, log exact frame index, bounding box, and confidence into `test_reports/empty_room_diagnostics.json` rather than claiming universal infallibility.
  3. Implement one-person test clip:
     - Asserts track ID assigned and visible count is 1.
- **Observable Pass Condition**:
  Both test clips run through `VideoSource -> PersonDetector -> PersonTracker` and pass criteria 4, 5, and 6.

---

## 4. Subplan Verification & Test Gates

| Test ID | Test Name | Command / Script | Expected Result | PRD Mapping |
|---|---|---|---|---|
| T2-01 | Local Video File Open | `pytest tests/test_video_source.py -k test_local_video_opens` | Opens and reads frames without error | FR-01, FR-02, AC-1 |
| T2-02 | IP Stream Open & Error Handling | `pytest tests/test_video_source.py -k test_invalid_source_error` | Raises clear `SourceOpenError`, no crash | FR-03, FR-04, AC-2 |
| T2-03 | Person-Only Detection | `pytest tests/test_detector.py -k test_person_only_classes` | Detects person; zero other classes returned | FR-07, FR-08, AC-4 |
| T2-04 | Empty-Room Clip Verification | `pytest tests/test_media_pipeline.py -k test_empty_room_clip` | 0 detections or documented false positive report | FR-07, FR-10, AC-5 |
| T2-05 | One-Person Occupancy & Track ID | `pytest tests/test_tracker.py -k test_single_person_occupancy` | Track confirmed, visible count updates to 1 | FR-09, FR-10, AC-6 |
| T2-06 | Track Expiration Lifetime | `pytest tests/test_tracker.py -k test_track_expiration` | Track removed after `max_missed_frames`, count becomes 0 | FR-10, FR-11, AC-11 |
| T2-07 | Clean Teardown on Interruption | `pytest tests/test_video_source.py -k test_resource_release_on_error` | `cap.release()` called, cleanly closed | FR-22, FR-23, AC-18 |
