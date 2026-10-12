# Smart Surveillance System — Phase 1 Master Implementation Plan

## 1. Executive Summary & Authoritative Scope Baseline
This master plan governs the implementation and verification of **Phase 1** for the **Smart Surveillance System**. As mandated by the user and the authoritative PRD (`PRD_PHASE1.md`), **all requirements—including genuine Ultralytics YOLO person detection, temporary person tracking, visible counts, directional line crossing, restricted-zone entry alerts, snapshots, SQLite logging, clean lifecycle, and CUDA RTX 5050 support where available—belong strictly to Phase 1**.

No features from the PRD are deferred to Phase 2. Any prior roadmap statements or messages suggesting otherwise are superseded by this baseline.

---

## 2. Plan Architecture & Subplans
The work is decomposed into three sequential, self-contained subplans:

1. **[Subplan 01: Environment, Runtime Truthfulness, and Model Loading](file:///home/boom/projects/SmartSurveillance/plans/01_environment_and_runtime.md)**
   - Focus: Python 3.14 venv audit, official PyTorch Linux+pip verification, genuine Ultralytics YOLO loading, truthful device reporting (`cuda:0` vs `cpu`), zero-HOG detector guardrails.
   - Tasks: 1.1 through 1.4.
2. **[Subplan 02: YOLO Person Detection, Tracking, and Annotated Display](file:///home/boom/projects/SmartSurveillance/plans/02_detection_and_tracking.md)**
   - Focus: Video source abstraction (local file and live IP camera), person-only detection filtering (class 0), centroid/IoU person tracking with ID assignment and expiration, HUD video annotation, empty-room and single-person test clip setup.
   - Tasks: 2.1 through 2.5.
3. **[Subplan 03: Analytics, Alerts, Storage, and Lifecycle](file:///home/boom/projects/SmartSurveillance/plans/03_analytics_alerts_and_lifecycle.md)**
   - Focus: Directional line-crossing counting, restricted-zone polygon transition detection, JPEG snapshot generation, SQLite event storage, main application orchestrator, 10-minute stability run, and >= 5 FPS performance benchmark.
   - Tasks: 3.1 through 3.5.

---

## 3. Full Requirements Coverage Matrix (FR-01 through FR-28)

| PRD FR ID | Requirement Description | Subplan | Task(s) | Verification Test |
|---|---|---|---|---|
| **FR-01** | Accept one configured video source at a time | Subplan 02 | Task 2.1 | `tests/test_video_source.py` |
| **FR-02** | Support local video file as source | Subplan 02 | Task 2.1 | `tests/test_video_source.py::test_local_video_opens` |
| **FR-03** | Support phone IP-camera video URL as source | Subplan 02 | Task 2.1 | `tests/test_video_source.py::test_ip_camera_stream` |
| **FR-04** | Report clear error if source cannot be opened | Subplan 02 | Task 2.1 | `tests/test_video_source.py::test_invalid_source_error` |
| **FR-05** | Load Ultralytics YOLO weights & report path/device truthfully | Subplan 01 | Task 1.3, 1.4 | `tests/test_model_load.py::test_valid_load` |
| **FR-06** | Fail actionably if YOLO fails to load; no silent fallback | Subplan 01 | Task 1.4 | `tests/test_model_load.py::test_invalid_load_fails_cleanly` |
| **FR-07** | Detect person class only (COCO class ID 0) | Subplan 02 | Task 2.2 | `tests/test_detector.py::test_person_only_classes` |
| **FR-08** | Filter detections using configurable confidence threshold | Subplan 02 | Task 2.2 | `tests/test_detector.py::test_confidence_filtering` |
| **FR-09** | Display box, label (`person`), confidence, and track ID | Subplan 02 | Task 2.3, 2.4 | `tests/test_visualizer.py::test_annotation_fields` |
| **FR-10** | Count only confirmed, recently observed person tracks | Subplan 02 | Task 2.3 | `tests/test_tracker.py::test_single_person_occupancy` |
| **FR-11** | Expire track after configurable consecutive missed frames | Subplan 02 | Task 2.3 | `tests/test_tracker.py::test_track_expiration` |
| **FR-12** | Display configured entry/exit line on video | Subplan 03 | Task 3.1, 3.4 | `tests/test_visualizer.py::test_line_rendering` |
| **FR-13** | Increment counter once on crossing in configured direction | Subplan 03 | Task 3.1 | `tests/test_line_crossing.py::test_directional_crossing` |
| **FR-14** | Repeated frames on same side do not increment counter again | Subplan 03 | Task 3.1 | `tests/test_line_crossing.py::test_same_side_suppression` |
| **FR-15** | Display configured restricted-zone polygon on video | Subplan 03 | Task 3.2, 3.4 | `tests/test_visualizer.py::test_zone_rendering` |
| **FR-16** | Zone alert created only on outside-to-inside transition | Subplan 03 | Task 3.2 | `tests/test_zone_monitor.py::test_entry_transition` |
| **FR-17** | Remaining inside zone does not repeatedly generate alerts | Subplan 03 | Task 3.2 | `tests/test_zone_monitor.py::test_stay_inside_no_duplicate` |
| **FR-18** | Alert contains timestamp, source, track ID, zone ID, snapshot | Subplan 03 | Task 3.3 | `tests/test_storage.py::test_alert_record_fields` |
| **FR-19** | Save one snapshot per zone-entry event | Subplan 03 | Task 3.3 | `tests/test_storage.py::test_snapshot_creation` |
| **FR-20** | Store each alert in a local SQLite database | Subplan 03 | Task 3.3 | `tests/test_storage.py::test_sqlite_persistence` |
| **FR-21** | Display source, model, device, visible count, and alert status | Subplan 02, 03 | Task 2.4, 3.4 | `tests/test_visualizer.py::test_hud_overlay` |
| **FR-22** | Clean application stop on operator command | Subplan 03 | Task 3.4 | `tests/test_e2e_pipeline.py::test_clean_operator_stop` |
| **FR-23** | Clean resource release on source interruption | Subplan 02, 03 | Task 2.1, 3.4 | `tests/test_video_source.py::test_resource_release_on_error` |
| **FR-24** | Check and truthfully report whether CUDA is available | Subplan 01 | Task 1.1, 1.3 | `tests/test_model_load.py::test_cuda_reporting` |
| **FR-25** | Run YOLO on CUDA device 0 when available and configured | Subplan 01 | Task 1.3, 1.4 | `tests/test_model_load.py::test_cuda_device_0_inference` |
| **FR-26** | Run YOLO on CPU and clearly report CPU device when fallback | Subplan 01 | Task 1.3, 1.4 | `tests/test_model_load.py::test_cpu_truthfulness` |
| **FR-27** | Never report GPU unless active model is using CUDA device 0 | Subplan 01 | Task 1.3 | `tests/test_model_load.py::test_device_truthfulness_guard` |
| **FR-28** | No OpenCV HOG or other detector in place of YOLO | Subplan 01 | Task 1.3, 1.4 | `tests/test_model_load.py::test_no_hog_detector_guard` |

---

## 4. Acceptance Criteria Verification Matrix (Criteria 1 through 18)

| AC # | Acceptance Criterion | Subplan | Test Fixture / Verification Task | Pass Condition |
|---|---|---|---|---|
| **AC-1** | Local video file opens and displays | Subplan 02 | `tests/test_video_source.py` | Frames decoded and displayed without error |
| **AC-2** | Live phone-camera stream opens and displays | Subplan 02 | `tests/test_video_source.py` | Stream frames read via IP Webcam URL |
| **AC-3** | Startup reports actual YOLO backend, weights path, device | Subplan 01 | `tests/test_model_load.py` | Exact backend, path, and device logged truthfully |
| **AC-4** | Person test clip displays labeled person detection | Subplan 02 | `tests/test_media_pipeline.py` | Bounding box, `person`, conf, track ID rendered |
| **AC-5** | Known empty-room test clip produces no false detections | Subplan 02 | `tests/test_media_pipeline.py` | 0 alerts/detections on `empty_room_test.mp4` (documented) |
| **AC-6** | One person entering frame updates visible count to 1 | Subplan 02 | `tests/test_tracker.py` | Track ID confirmed, visible occupancy = 1 |
| **AC-7** | Person crossing line once increments counter once | Subplan 03 | `tests/test_line_crossing.py` | Exactly 1 increment; repeated frames on side = 0 |
| **AC-8** | Person entering zone creates 1 alert and 1 snapshot | Subplan 03 | `tests/test_zone_monitor.py` | Exactly 1 alert event fired, 1 image file saved |
| **AC-9** | Remaining inside zone produces no repeated alerts | Subplan 03 | `tests/test_zone_monitor.py` | Zero new alerts while track remains in polygon |
| **AC-10** | Leaving and re-entering zone creates a new alert | Subplan 03 | `tests/test_zone_monitor.py` | Transition reset upon exit, new alert on re-entry |
| **AC-11** | Track expires after configured expiration period | Subplan 02 | `tests/test_tracker.py` | Active tracks count drops to 0 after `max_missed_frames` |
| **AC-12** | Event in SQLite and snapshot file exists | Subplan 03 | `tests/test_storage.py` | SQL record retrieved with correct columns; file exists |
| **AC-13** | 10-minute continuous run without crashing | Subplan 03 | `scripts/stability_run.py` | 600s execution completes cleanly; stable memory RSS |
| **AC-14** | Actionable failure if YOLO fails to load | Subplan 01 | `tests/test_model_load.py` | Aborts with diagnostic error; no HOG substitution |
| **AC-15** | Verified YOLO inference on CUDA device 0 (RTX 5050) | Subplan 01 | `tests/test_model_load.py` | Inference executes on device `cuda:0` when available |
| **AC-16** | Truthful CPU inference baseline/fallback | Subplan 01 | `tests/test_model_load.py` | Device strictly reported as `cpu` |
| **AC-17** | Target processing speed (>= 5 FPS) measured and reported | Subplan 03 | `scripts/benchmark_fps.py` | Average FPS >= 5.0 on test resolution |
| **AC-18** | Clean shutdown & source error resource release | Subplan 02, 03 | `tests/test_e2e_pipeline.py` | Video capture, GUI, and SQLite connection closed |

---

## 5. Strict Prohibitions Compliance Audit
The plan was rigorously checked to confirm the total exclusion of:
- Detectors for classes other than person: **STRICTLY EXCLUDED** (YOLO class filter pinned to 0).
- Carried object detection (bags, phones, purses, laptops): **STRICTLY EXCLUDED**.
- Face recognition or named person identification: **STRICTLY EXCLUDED**.
- Loitering, pose, action, fall, violence, or emotion analysis: **STRICTLY EXCLUDED**.
- Threat prediction, criminal intent, or risk scoring: **STRICTLY EXCLUDED**.
- Multi-camera orchestration or cross-camera re-identification: **STRICTLY EXCLUDED**.
- Remote APIs, cloud storage, or external security system hooks: **STRICTLY EXCLUDED**.
- Silent fallback to OpenCV HOG: **STRICTLY EXCLUDED** (explicit guardrail test T1-03).

---

## 6. Gated Tasks & User Decisions Required Before Execution
Execution of Phase 1 will require approval at specific gates:
1. **Gate 1: WSL2 & PyTorch Wheel Selection** (Task 1.1 & 1.2):
   - Review active Python 3.14 wheel availability via official PyTorch selector.
   - Confirm pip install command for `torch`, `torchvision`, `ultralytics`, and `opencv-python`.
   - Never install Linux NVIDIA display drivers inside WSL2.
2. **Gate 2: GPU Verification vs. CPU Fallback**:
   - If `torch.cuda.is_available()` returns `True`, test and log RTX 5050 on `cuda:0`.
   - If PyTorch prebuilt CUDA wheels are not yet available for Python 3.14 on Linux, explicitly confirm running Subplan 1–3 in verified CPU YOLO mode until PyTorch publishes Python 3.14 CUDA wheels.
