# Product Requirements Document
## Smart Surveillance System — Phase 1

### 1. Document Control

- Product: Smart Surveillance System
- Release: Phase 1
- Status: Requirements baseline
- Document scope: Phase 1 only
- Implementation environment: Antigravity IDE, WSL2 Ubuntu, Python 3.14
- Target hardware: NVIDIA GeForce RTX 5050 laptop GPU

### 2. Product Summary

The Smart Surveillance System is a local computer-vision application that processes one video source on a PC. The source may be a mobile-phone IP-camera stream or a local video file.

The system uses a genuine, verifiably loaded Ultralytics YOLO model to detect visible people, assigns temporary tracking IDs, displays the number of people currently visible, counts crossings of a configured virtual line, detects entry into a configured restricted area, and records alerts with timestamps and snapshots.

The application shall use the NVIDIA RTX 5050 through CUDA for YOLO inference when CUDA is available. CPU inference shall remain available as a supported fallback for testing or systems without accessible CUDA.

### 3. Problem Statement

A human operator may miss people entering a monitored area, miscount people passing through a doorway, or fail to notice entry into a restricted area. This application demonstrates how computer vision and explicit geometric rules can assist with those basic monitoring tasks.

### 4. Goals

The application shall:

- Read one live phone-camera stream or one local video file at a time.
- Detect people using a real, verifiably loaded Ultralytics YOLO model.
- Prefer and use CUDA device 0 on the NVIDIA RTX 5050 when available.
- Permit CPU inference as a supported fallback when CUDA is unavailable or explicitly selected.
- Draw a bounding box, person label, confidence score, and tracking ID for each confirmed visible person.
- Show the current visible-person count.
- Count crossings of a configured entry/exit line.
- Detect a person's transition from outside to inside a configured restricted zone.
- Show a visible alert for a restricted-zone entry.
- Save one snapshot and one event record per zone-entry event.
- Provide a clear way to start and stop monitoring.
- Report the actual model backend, weights path, and inference device truthfully.

### 5. Scope

#### Included

- Single-camera video input.
- Phone IP-camera URL input.
- Local video-file input.
- Ultralytics YOLO person detection.
- Person tracking with temporary IDs.
- Live visible-person count.
- Configurable line-crossing count.
- Configurable restricted-zone polygon.
- Restricted-zone entry alerts.
- Local snapshot storage.
- Local SQLite event storage.
- Basic video display with overlays.
- Python 3.14 project environment.
- CUDA-enabled PyTorch and RTX 5050 GPU inference when available.
- CPU inference as a supported fallback.
- Explicit startup checks and runtime reporting for Python, PyTorch, YOLO, CUDA, model weights, and selected inference device.
- OpenCV for video capture, frame handling, drawing, and display only; OpenCV HOG shall not be used as a person detector.

#### Out of scope

- Object detection other than the person class.
- Detection or association of bags, phones, purses, laptops, or other carried items.
- Face recognition or identifying a person by name.
- Loitering, action, posture, violence, or emotion analysis.
- Risk scoring, threat prediction, or criminal-intent inference.
- Multiple-camera processing or cross-camera identity matching.
- Cloud storage, remote notifications, or external security-service integration.
- Automatic law-enforcement or access-control decisions.
- OpenCV HOG or any other non-YOLO detector as a silent fallback.

### 6. Users

- Operator: selects a source, starts or stops monitoring, and views counts and alerts.
- Reviewer: examines locally saved event records and snapshots.
- Developer: configures the model, device, line, zone, and test input.

### 7. User Stories

- As an operator, I want to select a live phone-camera stream so I can view a monitored doorway.
- As an operator, I want to select a local video file so I can test without network access.
- As an operator, I want visible people outlined and labeled so I can verify detections.
- As an operator, I want a live count of visible people.
- As an operator, I want line crossings counted once per crossing.
- As an operator, I want an alert when a person enters the restricted zone.
- As a reviewer, I want each zone-entry alert to have a timestamp and snapshot.
- As a developer, I want to verify whether inference is running on CUDA or CPU.

### 8. Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | The application shall accept one configured video source at a time. |
| FR-02 | The application shall support a local video file as a source. |
| FR-03 | The application shall support a phone IP-camera video URL as a source. |
| FR-04 | The application shall report a clear error if the selected source cannot be opened. |
| FR-05 | The application shall load the configured Ultralytics YOLO weights and report the actual model path and inference device. |
| FR-06 | If YOLO cannot be imported or its weights cannot be loaded, the application shall report the full actionable error and stop; it shall not silently substitute another detector. |
| FR-07 | The application shall detect the person class only. |
| FR-08 | The application shall filter detections using a configurable confidence threshold. |
| FR-09 | The application shall display a bounding box, class label, confidence, and tracking ID for each confirmed person track. |
| FR-10 | The application shall count only currently confirmed, recently observed person tracks as visible people. |
| FR-11 | A track shall expire after a configurable number of consecutive frames without a matching detection. |
| FR-12 | The application shall display the configured entry/exit line on the video. |
| FR-13 | A person crossing the line in the configured direction shall increment the corresponding counter once. |
| FR-14 | Repeated frames on the same side of the line shall not increment the crossing counter again. |
| FR-15 | The application shall display the configured restricted-zone polygon on the video. |
| FR-16 | A zone alert shall be created only when a confirmed tracked person changes from outside the zone to inside it. |
| FR-17 | Remaining inside a zone shall not generate a new alert on every frame. |
| FR-18 | Each zone-entry alert shall contain timestamp, camera/source ID, track ID, zone ID, and snapshot path. |
| FR-19 | The application shall save one snapshot per zone-entry event. |
| FR-20 | The application shall store each alert in a local SQLite database. |
| FR-21 | The application shall display source mode, model status, actual inference device, current visible-person count, entry count, exit count, and recent alert status. |
| FR-22 | The operator shall be able to stop the application cleanly. |
| FR-23 | On source interruption, the application shall display a clear error and release resources cleanly. |
| FR-24 | The application shall check and report whether CUDA is available to the active PyTorch environment. |
| FR-25 | When CUDA is available and configured, the application shall run YOLO inference on CUDA device 0. |
| FR-26 | When CPU mode is configured or CUDA is unavailable, the application shall run YOLO inference on CPU and clearly report that device. |
| FR-27 | The application shall never report GPU inference unless the active YOLO model is actually using CUDA device 0. |
| FR-28 | The application shall not use OpenCV HOG or another detector in place of YOLO. |

### 9. Non-Functional Requirements

- The application shall run locally on the project computer.
- The application shall support Python 3.14 in the configured project environment.
- The application shall use the RTX 5050 CUDA GPU for YOLO inference when the GPU is visible to WSL and supported by the installed PyTorch CUDA build.
- CPU inference shall remain usable as a fallback.
- The application shall target at least 5 processed frames per second on the agreed test computer at the selected test resolution; performance shall be measured and reported separately for CPU and GPU when both are available.
- The application shall use configurable model confidence, image size, tracker lifetime, inference device, line, and zone settings.
- The application shall not upload video, snapshots, or event records to an external service.
- The application shall identify itself as a demonstration system and shall not claim to determine intent or criminality.
- The application shall close the camera and database cleanly after stop or error.
- The application shall fail clearly rather than silently changing detector backend or claiming GPU use when CUDA is unavailable.

### 10. Configuration Requirements

A configuration file shall define:

- Source mode: `video` or `live`.
- Video file path, when source mode is `video`.
- IP-camera URL, when source mode is `live`.
- YOLO weights path.
- Inference device: `cuda:0` / `0` or `cpu`.
- Whether automatic CUDA selection is enabled.
- Confidence threshold.
- Image size.
- Tracker settings.
- Entry/exit line coordinates and direction.
- Restricted-zone polygon coordinates.
- Snapshot directory.
- SQLite database path.

Coordinates shall be defined relative to the configured processing-frame dimensions, and the application shall document those dimensions.

### 11. Data Requirements

Each saved alert shall include:

- Unique event ID.
- Event type.
- Timestamp.
- Source ID.
- Track ID.
- Zone ID.
- Snapshot file path.
- Detection confidence.
- Inference device used for the event, if available.

Only data needed for Phase 1 operation shall be stored.

### 12. Acceptance Criteria

Phase 1 is accepted only when all applicable criteria pass:

1. A local video file opens and displays.
2. A live phone-camera stream opens and displays.
3. Startup reports the actual Ultralytics YOLO backend, actual weights path, and actual inference device.
4. A known person test clip displays at least one correctly labeled person detection.
5. A known empty-room test clip produces no person detections or intrusion alerts during the test interval; any observed false positives are documented and investigated.
6. One person entering the frame creates a confirmed track and changes the visible-person count to one.
7. A person crossing the configured line once increments exactly one direction counter once.
8. A person entering the restricted zone