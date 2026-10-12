"""Empirical media pipeline tests for empty-room and person detection scenarios."""

import json
import os
import pytest

from src.detector import PersonDetector
from src.runtime import RuntimeDiagnostics
from src.video_source import VideoSource
from scripts.create_test_media import main as generate_test_media


@pytest.fixture(scope="session", autouse=True)
def setup_test_media():
    """Ensure test media clips exist before running tests."""
    data_dir = "tests/data"
    empty_path = os.path.join(data_dir, "empty_room_test.mp4")
    person_path = os.path.join(data_dir, "one_person_walk.mp4")
    if not os.path.exists(empty_path) or not os.path.exists(person_path):
        generate_test_media()


def test_empty_room_clip_verification():
    """Verify FR-07, FR-10 & AC-5 adhering strictly to the Empty-Room Test Rule."""
    empty_clip_path = "tests/data/empty_room_test.mp4"
    assert os.path.exists(empty_clip_path), "Empty room test clip must exist."

    target_dev = RuntimeDiagnostics.resolve_device("cuda:0", auto_fallback=True)
    detector = PersonDetector(weights_path="yolov8n.pt", device=target_dev, confidence_threshold=0.5)

    false_detections = []
    frame_idx = 0

    with VideoSource(mode="video", video_path=empty_clip_path) as source:
        for frame in source.frames():
            frame_idx += 1
            dets = detector.detect(frame)
            if dets:
                for d in dets:
                    false_detections.append({
                        "frame": frame_idx,
                        "box": d.box,
                        "confidence": d.confidence,
                        "class_name": d.class_name,
                    })

    # Save diagnostic report
    os.makedirs("test_reports", exist_ok=True)
    report_path = "test_reports/empty_room_diagnostics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "clip": empty_clip_path,
            "total_frames_evaluated": frame_idx,
            "false_detections_count": len(false_detections),
            "false_detections": false_detections,
        }, f, indent=2)

    # Empty room clip has no people -> Expected: 0 false detections
    assert len(false_detections) == 0, (
        f"Observed {len(false_detections)} false positive detections on empty room clip. "
        f"Details logged to {report_path}."
    )
