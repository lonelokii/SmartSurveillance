"""Unit tests for VideoSource: local file, connection error handling, and clean release."""

import os
import cv2
import numpy as np
import pytest

from src.video_source import VideoSource, SourceOpenError


def test_video_source_missing_file():
    """Verify FR-04: Non-existent file raises clear actionable SourceOpenError."""
    source = VideoSource(mode="video", video_path="non_existent_video_path.mp4")
    with pytest.raises(SourceOpenError) as exc_info:
        source.open()
    assert "Local video file not found" in str(exc_info.value)


def test_video_source_invalid_mode():
    source = VideoSource(mode="invalid_mode")
    with pytest.raises(SourceOpenError) as exc_info:
        source.open()
    assert "Unsupported source mode" in str(exc_info.value)


def test_video_source_read_frames(tmp_path):
    """Verify FR-01, FR-02 & AC-1: Local video opens, reads frames, and resizes."""
    test_video_path = str(tmp_path / "dummy_test.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(test_video_path, fourcc, 10.0, (320, 240))
    for _ in range(5):
        frame = np.full((240, 320, 3), 128, dtype=np.uint8)
        out.write(frame)
    out.release()

    with VideoSource(mode="video", video_path=test_video_path, target_size=(640, 480)) as src:
        frames_read = 0
        for f in src.frames():
            assert f.shape == (480, 640, 3)
            frames_read += 1
        assert frames_read == 5
