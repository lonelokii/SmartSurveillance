"""Automated tests for YOLO model loading, runtime truthfulness, and zero-HOG enforcement."""

import os
import pytest
import numpy as np

from src.runtime import RuntimeDiagnostics, DeviceConfigurationError, ModelLoadError
from src.detector import load_yolo_model, PersonDetector


def test_cuda_status_reporting():
    """Verify that CUDA availability is reported truthfully."""
    cuda_avail, count, name, functional, diag = RuntimeDiagnostics.get_cuda_status()
    assert isinstance(cuda_avail, bool)
    assert isinstance(count, int)
    assert isinstance(functional, bool)
    if cuda_avail:
        assert count > 0
        assert name is not None
        assert "RTX" in name or "GeForce" in name or len(name) > 0
    else:
        assert count == 0
        assert name is None


def test_device_resolution_cpu():
    """Verify that 'cpu' request resolves to CPU truthfully."""
    dev = RuntimeDiagnostics.resolve_device("cpu")
    assert dev == "cpu"


def test_device_resolution_cuda():
    """Verify device resolution behavior for CUDA request."""
    cuda_avail, _, _, functional, _ = RuntimeDiagnostics.get_cuda_status()
    if cuda_avail and functional:
        dev = RuntimeDiagnostics.resolve_device("cuda:0", auto_fallback=False)
        assert dev == "cuda:0"
    else:
        # If CUDA is unavailable or kernels not functional, auto_fallback returns cpu
        dev = RuntimeDiagnostics.resolve_device("cuda:0", auto_fallback=True)
        assert dev == "cpu"
        # And disabling auto_fallback raises DeviceConfigurationError
        with pytest.raises(DeviceConfigurationError):
            RuntimeDiagnostics.resolve_device("cuda:0", auto_fallback=False)


def test_falsified_device_reporting_guard():
    """Verify that claiming CUDA when unavailable is rejected."""
    cuda_avail, _, _, functional, _ = RuntimeDiagnostics.get_cuda_status()
    if not (cuda_avail and functional):
        with pytest.raises(DeviceConfigurationError):
            RuntimeDiagnostics.get_runtime_info(active_device="cuda:0", weights_path="yolov8n.pt")


def test_invalid_weights_load_fails_cleanly():
    """Verify FR-06: Corrupted or missing weights raise actionable ModelLoadError with zero HOG fallback."""
    with pytest.raises(ModelLoadError) as exc_info:
        load_yolo_model(weights_path="non_existent_corrupted_weights.pt", device="cpu")
    assert "Failed to load YOLO model weights" in str(exc_info.value)
    assert "Zero silent fallbacks allowed" in str(exc_info.value)


def test_valid_yolo_model_load_and_predict():
    """Verify FR-05 & Criterion 3: Valid model loads, predicts on dummy frame, and filters to person."""
    # Using default pretrained weights yolov8n.pt
    target_device = RuntimeDiagnostics.resolve_device("cuda:0", auto_fallback=True)
    
    detector = PersonDetector(
        weights_path="yolov8n.pt",
        device=target_device,
        confidence_threshold=0.5,
    )
    assert detector.model is not None
    assert detector.device == target_device

    # Smoke test on blank frame (should detect 0 people)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    assert isinstance(detections, list)
    assert len(detections) == 0  # Blank frame has no people
