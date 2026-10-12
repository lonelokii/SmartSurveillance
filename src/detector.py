"""Ultralytics YOLO person-only detector implementation."""

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

from src.runtime import ModelLoadError, RuntimeDiagnostics

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coordinates
    confidence: float
    class_id: int
    class_name: str
    centroid: Tuple[int, int]


def load_yolo_model(weights_path: str, device: str):
    """Load the Ultralytics YOLO model truthfully and cleanly.
    
    Args:
        weights_path: Path to YOLO weights (e.g., 'yolov8n.pt').
        device: Verified target device ('cuda:0' or 'cpu').
        
    Returns:
        Loaded Ultralytics YOLO instance.
        
    Raises:
        ModelLoadError: If ultralytics cannot be imported or model fails to load.
    """
    # Enforce strictly no HOG fallback
    RuntimeDiagnostics.assert_no_hog()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ModelLoadError(
            f"Failed to import 'ultralytics': {e}. "
            "Please ensure ultralytics is installed in the active environment. "
            "Silent fallback to HOG or any other detector is strictly prohibited."
        ) from e

    try:
        logger.info(f"Loading YOLO model from '{weights_path}' onto device '{device}'...")
        model = YOLO(weights_path)
        # Verify model class mapping contains person (class 0)
        names = getattr(model, "names", {})
        if names.get(0) != "person":
            logger.warning(f"Class 0 name is '{names.get(0)}', expected 'person'.")
        
        logger.info(f"Successfully loaded YOLO model: {weights_path} on {device}")
        return model
    except Exception as e:
        raise ModelLoadError(
            f"Failed to load YOLO model weights from '{weights_path}': {e}. "
            "Zero silent fallbacks allowed. Halting execution."
        ) from e


class PersonDetector:
    """YOLO detector strictly configured for person-only detection."""

    def __init__(self, weights_path: str, device: str, confidence_threshold: float = 0.5):
        self.weights_path = weights_path
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        self.model = load_yolo_model(weights_path, device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect visible people in the given frame.
        
        Args:
            frame: BGR numpy image frame.
            
        Returns:
            List of Detection objects filtered strictly to class 0 ('person') and confidence >= threshold.
        """
        if frame is None or frame.size == 0:
            return []

        # Predict with classes=[0] to enforce person detection at model inference level
        results = self.model.predict(
            source=frame,
            classes=[0],
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = first_res.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            # Defense-in-depth: Assert class ID is strictly 0 (person)
            if cls_id != 0:
                continue

            if conf < self.confidence_threshold:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            detections.append(
                Detection(
                    box=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=0,
                    class_name="person",
                    centroid=(cx, cy),
                )
            )

        return detections
