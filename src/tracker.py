"""Centroid and IoU-based Person Tracker maintaining confirmed temporary IDs."""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np

from src.detector import Detection


@dataclass
class Track:
    track_id: int
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    centroid: Tuple[int, int]       # (cx, cy)
    confidence: float
    hits: int = 1
    missed_frames: int = 0
    history: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def is_confirmed(self) -> bool:
        """A track is confirmed if observed for at least min_hits consecutive hits."""
        return self.hits >= 2


def compute_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0.0


class PersonTracker:
    """Lightweight person tracker tracking temporary IDs across frames."""

    def __init__(self, max_missed_frames: int = 30, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_missed_frames = int(max_missed_frames)
        self.min_hits = int(min_hits)
        self.iou_threshold = float(iou_threshold)
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """Update tracks with detections from the current frame.
        
        Args:
            detections: List of Detection objects (class 0 'person' only).
            
        Returns:
            List of confirmed active Track objects.
        """
        # Match detections to existing tracks using IoU & centroid distance
        matched_tracks = set()
        matched_detections = set()

        if self.tracks and detections:
            track_ids = list(self.tracks.keys())
            iou_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = compute_iou(self.tracks[tid].box, det.box)

            # Greedy matching by highest IoU above threshold
            while True:
                max_iou = np.max(iou_matrix) if iou_matrix.size > 0 else 0.0
                if max_iou < self.iou_threshold:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = track_ids[i]
                det = detections[j]

                # Update matched track
                track = self.tracks[tid]
                track.box = det.box
                track.centroid = det.centroid
                track.confidence = det.confidence
                track.hits += 1
                track.missed_frames = 0
                track.history.append(det.centroid)
                if len(track.history) > 60:
                    track.history.pop(0)

                matched_tracks.add(tid)
                matched_detections.add(j)

                # Set row and col to -1 to avoid duplicate assignment
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

        # Handle unmatched detections -> create new tentative tracks
        for j, det in enumerate(detections):
            if j not in matched_detections:
                new_track = Track(
                    track_id=self._next_id,
                    box=det.box,
                    centroid=det.centroid,
                    confidence=det.confidence,
                    hits=1,
                    missed_frames=0,
                    history=[det.centroid],
                )
                self.tracks[self._next_id] = new_track
                self._next_id += 1

        # Handle unmatched existing tracks -> increment missed_frames
        tracks_to_delete = []
        for tid, track in self.tracks.items():
            if tid not in matched_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed_frames:
                    tracks_to_delete.append(tid)

        # Expire dead tracks
        for tid in tracks_to_delete:
            del self.tracks[tid]

        # Return only confirmed tracks that were detected or recently active
        confirmed_tracks = [
            t for t in self.tracks.values()
            if t.hits >= self.min_hits and t.missed_frames <= 3
        ]
        return confirmed_tracks

    def get_visible_count(self) -> int:
        """Return the number of confirmed active visible person tracks."""
        return len([t for t in self.tracks.values() if t.hits >= self.min_hits and t.missed_frames <= 3])
