"""Unit tests for PersonTracker: ID assignment, confirmation, and track expiration."""

import pytest
from src.detector import Detection
from src.tracker import PersonTracker, compute_iou


def test_compute_iou():
    box1 = (0, 0, 10, 10)
    box2 = (0, 0, 10, 10)
    assert compute_iou(box1, box2) == 1.0

    box3 = (20, 20, 30, 30)
    assert compute_iou(box1, box3) == 0.0

    box4 = (0, 0, 10, 5)  # 50 area, overlap 50, union 100
    assert compute_iou(box1, box4) == 0.5


def test_track_confirmation_and_occupancy():
    """Verify FR-09, FR-10 & AC-6: A person entering frame creates confirmed track and visible count updates to 1."""
    tracker = PersonTracker(max_missed_frames=10, min_hits=2)

    det_frame1 = [Detection(box=(100, 100, 150, 200), confidence=0.85, class_id=0, class_name="person", centroid=(125, 150))]
    # Frame 1: Tentative (hits=1)
    confirmed_tracks = tracker.update(det_frame1)
    assert len(confirmed_tracks) == 0
    assert tracker.get_visible_count() == 0

    # Frame 2: Confirmed (hits=2)
    det_frame2 = [Detection(box=(102, 101, 152, 201), confidence=0.88, class_id=0, class_name="person", centroid=(127, 151))]
    confirmed_tracks = tracker.update(det_frame2)
    assert len(confirmed_tracks) == 1
    assert confirmed_tracks[0].track_id == 1
    assert tracker.get_visible_count() == 1


def test_track_expiration():
    """Verify FR-11 & AC-11: Track expires after max_missed_frames and visible count returns to 0."""
    tracker = PersonTracker(max_missed_frames=5, min_hits=2)

    # Establish confirmed track
    for _ in range(3):
        tracker.update([Detection(box=(100, 100, 150, 200), confidence=0.9, class_id=0, class_name="person", centroid=(125, 150))])
    
    assert tracker.get_visible_count() == 1

    # Simulate missed frames
    for i in range(5):
        tracker.update([])
        # After 4 missed frames, it's inactive from visible count
        if i >= 4:
            assert tracker.get_visible_count() == 0

    # One more frame exceeds max_missed_frames (6 > 5)
    tracker.update([])
    assert len(tracker.tracks) == 0
    assert tracker.get_visible_count() == 0
