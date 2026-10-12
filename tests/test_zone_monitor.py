"""Unit tests for ZoneMonitor: outside-to-inside transition and anti-duplicate logic."""

import pytest
from src.tracker import Track
from src.zone_monitor import ZoneMonitor


def test_zone_entry_transition_and_no_duplicates():
    """Verify FR-15, FR-16, FR-17 & AC-8, AC-9, AC-10."""
    # Polygon box (200, 200) to (400, 400)
    polygon = [(200, 200), (400, 200), (400, 400), (200, 400)]
    monitor = ZoneMonitor(polygon_points=polygon, zone_id="TEST_ZONE")

    # Step 1: Person outside polygon at (100, 100)
    t = Track(track_id=1, box=(80, 60, 120, 140), centroid=(100, 100), confidence=0.9, hits=2)
    alerts = monitor.update([t])
    assert len(alerts) == 0

    # Step 2: Person enters polygon at (250, 250) -> Exactly 1 alert
    t.centroid = (250, 250)
    alerts = monitor.update([t])
    assert len(alerts) == 1
    assert alerts[0].track_id == 1
    assert alerts[0].zone_id == "TEST_ZONE"

    # Step 3: Person stays inside polygon across 5 frames -> 0 duplicate alerts
    for offset in range(1, 6):
        t.centroid = (250 + offset, 250 + offset)
        alerts = monitor.update([t])
        assert len(alerts) == 0

    # Step 4: Person exits polygon to (150, 150) -> 0 alerts
    t.centroid = (150, 150)
    alerts = monitor.update([t])
    assert len(alerts) == 0

    # Step 5: Person re-enters polygon at (300, 300) -> Exactly 1 new alert
    t.centroid = (300, 300)
    alerts = monitor.update([t])
    assert len(alerts) == 1
    assert alerts[0].track_id == 1
