"""Unit tests for LineCrossingEngine: directional counting and anti-duplicate logic."""

import pytest
from src.line_crossing import LineCrossingEngine
from src.tracker import Track


def test_line_crossing_directional_and_no_duplicates():
    """Verify FR-12, FR-13, FR-14 & AC-7: Exactly 1 increment per crossing, no repeat counts on same side."""
    # Horizontal line from (100, 240) to (500, 240)
    engine = LineCrossingEngine(start_point=(100, 240), end_point=(500, 240), entry_direction="top_to_bottom")

    assert engine.entry_count == 0
    assert engine.exit_count == 0

    # Step 1: Person at y=200 (top side)
    t = Track(track_id=1, box=(280, 160, 320, 240), centroid=(300, 200), confidence=0.9, hits=2, history=[(300, 180), (300, 200)])
    new_in, new_out = engine.update([t])
    assert new_in == 0 and new_out == 0
    assert engine.entry_count == 0

    # Step 2: Person moves to y=260 (crosses line to bottom side)
    t.history.append((300, 260))
    t.centroid = (300, 260)
    new_in, new_out = engine.update([t])
    assert new_in == 1 and new_out == 0
    assert engine.entry_count == 1

    # Step 3: Person stays at y=270, 280 (repeated frames on bottom side)
    t.history.append((300, 270))
    t.centroid = (300, 270)
    new_in, new_out = engine.update([t])
    assert new_in == 0 and new_out == 0
    assert engine.entry_count == 1  # No duplicate count!

    # Step 4: Reverse crossing (from y=280 back to y=200)
    t.history.append((300, 200))
    t.centroid = (300, 200)
    new_in, new_out = engine.update([t])
    assert new_in == 0 and new_out == 1
    assert engine.exit_count == 1
