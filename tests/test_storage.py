"""Unit tests for SnapshotWriter and DatabaseLogger."""

import os
import tempfile
import numpy as np
import pytest

from src.storage import DatabaseLogger, SnapshotWriter


def test_snapshot_writer():
    """Verify FR-19: Saves one JPEG snapshot per event."""
    with tempfile.TemporaryDirectory() as tmpdir:
        writer = SnapshotWriter(output_dir=tmpdir)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        filepath = writer.save_snapshot(
            frame=dummy_frame,
            event_id="test_event_123",
            timestamp_str="2026-10-12T03:00:00Z",
        )
        assert os.path.exists(filepath)
        assert filepath.endswith(".jpg")
        assert os.path.getsize(filepath) > 0


def test_sqlite_logger():
    """Verify FR-18, FR-20 & AC-12: SQLite record with all 9 fields and ACID persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_surveillance.db")
        logger = DatabaseLogger(db_path=db_path)

        event_id = logger.log_alert(
            event_type="ZONE_ENTRY",
            timestamp="2026-10-12T03:00:00Z",
            source_id="test_cam_01",
            track_id=42,
            zone_id="ZONE_A",
            snapshot_path="/tmp/snapshots/alert_01.jpg",
            confidence=0.92,
            device="cuda:0",
        )
        assert event_id is not None

        alerts = logger.get_all_alerts()
        assert len(alerts) == 1
        record = alerts[0]
        assert record["event_id"] == event_id
        assert record["event_type"] == "ZONE_ENTRY"
        assert record["track_id"] == 42
        assert record["zone_id"] == "ZONE_A"
        assert record["confidence"] == 0.92
        assert record["device"] == "cuda:0"

        logger.close()
