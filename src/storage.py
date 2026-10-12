"""Snapshot storage and SQLite event logging persistence module."""

import datetime
import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from src.zone_monitor import ZoneAlertEvent


class SnapshotWriter:
    """Manages saving image snapshots for alert events."""

    def __init__(self, output_dir: str = "snapshots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_snapshot(self, frame: np.ndarray, event_id: str, timestamp_str: str) -> str:
        """Save frame to disk as JPEG and return relative file path."""
        safe_time = timestamp_str.replace(":", "-").replace(".", "_")
        filename = f"alert_{event_id[:8]}_{safe_time}.jpg"
        filepath = os.path.join(self.output_dir, filename)

        success = cv2.imwrite(filepath, frame)
        if not success:
            raise IOError(f"Failed to write snapshot image to {filepath}")

        return filepath


class DatabaseLogger:
    """Manages SQLite storage for surveillance alert events."""

    def __init__(self, db_path: str = "surveillance.db"):
        self.db_path = db_path
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        """Create alerts table adhering strictly to PRD Section 11 Data Requirements."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                source_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                zone_id TEXT NOT NULL,
                snapshot_path TEXT NOT NULL,
                confidence REAL NOT NULL,
                device TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def log_alert(
        self,
        event_type: str,
        timestamp: str,
        source_id: str,
        track_id: int,
        zone_id: str,
        snapshot_path: str,
        confidence: float,
        device: str,
        event_id: Optional[str] = None,
    ) -> str:
        """Insert a single alert event record into the SQLite database."""
        if event_id is None:
            event_id = str(uuid.uuid4())

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO alerts (
                event_id, event_type, timestamp, source_id,
                track_id, zone_id, snapshot_path, confidence, device
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                event_type,
                timestamp,
                source_id,
                track_id,
                zone_id,
                snapshot_path,
                float(confidence),
                device,
            ),
        )
        self.conn.commit()
        return event_id

    def get_all_alerts(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded alerts as a list of dictionaries."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT event_id, event_type, timestamp, source_id, track_id, zone_id, snapshot_path, confidence, device FROM alerts ORDER BY timestamp ASC"
        )
        rows = cursor.fetchall()
        alerts = []
        for r in rows:
            alerts.append(
                {
                    "event_id": r[0],
                    "event_type": r[1],
                    "timestamp": r[2],
                    "source_id": r[3],
                    "track_id": r[4],
                    "zone_id": r[5],
                    "snapshot_path": r[6],
                    "confidence": r[7],
                    "device": r[8],
                }
            )
        return alerts

    def close(self):
        """Cleanly close database connection."""
        if self.conn:
            try:
                self.conn.commit()
                self.conn.close()
            except Exception:
                pass
            finally:
                self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
