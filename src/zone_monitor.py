"""Restricted-zone polygon monitor detecting outside-to-inside transitions."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.tracker import Track


@dataclass
class ZoneAlertEvent:
    track_id: int
    zone_id: str
    centroid: Tuple[int, int]
    box: Tuple[int, int, int, int]
    confidence: float


class ZoneMonitor:
    """Monitors configured polygon zone for strictly outside-to-inside entry transitions."""

    def __init__(self, polygon_points: List[Tuple[int, int]], zone_id: str = "ZONE_A"):
        """Initialize zone monitor.
        
        Args:
            polygon_points: List of (x, y) vertices defining the polygon.
            zone_id: Identifier for the monitored zone.
        """
        self.polygon = np.array(polygon_points, dtype=np.int32)
        self.zone_id = zone_id
        # State tracking per track: {track_id: is_inside: bool}
        self._inside_state: Dict[int, bool] = {}

    def is_point_inside(self, pt: Tuple[int, int]) -> bool:
        """Check whether point (x, y) lies inside the configured polygon."""
        if len(self.polygon) < 3:
            return False
        # pointPolygonTest returns > 0 (inside), 0 (on edge), < 0 (outside)
        res = cv2.pointPolygonTest(self.polygon, (float(pt[0]), float(pt[1])), measureDist=False)
        return res >= 0

    def update(self, tracks: List[Track]) -> List[ZoneAlertEvent]:
        """Evaluate active tracks and generate events on outside-to-inside transitions.
        
        Args:
            tracks: List of confirmed active tracks.
            
        Returns:
            List of new ZoneAlertEvent objects created during this frame.
        """
        alerts: List[ZoneAlertEvent] = []
        active_ids = {t.track_id for t in tracks}

        # Purge stale track IDs that are no longer active
        stale_ids = set(self._inside_state.keys()) - active_ids
        for sid in stale_ids:
            del self._inside_state[sid]

        for t in tracks:
            currently_inside = self.is_point_inside(t.centroid)
            was_inside = self._inside_state.get(t.track_id, False)

            # Strict transition logic:
            # Alert ONLY on outside -> inside (False -> True)
            if currently_inside and not was_inside:
                alerts.append(
                    ZoneAlertEvent(
                        track_id=t.track_id,
                        zone_id=self.zone_id,
                        centroid=t.centroid,
                        box=t.box,
                        confidence=t.confidence,
                    )
                )

            # Update state for next frame
            self._inside_state[t.track_id] = currently_inside

        return alerts
