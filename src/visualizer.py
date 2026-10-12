"""Frame visualizer rendering bounding boxes, IDs, HUD overlays, line, and zone."""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.tracker import Track


class FrameAnnotator:
    """Renders overlays, annotations, diagnostics HUD, line, and zone on OpenCV frames."""

    def __init__(
        self,
        source_mode: str,
        model_name: str,
        device_name: str,
        line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        zone_points: Optional[List[Tuple[int, int]]] = None,
    ):
        self.source_mode = source_mode
        self.model_name = model_name
        self.device_name = device_name
        self.line_coords = line_coords
        self.zone_points = np.array(zone_points, dtype=np.int32) if zone_points else None

    def annotate(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        visible_count: int,
        entry_count: int = 0,
        exit_count: int = 0,
        recent_alert: Optional[str] = None,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """Draw all visual annotations on a copy of the frame.
        
        Args:
            frame: Input BGR numpy image.
            tracks: Confirmed active tracks to draw.
            visible_count: Current visible person count.
            entry_count: Total line entry crossings.
            exit_count: Total line exit crossings.
            recent_alert: Text of latest alert, or None.
            fps: Current processed FPS, or None.
            
        Returns:
            Annotated BGR numpy image.
        """
        output = frame.copy()

        # 1. Draw Restricted Zone Polygon (semi-transparent red / orange fill & outline)
        if self.zone_points is not None and len(self.zone_points) >= 3:
            overlay = output.copy()
            cv2.fillPoly(overlay, [self.zone_points], (0, 0, 180))
            cv2.addWeighted(overlay, 0.25, output, 0.75, 0, output)
            cv2.polylines(output, [self.zone_points], isClosed=True, color=(0, 0, 255), thickness=2)
            # Label
            first_pt = tuple(self.zone_points[0])
            cv2.putText(output, "RESTRICTED ZONE", (first_pt[0] + 5, first_pt[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 2. Draw Entry/Exit Virtual Line (yellow line with direction marker)
        if self.line_coords is not None:
            pt1, pt2 = self.line_coords
            cv2.line(output, pt1, pt2, (0, 255, 255), 2)
            mid_x = int((pt1[0] + pt2[0]) / 2)
            mid_y = int((pt1[1] + pt2[1]) / 2)
            cv2.putText(output, "ENTRY / EXIT LINE", (mid_x - 60, mid_y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw Tracked People
        for t in tracks:
            x1, y1, x2, y2 = t.box
            color = (0, 255, 0)  # Green box for confirmed person track

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.circle(output, t.centroid, 4, (0, 0, 255), -1)

            # Label box with contrast background
            label = f"person #{t.track_id} ({t.confidence:.2f})"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - 20), (x1 + w + 6, y1), color, -1)
            cv2.putText(output, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # 4. HUD Diagnostics Header Bar
        h, w, _ = output.shape
        cv2.rectangle(output, (0, 0), (w, 54), (20, 20, 20), -1)
        cv2.line(output, (0, 54), (w, 54), (60, 60, 60), 1)

        # Left: Device & Model
        hud_left = f"Device: {self.device_name} | Model: {self.model_name} | Source: {self.source_mode.upper()}"
        cv2.putText(output, hud_left, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Right: FPS if available
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(output, fps_text, (w - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

        # Sub-row: Occupancy, Entry, Exit
        hud_metrics = (
            f"Occupancy: {visible_count}  |  "
            f"Entries: {entry_count}  |  "
            f"Exits: {exit_count}"
        )
        cv2.putText(output, hud_metrics, (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # 5. Alert Banner if triggered
        if recent_alert:
            alert_bar_y = h - 35
            cv2.rectangle(output, (0, alert_bar_y), (w, h), (0, 0, 200), -1)
            cv2.putText(output, f"ALERT: {recent_alert}", (15, alert_bar_y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return output
