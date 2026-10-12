"""Virtual line crossing engine detecting directional entry and exit transitions."""

from typing import Dict, List, Optional, Tuple
import numpy as np

from src.tracker import Track


def ccw(A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int]) -> bool:
    """Check if three points are listed in counter-clockwise order."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    A: Tuple[int, int], B: Tuple[int, int], C: Tuple[int, int], D: Tuple[int, int]
) -> bool:
    """Return True if line segment AB intersects line segment CD."""
    return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D))


class LineCrossingEngine:
    """Detects directional crossings of a configured line segment."""

    def __init__(
        self,
        start_point: Tuple[int, int],
        end_point: Tuple[int, int],
        entry_direction: str = "top_to_bottom",
    ):
        """Initialize line crossing engine.
        
        Args:
            start_point: (x, y) start of the virtual line.
            end_point: (x, y) end of the virtual line.
            entry_direction: 'top_to_bottom', 'bottom_to_top', 'left_to_right', 'right_to_left'.
        """
        self.p1 = start_point
        self.p2 = end_point
        self.entry_direction = entry_direction.lower()
        self.entry_count = 0
        self.exit_count = 0
        # Track history of side state per track ID: {track_id: side_sign (-1, 0, 1)}
        self._last_side: Dict[int, int] = {}
        # Crossed tracks cache to prevent double-counting across consecutive frames
        self._crossed_pairs: set[Tuple[int, int]] = set()

    def _get_side(self, pt: Tuple[int, int]) -> int:
        """Compute the signed side of pt relative to directed line p1 -> p2."""
        val = (self.p2[0] - self.p1[0]) * (pt[1] - self.p1[1]) - (self.p2[1] - self.p1[1]) * (pt[0] - self.p1[0])
        if val > 0:
            return 1
        elif val < 0:
            return -1
        return 0

    def update(self, tracks: List[Track]) -> Tuple[int, int]:
        """Process active tracks and update entry / exit counters.
        
        Args:
            tracks: List of confirmed active tracks.
            
        Returns:
            Tuple of (new_entries_this_frame, new_exits_this_frame).
        """
        new_entries = 0
        new_exits = 0

        active_ids = {t.track_id for t in tracks}
        # Clean up stale tracks from side cache
        stale_ids = set(self._last_side.keys()) - active_ids
        for sid in stale_ids:
            del self._last_side[sid]

        for t in tracks:
            curr_pt = t.centroid
            curr_side = self._get_side(curr_pt)

            if len(t.history) >= 2:
                prev_pt = t.history[-2]
                prev_side = self._last_side.get(t.track_id, self._get_side(prev_pt))

                # Check if segment between prev_pt and curr_pt intersects the virtual line
                if segments_intersect(prev_pt, curr_pt, self.p1, self.p2) and prev_side != 0 and curr_side != 0 and prev_side != curr_side:
                    pair_key = (t.track_id, prev_side, curr_side)
                    if pair_key not in self._crossed_pairs:
                        self._crossed_pairs.add(pair_key)
                        
                        # Determine direction based on side transition
                        if self.entry_direction in ("top_to_bottom", "left_to_right"):
                            if prev_side < 0 and curr_side > 0:
                                self.entry_count += 1
                                new_entries += 1
                            else:
                                self.exit_count += 1
                                new_exits += 1
                        else:
                            if prev_side > 0 and curr_side < 0:
                                self.entry_count += 1
                                new_entries += 1
                            else:
                                self.exit_count += 1
                                new_exits += 1

            self._last_side[t.track_id] = curr_side

        return new_entries, new_exits
