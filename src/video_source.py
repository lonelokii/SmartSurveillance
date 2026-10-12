"""Unified video source handler supporting local files and live IP Webcam streams."""

import logging
import os
from typing import Generator, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SourceOpenError(Exception):
    """Raised when a video source cannot be opened."""
    pass


class SourceInterruptedError(Exception):
    """Raised when an active video stream is interrupted or ends unexpectedly."""
    pass


class VideoSource:
    """Handles frame acquisition for local video files and live IP camera streams."""

    def __init__(
        self,
        mode: str,
        video_path: Optional[str] = None,
        live_url: Optional[str] = None,
        target_size: Optional[Tuple[int, int]] = (640, 480),
    ):
        """Initialize video source.
        
        Args:
            mode: 'video' or 'live'.
            video_path: File path if mode is 'video'.
            live_url: URL (HTTP/RTSP) if mode is 'live'.
            target_size: (width, height) to resize frames to, or None to keep native.
        """
        self.mode = mode.lower()
        self.video_path = video_path
        self.live_url = live_url
        self.target_size = target_size
        self.cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0

    def open(self):
        """Open the video stream or file and validate readable frames.
        
        Raises:
            SourceOpenError: If the source cannot be opened or read.
        """
        if self.mode == "video":
            if not self.video_path or not os.path.exists(self.video_path):
                raise SourceOpenError(
                    f"Local video file not found: '{self.video_path}'. "
                    "Please verify the file path exists."
                )
            source_target = self.video_path
        elif self.mode == "live":
            if not self.live_url:
                raise SourceOpenError("Live URL is empty. Please provide a valid IP camera stream URL.")
            source_target = self.live_url
        else:
            raise SourceOpenError(f"Unsupported source mode: '{self.mode}'. Must be 'video' or 'live'.")

        logger.info(f"Opening video source [{self.mode}]: {source_target}")
        self.cap = cv2.VideoCapture(source_target)

        if not self.cap.isOpened():
            self.release()
            raise SourceOpenError(
                f"Failed to open video source: '{source_target}'. "
                "Ensure codec is supported and the network URL or file is accessible."
            )

        # Probe first frame to verify readability
        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.release()
            raise SourceOpenError(
                f"Video source '{source_target}' opened, but failed to read initial frame."
            )
        
        # Rewind if local video so caller gets frame 0
        if self.mode == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self._frame_count = 0
        logger.info(f"Video source successfully opened and validated: {source_target}")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and optionally resize next frame.
        
        Returns:
            (success: bool, frame: np.ndarray or None)
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        self._frame_count += 1
        if self.target_size is not None:
            width, height = self.target_size
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

        return True, frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator yielding processed frames until end of stream or interruption."""
        if self.cap is None or not self.cap.isOpened():
            self.open()

        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield frame

    def release(self):
        """Release underlying video capture resource cleanly."""
        if self.cap is not None:
            logger.info("Releasing video capture resource.")
            try:
                self.cap.release()
            except Exception as e:
                logger.warning(f"Error releasing video capture: {e}")
            finally:
                self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
