"""Main surveillance application orchestrator coordinating capture, detection, tracking, analytics, and storage."""

import datetime
import logging
import signal
import sys
import time
from typing import Optional
import cv2

from src.config import AppConfig, load_config
from src.detector import PersonDetector
from src.line_crossing import LineCrossingEngine
from src.runtime import RuntimeDiagnostics
from src.storage import DatabaseLogger, SnapshotWriter
from src.tracker import PersonTracker
from src.video_source import VideoSource
from src.visualizer import FrameAnnotator
from src.zone_monitor import ZoneMonitor

logger = logging.getLogger(__name__)


class SurveillanceApp:
    """Core orchestration engine for Phase 1 Smart Surveillance System."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._running = False
        self._latest_alert_banner: Optional[str] = None
        self._latest_alert_time: float = 0.0

        # 1. Resolve runtime device truthfully
        self.device = RuntimeDiagnostics.resolve_device(
            requested_device=self.config.model.target_device,
            auto_fallback=self.config.model.auto_cuda_fallback,
        )
        self.runtime_info = RuntimeDiagnostics.get_runtime_info(
            active_device=self.device,
            weights_path=self.config.model.weights_path,
        )
        logger.info(
            f"Startup Diagnostics: Device={self.runtime_info.active_device} "
            f"({self.runtime_info.gpu_name or 'CPU'}), Model={self.runtime_info.weights_path}"
        )

        # 2. Initialize video source
        self.video_source = VideoSource(
            mode=self.config.source.mode,
            video_path=self.config.source.video_path,
            live_url=self.config.source.live_url,
            target_size=(self.config.source.processing_width, self.config.source.processing_height),
        )

        # 3. Initialize YOLO detector
        self.detector = PersonDetector(
            weights_path=self.config.model.weights_path,
            device=self.device,
            confidence_threshold=self.config.model.confidence_threshold,
        )

        # 4. Initialize Tracker
        self.tracker = PersonTracker(
            max_missed_frames=self.config.tracker.max_missed_frames,
            min_hits=self.config.tracker.min_hits,
            iou_threshold=self.config.tracker.iou_threshold,
        )

        # 5. Initialize Line Crossing Engine
        self.line_engine = LineCrossingEngine(
            start_point=self.config.line_crossing.start_point,
            end_point=self.config.line_crossing.end_point,
            entry_direction=self.config.line_crossing.entry_direction,
        )

        # 6. Initialize Restricted Zone Monitor
        self.zone_monitor = ZoneMonitor(
            polygon_points=self.config.restricted_zone.polygon_points,
            zone_id=self.config.restricted_zone.zone_id,
        )

        # 7. Initialize Storage
        self.snapshot_writer = SnapshotWriter(self.config.storage.snapshot_dir)
        self.db_logger = DatabaseLogger(self.config.storage.database_path)

        # 8. Initialize Visualizer
        dev_display = f"{self.device} ({self.runtime_info.gpu_name})" if self.runtime_info.gpu_name else self.device
        self.visualizer = FrameAnnotator(
            source_mode=self.config.source.mode,
            model_name=self.config.model.weights_path,
            device_name=dev_display,
            line_coords=(self.config.line_crossing.start_point, self.config.line_crossing.end_point),
            zone_points=self.config.restricted_zone.polygon_points,
        )

    def run(self, headless: bool = False, max_frames: Optional[int] = None):
        """Run the main surveillance processing loop."""
        self._running = True

        # Register termination signal handlers safely (only works in main thread)
        try:
            def signal_handler(sig, frame):
                logger.info("Termination signal received. Stopping application cleanly...")
                self._running = False

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
        except (ValueError, AttributeError):
            logger.debug("Signal handlers skipped (application not running in main thread).")

        # Auto-detect headless environments on Linux (e.g. Docker, CI, SSH without X11 forwarding)
        if not headless and sys.platform.startswith("linux"):
            import os
            if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
                logger.warning(
                    "No graphical display detected ($DISPLAY / $WAYLAND_DISPLAY not set). "
                    "Automatically switching to headless mode."
                )
                headless = True

        logger.info("Opening video source...")
        self.video_source.open()

        frame_index = 0
        fps_start_time = time.time()
        fps_counter = 0
        current_fps = 0.0

        try:
            while self._running:
                ret, frame = self.video_source.read_frame()
                if not ret or frame is None:
                    logger.info("End of video stream or no frame received.")
                    break

                frame_index += 1
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()

                # Step 1: Detect visible people (class 0 only)
                detections = self.detector.detect(frame)

                # Step 2: Track people across frames
                confirmed_tracks = self.tracker.update(detections)
                visible_count = self.tracker.get_visible_count()

                # Step 3: Virtual line crossing analytics
                self.line_engine.update(confirmed_tracks)

                # Step 4: Restricted zone analytics
                zone_alerts = self.zone_monitor.update(confirmed_tracks)

                # Step 5: Handle zone entry alerts
                timestamp_str = datetime.datetime.now(datetime.timezone.utc).isoformat()
                for alert in zone_alerts:
                    # Save snapshot
                    snapshot_path = self.snapshot_writer.save_snapshot(
                        frame=frame,
                        event_id=f"track_{alert.track_id}",
                        timestamp_str=timestamp_str,
                    )
                    # Insert record into SQLite
                    source_id = self.config.source.video_path if self.config.source.mode == "video" else self.config.source.live_url
                    event_id = self.db_logger.log_alert(
                        event_type="ZONE_ENTRY",
                        timestamp=timestamp_str,
                        source_id=source_id,
                        track_id=alert.track_id,
                        zone_id=alert.zone_id,
                        snapshot_path=snapshot_path,
                        confidence=alert.confidence,
                        device=self.device,
                    )
                    logger.warning(
                        f"ZONE ENTRY ALERT! Event ID: {event_id}, Track: #{alert.track_id}, "
                        f"Zone: {alert.zone_id}, Snapshot: {snapshot_path}"
                    )
                    self._latest_alert_banner = f"Zone {alert.zone_id} Intrusion: Track #{alert.track_id}"
                    self._latest_alert_time = time.time()

                # Expire alert banner after 3 seconds
                active_banner = None
                if self._latest_alert_banner and (time.time() - self._latest_alert_time < 3.0):
                    active_banner = self._latest_alert_banner

                # Step 6: Visual Annotations
                annotated_frame = self.visualizer.annotate(
                    frame=frame,
                    tracks=confirmed_tracks,
                    visible_count=visible_count,
                    entry_count=self.line_engine.entry_count,
                    exit_count=self.line_engine.exit_count,
                    recent_alert=active_banner,
                    fps=current_fps,
                )

                # Step 7: Display (if not headless)
                if not headless:
                    try:
                        cv2.imshow("Smart Surveillance System - Phase 1", annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q"):
                            logger.info("Operator pressed 'q'. Exiting cleanly.")
                            break
                    except cv2.error as e:
                        logger.warning(f"GUI display error ({e}). Automatically switching to headless mode.")
                        headless = True

                if max_frames and frame_index >= max_frames:
                    logger.info(f"Reached maximum requested frames ({max_frames}). Stopping.")
                    break

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.shutdown()

    def shutdown(self):
        """Cleanly release all hardware, camera, window, and database resources."""
        logger.info("Shutting down surveillance application...")
        self._running = False
        if self.video_source:
            self.video_source.release()
        if self.db_logger:
            self.db_logger.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("All resources successfully released.")
