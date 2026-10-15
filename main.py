"""CLI entrypoint for Smart Surveillance System."""

import argparse
import logging
import sys

from src.config import load_config
from src.app import SurveillanceApp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("SmartSurveillance")


def main():
    parser = argparse.ArgumentParser(description="Smart Surveillance System — Phase 1")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["video", "live"],
        help="Override source mode ('video' or 'live')",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        help="Override path to local video file",
    )
    parser.add_argument(
        "--live-url",
        type=str,
        help="Override phone IP Webcam URL",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Override inference device (e.g. 'cuda:0', 'cuda', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying OpenCV window (useful for automated testing/benchmarks)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop automatically after N frames",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)

        # Apply CLI overrides if specified
        if args.source:
            config.source.mode = args.source
        if args.video_path:
            config.source.video_path = args.video_path
        if args.live_url:
            config.source.live_url = args.live_url
        if args.device:
            config.model.target_device = args.device

        app = SurveillanceApp(config)
        app.run(headless=args.headless, max_frames=args.max_frames)

    except Exception as e:
        logger.error(f"Fatal error during execution: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
