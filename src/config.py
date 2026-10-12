"""Configuration loader and schema validator for Smart Surveillance System."""

import os
from dataclasses import dataclass, field
from typing import Any, List, Tuple
import yaml


@dataclass
class SourceConfig:
    mode: str
    video_path: str
    live_url: str
    processing_width: int
    processing_height: int


@dataclass
class ModelConfig:
    weights_path: str
    target_device: str
    auto_cuda_fallback: bool
    confidence_threshold: float
    image_size: int
    classes: List[int] = field(default_factory=lambda: [0])


@dataclass
class TrackerConfig:
    max_missed_frames: int
    min_hits: int
    iou_threshold: float


@dataclass
class LineCrossingConfig:
    enabled: bool
    start_point: Tuple[int, int]
    end_point: Tuple[int, int]
    entry_direction: str


@dataclass
class RestrictedZoneConfig:
    enabled: bool
    polygon_points: List[Tuple[int, int]]
    zone_id: str


@dataclass
class StorageConfig:
    snapshot_dir: str
    database_path: str


@dataclass
class AppConfig:
    source: SourceConfig
    model: ModelConfig
    tracker: TrackerConfig
    line_crossing: LineCrossingConfig
    restricted_zone: RestrictedZoneConfig
    storage: StorageConfig


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load and validate configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Validate source
    src_data = data.get("source", {})
    source_cfg = SourceConfig(
        mode=src_data.get("mode", "video"),
        video_path=src_data.get("video_path", ""),
        live_url=src_data.get("live_url", ""),
        processing_width=int(src_data.get("processing_width", 640)),
        processing_height=int(src_data.get("processing_height", 480)),
    )
    if source_cfg.mode not in ("video", "live"):
        raise ValueError(f"Invalid source mode: {source_cfg.mode}. Must be 'video' or 'live'.")

    # Validate model
    mod_data = data.get("model", {})
    classes = [int(c) for c in mod_data.get("classes", [0])]
    if classes != [0]:
        raise ValueError(f"Strict Phase 1 constraint: Only class 0 (person) is allowed. Configured: {classes}")

    model_cfg = ModelConfig(
        weights_path=mod_data.get("weights_path", "yolov8n.pt"),
        target_device=mod_data.get("target_device", "cuda:0"),
        auto_cuda_fallback=bool(mod_data.get("auto_cuda_fallback", True)),
        confidence_threshold=float(mod_data.get("confidence_threshold", 0.5)),
        image_size=int(mod_data.get("image_size", 640)),
        classes=classes,
    )

    # Validate tracker
    trk_data = data.get("tracker", {})
    tracker_cfg = TrackerConfig(
        max_missed_frames=int(trk_data.get("max_missed_frames", 30)),
        min_hits=int(trk_data.get("min_hits", 2)),
        iou_threshold=float(trk_data.get("iou_threshold", 0.3)),
    )

    # Validate line crossing
    ln_data = data.get("line_crossing", {})
    line_cfg = LineCrossingConfig(
        enabled=bool(ln_data.get("enabled", True)),
        start_point=tuple(ln_data.get("start_point", [100, 240])),
        end_point=tuple(ln_data.get("end_point", [540, 240])),
        entry_direction=ln_data.get("entry_direction", "top_to_bottom"),
    )

    # Validate restricted zone
    rz_data = data.get("restricted_zone", {})
    poly_raw = rz_data.get("polygon_points", [])
    poly_points = [tuple(p) for p in poly_raw]
    zone_cfg = RestrictedZoneConfig(
        enabled=bool(rz_data.get("enabled", True)),
        polygon_points=poly_points,
        zone_id=rz_data.get("zone_id", "ZONE_A"),
    )

    # Validate storage
    st_data = data.get("storage", {})
    storage_cfg = StorageConfig(
        snapshot_dir=st_data.get("snapshot_dir", "snapshots"),
        database_path=st_data.get("database_path", "surveillance.db"),
    )

    return AppConfig(
        source=source_cfg,
        model=model_cfg,
        tracker=tracker_cfg,
        line_crossing=line_cfg,
        restricted_zone=zone_cfg,
        storage=storage_cfg,
    )
