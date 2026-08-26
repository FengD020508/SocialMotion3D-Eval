from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def _scalar(value: np.ndarray | float | int, default: float) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def load_camera_trajectory(
    path: str | Path,
    *,
    source_frame_offset: int = 0,
    frame_numbers_are_source: bool = False,
    fps_override: float | None = None,
) -> dict[str, Any]:
    """Load the common fields emitted by the DROID and MegaSAM exporters."""
    with np.load(Path(path), allow_pickle=False) as data:
        keys = set(data.files)
        if "camera_center" in keys:
            center = np.asarray(data["camera_center"], dtype=np.float64)
        elif "T_c2w" in keys:
            center = np.asarray(data["T_c2w"], dtype=np.float64)[:, :3, 3]
        else:
            raise KeyError(f"{path}: missing camera_center/T_c2w")

        if "R_c2w" in keys:
            rotation = np.asarray(data["R_c2w"], dtype=np.float64)
        elif "T_c2w" in keys:
            rotation = np.asarray(data["T_c2w"], dtype=np.float64)[:, :3, :3]
        else:
            raise KeyError(f"{path}: missing R_c2w/T_c2w")

        n_frames = len(center)
        frame_numbers = (
            np.asarray(data["frame_numbers"], dtype=np.int64).reshape(-1)
            if "frame_numbers" in keys
            else np.arange(n_frames, dtype=np.int64)
        )
        if not frame_numbers_are_source:
            frame_numbers = frame_numbers + int(source_frame_offset)

        fps = float(fps_override) if fps_override else _scalar(data["fps"], 30.0) if "fps" in keys else 30.0
        if "timestamps" in keys:
            timestamps = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
            if len(timestamps) != n_frames or not np.all(np.isfinite(timestamps)):
                timestamps = np.arange(n_frames, dtype=np.float64) / fps
        else:
            timestamps = np.arange(n_frames, dtype=np.float64) / fps

        tracking_failed = None
        if "tracking_failed" in keys:
            tracking_failed = np.asarray(data["tracking_failed"]).astype(bool).reshape(-1)

    if center.shape != (n_frames, 3):
        raise ValueError(f"{path}: unexpected camera_center shape {center.shape}")
    if rotation.shape != (n_frames, 3, 3):
        raise ValueError(f"{path}: unexpected rotation shape {rotation.shape}")
    if len(frame_numbers) != n_frames or len(timestamps) != n_frames:
        raise ValueError(f"{path}: inconsistent frame-array lengths")
    if tracking_failed is not None and len(tracking_failed) != n_frames:
        tracking_failed = None

    return {
        "path": str(Path(path)),
        "camera_center": center,
        "rotation": rotation,
        "frame_numbers": frame_numbers,
        "timestamps": timestamps,
        "fps": fps,
        "tracking_failed": tracking_failed,
    }


def load_obd_speed_kmh(path: str | Path) -> dict[int, float]:
    """Read IDD-PeD OBD_speed values keyed by source-video frame id."""
    root = ET.parse(Path(path)).getroot()
    speeds: dict[int, float] = {}
    for element in root.iter():
        attributes = element.attrib
        if "id" not in attributes or "OBD_speed" not in attributes:
            continue
        try:
            frame_id = int(float(attributes["id"]))
            speed = float(attributes["OBD_speed"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(speed) and speed >= 0.0:
            speeds[frame_id] = speed
    if not speeds:
        raise ValueError(f"{path}: no valid OBD_speed records")
    return speeds


def align_obd_speed_mps(obd_by_frame_kmh: dict[int, float], frame_numbers: np.ndarray) -> np.ndarray:
    values = np.full(len(frame_numbers), np.nan, dtype=np.float64)
    for index, frame in enumerate(np.asarray(frame_numbers, dtype=np.int64)):
        speed = obd_by_frame_kmh.get(int(frame))
        if speed is not None:
            values[index] = speed / 3.6
    return values

