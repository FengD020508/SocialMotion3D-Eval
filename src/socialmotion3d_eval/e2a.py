from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_camera_trajectory, load_json, save_json


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_human(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    import torch

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    transl = _to_numpy(payload["body_params_incam"]["transl"]).astype(np.float64)
    source_frames = None
    target = payload.get("target_track")
    if isinstance(target, dict) and "source_frames" in target:
        source_frames = _to_numpy(target["source_frames"]).astype(np.int64).reshape(-1)
    if transl.ndim != 2 or transl.shape[1] != 3:
        raise ValueError(f"{path}: unexpected incam transl shape {transl.shape}")
    if source_frames is not None and len(source_frames) != len(transl):
        raise ValueError(f"{path}: source_frames and transl lengths differ")
    return transl, source_frames


def _align_camera(camera: dict[str, Any], source_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lookup = {int(frame): index for index, frame in enumerate(camera["frame_numbers"])}
    missing = [int(frame) for frame in source_frames if int(frame) not in lookup]
    if missing:
        raise ValueError(f"camera is missing {len(missing)} requested source frames; first={missing[0]}")
    indices = np.asarray([lookup[int(frame)] for frame in source_frames], dtype=np.int64)
    return camera["rotation"][indices], camera["camera_center"][indices], camera["timestamps"][indices]


def _trajectory_metrics(position: np.ndarray, timestamps: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    dt = np.diff(timestamps)
    step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=np.isfinite(dt) & (dt > 0))
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(np.isfinite(step), step, 0.0))))
    finite_speed = speed[np.isfinite(speed)]
    metrics = {
        "net_displacement_m": float(np.linalg.norm(position[-1] - position[0])),
        "path_length_m": float(np.sum(step)),
        "median_root_speed_mps": float(np.median(finite_speed)) if len(finite_speed) else float("nan"),
        "p95_root_speed_mps": float(np.percentile(finite_speed, 95)) if len(finite_speed) else float("nan"),
    }
    return metrics, speed, cumulative


def _plot_invariant(path: Path, title: str, trajectories: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for condition, item in trajectories.items():
        frame_time = item["timestamps"] - item["timestamps"][0]
        interval_time = 0.5 * (frame_time[:-1] + frame_time[1:])
        axes[0].plot(interval_time, item["speed"], label=condition)
        axes[1].plot(frame_time, item["cumulative_distance"], label=condition)
    axes[0].set_ylabel("root speed (m/s)")
    axes[1].set_ylabel("cumulative path (m)")
    axes[1].set_xlabel("clip time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(title + " (rotation/translation-invariant diagnostics)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_qualitative(path: Path, title: str, trajectories: dict[str, dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(trajectories), figsize=(5 * len(trajectories), 4.5))
    if len(trajectories) == 1:
        axes = [axes]
    for axis, (condition, item) in zip(axes, trajectories.items()):
        position = item["position"] - item["position"][0]
        axis.plot(position[:, 0], position[:, 2], linewidth=1.3)
        axis.scatter(position[0, 0], position[0, 2], marker="o", label="start")
        axis.scatter(position[-1, 0], position[-1, 2], marker="x", label="end")
        axis.set_title(condition)
        axis.set_xlabel("axis 0 (m)")
        axis.set_ylabel("axis 2 (m)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
    figure.suptitle(title + " (qualitative only; camera world axes are not cross-aligned)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_e2a(config_path: str | Path) -> dict[str, Any]:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_id = config["clip_id"]
    e3_report = load_json(config["e3_report"])
    clip_report = next(item for item in e3_report["clips"] if item["clip_id"] == clip_id)
    camera_scales = {
        method: float(result["scale_m_per_raw_unit"])
        for method, result in clip_report["methods"].items()
        if result.get("status") == "ok"
    }
    if set(config["camera_sources"]) - set(camera_scales):
        raise ValueError("E3 report does not contain valid scales for every E2a camera source")

    cameras: dict[str, dict[str, Any]] = {}
    for method, spec in config["camera_sources"].items():
        cameras[method] = load_camera_trajectory(
            spec["path"],
            source_frame_offset=int(spec.get("source_frame_offset", 0)),
            frame_numbers_are_source=bool(spec.get("frame_numbers_are_source", True)),
            fps_override=spec.get("fps"),
        )

    report: dict[str, Any] = {
        "experiment": "E2a_fixed_human_controlled_ego",
        "clip_id": clip_id,
        "design": "Each fixed GEM incam result is grounded with No ego, DROID, and MegaSAM; repeated for both GEM sources.",
        "metric_note": "Path length, net displacement, and root speed are rigid-coordinate invariant. Top-down plots are qualitative because backend world axes are not aligned.",
        "human_sources": {},
    }
    for human_method, human_spec in config["human_sources"].items():
        transl, source_frames = _load_human(human_spec["path"])
        if source_frames is None:
            reference = next(iter(cameras.values()))
            if len(reference["frame_numbers"]) != len(transl):
                raise ValueError(f"{human_method}: no source_frames and camera length does not match human length")
            source_frames = reference["frame_numbers"]

        trajectories: dict[str, dict[str, Any]] = {}
        reference_timestamps: np.ndarray | None = None
        for camera_method, camera in cameras.items():
            rotation, center, timestamps = _align_camera(camera, source_frames)
            position = np.einsum("nij,nj->ni", rotation, transl) + camera_scales[camera_method] * center
            metrics, speed, cumulative = _trajectory_metrics(position, timestamps)
            trajectories[camera_method] = {
                "position": position,
                "timestamps": timestamps,
                "speed": speed,
                "cumulative_distance": cumulative,
                "metrics": metrics,
            }
            if reference_timestamps is None:
                reference_timestamps = timestamps

        assert reference_timestamps is not None
        no_ego_metrics, no_ego_speed, no_ego_cumulative = _trajectory_metrics(transl, reference_timestamps)
        trajectories = {
            "no_ego": {
                "position": transl,
                "timestamps": reference_timestamps,
                "speed": no_ego_speed,
                "cumulative_distance": no_ego_cumulative,
                "metrics": no_ego_metrics,
            },
            **trajectories,
        }
        report["human_sources"][human_method] = {
            "n_frames": int(len(transl)),
            "source_frame_start": int(source_frames[0]),
            "source_frame_end": int(source_frames[-1]),
            "conditions": {condition: item["metrics"] for condition, item in trajectories.items()},
        }
        save_arrays: dict[str, np.ndarray] = {"source_frames": source_frames}
        for condition, item in trajectories.items():
            save_arrays[f"{condition}__position"] = item["position"]
            save_arrays[f"{condition}__timestamps"] = item["timestamps"]
            save_arrays[f"{condition}__speed"] = item["speed"]
        np.savez_compressed(output_dir / f"e2a__human_{human_method}.npz", **save_arrays)
        _plot_invariant(output_dir / f"e2a__human_{human_method}__invariant.png", f"E2a fixed human: {human_method}", trajectories)
        _plot_qualitative(output_dir / f"e2a__human_{human_method}__topdown.png", f"E2a fixed human: {human_method}", trajectories)

    save_json(output_dir / "e2a_report.json", report)
    return report

