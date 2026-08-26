from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io import load_camera_trajectory, load_json, save_json
from .metrics import robust_interval_validity


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


def _trajectory_series(position: np.ndarray, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = np.diff(timestamps)
    step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    speed = np.divide(step, dt, out=np.full_like(step, np.nan), where=np.isfinite(dt) & (dt > 0))
    return dt, step, speed


def _trajectory_metrics(
    position: np.ndarray,
    step: np.ndarray,
    speed: np.ndarray,
    common_valid: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    valid_speed = speed[common_valid]
    cumulative = np.concatenate(([0.0], np.cumsum(np.where(common_valid, step, 0.0))))
    plot_speed = np.where(common_valid, speed, np.nan)
    metrics = {
        "endpoint_displacement_m": float(np.linalg.norm(position[-1] - position[0])),
        "path_length_on_common_valid_intervals_m": float(np.sum(step[common_valid])),
        "raw_path_length_m": float(np.sum(step[np.isfinite(step)])),
        "median_root_speed_mps": float(np.median(valid_speed)),
        "p95_root_speed_mps": float(np.percentile(valid_speed, 95)),
        "common_valid_interval_ratio": float(np.mean(common_valid)),
    }
    return metrics, plot_speed, cumulative


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
    parameters = config.get("parameters", {})
    max_root_speed_mps = float(parameters.get("max_root_speed_mps", 8.0))
    jump_mad_factor = float(parameters.get("jump_mad_factor", 12.0))
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
        "metric_note": "Endpoint displacement, valid-step path length, and root speed are rigid-coordinate invariant. All conditions for one fixed human use the same validity mask. Top-down plots are qualitative because backend world axes are not aligned.",
        "validity_rule": {
            "max_root_speed_mps": max_root_speed_mps,
            "jump_mad_factor": jump_mad_factor,
            "support": "intersection across No ego, DROID, and MegaSAM for each fixed human",
        },
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
            dt, step, speed = _trajectory_series(position, timestamps)
            trajectories[camera_method] = {
                "position": position,
                "timestamps": timestamps,
                "dt": dt,
                "step": step,
                "speed": speed,
            }
            if reference_timestamps is None:
                reference_timestamps = timestamps

        assert reference_timestamps is not None
        no_ego_dt, no_ego_step, no_ego_speed = _trajectory_series(transl, reference_timestamps)
        trajectories = {
            "no_ego": {
                "position": transl,
                "timestamps": reference_timestamps,
                "dt": no_ego_dt,
                "step": no_ego_step,
                "speed": no_ego_speed,
            },
            **trajectories,
        }
        common_valid = np.ones(len(transl) - 1, dtype=bool)
        condition_validity = {}
        for condition, item in trajectories.items():
            robust_valid, jump_threshold = robust_interval_validity(item["step"], jump_mad_factor)
            valid = (
                robust_valid
                & np.isfinite(item["speed"])
                & np.isfinite(item["dt"])
                & (item["dt"] > 0)
                & (item["speed"] <= max_root_speed_mps)
            )
            condition_validity[condition] = {
                "independent_valid_interval_ratio": float(np.mean(valid)),
                "jump_threshold_m": jump_threshold,
            }
            common_valid &= valid
        if not np.any(common_valid):
            raise ValueError(f"{human_method}: no common valid E2a intervals")
        for item in trajectories.values():
            metrics, plot_speed, cumulative = _trajectory_metrics(
                item["position"], item["step"], item["speed"], common_valid
            )
            item["metrics"] = metrics
            item["speed"] = plot_speed
            item["cumulative_distance"] = cumulative
        report["human_sources"][human_method] = {
            "n_frames": int(len(transl)),
            "source_frame_start": int(source_frames[0]),
            "source_frame_end": int(source_frames[-1]),
            "common_valid_intervals": int(np.sum(common_valid)),
            "common_valid_interval_ratio": float(np.mean(common_valid)),
            "condition_validity": condition_validity,
            "conditions": {condition: item["metrics"] for condition, item in trajectories.items()},
        }
        save_arrays: dict[str, np.ndarray] = {"source_frames": source_frames}
        for condition, item in trajectories.items():
            save_arrays[f"{condition}__position"] = item["position"]
            save_arrays[f"{condition}__timestamps"] = item["timestamps"]
            save_arrays[f"{condition}__speed"] = item["speed"]
        save_arrays["common_valid_interval"] = common_valid
        np.savez_compressed(output_dir / f"e2a__human_{human_method}.npz", **save_arrays)
        _plot_invariant(output_dir / f"e2a__human_{human_method}__invariant.png", f"E2a fixed human: {human_method}", trajectories)
        _plot_qualitative(output_dir / f"e2a__human_{human_method}__topdown.png", f"E2a fixed human: {human_method}", trajectories)

    save_json(output_dir / "e2a_report.json", report)
    return report
