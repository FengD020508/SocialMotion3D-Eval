from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .io import align_obd_speed_mps, load_camera_trajectory, load_json, load_obd_speed_kmh, save_json
from .metrics import build_motion_series, evaluate_scaled_series


def _plot_clip(path: Path, clip_id: str, series_by_method: dict[str, dict[str, Any]], arrays_by_method: dict[str, dict[str, np.ndarray]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(series_by_method), 1, figsize=(11, 3.3 * len(series_by_method)), sharex=True)
    if len(series_by_method) == 1:
        axes = [axes]
    for axis, (method, series) in zip(axes, series_by_method.items()):
        arrays = arrays_by_method[method]
        time = 0.5 * (series["timestamps"][:-1] + series["timestamps"][1:])
        time = time - time[0]
        axis.plot(time, arrays["obd_speed_mps"], label="OBD", color="black", linewidth=1.5)
        axis.plot(time, arrays["prediction_speed_mps"], label=method, linewidth=1.2)
        split_locations = time[arrays["calibration_mask"]]
        if len(split_locations):
            axis.axvline(split_locations[-1], color="gray", linestyle="--", linewidth=1, label="calibration end")
        axis.set_ylabel("speed (m/s)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("clip time (s)")
    figure.suptitle(f"E3 held-out speed comparison: {clip_id}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _flat_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in report["clips"]:
        for method, result in clip.get("methods", {}).items():
            if result.get("status") != "ok":
                rows.append({"clip_id": clip["clip_id"], "method": method, "status": "error", "error": result.get("error")})
                continue
            accuracy = result["accuracy"]
            validity = result["validity"]
            rotation = result["rotation_stability"]
            row = {
                "clip_id": clip["clip_id"],
                "method": method,
                "status": "ok",
                "scale_m_per_raw_unit": result["scale_m_per_raw_unit"],
                "scale_estimator": result["scale_calibration"]["estimator"],
                "scale_calibration_samples": result["scale_calibration"]["n_samples"],
                "scale_calibration_iqr_raw_units": result["scale_calibration"]["scale_ratio_iqr_raw_units"],
                "scale_ols_diagnostic": result["scale_calibration"]["ols_scale_diagnostic"],
                "mae_mps": accuracy["mae_mps"],
                "rmse_mps": accuracy["rmse_mps"],
                "pearson_r": accuracy["pearson_r"],
                "frame_valid_ratio": validity["frame_valid_ratio"],
                "interval_valid_ratio": validity["interval_valid_ratio"],
                "common_interval_valid_ratio": validity["common_interval_valid_ratio"],
                "backend_reported_failed_ratio": validity["backend_reported_failed_ratio"],
                "rotation_step_p95_deg": rotation["relative_rotation_step_deg"]["p95"],
                "angular_speed_p95_deg_s": rotation["angular_speed_deg_s"]["p95"],
                "angular_acceleration_p95_deg_s2": rotation["angular_acceleration_deg_s2"]["p95"],
                "tilt_from_initial_max_deg": rotation["tilt_from_initial_deg"]["max"],
                "yaw_net_deg": rotation["yaw_deg"]["net"],
                "yaw_range_deg": rotation["yaw_deg"]["range"],
            }
            for window, metrics in result["wrde"].items():
                row[f"wrde_{window}_mean"] = metrics["mean"]
                row[f"wrde_{window}_n"] = metrics["n_windows"]
            for window, metrics in result["window_scale_stability"].items():
                row[f"scale_{window}_median"] = metrics["median"]
                row[f"scale_{window}_iqr"] = metrics["iqr"]
                row[f"scale_{window}_cv"] = metrics["cv"]
            rows.append(row)
    return rows


def _aggregate(report: dict[str, Any]) -> dict[str, Any]:
    methods = sorted({method for clip in report["clips"] for method in clip.get("methods", {})})
    output: dict[str, Any] = {"methods": {}, "paired_megasam_minus_droid": {}}
    for method in methods:
        results = [
            clip["methods"][method]
            for clip in report["clips"]
            if method in clip.get("methods", {}) and clip["methods"][method].get("status") == "ok"
        ]
        if not results:
            continue
        output["methods"][method] = {
            "n_clips": len(results),
            "mae_mps_mean": float(np.mean([item["accuracy"]["mae_mps"] for item in results])),
            "mae_mps_median": float(np.median([item["accuracy"]["mae_mps"] for item in results])),
            "rmse_mps_mean": float(np.mean([item["accuracy"]["rmse_mps"] for item in results])),
            "rmse_mps_median": float(np.median([item["accuracy"]["rmse_mps"] for item in results])),
            "interval_valid_ratio_mean": float(np.mean([item["validity"]["interval_valid_ratio"] for item in results])),
            "rotation_step_p95_deg_median": float(
                np.median([item["rotation_stability"]["relative_rotation_step_deg"]["p95"] for item in results])
            ),
            "angular_speed_p95_deg_s_median": float(
                np.median([item["rotation_stability"]["angular_speed_deg_s"]["p95"] for item in results])
            ),
            "angular_acceleration_p95_deg_s2_median": float(
                np.median([item["rotation_stability"]["angular_acceleration_deg_s2"]["p95"] for item in results])
            ),
            "tilt_from_initial_max_deg_median": float(
                np.median([item["rotation_stability"]["tilt_from_initial_deg"]["max"] for item in results])
            ),
        }
        pearson = [item["accuracy"]["pearson_r"] for item in results if item["accuracy"]["pearson_r"] is not None]
        output["methods"][method]["pearson_r_mean"] = float(np.mean(pearson)) if pearson else None

    paired = []
    for clip in report["clips"]:
        droid = clip.get("methods", {}).get("droid", {})
        megasam = clip.get("methods", {}).get("megasam", {})
        if droid.get("status") == "ok" and megasam.get("status") == "ok":
            paired.append((droid, megasam))
    if paired:
        output["paired_megasam_minus_droid"] = {
            "n_clips": len(paired),
            "mae_mps_mean_delta": float(np.mean([m["accuracy"]["mae_mps"] - d["accuracy"]["mae_mps"] for d, m in paired])),
            "mae_mps_median_delta": float(np.median([m["accuracy"]["mae_mps"] - d["accuracy"]["mae_mps"] for d, m in paired])),
            "rmse_mps_mean_delta": float(np.mean([m["accuracy"]["rmse_mps"] - d["accuracy"]["rmse_mps"] for d, m in paired])),
            "rmse_mps_median_delta": float(np.median([m["accuracy"]["rmse_mps"] - d["accuracy"]["rmse_mps"] for d, m in paired])),
            "interval_valid_ratio_mean_delta": float(
                np.mean([m["validity"]["interval_valid_ratio"] - d["validity"]["interval_valid_ratio"] for d, m in paired])
            ),
            "rotation_step_p95_deg_median_delta": float(
                np.median(
                    [
                        m["rotation_stability"]["relative_rotation_step_deg"]["p95"]
                        - d["rotation_stability"]["relative_rotation_step_deg"]["p95"]
                        for d, m in paired
                    ]
                )
            ),
            "tilt_from_initial_max_deg_median_delta": float(
                np.median(
                    [
                        m["rotation_stability"]["tilt_from_initial_deg"]["max"]
                        - d["rotation_stability"]["tilt_from_initial_deg"]["max"]
                        for d, m in paired
                    ]
                )
            ),
            "interpretation": "descriptive pilot delta only; no significance claim",
        }
    return output


def run_e3(config_path: str | Path) -> dict[str, Any]:
    config = load_json(config_path)
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = config.get("parameters", {})
    smooth_window = int(parameters.get("smooth_window_frames", 5))
    jump_mad_factor = float(parameters.get("jump_mad_factor", 12.0))
    orthogonality_tolerance = float(parameters.get("orthogonality_tolerance", 0.1))
    determinant_tolerance = float(parameters.get("determinant_tolerance", 0.05))
    calibration_fraction = float(parameters.get("calibration_fraction", 0.4))
    windows_seconds = [float(value) for value in parameters.get("wrde_windows_seconds", [1, 2, 3])]
    min_target_distance = float(parameters.get("min_target_distance_m", 0.5))
    min_calibration_speed = float(parameters.get("min_calibration_speed_mps", 0.5))

    report: dict[str, Any] = {
        "experiment": "E3_camera_speed_vs_obd",
        "protocol": {
            "scale_calibration": f"first {calibration_fraction:.0%} of intervals",
            "evaluation": f"remaining {1.0 - calibration_fraction:.0%}",
            "accuracy_support": "intersection of recomputed DROID/MegaSAM validity and finite OBD",
            "backend_failure_flags": "diagnostic only; not used to define validity",
            "rotation_stability": "relative rotation, angular speed/acceleration, tilt and yaw are diagnostics only because no rotation ground truth is available",
            "direction_boundary": "OBD speed is scalar: it validates scale/speed magnitude, not world-axis sign. Direction is audited structurally and E2a quantitative metrics are rigid-coordinate invariant.",
            "parameters": parameters,
        },
        "clips": [],
    }

    for clip in config["clips"]:
        clip_id = clip["clip_id"]
        clip_report: dict[str, Any] = {"clip_id": clip_id, "methods": {}}
        series_by_method: dict[str, dict[str, Any]] = {}
        obd = load_obd_speed_kmh(clip["obd_xml"])
        expected_frames: np.ndarray | None = None
        for method, camera_spec in clip["cameras"].items():
            try:
                camera = load_camera_trajectory(
                    camera_spec["path"],
                    source_frame_offset=int(camera_spec.get("source_frame_offset", clip.get("source_frame_offset", 0))),
                    frame_numbers_are_source=bool(camera_spec.get("frame_numbers_are_source", False)),
                    fps_override=camera_spec.get("fps"),
                )
                if expected_frames is None:
                    expected_frames = camera["frame_numbers"]
                elif not np.array_equal(expected_frames, camera["frame_numbers"]):
                    raise ValueError("DROID and MegaSAM frame_numbers are not identical; explicit re-alignment is required")
                obd_speed = align_obd_speed_mps(obd, camera["frame_numbers"])
                series_by_method[method] = build_motion_series(
                    camera,
                    obd_speed,
                    smooth_window_frames=smooth_window,
                    jump_mad_factor=jump_mad_factor,
                    orthogonality_tolerance=orthogonality_tolerance,
                    determinant_tolerance=determinant_tolerance,
                )
            except Exception as exc:  # Keep the remaining pilot clips auditable.
                clip_report["methods"][method] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

        if len(series_by_method) == len(clip["cameras"]):
            reference_series = next(iter(series_by_method.values()))
            clip_report["source_frame_range_inclusive"] = [
                int(reference_series["frame_numbers"][0]),
                int(reference_series["frame_numbers"][-1]),
            ]
            clip_report["obd_interval_coverage_ratio"] = float(np.mean(np.isfinite(reference_series["obd_speed"])))
            common_mask = np.ones(len(next(iter(series_by_method.values()))["dt"]), dtype=bool)
            for series in series_by_method.values():
                common_mask &= series["interval_valid"]
                common_mask &= np.isfinite(series["obd_speed"])
                common_mask &= np.isfinite(series["visual_speed"])
            arrays_by_method: dict[str, dict[str, np.ndarray]] = {}
            for method, series in series_by_method.items():
                try:
                    method_report, arrays = evaluate_scaled_series(
                        series,
                        common_mask=common_mask,
                        calibration_fraction=calibration_fraction,
                        fps=float(clip.get("fps", 30.0)),
                        windows_seconds=windows_seconds,
                        min_target_distance_m=min_target_distance,
                        min_calibration_speed_mps=min_calibration_speed,
                    )
                    method_report["status"] = "ok"
                    clip_report["methods"][method] = method_report
                    arrays_by_method[method] = arrays
                    rotation_arrays = series["rotation_arrays"]
                    np.savez_compressed(
                        output_dir / f"{clip_id}__{method}__series.npz",
                        frame_numbers=series["frame_numbers"],
                        timestamps=series["timestamps"],
                        **arrays,
                        **rotation_arrays,
                    )

                    scale = float(method_report["scale_m_per_raw_unit"])
                    metric_c2w = np.repeat(np.eye(4, dtype=np.float64)[None], len(series["camera_center"]), axis=0)
                    metric_c2w[:, :3, :3] = series["rotation"]
                    metric_c2w[:, :3, 3] = scale * series["camera_center"]
                    np.savez_compressed(
                        output_dir / f"{clip_id}__{method}__camera_metric.npz",
                        frame_numbers=series["frame_numbers"],
                        timestamps=series["timestamps"],
                        fps=np.asarray([float(clip.get("fps", 30.0))], dtype=np.float64),
                        scale_m_per_raw_unit=np.asarray([scale], dtype=np.float64),
                        camera_center_m=metric_c2w[:, :3, 3],
                        R_c2w=metric_c2w[:, :3, :3],
                        T_c2w=metric_c2w,
                        T_w2c=np.linalg.inv(metric_c2w),
                    )
                except Exception as exc:
                    clip_report["methods"][method] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            if len(arrays_by_method) == len(series_by_method):
                _plot_clip(output_dir / f"{clip_id}__speed.png", clip_id, series_by_method, arrays_by_method)
        report["clips"].append(clip_report)

    report["summary"] = _aggregate(report)
    save_json(output_dir / "e3_report.json", report)
    rows = _flat_rows(report)
    if rows:
        fieldnames = sorted({key for row in rows for key in row})
        with (output_dir / "e3_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    return report
