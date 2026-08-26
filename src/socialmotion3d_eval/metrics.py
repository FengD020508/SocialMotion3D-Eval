from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def masked_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    window = int(window)
    kernel = np.ones(window, dtype=np.float64)
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    counts = np.convolve(finite.astype(np.float64), kernel, mode="same")
    output = np.full_like(values, np.nan)
    np.divide(sums, counts, out=output, where=counts > 0)
    return output


def rotation_validity(rotation: np.ndarray, orthogonality_tolerance: float, determinant_tolerance: float) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    finite = np.all(np.isfinite(rotation), axis=(1, 2))
    identity = np.eye(3, dtype=np.float64)
    orthogonality_error = np.linalg.norm(
        np.einsum("nji,njk->nik", rotation, rotation) - identity,
        axis=(1, 2),
    )
    determinant_error = np.abs(np.linalg.det(rotation) - 1.0)
    return finite & (orthogonality_error <= orthogonality_tolerance) & (determinant_error <= determinant_tolerance)


def robust_interval_validity(displacement: np.ndarray, mad_factor: float) -> tuple[np.ndarray, float]:
    displacement = np.asarray(displacement, dtype=np.float64)
    finite = np.isfinite(displacement)
    sample = displacement[finite]
    if sample.size == 0:
        return finite & False, float("nan")
    median = float(np.median(sample))
    mad = float(np.median(np.abs(sample - median)))
    robust_sigma = 1.4826 * mad
    threshold = median + float(mad_factor) * robust_sigma
    if robust_sigma <= 1e-12:
        threshold = max(median * 10.0, 1e-9)
    return finite & (displacement <= threshold), float(threshold)


def build_motion_series(
    camera: dict[str, Any],
    obd_speed_frame_mps: np.ndarray,
    *,
    smooth_window_frames: int,
    jump_mad_factor: float,
    orthogonality_tolerance: float,
    determinant_tolerance: float,
) -> dict[str, Any]:
    center = np.asarray(camera["camera_center"], dtype=np.float64)
    rotation = np.asarray(camera["rotation"], dtype=np.float64)
    timestamps = np.asarray(camera["timestamps"], dtype=np.float64)
    frame_numbers = np.asarray(camera["frame_numbers"], dtype=np.int64)
    obd_frame = np.asarray(obd_speed_frame_mps, dtype=np.float64)

    if len(center) < 3:
        raise ValueError("camera trajectory must contain at least three frames")
    dt = np.diff(timestamps)
    displacement = np.linalg.norm(np.diff(center, axis=0), axis=1)
    raw_speed = np.divide(
        displacement,
        dt,
        out=np.full_like(displacement, np.nan),
        where=np.isfinite(dt) & (dt > 0),
    )

    frame_valid = np.all(np.isfinite(center), axis=1)
    frame_valid &= rotation_validity(rotation, orthogonality_tolerance, determinant_tolerance)
    jump_valid, jump_threshold = robust_interval_validity(displacement, jump_mad_factor)
    interval_valid = frame_valid[:-1] & frame_valid[1:] & np.isfinite(dt) & (dt > 0) & jump_valid

    obd_smoothed_frame = masked_moving_average(obd_frame, smooth_window_frames)
    obd_interval = 0.5 * (obd_smoothed_frame[:-1] + obd_smoothed_frame[1:])
    visual_interval = masked_moving_average(raw_speed, smooth_window_frames)

    tracking_failed = camera.get("tracking_failed")
    backend_failed_ratio = None
    if tracking_failed is not None:
        backend_failed_ratio = float(np.mean(np.asarray(tracking_failed, dtype=bool)))

    return {
        "frame_numbers": frame_numbers,
        "timestamps": timestamps,
        "dt": dt,
        "raw_displacement": displacement,
        "visual_speed_raw": raw_speed,
        "visual_speed": visual_interval,
        "obd_speed": obd_interval,
        "frame_valid": frame_valid,
        "interval_valid": interval_valid,
        "jump_threshold_raw_units": jump_threshold,
        "backend_failed_ratio": backend_failed_ratio,
    }


def fit_nonnegative_scale(visual_speed: np.ndarray, obd_speed: np.ndarray, mask: np.ndarray) -> float:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(visual_speed) & np.isfinite(obd_speed)
    if int(np.sum(valid)) < 10:
        raise ValueError("fewer than 10 valid calibration intervals")
    x = np.asarray(visual_speed, dtype=np.float64)[valid]
    y = np.asarray(obd_speed, dtype=np.float64)[valid]
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        raise ValueError("visual motion is degenerate in calibration interval")
    return max(0.0, float(np.dot(x, y) / denominator))


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or float(np.std(x)) <= 1e-8 or float(np.std(y)) <= 1e-8:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def accuracy_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    if int(np.sum(valid)) == 0:
        raise ValueError("no valid evaluation intervals")
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    truth = np.asarray(target, dtype=np.float64)[valid]
    error = pred - truth
    return {
        "n_intervals": int(len(pred)),
        "mae_mps": float(np.mean(np.abs(error))),
        "rmse_mps": float(np.sqrt(np.mean(error**2))),
        "pearson_r": safe_pearson(pred, truth),
    }


def windowed_relative_distance_error(
    prediction_speed: np.ndarray,
    target_speed: np.ndarray,
    dt: np.ndarray,
    mask: np.ndarray,
    *,
    fps: float,
    windows_seconds: Iterable[float],
    min_target_distance_m: float,
) -> dict[str, dict[str, Any]]:
    prediction_speed = np.asarray(prediction_speed, dtype=np.float64)
    target_speed = np.asarray(target_speed, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    base_valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(prediction_speed)
        & np.isfinite(target_speed)
        & np.isfinite(dt)
        & (dt > 0)
    )
    output: dict[str, dict[str, Any]] = {}
    for seconds in windows_seconds:
        length = max(1, int(round(float(seconds) * float(fps))))
        errors: list[float] = []
        for start in range(0, len(dt) - length + 1):
            stop = start + length
            if not np.all(base_valid[start:stop]):
                continue
            target_distance = float(np.sum(target_speed[start:stop] * dt[start:stop]))
            if target_distance < min_target_distance_m:
                continue
            predicted_distance = float(np.sum(prediction_speed[start:stop] * dt[start:stop]))
            errors.append(abs(predicted_distance - target_distance) / target_distance)
        key = f"{float(seconds):g}s"
        output[key] = {
            "n_windows": len(errors),
            "mean": float(np.mean(errors)) if errors else None,
            "median": float(np.median(errors)) if errors else None,
        }
    return output


def window_scale_stability(
    calibrated_prediction_speed: np.ndarray,
    target_speed: np.ndarray,
    dt: np.ndarray,
    mask: np.ndarray,
    *,
    fps: float,
    windows_seconds: Iterable[float],
    min_target_distance_m: float,
) -> dict[str, dict[str, Any]]:
    calibrated_prediction_speed = np.asarray(calibrated_prediction_speed, dtype=np.float64)
    target_speed = np.asarray(target_speed, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    base_valid = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(calibrated_prediction_speed)
        & np.isfinite(target_speed)
        & np.isfinite(dt)
        & (dt > 0)
    )
    output: dict[str, dict[str, Any]] = {}
    for seconds in windows_seconds:
        length = max(1, int(round(float(seconds) * float(fps))))
        scales: list[float] = []
        for start in range(0, len(dt) - length + 1):
            stop = start + length
            if not np.all(base_valid[start:stop]):
                continue
            target_distance = float(np.sum(target_speed[start:stop] * dt[start:stop]))
            predicted_distance = float(np.sum(calibrated_prediction_speed[start:stop] * dt[start:stop]))
            if target_distance < min_target_distance_m or predicted_distance < min_target_distance_m:
                continue
            scales.append(target_distance / predicted_distance)
        key = f"{float(seconds):g}s"
        if scales:
            values = np.asarray(scales, dtype=np.float64)
            q25, q75 = np.percentile(values, [25, 75])
            mean = float(np.mean(values))
            output[key] = {
                "n_windows": len(scales),
                "median": float(np.median(values)),
                "iqr": float(q75 - q25),
                "cv": float(np.std(values) / abs(mean)) if abs(mean) > 1e-12 else None,
            }
        else:
            output[key] = {"n_windows": 0, "median": None, "iqr": None, "cv": None}
    return output


def evaluate_scaled_series(
    series: dict[str, Any],
    *,
    common_mask: np.ndarray,
    calibration_fraction: float,
    fps: float,
    windows_seconds: Iterable[float],
    min_target_distance_m: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_intervals = len(series["dt"])
    split = min(n_intervals - 1, max(1, int(np.floor(n_intervals * calibration_fraction))))
    indices = np.arange(n_intervals)
    common_mask = np.asarray(common_mask, dtype=bool)
    calibration_mask = common_mask & (indices < split)
    evaluation_mask = common_mask & (indices >= split)

    scale = fit_nonnegative_scale(series["visual_speed"], series["obd_speed"], calibration_mask)
    prediction = scale * np.asarray(series["visual_speed"], dtype=np.float64)
    accuracy = accuracy_metrics(prediction, series["obd_speed"], evaluation_mask)
    wrde = windowed_relative_distance_error(
        prediction,
        series["obd_speed"],
        series["dt"],
        evaluation_mask,
        fps=fps,
        windows_seconds=windows_seconds,
        min_target_distance_m=min_target_distance_m,
    )
    stability = window_scale_stability(
        prediction,
        series["obd_speed"],
        series["dt"],
        evaluation_mask,
        fps=fps,
        windows_seconds=windows_seconds,
        min_target_distance_m=min_target_distance_m,
    )
    report = {
        "scale_m_per_raw_unit": scale,
        "calibration_intervals": int(np.sum(calibration_mask)),
        "evaluation_intervals": int(np.sum(evaluation_mask)),
        "accuracy": accuracy,
        "wrde": wrde,
        "window_scale_stability": stability,
        "speed_support": {
            "calibration_obd_mean_mps": float(np.mean(np.asarray(series["obd_speed"])[calibration_mask])),
            "calibration_obd_std_mps": float(np.std(np.asarray(series["obd_speed"])[calibration_mask])),
            "evaluation_obd_mean_mps": float(np.mean(np.asarray(series["obd_speed"])[evaluation_mask])),
            "evaluation_obd_std_mps": float(np.std(np.asarray(series["obd_speed"])[evaluation_mask])),
        },
        "validity": {
            "frame_valid_ratio": float(np.mean(series["frame_valid"])),
            "interval_valid_ratio": float(np.mean(series["interval_valid"])),
            "common_interval_valid_ratio": float(np.mean(common_mask)),
            "backend_reported_failed_ratio": series["backend_failed_ratio"],
            "jump_threshold_raw_units": series["jump_threshold_raw_units"],
        },
    }
    arrays = {
        "prediction_speed_mps": prediction,
        "obd_speed_mps": np.asarray(series["obd_speed"], dtype=np.float64),
        "visual_speed_raw": np.asarray(series["visual_speed_raw"], dtype=np.float64),
        "calibration_mask": calibration_mask,
        "evaluation_mask": evaluation_mask,
    }
    return report, arrays
