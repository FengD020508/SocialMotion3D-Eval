from __future__ import annotations

from dataclasses import dataclass

import numpy as np


JOINT_NAMES = [
    "pelvis", "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee",
    "left_ankle", "spine", "thorax", "neck", "head", "left_shoulder", "left_elbow",
    "left_wrist", "right_shoulder", "right_elbow", "right_wrist",
]

BONES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
    (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16),
]


@dataclass(frozen=True)
class AlignedMotion:
    local_frames: np.ndarray
    motionbert: np.ndarray
    gem: np.ndarray
    valid_motionbert: np.ndarray
    valid_gem: np.ndarray


def align_by_local_frame(
    motionbert: np.ndarray,
    motionbert_frames: np.ndarray,
    gem: np.ndarray,
    gem_frames: np.ndarray,
    gem_valid: np.ndarray | None = None,
) -> AlignedMotion:
    motionbert = np.asarray(motionbert, dtype=np.float64)
    gem = np.asarray(gem, dtype=np.float64)
    motionbert_frames = np.asarray(motionbert_frames, dtype=np.int64)
    gem_frames = np.asarray(gem_frames, dtype=np.int64)
    if motionbert.shape[1:] != (17, 3) or gem.shape[1:] != (17, 3):
        raise ValueError("E1 inputs must have shape [T, 17, 3]")
    if len(motionbert) != len(motionbert_frames) or len(gem) != len(gem_frames):
        raise ValueError("frame index arrays must match their motions")
    common = np.intersect1d(motionbert_frames, gem_frames)
    if len(common) == 0:
        raise ValueError("MotionBERT and GEM have no common local frames")
    motionbert_index = {int(frame): index for index, frame in enumerate(motionbert_frames)}
    gem_index = {int(frame): index for index, frame in enumerate(gem_frames)}
    mb = motionbert[[motionbert_index[int(frame)] for frame in common]]
    gm = gem[[gem_index[int(frame)] for frame in common]]
    valid_mb = np.isfinite(mb).all(axis=(1, 2))
    valid_gm = np.isfinite(gm).all(axis=(1, 2))
    if gem_valid is not None:
        source_valid = np.asarray(gem_valid, dtype=bool)
        valid_gm &= source_valid[[gem_index[int(frame)] for frame in common]]
    return AlignedMotion(common, mb, gm, valid_mb, valid_gm)


def root_relative(motion: np.ndarray) -> np.ndarray:
    motion = np.asarray(motion, dtype=np.float64)
    return motion - motion[:, :1, :]


def body_scale(motion: np.ndarray, valid: np.ndarray | None = None) -> float:
    motion = np.asarray(motion, dtype=np.float64)
    mask = np.ones(len(motion), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    lengths = np.stack([np.linalg.norm(motion[:, a] - motion[:, b], axis=-1) for a, b in BONES], axis=1)
    values = lengths[mask & np.isfinite(lengths).all(axis=1)]
    if not len(values):
        return float("nan")
    scale = float(np.median(values.sum(axis=1)))
    return scale if scale > 1e-8 else float("nan")


def normalized_root_relative(motion: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    relative = root_relative(motion)
    scale = body_scale(relative, valid)
    if not np.isfinite(scale):
        return np.full_like(relative, np.nan), scale
    return relative / scale, scale


def robust_temporal_metrics(motion: np.ndarray, valid: np.ndarray, fps: float) -> dict[str, float | int]:
    motion = np.asarray(motion, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(motion).all(axis=(1, 2))
    normalized, scale = normalized_root_relative(motion, valid)
    lengths = np.stack([np.linalg.norm(motion[:, a] - motion[:, b], axis=-1) for a, b in BONES], axis=1)
    bone_cv_values = []
    for column in range(lengths.shape[1]):
        values = lengths[valid, column]
        if len(values) and np.mean(values) > 1e-8:
            bone_cv_values.append(float(np.std(values) / np.mean(values)))

    pair_valid = valid[1:] & valid[:-1]
    velocity = np.diff(normalized, axis=0) * fps
    speed = np.linalg.norm(velocity, axis=-1)
    speed_values = speed[pair_valid]
    if len(speed_values):
        median = float(np.median(speed_values))
        mad = float(np.median(np.abs(speed_values - median)))
        threshold = median + 6.0 * max(1.4826 * mad, 1e-8)
        temporal_outlier_ratio = float(np.mean(speed_values > threshold))
        velocity_p95 = float(np.percentile(speed_values, 95))
    else:
        temporal_outlier_ratio = float("nan")
        velocity_p95 = float("nan")

    triple_valid = valid[2:] & valid[1:-1] & valid[:-2]
    acceleration = np.diff(normalized, n=2, axis=0) * fps**2
    acceleration_values = np.linalg.norm(acceleration, axis=-1)[triple_valid]
    quad_valid = valid[3:] & valid[2:-1] & valid[1:-2] & valid[:-3]
    jerk = np.diff(normalized, n=3, axis=0) * fps**3
    jerk_values = np.linalg.norm(jerk, axis=-1)[quad_valid]
    return {
        "frame_count": int(len(motion)),
        "valid_frames": int(valid.sum()),
        "valid_ratio": float(valid.mean()) if len(valid) else float("nan"),
        "body_scale": scale,
        "bone_length_cv_median": float(np.median(bone_cv_values)) if bone_cv_values else float("nan"),
        "bone_length_cv_p95": float(np.percentile(bone_cv_values, 95)) if bone_cv_values else float("nan"),
        "joint_velocity_p95_normalized_per_s": velocity_p95,
        "joint_acceleration_p95_normalized_per_s2": (
            float(np.percentile(acceleration_values, 95)) if len(acceleration_values) else float("nan")
        ),
        "joint_jerk_p95_normalized_per_s3": (
            float(np.percentile(jerk_values, 95)) if len(jerk_values) else float("nan")
        ),
        "temporal_outlier_ratio": temporal_outlier_ratio,
    }


def kabsch_align(source: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_normalized, _ = normalized_root_relative(source, valid)
    target_normalized, _ = normalized_root_relative(target, valid)
    x = source_normalized[valid].reshape(-1, 3)
    y = target_normalized[valid].reshape(-1, 3)
    if len(x) < 3:
        raise ValueError("at least three valid points are required for alignment")
    covariance = x.T @ y
    u, _, vt = np.linalg.svd(covariance)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return source_normalized @ rotation, target_normalized


def method_disagreement(motionbert: np.ndarray, gem: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    aligned_mb, normalized_gem = kabsch_align(motionbert, gem, valid)
    distance = np.linalg.norm(aligned_mb - normalized_gem, axis=-1)
    values = distance[valid]
    return {
        "common_valid_frames": int(valid.sum()),
        "common_valid_ratio": float(valid.mean()),
        "normalized_mpjpe_median": float(np.median(values)),
        "normalized_mpjpe_p95": float(np.percentile(values, 95)),
    }
