from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .e1 import body_scale, kabsch_align, root_relative


@dataclass(frozen=True)
class SharedRootVariants:
    gem_native: np.ndarray
    motionbert_shared_root: np.ndarray
    valid: np.ndarray
    gem_body_scale: float


@dataclass(frozen=True)
class DesynchronizedMotion:
    native: np.ndarray
    desynchronized: np.ndarray
    trajectory_indices: np.ndarray
    pose_indices: np.ndarray
    valid: np.ndarray
    offset_frames: int


@dataclass(frozen=True)
class RecoveredCamera:
    """Rigid camera-to-world transforms recovered from paired joint sets.

    ``rotation_row`` follows NumPy's row-vector convention:
    ``points_world = points_camera @ rotation_row + camera_center``.
    """

    camera_center: np.ndarray
    rotation_row: np.ndarray
    fit_rmse: np.ndarray
    valid: np.ndarray


def recover_camera_from_joint_pairs(
    joints_incam: np.ndarray,
    joints_global: np.ndarray,
    valid: np.ndarray,
) -> RecoveredCamera:
    """Recover per-frame camera pose from matching in-camera/global joints.

    GEM stores the same articulated body in both coordinate systems. A proper
    rigid transform between those paired joints therefore recovers the source
    camera location and orientation without rerunning the camera backend.
    Reflections are explicitly rejected.
    """
    joints_incam = np.asarray(joints_incam, dtype=np.float64)
    joints_global = np.asarray(joints_global, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if joints_incam.shape != joints_global.shape or joints_incam.ndim != 3 or joints_incam.shape[-1] != 3:
        raise ValueError("paired joint arrays must have matching [T,J,3] shapes")
    if len(valid) != len(joints_incam):
        raise ValueError("valid mask must match the number of frames")

    frame_count = len(joints_incam)
    centers = np.full((frame_count, 3), np.nan, dtype=np.float64)
    rotations = np.full((frame_count, 3, 3), np.nan, dtype=np.float64)
    fit_rmse = np.full(frame_count, np.nan, dtype=np.float64)
    recovered = valid & np.isfinite(joints_incam).all(axis=(1, 2)) & np.isfinite(joints_global).all(axis=(1, 2))

    for frame_index in np.flatnonzero(recovered):
        source = joints_incam[frame_index]
        target = joints_global[frame_index]
        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)
        source_centered = source - source_mean
        target_centered = target - target_mean
        if float(np.linalg.norm(source_centered)) < 1e-10 or float(np.linalg.norm(target_centered)) < 1e-10:
            recovered[frame_index] = False
            continue
        u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        center = target_mean - source_mean @ rotation
        predicted = source @ rotation + center
        centers[frame_index] = center
        rotations[frame_index] = rotation
        fit_rmse[frame_index] = float(np.sqrt(np.mean(np.sum((predicted - target) ** 2, axis=-1))))

    return RecoveredCamera(centers, rotations, fit_rmse, recovered)


def construct_shared_root_variants(
    motionbert: np.ndarray,
    gem_global: np.ndarray,
    valid: np.ndarray,
) -> SharedRootVariants:
    """Place MotionBERT articulation on the exact GEM pelvis trajectory.

    One sequence-level Kabsch rotation and one body-scale factor are used. No
    per-frame transform is allowed because that would leak GEM articulation into
    the MotionBERT condition.
    """
    motionbert = np.asarray(motionbert, dtype=np.float64)
    gem_global = np.asarray(gem_global, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if motionbert.shape != gem_global.shape or motionbert.shape[1:] != (17, 3):
        raise ValueError("shared-root inputs must have matching [T,17,3] shapes")
    valid &= np.isfinite(motionbert).all(axis=(1, 2)) & np.isfinite(gem_global).all(axis=(1, 2))
    if valid.sum() < 3:
        raise ValueError("at least three valid frames are required")
    aligned_motionbert, _ = kabsch_align(motionbert, gem_global, valid)
    gem_scale = body_scale(gem_global, valid)
    if not np.isfinite(gem_scale):
        raise ValueError("GEM body scale is not finite")
    root = gem_global[:, :1, :]
    shared = root + aligned_motionbert * gem_scale
    return SharedRootVariants(gem_global.copy(), shared, valid, float(gem_scale))


def desynchronize_articulation(
    motion: np.ndarray,
    valid: np.ndarray,
    offset_frames: int,
) -> DesynchronizedMotion:
    """Pair the native trajectory at t with articulation at t + offset.

    The sequence is cropped rather than wrapped or padded. Therefore path,
    scale, duration sampling, and camera convention remain unchanged on the
    retained interval, while only temporal synchronization is altered.
    """
    motion = np.asarray(motion, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if motion.ndim != 3 or motion.shape[1:] != (17, 3) or len(valid) != len(motion):
        raise ValueError("motion must be [T,17,3] and valid must be [T]")
    offset_frames = int(offset_frames)
    if offset_frames == 0:
        indices = np.arange(len(motion), dtype=np.int64)
        return DesynchronizedMotion(
            motion.copy(), motion.copy(), indices, indices.copy(), valid.copy(), offset_frames
        )
    amount = abs(offset_frames)
    if amount >= len(motion) - 2:
        raise ValueError("temporal offset leaves fewer than three frames")
    if offset_frames > 0:
        trajectory_indices = np.arange(0, len(motion) - amount, dtype=np.int64)
        pose_indices = trajectory_indices + amount
    else:
        trajectory_indices = np.arange(amount, len(motion), dtype=np.int64)
        pose_indices = trajectory_indices - amount
    root = motion[trajectory_indices, :1, :]
    shifted_local = root_relative(motion)[pose_indices]
    desynchronized = root + shifted_local
    paired_valid = valid[trajectory_indices] & valid[pose_indices]
    return DesynchronizedMotion(
        native=motion[trajectory_indices].copy(),
        desynchronized=desynchronized,
        trajectory_indices=trajectory_indices,
        pose_indices=pose_indices,
        valid=paired_valid,
        offset_frames=offset_frames,
    )


def infer_ankle_contacts(
    motion: np.ndarray,
    valid: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Infer left/right ankle contact from low height and low vertical speed."""
    motion = np.asarray(motion, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(motion).all(axis=(1, 2))
    if motion.ndim != 3 or motion.shape[1:] != (17, 3) or len(valid) != len(motion):
        raise ValueError("motion must be [T,17,3] and valid must be [T]")
    scale = body_scale(motion, valid)
    contacts = np.zeros((len(motion), 2), dtype=bool)
    if not np.isfinite(scale) or float(fps) <= 0:
        return contacts
    local = motion - motion[:, :1, :]
    for column, ankle in enumerate((3, 6)):
        relative_height = local[:, ankle, 1]
        height_values = relative_height[valid]
        if not len(height_values):
            continue
        low_height = float(np.percentile(height_values, 35)) + 0.02 * scale
        relative_vertical_speed = np.zeros(len(motion), dtype=np.float64)
        relative_vertical_speed[1:] = np.abs(np.diff(relative_height)) * float(fps)
        contacts[:, column] = (
            valid & (relative_height <= low_height) & (relative_vertical_speed <= 0.12 * scale)
        )
    return contacts


def coupling_metrics(motion: np.ndarray, valid: np.ndarray, fps: float) -> dict[str, float | int]:
    """Trajectory/articulation diagnostics in body-scale-normalized units.

    Contact is inferred from common-17 ankles, so these are diagnostic
    consistency measures rather than ground-truth physical contact labels.
    """
    motion = np.asarray(motion, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(motion).all(axis=(1, 2))
    fps = float(fps)
    if motion.ndim != 3 or motion.shape[1:] != (17, 3) or len(valid) != len(motion):
        raise ValueError("motion must be [T,17,3] and valid must be [T]")
    if fps <= 0:
        raise ValueError("fps must be positive")
    scale = body_scale(motion, valid)
    if not np.isfinite(scale) or len(motion) < 3:
        return {
            "frame_count": int(len(motion)),
            "valid_frames": int(valid.sum()),
            "contact_samples": 0,
            "contact_foot_speed_median_body_scale_per_s": float("nan"),
            "contact_foot_speed_p95_body_scale_per_s": float("nan"),
            "foot_sliding_ratio_above_0p05": float("nan"),
            "path_lateral_axis_misalignment_median_deg": float("nan"),
            "path_lateral_axis_misalignment_p95_deg": float("nan"),
            "root_speed_median_body_scale_per_s": float("nan"),
            "root_speed_p95_body_scale_per_s": float("nan"),
            "root_leg_activity_pearson": float("nan"),
        }

    root = motion[:, 0]
    local = motion - root[:, None, :]
    pair_valid = valid[1:] & valid[:-1]
    root_velocity = np.diff(root[:, [0, 2]], axis=0) * fps / scale
    root_speed = np.linalg.norm(root_velocity, axis=-1)
    root_speed_values = root_speed[pair_valid]

    contact_speeds = []
    contacts = infer_ankle_contacts(motion, valid, fps)
    for column, ankle in enumerate((3, 6)):
        contact = contacts[:, column]
        contact_pair = contact[1:] & contact[:-1] & pair_valid
        foot_velocity = np.diff(motion[:, ankle, [0, 2]], axis=0) * fps / scale
        foot_speed = np.linalg.norm(foot_velocity, axis=-1)
        contact_speeds.extend(foot_speed[contact_pair].tolist())
    contact_speeds_array = np.asarray(contact_speeds, dtype=np.float64)

    hip_axis = motion[1:, 4, [0, 2]] - motion[1:, 1, [0, 2]]
    hip_norm = np.linalg.norm(hip_axis, axis=-1)
    moving = pair_valid & (root_speed > 0.03) & (hip_norm > 1e-8)
    if moving.any():
        cosine = np.abs(np.sum(hip_axis[moving] * root_velocity[moving], axis=-1))
        cosine /= hip_norm[moving] * np.maximum(root_speed[moving], 1e-8)
        misalignment = np.degrees(np.arcsin(np.clip(cosine, 0.0, 1.0)))
    else:
        misalignment = np.asarray([], dtype=np.float64)

    lower_body = local[:, [2, 3, 5, 6], :]
    leg_velocity = np.diff(lower_body, axis=0) * fps / scale
    leg_activity = np.mean(np.linalg.norm(leg_velocity, axis=-1), axis=1)
    correlation_mask = pair_valid & np.isfinite(root_speed) & np.isfinite(leg_activity)
    if correlation_mask.sum() >= 3:
        root_values = root_speed[correlation_mask]
        leg_values = leg_activity[correlation_mask]
        if np.std(root_values) > 1e-8 and np.std(leg_values) > 1e-8:
            correlation = float(np.corrcoef(root_values, leg_values)[0, 1])
        else:
            correlation = float("nan")
    else:
        correlation = float("nan")

    return {
        "frame_count": int(len(motion)),
        "valid_frames": int(valid.sum()),
        "contact_samples": int(len(contact_speeds_array)),
        "contact_foot_speed_median_body_scale_per_s": (
            float(np.median(contact_speeds_array)) if len(contact_speeds_array) else float("nan")
        ),
        "contact_foot_speed_p95_body_scale_per_s": (
            float(np.percentile(contact_speeds_array, 95)) if len(contact_speeds_array) else float("nan")
        ),
        "foot_sliding_ratio_above_0p05": (
            float(np.mean(contact_speeds_array > 0.05)) if len(contact_speeds_array) else float("nan")
        ),
        "path_lateral_axis_misalignment_median_deg": (
            float(np.median(misalignment)) if len(misalignment) else float("nan")
        ),
        "path_lateral_axis_misalignment_p95_deg": (
            float(np.percentile(misalignment, 95)) if len(misalignment) else float("nan")
        ),
        "root_speed_median_body_scale_per_s": (
            float(np.median(root_speed_values)) if len(root_speed_values) else float("nan")
        ),
        "root_speed_p95_body_scale_per_s": (
            float(np.percentile(root_speed_values, 95)) if len(root_speed_values) else float("nan")
        ),
        "root_leg_activity_pearson": correlation,
    }
