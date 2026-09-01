"""Camera and coordinate utilities for IDD-PeD joint reconstruction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gem.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
)


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    q /= np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = np.moveaxis(q, -1, 0)
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out.astype(np.float32)


def droid_pose7_to_c2w(poses: np.ndarray) -> np.ndarray:
    """Convert DROID's [tx,ty,tz,qx,qy,qz,qw] camera poses to matrices."""
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected DROID trajectory (N,7), got {poses.shape}")
    T = np.repeat(np.eye(4, dtype=np.float32)[None], len(poses), axis=0)
    T[:, :3, :3] = quat_xyzw_to_matrix(poses[:, 3:])
    T[:, :3, 3] = poses[:, :3]
    return T


def project_to_so3(R: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(R.astype(np.float64))
    out = u @ vt
    bad = np.linalg.det(out) < 0
    if np.any(bad):
        u[bad, :, -1] *= -1
        out = u @ vt
    return out.astype(np.float32)


def normalize_and_y_up(T_c2w_droid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-frame normalize and express poses in GENMO's Y-up basis.

    DROID/OpenCV camera coordinates use +Y down. A 180-degree rotation about Z
    maps the trajectory into a proper-rotation Y-up basis while retaining +Z as
    the longitudinal axis. Unity mirroring is applied later, exactly once.
    """
    T_c2w = torch.from_numpy(np.asarray(T_c2w_droid, dtype=np.float32))
    T_w2c = torch.linalg.inv(T_c2w)
    T_w2c = normalize_T_w2c(T_w2c)
    T_c2w = torch.linalg.inv(T_w2c).numpy()

    basis = np.eye(4, dtype=np.float32)
    basis[:3, :3] = np.diag([-1.0, -1.0, 1.0])
    T_c2w = basis[None] @ T_c2w @ basis[None]
    T_c2w[:, :3, :3] = project_to_so3(T_c2w[:, :3, :3])
    T_c2w[0] = np.eye(4, dtype=np.float32)
    T_w2c = np.linalg.inv(T_c2w).astype(np.float32)
    return T_c2w.astype(np.float32), T_w2c


def make_K(intrinsics: np.ndarray, n: int) -> np.ndarray:
    intr = np.asarray(intrinsics, dtype=np.float32)
    if intr.ndim == 1:
        intr = np.repeat(intr[None], n, axis=0)
    K = np.zeros((n, 3, 3), dtype=np.float32)
    K[:, 0, 0], K[:, 1, 1] = intr[:, 0], intr[:, 1]
    K[:, 0, 2], K[:, 1, 2] = intr[:, 2], intr[:, 3]
    K[:, 2, 2] = 1
    return K


def validate_camera(T_c2w: np.ndarray, T_w2c: np.ndarray) -> dict:
    R = T_c2w[:, :3, :3]
    finite = bool(np.isfinite(T_c2w).all() and np.isfinite(T_w2c).all())
    det_err = np.abs(np.linalg.det(R) - 1)
    inv_err = np.abs(T_c2w @ T_w2c - np.eye(4)).max()
    first_err = np.abs(T_c2w[0] - np.eye(4)).max()
    return {
        "finite": finite,
        "max_rotation_det_error": float(det_err.max()),
        "max_inverse_error": float(inv_err),
        "first_pose_identity_error": float(first_err),
        "passed": bool(finite and det_err.max() < 1e-3 and inv_err < 1e-3 and first_err < 1e-3),
    }


def save_camera_npz(
    path: Path,
    T_c2w: np.ndarray,
    T_w2c: np.ndarray,
    intrinsics: np.ndarray,
    frame_numbers: np.ndarray,
    timestamps: np.ndarray,
    tracking_confidence: np.ndarray,
    tracking_failed: np.ndarray,
    fps: float,
    metadata: dict,
) -> None:
    R_w2c = torch.from_numpy(T_w2c[:, :3, :3])
    cam_angvel = compute_cam_angvel(R_w2c, padding_last=True).numpy()
    cam_tvel = compute_cam_tvel(torch.from_numpy(T_w2c[:, :3, 3])).numpy()
    K = make_K(intrinsics, len(T_w2c))
    np.savez_compressed(
        path,
        T_c2w=T_c2w,
        T_w2c=T_w2c,
        R_c2w=T_c2w[:, :3, :3],
        R_w2c=T_w2c[:, :3, :3],
        camera_center=T_c2w[:, :3, 3],
        intrinsics=np.asarray(intrinsics, dtype=np.float32),
        K_fullimg=K,
        frame_numbers=np.asarray(frame_numbers, dtype=np.int64),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        tracking_confidence=np.asarray(tracking_confidence, dtype=np.float32),
        tracking_failed=np.asarray(tracking_failed, dtype=bool),
        cam_angvel=cam_angvel,
        cam_tvel=cam_tvel,
        fps=np.float64(fps),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
