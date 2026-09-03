#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from socialmotion3d_eval.e1 import BONES, align_by_local_frame
from socialmotion3d_eval.e1c import (
    canonicalize_world_up,
    construct_shared_root_variants,
    coupling_metrics,
    desynchronize_articulation,
    infer_ankle_contacts,
    recover_camera_from_joint_pairs,
)


class FfmpegFrameReader:
    """Minimal sequential RGB reader using the system FFmpeg binaries."""

    def __init__(self, path: Path):
        ffmpeg = shutil.which("ffmpeg")
        ffprobe = shutil.which("ffprobe")
        if ffmpeg is None or ffprobe is None:
            raise RuntimeError("ffmpeg and ffprobe are required to render E1c reference videos")
        probe = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                "stream=width,height", "-of", "json", str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        stream = json.loads(probe.stdout)["streams"][0]
        self.width = int(stream["width"])
        self.height = int(stream["height"])
        self.frame_bytes = self.width * self.height * 3
        self.process = subprocess.Popen(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.index = -1
        self.frame = None

    def get_data(self, target_index: int) -> np.ndarray:
        target_index = int(target_index)
        if target_index < self.index:
            raise ValueError("FfmpegFrameReader only supports monotonically increasing frame indices")
        while self.index < target_index:
            if self.process.stdout is None:
                raise RuntimeError("FFmpeg stdout is unavailable")
            raw = self.process.stdout.read(self.frame_bytes)
            if len(raw) != self.frame_bytes:
                raise IndexError(f"video ended before frame {target_index}")
            self.index += 1
            self.frame = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 3)
        if self.frame is None:
            raise IndexError("video contains no frames")
        return self.frame

    def close(self) -> None:
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def finite_or_none(value):
    if isinstance(value, dict):
        return {key: finite_or_none(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_or_none(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def cue_label(clip: dict) -> str:
    pieces = []
    stratum = clip.get("stratum") or {}
    if stratum.get("kind"):
        pieces.append(str(stratum["kind"]))
    traffic = clip.get("traffic_interaction")
    if traffic and traffic != "N/A":
        pieces.append(str(traffic))
    crossing = clip.get("crossing_behavior")
    if crossing and crossing != "N/A":
        pieces.append(str(crossing))
    return "+".join(dict.fromkeys(pieces)) or "unspecified"


def deterministic_sign(seed: str, scene: str) -> int:
    digest = hashlib.sha256(f"{seed}:shift:{scene}".encode()).hexdigest()
    return 1 if int(digest[:8], 16) % 2 == 0 else -1


def blind_pair(seed: str, scene: str, comparison: str, left: str, right: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{comparison}:{scene}".encode()).hexdigest()
    return (left, right) if int(digest[:8], 16) % 2 == 0 else (right, left)


LEFT_BONES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT_BONES = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}


def _plot_coordinates(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points[..., [0, 2, 1]]


def _cylinder_faces(start: np.ndarray, end: np.ndarray, radius: float, sides: int = 7) -> list[np.ndarray]:
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    direction = end - start
    length = float(np.linalg.norm(direction))
    if length < 1e-8:
        return []
    axis = direction / length
    helper = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(axis, helper))) > 0.9:
        helper = np.asarray([1.0, 0.0, 0.0])
    basis_a = np.cross(axis, helper)
    basis_a /= np.linalg.norm(basis_a)
    basis_b = np.cross(axis, basis_a)
    angles = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    ring_start = np.stack(
        [start + radius * (np.cos(angle) * basis_a + np.sin(angle) * basis_b) for angle in angles]
    )
    ring_end = ring_start + direction
    faces = []
    for index in range(sides):
        next_index = (index + 1) % sides
        faces.append(
            _plot_coordinates(
                np.stack([ring_start[index], ring_start[next_index], ring_end[next_index], ring_end[index]])
            )
        )
    faces.append(_plot_coordinates(ring_start[::-1]))
    faces.append(_plot_coordinates(ring_end))
    return faces


def _box_faces(
    center: np.ndarray,
    forward: np.ndarray,
    half_length: float,
    half_width: float,
    half_height: float,
) -> list[np.ndarray]:
    forward = np.asarray(forward, dtype=np.float64)
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-8:
        forward = np.asarray([0.0, 0.0, 1.0])
    else:
        forward /= norm
    side = np.asarray([-forward[2], 0.0, forward[0]])
    up = np.asarray([0.0, 1.0, 0.0])
    corners = []
    for length_sign, width_sign, height_sign in (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ):
        corners.append(
            center
            + length_sign * half_length * forward
            + width_sign * half_width * side
            + height_sign * half_height * up
        )
    corners = np.asarray(corners)
    indices = ((0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0))
    return [_plot_coordinates(corners[list(face)]) for face in indices]


def _floor_material(axis, root: np.ndarray, ground: float, radius: float, spacing: float) -> None:
    spacing = max(float(spacing), 1e-3)
    x0 = np.floor((root[0] - radius) / spacing) * spacing
    z0 = np.floor((root[2] - radius) / spacing) * spacing
    values_x = np.arange(x0, root[0] + radius + spacing, spacing)
    values_z = np.arange(z0, root[2] + radius + spacing, spacing)
    faces = []
    colors = []
    for x_index, x_value in enumerate(values_x[:-1]):
        for z_index, z_value in enumerate(values_z[:-1]):
            faces.append(
                np.asarray(
                    [
                        [x_value, z_value, ground],
                        [x_value + spacing, z_value, ground],
                        [x_value + spacing, z_value + spacing, ground],
                        [x_value, z_value + spacing, ground],
                    ]
                )
            )
            colors.append("#343a40" if (x_index + z_index) % 2 == 0 else "#454c55")
    # A single opaque 3D polygon collection can be depth-sorted in front of the
    # mannequin by mplot3d for oblique fixed cameras. Keep the material
    # translucent so it remains a ground cue without hiding the motion.
    collection = Poly3DCollection(
        faces, facecolors=colors, edgecolors="#77818d", linewidths=0.45, alpha=0.34
    )
    axis.add_collection3d(collection)


def _path_direction(root_path: np.ndarray, frame_index: int) -> np.ndarray:
    previous_index = max(frame_index - 2, 0)
    next_index = min(frame_index + 2, len(root_path) - 1)
    direction = root_path[next_index] - root_path[previous_index]
    direction[1] = 0.0
    return direction


def _fixed_world_camera(
    root_path: np.ndarray,
    valid: np.ndarray,
    valid_data: np.ndarray,
    body_height: float,
    ground: float,
    source_camera_centers: np.ndarray | None = None,
) -> dict[str, float | str | tuple[float, float, float]]:
    """Choose one world-fixed camera, viewed from the source-camera side."""
    horizontal = np.asarray(root_path[valid][:, [0, 2]], dtype=np.float64)
    horizontal = horizontal[np.isfinite(horizontal).all(axis=1)]
    if not len(horizontal):
        horizontal = np.zeros((1, 2), dtype=np.float64)

    centered = horizontal - np.mean(horizontal, axis=0, keepdims=True)
    view_policy = "trajectory_side_fallback"
    if len(horizontal) >= 2 and float(np.linalg.norm(centered)) > 1e-8:
        _, _, principal_axes = np.linalg.svd(centered, full_matrices=False)
        direction = principal_axes[0]
        endpoint_direction = horizontal[-1] - horizontal[0]
        if float(np.dot(direction, endpoint_direction)) < 0.0:
            direction *= -1.0
        path_azimuth = float(np.degrees(np.arctan2(direction[1], direction[0])))
        azimuth = path_azimuth + 90.0
    else:
        azimuth = -65.0

    if source_camera_centers is not None:
        source_camera_centers = np.asarray(source_camera_centers, dtype=np.float64)
        if source_camera_centers.shape == root_path.shape:
            camera_valid = (
                np.asarray(valid, dtype=bool)
                & np.isfinite(source_camera_centers).all(axis=1)
                & np.isfinite(root_path).all(axis=1)
            )
            observer = source_camera_centers[camera_valid][:, [0, 2]] - root_path[camera_valid][:, [0, 2]]
            observer_norm = np.linalg.norm(observer, axis=1)
            observer = observer[observer_norm > 1e-8]
            observer_norm = observer_norm[observer_norm > 1e-8]
            if len(observer):
                unit_observer = observer / observer_norm[:, None]
                representative = np.median(unit_observer, axis=0)
                if float(np.linalg.norm(representative)) < 0.25:
                    representative = unit_observer[len(unit_observer) // 2]
                representative /= np.linalg.norm(representative)
                azimuth = float(np.degrees(np.arctan2(representative[1], representative[0])))
                view_policy = "source_camera_side"

    padding = 0.62 * body_height
    x_min = float(np.min(horizontal[:, 0]) - padding)
    x_max = float(np.max(horizontal[:, 0]) + padding)
    z_min = float(np.min(horizontal[:, 1]) - padding)
    z_max = float(np.max(horizontal[:, 1]) + padding)
    minimum_span = 1.55 * body_height
    if x_max - x_min < minimum_span:
        midpoint = 0.5 * (x_min + x_max)
        x_min, x_max = midpoint - minimum_span / 2.0, midpoint + minimum_span / 2.0
    if z_max - z_min < minimum_span:
        midpoint = 0.5 * (z_min + z_max)
        z_min, z_max = midpoint - minimum_span / 2.0, midpoint + minimum_span / 2.0

    y_min = float(ground - 0.06 * body_height)
    finite_y = valid_data[..., 1][np.isfinite(valid_data[..., 1])]
    y_max = float(np.percentile(finite_y, 99) + 0.12 * body_height) if len(finite_y) else ground + body_height
    y_max = max(y_max, ground + 1.12 * body_height)
    spans = (x_max - x_min, z_max - z_min, y_max - y_min)
    return {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "y_min": y_min,
        "y_max": y_max,
        "azimuth": azimuth,
        "elevation": 19.0,
        "view_policy": view_policy,
        "box_aspect": spans,
        "floor_radius": 0.5 * max(spans[0], spans[1]),
        "floor_x": 0.5 * (x_min + x_max),
        "floor_z": 0.5 * (z_min + z_max),
    }


def _body_forward(joints: np.ndarray, path_direction: np.ndarray) -> np.ndarray:
    hip_axis = joints[4] - joints[1]
    shoulder_axis = joints[11] - joints[14]
    lateral_axis = 0.45 * hip_axis + 0.55 * shoulder_axis
    lateral_axis[1] = 0.0
    forward = np.asarray([-lateral_axis[2], 0.0, lateral_axis[0]])
    if np.linalg.norm(forward) < 1e-8:
        forward = np.asarray(path_direction, dtype=np.float64)
    if np.linalg.norm(forward) < 1e-8:
        forward = np.asarray([0.0, 0.0, 1.0])
    forward /= np.linalg.norm(forward)
    return forward


def _render_mannequin(
    axis,
    joints: np.ndarray,
    body_height: float,
    ground: float,
    path_direction: np.ndarray,
    contacts: np.ndarray,
) -> None:
    all_faces = []
    all_colors = []
    for bone in BONES:
        if bone in LEFT_BONES:
            color = "#38bdf8"
        elif bone in RIGHT_BONES:
            color = "#fb923c"
        else:
            color = "#dbe4ee"
        radius = body_height * (0.032 if bone in LEFT_BONES or bone in RIGHT_BONES else 0.045)
        faces = _cylinder_faces(joints[bone[0]], joints[bone[1]], radius)
        all_faces.extend(faces)
        all_colors.extend([color] * len(faces))

    torso_faces = _cylinder_faces(joints[0], joints[8], body_height * 0.105, sides=8)
    pelvis_faces = _cylinder_faces(joints[1], joints[4], body_height * 0.075, sides=8)
    shoulder_faces = _cylinder_faces(joints[11], joints[14], body_height * 0.065, sides=8)
    for faces, color in (
        (torso_faces, "#aebdca"), (pelvis_faces, "#94a3b8"), (shoulder_faces, "#cbd5e1")
    ):
        all_faces.extend(faces)
        all_colors.extend([color] * len(faces))

    forward = _body_forward(joints, path_direction)
    chest_center = joints[8] + forward * (0.108 * body_height)
    chest_faces = _box_faces(
        chest_center, forward, half_length=0.014 * body_height,
        half_width=0.060 * body_height, half_height=0.070 * body_height,
    )
    all_faces.extend(chest_faces)
    all_colors.extend(["#facc15"] * len(chest_faces))
    for ankle, color in ((3, "#fb923c"), (6, "#38bdf8")):
        center = joints[ankle] + forward * (0.055 * body_height)
        center[1] = max(center[1], ground + 0.028 * body_height)
        foot_faces = _box_faces(
            center, forward, half_length=0.095 * body_height,
            half_width=0.045 * body_height, half_height=0.025 * body_height,
        )
        all_faces.extend(foot_faces)
        all_colors.extend([color] * len(foot_faces))

    axis.add_collection3d(
        Poly3DCollection(
            all_faces, facecolors=all_colors, edgecolors="#1f2937", linewidths=0.18, alpha=1.0
        )
    )

    head_center = joints[10]
    u = np.linspace(0.0, 2.0 * np.pi, 9)
    v = np.linspace(0.0, np.pi, 6)
    head_radius = body_height * 0.072
    head_x = head_center[0] + head_radius * np.outer(np.cos(u), np.sin(v))
    head_y = head_center[1] + head_radius * np.outer(np.ones_like(u), np.cos(v))
    head_z = head_center[2] + head_radius * np.outer(np.sin(u), np.sin(v))
    axis.plot_surface(
        head_x, head_z, head_y, color="#f0b88c", edgecolor="#1f2937", linewidth=0.15, shade=True
    )

    for joint_a, joint_b in BONES:
        segment = joints[[joint_a, joint_b]]
        axis.plot(
            segment[:, 0], segment[:, 2], np.full(2, ground + 0.004 * body_height),
            color="#0b0f14", linewidth=3.8, alpha=0.22,
        )
    for contact, ankle, color in zip(contacts, (3, 6), ("#fb923c", "#38bdf8")):
        if contact:
            axis.scatter(
                [joints[ankle, 0]], [joints[ankle, 2]], [ground + 0.012 * body_height],
                s=82, marker="o", facecolors="none", edgecolors=color,
                linewidths=2.0, depthshade=False,
            )


def render_blind_pair_video(
    output: Path,
    scene: str,
    reference_video: Path,
    reference_frame_indices: np.ndarray,
    condition_data: dict[str, np.ndarray],
    method_a: str,
    method_b: str,
    valid: np.ndarray,
    fps: float,
    stride: int,
    source_camera_centers: np.ndarray | None = None,
) -> None:
    data_a = condition_data[method_a]
    data_b = condition_data[method_b]
    if data_a.shape != data_b.shape or len(data_a) != len(reference_frame_indices):
        raise ValueError("blind video inputs must have equal lengths")
    valid = np.asarray(valid, dtype=bool)
    root_path = data_a[:, 0]
    valid_data = np.concatenate([data_a[valid], data_b[valid]], axis=0)
    vertical_extent = np.ptp(valid_data[..., 1], axis=1) if len(valid_data) else np.asarray([1.0])
    body_height = max(float(np.median(vertical_extent)), 0.5)
    ground = float(np.percentile(valid_data[:, [3, 6], 1], 5)) if len(valid_data) else 0.0
    camera = _fixed_world_camera(
        root_path, valid, valid_data, body_height, ground,
        source_camera_centers=source_camera_centers,
    )
    frame_indices = np.flatnonzero(valid)[::max(int(stride), 1)]
    if not len(frame_indices):
        raise ValueError(f"{scene}: no valid frames to render")
    contacts_by_condition = {
        method_a: infer_ankle_contacts(data_a, valid, fps),
        method_b: infer_ankle_contacts(data_b, valid, fps),
    }

    figure = plt.figure(figsize=(12, 6), dpi=110, facecolor="#111827")
    grid = figure.add_gridspec(2, 3, height_ratios=(4.2, 1.15), hspace=0.04, wspace=0.05)
    reference_axis = figure.add_subplot(grid[0, 0])
    skeleton_axes = [figure.add_subplot(grid[0, column], projection="3d") for column in (1, 2)]
    path_axis = figure.add_subplot(grid[1, :])

    reference_axis.set_title("Reference video", color="#f8fafc")
    reference_axis.set_facecolor("#111827")
    reference_axis.set_axis_off()
    reference_artist = reference_axis.imshow(np.zeros((360, 360, 3), dtype=np.uint8))

    for label, axis in zip(("A", "B"), skeleton_axes):
        axis.set_title(label, color="#f8fafc")
        axis.set_facecolor("#1f2937")
        axis.set_proj_type("ortho")
        axis.set_box_aspect(camera["box_aspect"])
        axis.view_init(elev=camera["elevation"], azim=camera["azimuth"])
        axis.set_axis_off()

    path_axis.set_facecolor("#1f2937")
    path_axis.plot(root_path[:, 0], root_path[:, 2], color="#94a3b8", linewidth=2.0)
    traversed_line = path_axis.plot([], [], color="#38bdf8", linewidth=2.8)[0]
    path_point = path_axis.scatter([], [], color="#f43f5e", s=34)
    path_axis.set_title("Shared root trajectory (top view; red = current)", fontsize=9, color="#f8fafc")
    path_axis.set_aspect("auto")
    path_axis.grid(True, color="#475569", linewidth=0.6)
    path_axis.tick_params(labelsize=7, colors="#cbd5e1")
    for spine in path_axis.spines.values():
        spine.set_color("#64748b")
    path_axis.set_xlim(camera["x_min"], camera["x_max"])
    path_axis.set_ylim(camera["z_min"], camera["z_max"])
    time_artist = path_axis.text(
        0.99, 0.86, "", transform=path_axis.transAxes, ha="right", fontsize=9, color="#f8fafc"
    )
    figure.suptitle(f"{scene} | E1c blinded pair", color="#f8fafc")

    capture = FfmpegFrameReader(reference_video)
    writer = FFMpegWriter(
        fps=max(int(round(fps / max(int(stride), 1))), 1),
        codec="libx264",
        bitrate=2600,
        extra_args=[
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-level", "4.0",
            "-movflags", "+faststart",
        ],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with writer.saving(figure, str(output), dpi=110):
            for frame_index in frame_indices:
                try:
                    frame = capture.get_data(int(reference_frame_indices[frame_index]))
                    reference_artist.set_data(frame)
                except (IndexError, RuntimeError):
                    pass
                current_root = root_path[frame_index]
                direction = _path_direction(root_path, frame_index)
                for axis, motion, method in zip(skeleton_axes, (data_a, data_b), (method_a, method_b)):
                    axis.cla()
                    axis.set_title("A" if axis is skeleton_axes[0] else "B", color="#f8fafc")
                    axis.set_facecolor("#1f2937")
                    axis.set_proj_type("ortho")
                    axis.set_box_aspect(camera["box_aspect"])
                    axis.view_init(elev=camera["elevation"], azim=camera["azimuth"])
                    axis.set_axis_off()
                    axis.set_xlim(camera["x_min"], camera["x_max"])
                    axis.set_ylim(camera["z_min"], camera["z_max"])
                    axis.set_zlim(camera["y_min"], camera["y_max"])
                    floor_center = np.asarray([camera["floor_x"], ground, camera["floor_z"]])
                    floor_spacing = max(
                        body_height * 0.28,
                        2.0 * float(camera["floor_radius"]) / 14.0,
                    )
                    _floor_material(
                        axis, floor_center, ground - 0.025 * body_height,
                        float(camera["floor_radius"]), floor_spacing,
                    )
                    path_height = ground + 0.009 * body_height
                    axis.plot(
                        root_path[:, 0], root_path[:, 2], np.full(len(root_path), path_height),
                        color="#94a3b8", linewidth=1.5, alpha=0.85,
                    )
                    axis.plot(
                        root_path[: frame_index + 1, 0], root_path[: frame_index + 1, 2],
                        np.full(frame_index + 1, path_height + 0.003 * body_height),
                        color="#22d3ee", linewidth=2.4, alpha=0.95,
                    )
                    _render_mannequin(
                        axis, motion[frame_index], body_height, ground, direction,
                        contacts_by_condition[method][frame_index],
                    )
                    axis.scatter(
                        [motion[frame_index, 0, 0]], [motion[frame_index, 0, 2]],
                        [motion[frame_index, 0, 1]], color="#f43f5e", s=13, depthshade=False,
                    )
                traversed_line.set_data(root_path[: frame_index + 1, 0], root_path[: frame_index + 1, 2])
                path_point.set_offsets([[current_root[0], current_root[2]]])
                time_artist.set_text(f"t = {frame_index / fps:.2f} s")
                writer.grab_frame()
    finally:
        capture.close()
        plt.close(figure)


def aggregate_condition_rows(rows: list[dict]) -> dict:
    result = {}
    conditions = sorted({row["condition"] for row in rows})
    ignored = {"scene", "cue", "condition", "shift_frames"}
    for condition in conditions:
        subset = [row for row in rows if row["condition"] == condition]
        metrics = sorted(set().union(*(set(row) - ignored for row in subset)))
        summary = {"clip_count": len(subset)}
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in subset if isinstance(row.get(metric), (int, float)) and np.isfinite(row[metric])],
                dtype=np.float64,
            )
            if len(values):
                summary[metric] = {"mean": float(values.mean()), "median": float(np.median(values))}
        result[condition] = summary
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--render-blind", action="store_true")
    parser.add_argument("--overwrite-videos", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scene")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    motionbert_root = Path(config["motionbert_root"])
    gem_root = Path(config["gem_root"])
    output_root = args.output_root or Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    fps_default = float(config.get("fps", 30.0))
    primary_shift = abs(int(config.get("primary_shift_frames", 15)))
    diagnostic_shifts = [int(value) for value in config.get("diagnostic_shift_frames", [-30, -15, -8, 0, 8, 15, 30])]
    seed = str(config.get("blind_seed", "e1c-batch23-v1"))
    render_stride = max(int(config.get("render_stride", 3)), 1)

    condition_rows = []
    dose_rows = []
    blind_key = []
    shared_rating_rows = []
    desync_rating_rows = []
    scene_reports = []

    clips = manifest["clips"]
    if args.scene is not None:
        clips = [clip for clip in clips if Path(clip["output"]).stem == args.scene]
        if not clips:
            raise ValueError(f"scene not found in manifest: {args.scene}")
    if args.limit is not None:
        clips = clips[: args.limit]
    for clip in clips:
        scene = Path(clip["output"]).stem
        cue = cue_label(clip)
        mb_dir = motionbert_root / scene
        mb_path = mb_dir / "motionbert_lifting" / "X3D.npy"
        mb_meta_path = mb_dir / "crop_meta.npz"
        gem_path = gem_root / scene / "gem_common17.npz"
        if not mb_path.is_file() or not mb_meta_path.is_file() or not gem_path.is_file():
            raise FileNotFoundError(f"missing E1c input for {scene}")
        motionbert = np.load(mb_path)
        with np.load(mb_meta_path, allow_pickle=False) as meta:
            motionbert_frames = np.asarray(meta["local_frames"], dtype=np.int64)
            fps = float(meta["fps"])
        with np.load(gem_path, allow_pickle=False) as gem_data:
            gem_incam = np.asarray(gem_data["joints_incam"], dtype=np.float64)
            gem_global_raw = np.asarray(gem_data["joints_global"], dtype=np.float64)
            gem_frames = np.asarray(gem_data["local_frames"], dtype=np.int64)
            gem_valid = np.asarray(gem_data["valid_mask"], dtype=bool)
        gem_global, world_up_corrected = canonicalize_world_up(gem_global_raw, gem_valid)

        aligned = align_by_local_frame(
            motionbert, motionbert_frames, gem_global, gem_frames, gem_valid
        )
        valid = aligned.valid_motionbert & aligned.valid_gem
        variants = construct_shared_root_variants(aligned.motionbert, aligned.gem, valid)
        sign = deterministic_sign(seed, scene)
        signed_shift = sign * primary_shift
        primary = desynchronize_articulation(variants.gem_native, variants.valid, signed_shift)
        native = primary.native
        motionbert_shared = variants.motionbert_shared_root[primary.trajectory_indices]
        desynchronized = primary.desynchronized
        common_valid = primary.valid & variants.valid[primary.trajectory_indices]

        gem_frame_index = {int(frame): index for index, frame in enumerate(gem_frames)}
        incam_aligned = gem_incam[
            [gem_frame_index[int(frame)] for frame in aligned.local_frames]
        ]
        incam_retained = incam_aligned[primary.trajectory_indices]
        recovered_camera = recover_camera_from_joint_pairs(
            incam_retained, native, common_valid
        )

        mb_index = {int(frame): index for index, frame in enumerate(motionbert_frames)}
        reference_indices = np.asarray(
            [mb_index[int(aligned.local_frames[index])] for index in primary.trajectory_indices],
            dtype=np.int64,
        )
        retained_local_frames = aligned.local_frames[primary.trajectory_indices]
        scene_dir = output_root / "variants" / scene
        scene_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scene_dir / "e1c_variants.npz",
            gem_native=native.astype(np.float32),
            motionbert_shared_root=motionbert_shared.astype(np.float32),
            gem_desynchronized=desynchronized.astype(np.float32),
            local_frames=retained_local_frames,
            reference_frame_indices=reference_indices,
            valid_mask=common_valid,
            primary_shift_frames=np.asarray(signed_shift, dtype=np.int64),
            fps=np.asarray(fps or fps_default, dtype=np.float32),
            coordinate=np.asarray("GEM global coordinates before OBD metric registration"),
        )

        current_conditions = {
            "GEM_native": native,
            "MotionBERT_shared_root": motionbert_shared,
            "GEM_desynchronized": desynchronized,
        }
        for condition, motion in current_conditions.items():
            condition_rows.append(
                {
                    "scene": scene,
                    "cue": cue,
                    "condition": condition,
                    "shift_frames": signed_shift if condition == "GEM_desynchronized" else 0,
                    **coupling_metrics(motion, common_valid, fps or fps_default),
                }
            )

        for offset in diagnostic_shifts:
            if abs(offset) >= len(variants.gem_native) - 2:
                continue
            shifted = desynchronize_articulation(variants.gem_native, variants.valid, offset)
            native_metrics = coupling_metrics(shifted.native, shifted.valid, fps or fps_default)
            shifted_metrics = coupling_metrics(shifted.desynchronized, shifted.valid, fps or fps_default)
            dose_rows.append(
                {
                    "scene": scene,
                    "cue": cue,
                    "shift_frames": offset,
                    **{f"native_{key}": value for key, value in native_metrics.items()},
                    **{f"desync_{key}": value for key, value in shifted_metrics.items()},
                }
            )

        shared_a, shared_b = blind_pair(
            seed, scene, "shared_trajectory", "GEM_native", "MotionBERT_shared_root"
        )
        desync_a, desync_b = blind_pair(
            seed, scene, "desync_sensitivity", "GEM_native", "GEM_desynchronized"
        )
        shared_video = output_root / "blind_shared_trajectory" / f"{scene}_blind.mp4"
        desync_video = output_root / "blind_desync_sensitivity" / f"{scene}_blind.mp4"
        blind_key.extend(
            [
                {
                    "scene": scene,
                    "comparison": "shared_trajectory",
                    "A": shared_a,
                    "B": shared_b,
                    "video": str(shared_video),
                },
                {
                    "scene": scene,
                    "comparison": "desync_sensitivity",
                    "A": desync_a,
                    "B": desync_b,
                    "signed_shift_frames": signed_shift,
                    "video": str(desync_video),
                },
            ]
        )
        common_rating = {
            "scene": scene,
            "cue": cue,
            "A_technical_success_0_1": "",
            "B_technical_success_0_1": "",
            "A_action_path_coherence_1_5": "",
            "B_action_path_coherence_1_5": "",
            "A_foot_contact_naturalness_1_5": "",
            "B_foot_contact_naturalness_1_5": "",
            "A_cue_or_onset_fidelity_1_5": "",
            "B_cue_or_onset_fidelity_1_5": "",
            "preference_A_B_tie": "",
            "notes": "",
        }
        shared_rating_rows.append({"video": str(shared_video), **common_rating})
        desync_rating_rows.append({"video": str(desync_video), **common_rating})

        if args.render_blind:
            reference_video = mb_dir / "focus_crop.mp4"
            if args.overwrite_videos or not shared_video.is_file():
                print(f"rendering shared-trajectory blind pair: {scene}")
                render_blind_pair_video(
                    shared_video, scene, reference_video, reference_indices,
                    current_conditions, shared_a, shared_b, common_valid,
                    fps or fps_default, render_stride,
                    source_camera_centers=recovered_camera.camera_center,
                )
            if args.overwrite_videos or not desync_video.is_file():
                print(f"rendering desynchronization blind pair: {scene}")
                render_blind_pair_video(
                    desync_video, scene, reference_video, reference_indices,
                    current_conditions, desync_a, desync_b, common_valid,
                    fps or fps_default, render_stride,
                    source_camera_centers=recovered_camera.camera_center,
                )

        scene_reports.append(
            {
                "scene": scene,
                "cue": cue,
                "input_frames": int(len(aligned.local_frames)),
                "retained_frames": int(len(native)),
                "common_valid_frames": int(common_valid.sum()),
                "primary_shift_frames": signed_shift,
                "gem_body_scale": variants.gem_body_scale,
                "source_camera_recovered_frames": int(recovered_camera.valid.sum()),
                "source_camera_fit_rmse_median": (
                    float(np.nanmedian(recovered_camera.fit_rmse[recovered_camera.valid]))
                    if recovered_camera.valid.any() else None
                ),
                "world_up_canonicalization": "rotate_x_180" if world_up_corrected else "unchanged",
            }
        )
        print(f"E1c prepared {scene}: {int(common_valid.sum())}/{len(common_valid)} valid")

    def write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(finite_or_none(rows))

    write_csv(output_root / "e1c_metrics.csv", condition_rows)
    write_csv(output_root / "e1c_shift_dose.csv", dose_rows)
    write_csv(output_root / "blind_shared_trajectory_rating.csv", shared_rating_rows)
    write_csv(output_root / "blind_desync_sensitivity_rating.csv", desync_rating_rows)
    (output_root / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = {
        "experiment": "E1c action-trajectory coupling pilot",
        "clip_count": len(scene_reports),
        "primary_shift_frames": primary_shift,
        "primary_shift_policy": "deterministic blinded sign per clip; no wrap or padding",
        "render_view_policy": (
            "World-fixed orthographic camera placed on the recovered source-camera side; "
            "trajectory-side view is used only as a fallback. Anatomical front is not forced "
            "to agree with the path direction."
        ),
        "comparison_boundary": (
            "MotionBERT_shared_root is an assembly control using GEM root trajectory, not a native "
            "MotionBERT trajectory. Coupling metrics use inferred common-17 ankle contacts and are "
            "diagnostics without world-trajectory ground truth."
        ),
        "conditions": {
            "GEM_native": "GEM global common-17 articulation and native root trajectory",
            "MotionBERT_shared_root": (
                "MotionBERT root-relative articulation after one sequence-level rotation and scale, "
                "placed on the exact GEM root trajectory"
            ),
            "GEM_desynchronized": (
                "GEM root trajectory at t plus GEM root-relative articulation at t+offset"
            ),
        },
        "aggregate": aggregate_condition_rows(condition_rows),
        "scenes": finite_or_none(scene_reports),
        "blind_shared_rating": str(output_root / "blind_shared_trajectory_rating.csv"),
        "blind_desync_rating": str(output_root / "blind_desync_sensitivity_rating.csv"),
        "blind_key": str(output_root / "blind_key.json"),
    }
    (output_root / "e1c_report.json").write_text(
        json.dumps(finite_or_none(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
