#!/usr/bin/env python3
"""Render E1d native-output review videos.

E1d deliberately compares the information delivered by each pipeline rather
than forcing both methods into the common-17 representation used by E1c:

* GEM: native SMPL-X surface and native global root trajectory.
* MotionBERT: native 17-joint root-relative motion, with no imputed trajectory.

A single clip-level rigid display alignment and body-scale factor are applied
to MotionBERT.  They make the view readable but do not add per-frame pose or
trajectory information.  The reference video and the two anonymous outputs are
then encoded into one Windows-compatible H.264 video.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from socialmotion3d_eval.e1 import BONES, align_by_local_frame
from socialmotion3d_eval.e1c import construct_shared_root_variants, recover_camera_from_joint_pairs


LEFT_BONES = {(0, 4), (4, 5), (5, 6), (8, 11), (11, 12), (12, 13)}
RIGHT_BONES = {(0, 1), (1, 2), (2, 3), (8, 14), (14, 15), (15, 16)}


class RawVideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps:.8g}", "-i", "-", "-an",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-profile:v", "high", "-level", "4.0", "-movflags", "+faststart", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg exited with status {return_code}")


def blind_pair(seed: str, scene: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:native-output:{scene}".encode()).hexdigest()
    return ("GEM_native", "MotionBERT_native") if int(digest[:8], 16) % 2 == 0 else (
        "MotionBERT_native", "GEM_native"
    )


def video_frame(capture: cv2.VideoCapture, index: int, size: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        return np.full((size, size, 3), 24, dtype=np.uint8)
    height, width = frame.shape[:2]
    scale = min(size / max(width, 1), size / max(height, 1))
    resized = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    canvas = np.full((size, size, 3), 18, dtype=np.uint8)
    y0 = (size - resized.shape[0]) // 2
    x0 = (size - resized.shape[1]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def load_body_model(gem_code_root: Path, body_model_root: Path):
    sys.path.insert(0, str(gem_code_root))
    from gem.utils.smplx_utils import make_smplx

    return make_smplx("supermotion", model_path=str(body_model_root)).cpu().eval()


def smpl_vertices_and_joints(body_model, params: dict, indices: np.ndarray, batch_size: int = 48):
    vertices = []
    joints = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            chosen = torch.as_tensor(indices[start : start + batch_size], dtype=torch.long)
            batch = {key: value.index_select(0, chosen).cpu() for key, value in params.items()}
            output = body_model(**batch)
            vertices.append(output.vertices.cpu())
            joints.append(output.joints.cpu())
    return torch.cat(vertices).numpy(), torch.cat(joints).numpy()


def source_observer(common17_path: Path) -> np.ndarray:
    with np.load(common17_path, allow_pickle=False) as data:
        incam = np.asarray(data["joints_incam"], dtype=np.float64)
        world = np.asarray(data["joints_global"], dtype=np.float64)
        valid = np.asarray(data["valid_mask"], dtype=bool)
    recovered = recover_camera_from_joint_pairs(incam, world, valid)
    observer = recovered.camera_center[recovered.valid] - world[recovered.valid, 0]
    horizontal = np.median(observer[:, [0, 2]], axis=0)
    if not np.isfinite(horizontal).all() or np.linalg.norm(horizontal) < 1e-8:
        horizontal = np.asarray([1.0, -2.0])
    return horizontal / np.linalg.norm(horizontal)


class MeshRenderer:
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, roots: np.ndarray, observer: np.ndarray, size: int):
        self.size = size
        finite = vertices[np.isfinite(vertices).all(axis=-1)]
        y_min = float(np.min(finite[:, 1]))
        self.offset = np.asarray([roots[0, 0], y_min, roots[0, 2]], dtype=np.float64)
        shifted = vertices - self.offset
        self.path_points = roots - self.offset
        self.right, self.up = camera_basis(observer)
        eye_direction = np.asarray([observer[0], 0.42, observer[1]], dtype=np.float64)
        self.forward = -eye_direction / np.linalg.norm(eye_direction)

        projected = np.stack([shifted @ self.right, shifted @ self.up], axis=-1)
        low = np.nanpercentile(projected.reshape(-1, 2), 0.4, axis=0)
        high = np.nanpercentile(projected.reshape(-1, 2), 99.6, axis=0)
        center = 0.5 * (low + high)
        span = max(float(np.max(high - low)), 1.7)
        self.pixel_scale = 0.82 * size / span
        self.project_center = center

        local_template = shifted[0] - self.path_points[0]
        self.vertex_groups, self.faces = self._cluster_topology(local_template, np.asarray(faces, dtype=np.int32))
        self.group_counts = np.bincount(self.vertex_groups).astype(np.float64)
        self.ground_x = (float(shifted[..., 0].min()), float(shifted[..., 0].max()))
        self.ground_z = (float(shifted[..., 2].min()), float(shifted[..., 2].max()))

    @staticmethod
    def _cluster_topology(template: np.ndarray, faces: np.ndarray, target_vertices: int = 1450):
        lower = template.min(axis=0)
        best = None
        for voxel in np.geomspace(0.012, 0.11, 22):
            cells = np.floor((template - lower) / voxel).astype(np.int32)
            _, inverse = np.unique(cells, axis=0, return_inverse=True)
            score = abs(int(inverse.max()) + 1 - target_vertices)
            if best is None or score < best[0]:
                best = (score, inverse)
        inverse = best[1]
        remapped = inverse[faces]
        remapped = remapped[
            (remapped[:, 0] != remapped[:, 1])
            & (remapped[:, 1] != remapped[:, 2])
            & (remapped[:, 0] != remapped[:, 2])
        ]
        canonical = np.sort(remapped, axis=1)
        _, keep = np.unique(canonical, axis=0, return_index=True)
        return inverse, remapped[np.sort(keep)]

    def _simplify_vertices(self, vertices: np.ndarray) -> np.ndarray:
        shifted = np.asarray(vertices, dtype=np.float64) - self.offset
        simplified = np.zeros((len(self.group_counts), 3), dtype=np.float64)
        np.add.at(simplified, self.vertex_groups, shifted)
        return simplified / self.group_counts[:, None]

    def _project(self, points: np.ndarray) -> np.ndarray:
        uv = np.stack([points @ self.right, points @ self.up], axis=-1)
        normalized = uv - self.project_center
        return np.asarray([self.size * 0.5, self.size * 0.52]) + normalized * np.asarray(
            [self.pixel_scale, -self.pixel_scale]
        )

    def _draw_ground(self, image: np.ndarray) -> None:
        x_min, x_max = self.ground_x
        z_min, z_max = self.ground_z
        padding = 0.38 * max(x_max - x_min, z_max - z_min, 1.7)
        x_min, x_max = x_min - padding, x_max + padding
        z_min, z_max = z_min - padding, z_max + padding
        for value in np.linspace(x_min, x_max, 13):
            points = self._project(np.asarray([[value, 0.0, z_min], [value, 0.0, z_max]]))
            cv2.line(image, tuple(np.round(points[0]).astype(int)), tuple(np.round(points[1]).astype(int)), (66, 75, 88), 1, cv2.LINE_AA)
        for value in np.linspace(z_min, z_max, 13):
            points = self._project(np.asarray([[x_min, 0.0, value], [x_max, 0.0, value]]))
            cv2.line(image, tuple(np.round(points[0]).astype(int)), tuple(np.round(points[1]).astype(int)), (66, 75, 88), 1, cv2.LINE_AA)

    def frame(self, vertices: np.ndarray, frame_index: int) -> np.ndarray:
        image = np.full((self.size, self.size, 3), (22, 28, 37), dtype=np.uint8)
        self._draw_ground(image)
        path_2d = np.round(self._project(self.path_points)).astype(np.int32)
        if len(path_2d) > 1:
            cv2.polylines(image, [path_2d], False, (132, 142, 156), 2, cv2.LINE_AA)
            cv2.polylines(image, [path_2d[: frame_index + 1]], False, (224, 198, 42), 3, cv2.LINE_AA)
        cv2.circle(image, tuple(path_2d[frame_index]), 5, (83, 70, 244), -1, cv2.LINE_AA)

        points = self._simplify_vertices(vertices)
        triangles = points[self.faces]
        projected = np.round(self._project(points)).astype(np.int32)
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_norm = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(normal_norm[:, None], 1e-10)
        light = np.asarray([0.35, 0.84, -0.41])
        light /= np.linalg.norm(light)
        intensity = 0.38 + 0.62 * np.clip(np.abs(normals @ light), 0.0, 1.0)
        depth = triangles.mean(axis=1) @ self.forward
        for face_index in np.argsort(depth)[::-1]:
            polygon = projected[self.faces[face_index]]
            if abs(float(np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0]))) < 0.7:
                continue
            value = float(intensity[face_index])
            color = tuple(int(channel * value) for channel in (246, 128, 202))
            cv2.fillConvexPoly(image, polygon, color, cv2.LINE_AA)
        return image


def camera_basis(observer: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eye_direction = np.asarray([observer[0], 0.42, observer[1]], dtype=np.float64)
    forward = -eye_direction / np.linalg.norm(eye_direction)
    right = np.cross(forward, np.asarray([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return right, up


def render_skeleton(joints: np.ndarray, all_joints: np.ndarray, observer: np.ndarray, size: int) -> np.ndarray:
    right, up = camera_basis(observer)
    projected_all = np.stack([all_joints @ right, all_joints @ up], axis=-1)
    extent = np.nanpercentile(np.abs(projected_all), 99.5)
    extent = max(float(extent), 0.8)
    projected = np.stack([joints @ right, joints @ up], axis=-1)
    scale = 0.39 * size / extent
    center = np.asarray([size * 0.5, size * 0.54])
    points = center + projected * np.asarray([scale, -scale])

    image = np.full((size, size, 3), (22, 28, 37), dtype=np.uint8)
    grid_color = (56, 65, 77)
    for value in np.linspace(0.1, 0.9, 9):
        coordinate = int(round(value * size))
        cv2.line(image, (coordinate, int(size * 0.12)), (coordinate, int(size * 0.92)), grid_color, 1)
        cv2.line(image, (int(size * 0.08), coordinate), (int(size * 0.92), coordinate), grid_color, 1)
    cv2.line(image, (int(size * 0.08), int(size * 0.89)), (int(size * 0.92), int(size * 0.89)), (105, 117, 130), 2)

    for joint_a, joint_b in BONES:
        bone = (joint_a, joint_b)
        color = (248, 146, 56) if bone in LEFT_BONES else (72, 184, 249) if bone in RIGHT_BONES else (226, 232, 240)
        point_a = tuple(np.round(points[joint_a]).astype(int))
        point_b = tuple(np.round(points[joint_b]).astype(int))
        cv2.line(image, point_a, point_b, color, max(2, round(size / 150)), cv2.LINE_AA)
    for index, point in enumerate(points):
        color = (248, 146, 56) if index in (4, 5, 6, 11, 12, 13) else (72, 184, 249) if index in (1, 2, 3, 14, 15, 16) else (226, 232, 240)
        cv2.circle(image, tuple(np.round(point).astype(int)), max(3, round(size / 110)), color, -1, cv2.LINE_AA)
    return image


def panel_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 48), (10, 14, 20), -1)
    cv2.putText(output, label, (20, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (244, 247, 251), 2, cv2.LINE_AA)
    return output


def render_scene(args, body_model, scene: str, output: Path) -> dict:
    mb_dir = args.motionbert_root / scene
    gem_dir = args.gem_smpl_root / scene
    common17_path = args.gem_common17_root / scene / "gem_common17.npz"
    motionbert = np.load(mb_dir / "motionbert_lifting" / "X3D.npy")
    with np.load(mb_dir / "crop_meta.npz", allow_pickle=False) as metadata:
        mb_frames = np.asarray(metadata["local_frames"], dtype=np.int64)
        fps = float(metadata["fps"])
    with np.load(common17_path, allow_pickle=False) as common:
        gem_global = np.asarray(common["joints_global"], dtype=np.float64)
        gem_frames = np.asarray(common["local_frames"], dtype=np.int64)
        gem_valid = np.asarray(common["valid_mask"], dtype=bool)

    aligned = align_by_local_frame(motionbert, mb_frames, gem_global, gem_frames, gem_valid)
    variants = construct_shared_root_variants(
        aligned.motionbert, aligned.gem, aligned.valid_motionbert & aligned.valid_gem
    )
    mb_local = variants.motionbert_shared_root - variants.motionbert_shared_root[:, :1]
    valid_indices = np.flatnonzero(variants.valid)[:: max(args.stride, 1)]
    if not len(valid_indices):
        raise ValueError(f"{scene}: no valid aligned frames")

    try:
        raw = torch.load(gem_dir / "smpl_params.pt", map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument was introduced.
        raw = torch.load(gem_dir / "smpl_params.pt", map_location="cpu")
    raw_frames = np.asarray(raw["target_track"]["local_frames"], dtype=np.int64)
    raw_index = {int(frame): index for index, frame in enumerate(raw_frames)}
    mb_index = {int(frame): index for index, frame in enumerate(mb_frames)}
    selected_local = aligned.local_frames[valid_indices]
    selected_raw = np.asarray([raw_index[int(frame)] for frame in selected_local], dtype=np.int64)
    selected_reference = np.asarray([mb_index[int(frame)] for frame in selected_local], dtype=np.int64)

    params = raw["body_params_global"]
    vertices, joints = smpl_vertices_and_joints(body_model, params, selected_raw)
    roots = joints[:, 0]
    observer = source_observer(common17_path)
    mesh_renderer = MeshRenderer(vertices, body_model.faces, roots, observer, args.panel_size)

    local_selected = mb_local[valid_indices]
    ground = float(np.nanpercentile(local_selected[:, [3, 6], 1], 4))
    local_selected = local_selected.copy()
    local_selected[..., 1] -= ground
    reference_capture = cv2.VideoCapture(str(mb_dir / "focus_crop.mp4"))
    if not reference_capture.isOpened():
        raise RuntimeError(f"cannot open reference video for {scene}")

    method_a, method_b = blind_pair(args.seed, scene)
    writer = RawVideoWriter(
        output, 3 * args.panel_size, args.panel_size,
        max(fps / max(args.stride, 1), 1.0),
    )
    try:
        for output_index, reference_index in enumerate(selected_reference):
            reference = panel_label(video_frame(reference_capture, int(reference_index), args.panel_size), "Reference")
            gem_frame = mesh_renderer.frame(vertices[output_index], output_index)
            mb_frame = render_skeleton(local_selected[output_index], local_selected, observer, args.panel_size)
            frames = {"GEM_native": gem_frame, "MotionBERT_native": mb_frame}
            a = panel_label(frames[method_a], "A")
            b = panel_label(frames[method_b], "B")
            writer.write(np.concatenate([reference, a, b], axis=1))
    finally:
        reference_capture.release()
        writer.close()

    return {
        "scene": scene,
        "comparison": "native_end_to_end_output",
        "A": method_a,
        "B": method_b,
        "video": output.name,
        "frame_count": int(len(selected_local)),
        "fps": float(max(fps / max(args.stride, 1), 1.0)),
        "motionbert_display_adapter": "single clip-level rigid alignment + body scale; root trajectory removed",
        "gem_condition": "native SMPL-X body_params_global surface and trajectory",
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motionbert-root", type=Path, required=True)
    parser.add_argument("--gem-common17-root", type=Path, required=True)
    parser.add_argument("--gem-smpl-root", type=Path, required=True)
    parser.add_argument("--gem-code-root", type=Path, required=True)
    parser.add_argument("--body-model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blind-key", type=Path, required=True)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--seed", default="e1d-native-pilot-v1")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--panel-size", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenes = args.scene or sorted(
        path.name for path in args.gem_smpl_root.iterdir()
        if path.is_dir() and (path / "smpl_params.pt").is_file()
    )
    body_model = load_body_model(args.gem_code_root, args.body_model_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for index, scene in enumerate(scenes, start=1):
        output = args.output_root / f"{scene}_blind.mp4"
        if output.is_file() and not args.overwrite:
            print(f"[{index}/{len(scenes)}] reuse {output.name}", flush=True)
            records.append({"scene": scene, "video": output.name, "status": "reused"})
            continue
        print(f"[{index}/{len(scenes)}] render {scene}", flush=True)
        records.append(render_scene(args, body_model, scene, output))
    args.blind_key.parent.mkdir(parents=True, exist_ok=True)
    args.blind_key.write_text(
        json.dumps({"experiment": "E1d", "seed": args.seed, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"blind key: {args.blind_key}")


if __name__ == "__main__":
    main()
