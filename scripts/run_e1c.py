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

from socialmotion3d_eval.e1 import BONES, align_by_local_frame
from socialmotion3d_eval.e1c import (
    construct_shared_root_variants,
    coupling_metrics,
    desynchronize_articulation,
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


def _floor_grid(axis, root: np.ndarray, ground: float, radius: float, spacing: float) -> None:
    spacing = max(float(spacing), 1e-3)
    x0 = np.floor((root[0] - radius) / spacing) * spacing
    z0 = np.floor((root[2] - radius) / spacing) * spacing
    values_x = np.arange(x0, root[0] + radius + spacing, spacing)
    values_z = np.arange(z0, root[2] + radius + spacing, spacing)
    for value in values_x:
        axis.plot(
            [value, value], [root[2] - radius, root[2] + radius], [ground, ground],
            color="#d1d5db", linewidth=0.45, alpha=0.75,
        )
    for value in values_z:
        axis.plot(
            [root[0] - radius, root[0] + radius], [value, value], [ground, ground],
            color="#d1d5db", linewidth=0.45, alpha=0.75,
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
    radius = max(body_height * 0.85, 0.65)
    ground = float(np.percentile(valid_data[:, [3, 6], 1], 5)) if len(valid_data) else 0.0
    frame_indices = np.flatnonzero(valid)[::max(int(stride), 1)]
    if not len(frame_indices):
        raise ValueError(f"{scene}: no valid frames to render")

    figure = plt.figure(figsize=(12, 6), dpi=110)
    grid = figure.add_gridspec(2, 3, height_ratios=(4.2, 1.15), hspace=0.04, wspace=0.05)
    reference_axis = figure.add_subplot(grid[0, 0])
    skeleton_axes = [figure.add_subplot(grid[0, column], projection="3d") for column in (1, 2)]
    path_axis = figure.add_subplot(grid[1, :])

    reference_axis.set_title("Reference video")
    reference_axis.set_axis_off()
    reference_artist = reference_axis.imshow(np.zeros((360, 360, 3), dtype=np.uint8))

    for label, axis in zip(("A", "B"), skeleton_axes):
        axis.set_title(label)
        axis.set_box_aspect((1.0, 1.0, 1.3))
        axis.view_init(elev=17, azim=-65)
        axis.set_axis_off()

    path_axis.plot(root_path[:, 0], root_path[:, 2], color="#9ca3af", linewidth=2.0)
    traversed_line = path_axis.plot([], [], color="#2563eb", linewidth=2.6)[0]
    path_point = path_axis.scatter([], [], color="#ef4444", s=30)
    path_axis.set_title("Shared root trajectory (top view; red = current)", fontsize=9)
    path_axis.set_aspect("auto")
    path_axis.grid(True, color="#e5e7eb", linewidth=0.6)
    path_axis.tick_params(labelsize=7)
    margin = max(body_height * 0.25, 0.1)
    path_axis.set_xlim(float(root_path[:, 0].min() - margin), float(root_path[:, 0].max() + margin))
    path_axis.set_ylim(float(root_path[:, 2].min() - margin), float(root_path[:, 2].max() + margin))
    time_artist = path_axis.text(0.99, 0.86, "", transform=path_axis.transAxes, ha="right", fontsize=9)
    figure.suptitle(f"{scene} | E1c blinded pair")

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
                for axis, motion in zip(skeleton_axes, (data_a, data_b)):
                    axis.cla()
                    axis.set_title("A" if axis is skeleton_axes[0] else "B")
                    axis.set_box_aspect((1.0, 1.0, 1.3))
                    axis.view_init(elev=17, azim=-65)
                    axis.set_axis_off()
                    axis.set_xlim(current_root[0] - radius, current_root[0] + radius)
                    axis.set_ylim(current_root[2] - radius, current_root[2] + radius)
                    axis.set_zlim(ground - 0.08 * body_height, ground + 1.35 * body_height)
                    _floor_grid(axis, current_root, ground, radius, body_height * 0.35)
                    for joint_a, joint_b in BONES:
                        segment = motion[frame_index, [joint_a, joint_b]]
                        axis.plot(
                            segment[:, 0], segment[:, 2], segment[:, 1],
                            color="#2563eb", linewidth=2.4,
                        )
                    axis.scatter(
                        [motion[frame_index, 0, 0]], [motion[frame_index, 0, 2]],
                        [motion[frame_index, 0, 1]], color="#ef4444", s=12, depthshade=False,
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

    clips = manifest["clips"][: args.limit] if args.limit is not None else manifest["clips"]
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
            gem_global = np.asarray(gem_data["joints_global"], dtype=np.float64)
            gem_frames = np.asarray(gem_data["local_frames"], dtype=np.int64)
            gem_valid = np.asarray(gem_data["valid_mask"], dtype=bool)

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
                )
            if args.overwrite_videos or not desync_video.is_file():
                print(f"rendering desynchronization blind pair: {scene}")
                render_blind_pair_video(
                    desync_video, scene, reference_video, reference_indices,
                    current_conditions, desync_a, desync_b, common_valid,
                    fps or fps_default, render_stride,
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
