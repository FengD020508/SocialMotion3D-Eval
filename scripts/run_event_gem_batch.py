#!/usr/bin/env python3
"""Run one fixed-camera GEM reconstruction per IDD-PeD interaction event."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def _valid_smpl(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        return "body_params_incam" in payload and "body_params_global" in payload
    except Exception:
        return False


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _find_video(inputs: Path, output_name: str) -> Path:
    path = inputs / output_name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _rebase_camera_trajectory(T_c2w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express a sliced camera trajectory in its first camera frame."""
    T = np.asarray(T_c2w, dtype=np.float64)
    if T.ndim != 3 or T.shape[1:] != (4, 4) or len(T) == 0:
        raise ValueError(f"invalid T_c2w shape: {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError("T_c2w contains non-finite values")
    anchor = np.linalg.inv(T[0])
    rebased = np.einsum("ij,njk->nik", anchor, T).astype(np.float32)
    rebased[0] = np.eye(4, dtype=np.float32)
    inverse = np.linalg.inv(rebased).astype(np.float32)
    return rebased, inverse


def _slice_camera(
    source: Path,
    output: Path,
    source_frames: np.ndarray,
    save_camera_npz,
    validate_camera,
) -> None:
    with np.load(source, allow_pickle=False) as camera:
        if "frame_numbers" not in camera.files:
            raise ValueError(f"{source}: missing frame_numbers")
        available = np.asarray(camera["frame_numbers"], dtype=np.int64).reshape(-1)
        lookup = {int(frame): index for index, frame in enumerate(available)}
        missing = [int(frame) for frame in source_frames if int(frame) not in lookup]
        if missing:
            raise ValueError(f"{source}: missing {len(missing)} target frames; first={missing[0]}")
        indices = np.asarray([lookup[int(frame)] for frame in source_frames], dtype=np.int64)
        T_c2w, T_w2c = _rebase_camera_trajectory(camera["T_c2w"][indices])
        fps = float(camera["fps"])
        metadata = json.loads(str(camera["metadata_json"].item()))
        metadata.update(
            {
                "normalization_origin": "event_first_frame",
                "normalization_source_frame": int(source_frames[0]),
                "source_scene_camera": str(source),
            }
        )
        intrinsics = camera["intrinsics"]
        timestamps = camera["timestamps"][indices]
        confidence = camera["tracking_confidence"][indices]
        failed = camera["tracking_failed"][indices]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    save_camera_npz(
        temporary,
        T_c2w,
        T_w2c,
        intrinsics,
        source_frames,
        timestamps,
        confidence,
        failed,
        fps,
        metadata,
    )
    validation = validate_camera(T_c2w, T_w2c)
    if not validation["passed"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"rebased event camera failed validation: {validation}")
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genmo-root", type=Path, required=True)
    parser.add_argument("--event-inputs", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--camera-method", choices=("droid", "megasam"), default="megasam")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--hmr2-ckpt", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-target-video", action="store_true")
    args = parser.parse_args()

    genmo = args.genmo_root.resolve()
    sys.path.insert(0, str(genmo / "scripts"))
    from idd_ped.target_annotation import (  # type: ignore
        load_target_track,
        save_target_npz,
        write_target_video_and_overlay,
    )
    from idd_ped.camera_geometry import save_camera_npz, validate_camera  # type: ignore

    event_manifest = json.loads(
        (args.event_inputs / "selection_manifest.json").read_text(encoding="utf-8")
    )
    scene_manifest = json.loads(args.scene_manifest.read_text(encoding="utf-8"))
    scene_by_id = {entry["scene_id"]: entry for entry in scene_manifest["clips"]}
    selected = set(args.ids or [])
    entries = [
        entry for entry in event_manifest["clips"] if not selected or entry["clip_id"] in selected
    ]
    rows: list[dict] = []
    summary_path = args.output_root / "run_summary.json"
    fps = float(event_manifest["fps"])

    for index, entry in enumerate(entries, start=1):
        clip_id = entry["clip_id"]
        output_dir = args.output_root / clip_id
        smpl_path = output_dir / "smpl_params.pt"
        row = {
            "clip_id": clip_id,
            "event_id": entry["event_id"],
            "scene_id": entry["scene_id"],
            "camera_method": args.camera_method,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.resume and _valid_smpl(smpl_path) and (output_dir / "target_track.npz").is_file():
            row["status"] = "skipped_valid"
            rows.append(row)
            _write_summary(summary_path, rows)
            print(f"[{index}/{len(entries)}] {clip_id}: already valid", flush=True)
            continue

        started = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            video = _find_video(args.event_inputs, entry["output"])
            track = load_target_track(args.database, entry)
            target_track = output_dir / "target_track.npz"
            save_target_npz(target_track, track)
            target_video = output_dir / "target_video.mp4"
            overlay = output_dir / "target_overlay.mp4"
            write_target_video_and_overlay(video, track, target_video, overlay, fps)

            scene = scene_by_id[entry["scene_id"]]
            source_frames = np.asarray(track["source_frames"], dtype=np.int64)
            target_cameras = {}
            for method in ("droid", "megasam"):
                scene_camera = (
                    args.camera_root / method / scene["clip_id"] / "camera_trajectory.npz"
                )
                target_cameras[method] = output_dir / f"camera_{method}.npz"
                _slice_camera(
                    scene_camera,
                    target_cameras[method],
                    source_frames,
                    save_camera_npz,
                    validate_camera,
                )
            target_camera = target_cameras[args.camera_method]

            demo_root = output_dir / "_genmo"
            command = [
                sys.executable,
                "scripts/demo/demo_smpl_hpe.py",
                "--video",
                str(target_video),
                "--camera_traj",
                str(target_camera),
                "--target_track",
                str(target_track),
                "--hmr2_ckpt",
                str(args.hmr2_ckpt),
                "--output_root",
                str(demo_root),
                "--no_render",
            ]
            if args.ckpt:
                command += ["--ckpt_path", str(args.ckpt)]
            print(f"[{index}/{len(entries)}] {clip_id}: starting GEM", flush=True)
            subprocess.run(command, cwd=genmo, check=True)
            generated = demo_root / target_video.stem
            for item in generated.iterdir():
                destination = output_dir / item.name
                if destination.exists():
                    if destination.is_dir():
                        shutil.rmtree(destination)
                    else:
                        destination.unlink()
                shutil.move(str(item), str(destination))
            shutil.rmtree(demo_root, ignore_errors=True)
            if not _valid_smpl(smpl_path):
                raise RuntimeError(f"invalid GEM result: {smpl_path}")
            if not args.keep_target_video:
                target_video.unlink(missing_ok=True)
            row["status"] = "ok"
            row["frames"] = int(len(track["source_frames"]))
            row["elapsed_seconds"] = time.monotonic() - started
        except Exception as error:
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
            rows.append(row)
            _write_summary(summary_path, rows)
            raise
        row["finished_at"] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        _write_summary(summary_path, rows)
    print(f"completed {len(rows)} GEM event runs", flush=True)


if __name__ == "__main__":
    main()
