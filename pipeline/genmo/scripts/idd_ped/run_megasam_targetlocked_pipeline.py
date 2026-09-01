#!/usr/bin/env python3
"""MegaSAM camera + annotation-locked GENMO joint reconstruction for IDD-PeD."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.idd_ped.camera_geometry import save_camera_npz, validate_camera  # noqa:E402
from scripts.idd_ped.target_annotation import load_target_track, save_target_npz, write_target_video_and_overlay  # noqa:E402


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def cid(entry: dict) -> str:
    return entry["output"].split("_", 1)[0]


def rebase_target_camera(full_path: Path, target_path: Path, track: dict) -> None:
    with np.load(full_path, allow_pickle=False) as z:
        full = {k: z[k] for k in z.files}
    idx = track["local_frames"].astype(int)
    if idx.min() < 0 or idx.max() >= len(full["T_c2w"]):
        raise ValueError("Target local frames fall outside MegaSAM camera trajectory")
    anchor = np.linalg.inv(full["T_c2w"][idx[0]].astype(np.float64))
    T = np.einsum("ij,njk->nik", anchor, full["T_c2w"].astype(np.float64)).astype(np.float32)
    Tw = np.linalg.inv(T).astype(np.float32)
    meta = json.loads(str(full["metadata_json"])); meta.update({
        "normalization_origin": "target_first_frame",
        "normalization_source_frame": int(track["source_frames"][0]),
        "scope": "target_track", "bbox_fingerprint": track["fingerprint"],
        "target_key": [track["set_id"], track["video_id"], track["pedestrian_id"]],
    })
    save_camera_npz(target_path, T[idx], Tw[idx], full["intrinsics"], track["source_frames"],
                    track["source_frames"] / float(full["fps"]), full["tracking_confidence"][idx],
                    full["tracking_failed"][idx], float(full["fps"]), meta)
    # Keep the complete ego trajectory in the same target-normalized world.
    full_meta = json.loads(str(full["metadata_json"])); full_meta.update({
        "normalization_origin": "target_first_frame",
        "normalization_source_frame": int(track["source_frames"][0]),
    })
    save_camera_npz(full_path, T, Tw, full["intrinsics"], full["frame_numbers"], full["timestamps"],
                    full["tracking_confidence"], full["tracking_failed"], float(full["fps"]), full_meta)


def validate_camera_file(
    path: Path,
    expected_frames: int | None = None,
    *,
    require_first_identity: bool = True,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as z:
        if "T_c2w" not in z.files or "T_w2c" not in z.files:
            raise KeyError(f"{path}: missing T_c2w/T_w2c")
        T_c2w = z["T_c2w"]
        T_w2c = z["T_w2c"]
        frame_numbers = z["frame_numbers"]
    if len(T_c2w) != len(T_w2c) or len(T_c2w) != len(frame_numbers):
        raise ValueError(f"{path}: inconsistent camera array lengths")
    if expected_frames is not None and len(T_c2w) != expected_frames:
        raise ValueError(f"{path}: {len(T_c2w)} poses, expected {expected_frames}")
    validation = validate_camera(T_c2w, T_w2c)
    validation["first_pose_identity_required"] = require_first_identity
    validation["passed"] = bool(
        validation["finite"]
        and validation["max_rotation_det_error"] < 1e-3
        and validation["max_inverse_error"] < 1e-3
        and (not require_first_identity or validation["first_pose_identity_error"] < 1e-3)
    )
    if not validation["passed"]:
        raise RuntimeError(f"{path}: camera validation failed: {validation}")
    return {"path": str(path), "frames": int(len(T_c2w)), "validation": validation}


def path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return 0


def cleanup_megasam_cache(work_root: Path, scene: str, cache_policy: str) -> dict:
    work = work_root / scene
    official_raw = ROOT / "third_party" / "mega-sam" / "outputs" / f"{scene}_droid.npz"
    targets = [work / "frames", work / "mono_depth", work / "megasam_raw.npz", official_raw]
    if cache_policy == "delete_all":
        targets.append(work / "metric_depth")
    removed = []
    for target in targets:
        if not target.exists():
            continue
        size = path_bytes(target)
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        removed.append({"path": str(target), "bytes": int(size)})
    metric = work / "metric_depth"
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scene": scene,
        "policy": cache_policy,
        "work_root": str(work_root),
        "removed": removed,
        "removed_bytes": int(sum(item["bytes"] for item in removed)),
        "metric_depth_retained_until_job_end": str(metric) if cache_policy == "metric_tmp" and metric.exists() else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=ROOT / "inputs/idd-ped_inputs")
    p.add_argument("--output_root", type=Path, default=ROOT / "outputs/idd_ped_joint_megasam_targetlocked")
    p.add_argument("--clips", nargs="+", default=["01", "03", "02", "04", "05"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--work_root", type=Path)
    p.add_argument("--cache_policy", choices=["keep", "metric_tmp", "delete_all"], default="keep")
    p.add_argument("--ckpt", type=Path, default=ROOT / "inputs/pretrained/gem_smpl.ckpt")
    p.add_argument("--hmr2_ckpt", type=Path, default=ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt")
    args = p.parse_args()
    db = args.inputs / "iddp_database.pkl"
    if not db.is_file():
        raise FileNotFoundError(f"Required annotation database missing: {db}")
    manifest = json.loads((args.inputs / "selection_manifest.json").read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    slurm_tmp = Path(os.environ["SLURM_TMPDIR"]).resolve() if os.environ.get("SLURM_TMPDIR") else None
    if args.work_root is not None:
        work_root = args.work_root.resolve()
    elif args.cache_policy == "metric_tmp":
        if slurm_tmp is None:
            raise RuntimeError("cache_policy=metric_tmp requires SLURM_TMPDIR (submit this as a Slurm job)")
        work_root = slurm_tmp / "socialmotion3d_megasam_work"
    else:
        work_root = args.output_root / "_megasam_work"
    if args.cache_policy == "metric_tmp" and (slurm_tmp is None or not work_root.is_relative_to(slurm_tmp)):
        raise ValueError("cache_policy=metric_tmp requires work_root to be inside SLURM_TMPDIR")
    work_root.mkdir(parents=True, exist_ok=True)
    cleanup_log = args.output_root / "megasam_cleanup_log.jsonl"
    summary = []
    for clip_id in args.clips:
        entry = next(x for x in manifest["clips"] if cid(x) == clip_id)
        video = args.inputs / entry["output"]; out = args.output_root / video.stem
        out.mkdir(parents=True, exist_ok=True)
        try:
            track = load_target_track(db, entry)
            save_target_npz(out / "target_track.npz", track)
            target_video = out / "target_track_fullframe.mp4"
            write_target_video_and_overlay(video, track, target_video, out / "target_bbox_overlay.mp4", float(manifest["fps"]))
            camera = out / "camera_trajectory.npz"
            if not (args.resume and camera.is_file()):
                run([sys.executable, "scripts/idd_ped/run_megasam.py", "--inputs", str(args.inputs),
                     "--clip_id", clip_id, "--output", str(camera), "--work_root", str(work_root), "--resume"])
            target_camera = out / "camera_trajectory_target.npz"
            rebase_target_camera(camera, target_camera, track)
            # The full trajectory is rebased at the first target frame, so
            # its video-first pose is generally not identity.  The target
            # trajectory below must still begin at identity.
            full_validation = validate_camera_file(camera, require_first_identity=False)
            target_validation = validate_camera_file(target_camera, expected_frames=len(track["source_frames"]))
            if args.cache_policy != "keep":
                cleanup = cleanup_megasam_cache(work_root, video.stem, args.cache_policy)
                cleanup["full_camera_validation"] = full_validation
                cleanup["target_camera_validation"] = target_validation
                with cleanup_log.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(cleanup) + "\n")

            cache_name = f"{track['pedestrian_id']}_{track['fingerprint'][:16]}"
            demo_root = out / f"_genmo_megasam_{cache_name}"
            generated = demo_root / target_video.stem
            old_cache = ROOT / "outputs/idd_ped_joint_targetlocked" / video.stem / "preprocess_target" / cache_name
            if old_cache.is_dir() and not (generated / "preprocess").exists():
                (generated / "preprocess").parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(old_cache, generated / "preprocess")
            run([sys.executable, "scripts/demo/demo_smpl_hpe.py", "--video", str(target_video),
                 "--camera_traj", str(target_camera), "--target_track", str(out / "target_track.npz"),
                 "--ckpt_path", str(args.ckpt), "--hmr2_ckpt", str(args.hmr2_ckpt),
                 "--output_root", str(demo_root), "--reprojection_only"])
            cache_dir = out / "preprocess_target" / cache_name
            if cache_dir.exists(): shutil.rmtree(cache_dir)
            cache_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(generated / "preprocess"), str(cache_dir))
            for name in ("smpl_params.pt", "used_bbx_xys.pt", "1_incam.mp4"):
                shutil.move(str(generated / name), str(out / name))
            shutil.copy2(out / "1_incam.mp4", out / "preview_reprojection.mp4")
            shutil.rmtree(demo_root)
            run([sys.executable, "scripts/idd_ped/export_targetlocked.py", "--clip_id", clip_id,
                 "--target_video", str(target_video), "--output_dir", str(out),
                 "--preprocess_dir", str(cache_dir)])
            report = json.loads((out / "quality_report.json").read_text())
            if not report["passed"]: raise RuntimeError("Quality report failed")
            summary.append({"clip_id": clip_id, "status": "ok", "output": str(out)})
        except Exception as exc:
            summary.append({"clip_id": clip_id, "status": "failed", "error": str(exc), "output": str(out)})
            (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
            raise
        (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2)+"\n")


if __name__ == "__main__":
    main()
