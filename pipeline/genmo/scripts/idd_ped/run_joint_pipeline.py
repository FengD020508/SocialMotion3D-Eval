#!/usr/bin/env python3
"""End-to-end IDD-PeD DROID + GENMO joint reconstruction driver."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def find_clip(inputs: Path, clip_id: str) -> Path:
    matches = sorted(inputs.glob(f"{clip_id}_*.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one {clip_id}_*.mp4; got {matches}")
    return matches[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=ROOT / "inputs/idd-ped_inputs")
    p.add_argument("--output_root", type=Path, default=ROOT / "outputs/idd_ped_joint")
    p.add_argument("--clips", nargs="+", default=["01", "03", "02", "04", "05"])
    p.add_argument("--source_root", type=Path, default=None)
    p.add_argument("--dynamic_mask", action="store_true")
    p.add_argument("--scale_mode", choices=["relative", "camera_height"], default="relative")
    p.add_argument("--camera_height", type=float, default=1.5)
    p.add_argument("--ckpt", type=Path, default=ROOT / "inputs/pretrained/gem_smpl.ckpt")
    p.add_argument("--hmr2_ckpt", type=Path,
                   default=ROOT / "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt")
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    manifest = args.inputs / "selection_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for clip_id in args.clips:
        video = find_clip(args.inputs, clip_id)
        out = args.output_root / video.stem
        out.mkdir(parents=True, exist_ok=True)
        camera = out / "camera_trajectory.npz"
        report = out / "quality_report.json"
        try:
            if not (args.resume and camera.is_file()):
                cmd = [sys.executable, "scripts/idd_ped/run_droid_slam.py",
                       "--inputs", str(args.inputs), "--clip_id", clip_id,
                       "--output", str(camera)]
                if args.source_root:
                    cmd += ["--source_root", str(args.source_root)]
                if args.calib:
                    cmd += ["--calib", str(args.calib)]
                if args.dynamic_mask:
                    cmd += ["--dynamic_mask"]
                run(cmd)

            smpl = out / "smpl_params.pt"
            if not (args.resume and smpl.is_file()):
                # demo output convention appends video stem; use a temporary
                # root then move the complete result into the joint clip dir.
                demo_root = out / "_genmo"
                run([sys.executable, "scripts/demo/demo_smpl_hpe.py",
                     "--video", str(video), "--camera_traj", str(camera),
                     "--ckpt_path", str(args.ckpt), "--hmr2_ckpt", str(args.hmr2_ckpt),
                     "--output_root", str(demo_root), "--reprojection_only"])
                generated = demo_root / video.stem
                for item in generated.iterdir():
                    target = out / item.name
                    if item.is_dir():
                        if target.exists():
                            shutil.rmtree(target)
                        shutil.move(str(item), str(target))
                    else:
                        shutil.move(str(item), str(target))
                shutil.rmtree(demo_root)
            if (out / "1_incam.mp4").is_file():
                shutil.copy2(out / "1_incam.mp4", out / "preview_reprojection.mp4")

            run([sys.executable, "scripts/idd_ped/export_joint_reconstruction.py",
                 "--clip_id", clip_id, "--video", str(video), "--output_dir", str(out),
                 "--manifest", str(manifest), "--scale_mode", args.scale_mode,
                 "--camera_height", str(args.camera_height),
                 "--preprocess_dir", str(out / "preprocess")])
            required = ["smpl_params.pt", "camera_trajectory.npz", "ego_trajectory_unity.json",
                        "human_trajectory_unity.json", "interaction_labels.json",
                        "preview_reprojection.mp4", "preview_global.mp4", "quality_report.json"]
            missing = [name for name in required if not (out / name).is_file()]
            if missing:
                raise RuntimeError(f"Missing required outputs: {missing}")
            summary.append({"clip_id": clip_id, "status": "ok", "output": str(out)})
        except Exception as exc:
            # Per-clip failure is visible and stops the prioritized validation;
            # never synthesize a static trajectory.
            summary.append({"clip_id": clip_id, "status": "failed", "error": str(exc), "output": str(out)})
            (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            raise
        (args.output_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
