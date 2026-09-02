#!/usr/bin/env python3
"""Run official MegaSAM on one full-frame IDD-PeD clip and export GENMO camera NPZ."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MEGASAM = ROOT / "third_party" / "mega-sam"
MEGASAM_PYTHON = Path(
    os.environ.get("MEGASAM_PYTHON", str(ROOT.parent / "envs/mega_sam/bin/python"))
)
sys.path.insert(0, str(ROOT))

from scripts.idd_ped.camera_geometry import normalize_and_y_up, save_camera_npz, validate_camera  # noqa:E402


def run(cmd: list[str], cwd: Path = MEGASAM) -> None:
    env = os.environ.copy()
    torch_lib = MEGASAM_PYTHON.parent.parent / "lib/python3.10/site-packages/torch/lib"
    env["LD_LIBRARY_PATH"] = f"{MEGASAM_PYTHON.parent.parent / 'lib'}:{torch_lib}:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = f"{MEGASAM / 'UniDepth'}:{MEGASAM / 'base/droid_slam'}:" + env.get("PYTHONPATH", "")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def load_camera_result(
    raw_out: Path, expected_frames: int, width: int, height: int
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an official MegaSAM result and convert intrinsics to input pixels."""
    if not raw_out.is_file():
        raise FileNotFoundError(f"MegaSAM did not produce {raw_out}")
    with np.load(raw_out, allow_pickle=False) as z:
        raw = {key: z[key] for key in z.files}
    T_raw = raw["cam_c2w"].astype(np.float32)
    if len(T_raw) != expected_frames:
        raise RuntimeError(
            f"MegaSAM pose length {len(T_raw)} != video length {expected_frames}"
        )
    T_c2w, T_w2c = normalize_and_y_up(T_raw)
    K = raw["intrinsic"].astype(np.float32)
    tracked_h, tracked_w = raw["images"].shape[1:3]
    K[0, :] *= width / float(tracked_w)
    K[1, :] *= height / float(tracked_h)
    intr = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], np.float32)
    return raw, T_c2w, T_w2c, K, intr


def valid_focal(K: np.ndarray) -> bool:
    focal = np.asarray([K[0, 0], K[1, 1]], dtype=np.float64)
    return bool(np.isfinite(focal).all() and np.all(focal > 0))


def clip_entry(inputs: Path, clip_id: str) -> tuple[dict, Path, float, int, int, int]:
    manifest = json.loads((inputs / "selection_manifest.json").read_text())
    entry = next(x for x in manifest["clips"] if x["output"].split("_", 1)[0] == clip_id)
    video = inputs / entry["output"]
    if not video.is_file():
        raise FileNotFoundError(video)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return entry, video, fps, n, width, height


def extract_frames(video: Path, frames: Path, expected: int) -> None:
    existing = sorted(frames.glob("*.png")) if frames.is_dir() else []
    if len(existing) == expected:
        return
    if frames.exists():
        shutil.rmtree(frames)
    frames.mkdir(parents=True)
    cap = cv2.VideoCapture(str(video)); index = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if not cv2.imwrite(str(frames / f"{index:06d}.png"), image):
            raise IOError(f"Failed writing frame {index}")
        index += 1
    cap.release()
    if index != expected:
        raise RuntimeError(f"Decoded {index} frames, expected {expected}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=ROOT / "inputs/idd-ped_inputs")
    p.add_argument("--clip_id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--work_root", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--disable_focal_optimization",
        action="store_true",
        help="hold the UniDepth FoV focal fixed (targeted recovery/debugging)",
    )
    args = p.parse_args()
    if not MEGASAM_PYTHON.is_file():
        raise FileNotFoundError(f"MegaSAM environment missing: {MEGASAM_PYTHON}")
    da_ckpt = MEGASAM / "Depth-Anything/checkpoints/depth_anything_vitl14.pth"
    mega_ckpt = MEGASAM / "checkpoints/megasam_final.pth"
    for required in (da_ckpt, mega_ckpt):
        if not required.is_file():
            raise FileNotFoundError(required)

    entry, video, fps, n, width, height = clip_entry(args.inputs, args.clip_id)
    scene = video.stem
    work = args.work_root / scene
    frames = work / "frames"
    mono_root = work / "mono_depth"; metric_root = work / "metric_depth"
    extract_frames(video, frames, n)
    mono_scene = mono_root / scene; metric_scene = metric_root / scene
    if not (args.resume and len(list(mono_scene.glob("*.npy"))) == n):
        mono_scene.mkdir(parents=True, exist_ok=True)
        run([str(MEGASAM_PYTHON), "Depth-Anything/run_videos.py", "--encoder", "vitl",
             "--load-from", str(da_ckpt), "--img-path", str(frames), "--outdir", str(mono_scene)])
    if not (args.resume and len(list(metric_scene.glob("*.npz"))) == n):
        metric_scene.mkdir(parents=True, exist_ok=True)
        run([str(MEGASAM_PYTHON), "UniDepth/scripts/demo_mega-sam.py", "--scene-name", scene,
             "--img-path", str(frames), "--outdir", str(metric_root)])
    if len(list(mono_scene.glob("*.npy"))) != n or len(list(metric_scene.glob("*.npz"))) != n:
        raise RuntimeError("Incomplete MegaSAM depth cache")

    raw_out = MEGASAM / "outputs" / f"{scene}_droid.npz"
    tracking_command = [
        str(MEGASAM_PYTHON), "camera_tracking_scripts/test_demo.py",
        "--datapath", str(frames), "--weights", str(mega_ckpt),
        "--scene_name", scene, "--mono_depth_path", str(mono_root),
        "--metric_depth_path", str(metric_root), "--disable_vis",
    ]
    if raw_out.exists():
        raw_out.unlink()
    if args.disable_focal_optimization:
        tracking_command.append("--disable_focal_optimization")
    run(tracking_command)
    raw, T_c2w, T_w2c, K, intr = load_camera_result(raw_out, n, width, height)
    focal_fallback = args.disable_focal_optimization
    focal_fallback_reason = "explicitly_disabled" if focal_fallback else None
    rejected_intrinsics = None
    if not valid_focal(K):
        if args.disable_focal_optimization:
            raise RuntimeError(
                "MegaSAM returned invalid focal length with focal "
                f"optimization disabled: {K}"
            )
        # MegaSAM's optional BA focal update is additive and unconstrained.  On
        # weakly observable clips it can cross zero.  Never hide that failure by
        # taking abs(f): retry the official tracker with the positive UniDepth
        # FoV initialization held fixed instead.
        rejected_intrinsics = K.tolist()
        print(
            "MegaSAM focal optimization produced invalid intrinsics; "
            "retrying with UniDepth FoV focal fixed: " + repr(K),
            flush=True,
        )
        raw_out.unlink()
        run(tracking_command + ["--disable_focal_optimization"])
        raw, T_c2w, T_w2c, K, intr = load_camera_result(raw_out, n, width, height)
        focal_fallback = True
        focal_fallback_reason = "invalid_optimized_focal"
        if not valid_focal(K):
            raise RuntimeError(
                "MegaSAM returned invalid focal length even with focal "
                f"optimization disabled: {K}"
            )
    centers = T_c2w[:, :3, 3]
    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    med = float(np.median(step)); mad = float(np.median(np.abs(step-med)))
    jump = step > max(med + 12 * max(mad, 1e-7), 20 * max(med, 1e-7))
    finite = np.isfinite(T_c2w).all(axis=(1, 2)); failed = ~finite; failed[1:] |= jump
    confidence = np.ones(n, np.float32); confidence[1:][jump] = 0.0
    validation = validate_camera(T_c2w, T_w2c)
    if not validation["passed"]:
        raise RuntimeError(f"Invalid MegaSAM camera trajectory: {validation}")
    clip_start, clip_end = map(int, entry["clip_frames"])
    frame_numbers = np.arange(clip_start, clip_start+n, dtype=np.int64)
    metadata = {
        "backend": "MegaSAM", "official_repository": "https://github.com/mega-sam/mega-sam",
        "video_used": str(video), "full_spatial_frame": True,
        "temporal_context": "30 frames before and after target track from distributed clip",
        "clip_source_frame_range_inclusive": [clip_start, clip_end],
        "depth_prior": "Depth-Anything ViT-L + UniDepth V2 ViT-L14",
        "tracking_confidence_kind": "finite_and_robust_translation_discontinuity",
        "tracking_failure_kind": "nonfinite_or_robust_translation_discontinuity",
        "validation": validation, "input_resolution": [width, height],
        "intrinsics_source": (
            "UniDepth median FoV held fixed after invalid MegaSAM focal BA"
            if focal_fallback else "MegaSAM focal BA initialized by UniDepth median FoV"
        ),
        "focal_optimization_used": not focal_fallback,
        "focal_fallback_used": focal_fallback,
        "focal_fallback_reason": focal_fallback_reason,
    }
    if rejected_intrinsics is not None:
        metadata["rejected_optimized_intrinsics"] = rejected_intrinsics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_camera_npz(args.output, T_c2w, T_w2c, intr, frame_numbers,
                    frame_numbers / fps, confidence, failed, fps, metadata)
    shutil.copy2(raw_out, work / "megasam_raw.npz")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
