#!/usr/bin/env python3
"""Run DROID-SLAM on IDD-PeD full-frame clips and write GENMO camera NPZ."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DROID_ROOT = ROOT / "third_party" / "DROID-SLAM"
sys.path.insert(0, str(DROID_ROOT / "droid_slam"))
sys.path.insert(0, str(ROOT))

from scripts.idd_ped.camera_geometry import (  # noqa: E402
    droid_pose7_to_c2w,
    normalize_and_y_up,
    save_camera_npz,
    validate_camera,
)


def clip_file(inputs: Path, clip_id: str) -> Path:
    matches = sorted(inputs.glob(f"{clip_id}_*.mp4"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one clip matching {clip_id}_*.mp4, found {matches}")
    return matches[0]


def entry_clip_id(entry: dict) -> str:
    return str(entry.get("clip_id") or entry["output"].split("_", 1)[0])


def entry_source(entry: dict) -> str:
    return str(entry.get("source_video") or entry["source"])


def resolve_video(inputs: Path, entry: dict, source_root: Path | None) -> tuple[Path, int, int, bool]:
    """Return video, inclusive read interval, and whether temporal context exists."""
    clip_start, clip_end = map(int, entry["clip_frames"])
    if source_root is not None:
        candidate = source_root / entry_source(entry)
        if candidate.is_file():
            return candidate, max(0, clip_start - 90), clip_end + 90, True
    # Distributed test inputs are spatially full-frame but temporally pre-cut.
    video = clip_file(inputs, entry_clip_id(entry))
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return video, 0, n - 1, False


def default_intrinsics(width: int, height: int, focal_factor: float) -> np.ndarray:
    f = float(focal_factor * max(width, height))
    return np.array([f, f, width / 2.0, height / 2.0], dtype=np.float32)


class DynamicMasker:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.model = None
        if enabled:
            from ultralytics import YOLO

            self.model = YOLO("yolov8x.pt")

    def __call__(self, image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        if self.model is None:
            return image_bgr, 0.0
        # COCO: person, car, motorcycle, bus, truck.
        result = self.model(image_bgr, classes=[0, 2, 3, 5, 7], device=0, verbose=False)[0]
        mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        for box in result.boxes.xyxy.detach().cpu().numpy():
            x1, y1, x2, y2 = np.round(box).astype(int)
            pad = max(4, int(0.03 * max(x2 - x1, y2 - y1)))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(mask.shape[1] - 1, x2 + pad), min(mask.shape[0] - 1, y2 + pad)
            mask[y1 : y2 + 1, x1 : x2 + 1] = 255
        ratio = float((mask > 0).mean())
        if ratio == 0:
            return image_bgr, ratio
        return cv2.inpaint(image_bgr, mask, 5, cv2.INPAINT_TELEA), ratio


def make_stream(
    video: Path,
    start: int,
    end: int,
    intrinsics: np.ndarray,
    masker: DynamicMasker,
    collect_mask_ratio: list[float] | None = None,
):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    index = start
    local = 0
    while index <= end:
        ok, image = cap.read()
        if not ok:
            break
        image, ratio = masker(image)
        if collect_mask_ratio is not None:
            collect_mask_ratio.append(ratio)
        h0, w0 = image.shape[:2]
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
        image = cv2.resize(image, (w1, h1))
        image = image[: h1 - h1 % 8, : w1 - w1 % 8]
        tensor = torch.as_tensor(image).permute(2, 0, 1)[None]
        intr = torch.as_tensor(intrinsics.copy())
        intr[0::2] *= w1 / w0
        intr[1::2] *= h1 / h0
        yield local, tensor, intr
        index += 1
        local += 1
    cap.release()


def droid_args(weights: Path, n_frames: int) -> SimpleNamespace:
    return SimpleNamespace(
        weights=str(weights), buffer=max(512, n_frames + 32), image_size=[240, 320],
        disable_vis=True, beta=0.3, filter_thresh=2.4, warmup=8,
        keyframe_thresh=4.0, frontend_thresh=16.0, frontend_window=25,
        frontend_radius=2, frontend_nms=1, backend_thresh=22.0,
        backend_radius=2, backend_nms=3, upsample=False, stereo=False,
        asynchronous=False, frontend_device="cuda", backend_device="cuda",
    )


def estimate_road_plane(droid, T_c2w_raw: np.ndarray, keyframe_times: np.ndarray) -> dict:
    """Estimate a road plane from lower-image DROID keyframe depths."""
    count = int(droid.video.counter.value)
    disps = droid.video.disps[:count].detach().cpu().numpy()
    intrs = droid.video.intrinsics[:count].detach().cpu().numpy()
    rng = np.random.default_rng(2026)
    clouds = []
    for k, source_idx in enumerate(keyframe_times):
        if source_idx < 0 or source_idx >= len(T_c2w_raw):
            continue
        disp = disps[k]
        h, w = disp.shape[-2:]
        yy, xx = np.mgrid[int(0.58*h):h, :w]
        dd = disp[int(0.58*h):h]
        valid = np.isfinite(dd) & (dd > 1e-5)
        if valid.sum() < 50:
            continue
        x, y, d = xx[valid], yy[valid], dd[valid]
        take = rng.choice(len(d), size=min(1200, len(d)), replace=False)
        x, y, z = x[take], y[take], 1.0 / d[take]
        fx, fy, cx, cy = intrs[k]
        pc = np.stack([(x-cx)/fx*z, (y-cy)/fy*z, z], axis=-1)
        T = T_c2w_raw[source_idx]
        pw = pc @ T[:3,:3].T + T[:3,3]
        clouds.append(pw)
    if len(clouds) < 3:
        return {"stable": False, "reason": "insufficient_depth_keyframes"}
    points = np.concatenate(clouds)
    if len(points) > 10000:
        points = points[rng.choice(len(points), 10000, replace=False)]
    scale = float(np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1)))
    threshold = max(0.01, 0.025 * scale)
    best = None
    for _ in range(350):
        tri = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(tri[1]-tri[0], tri[2]-tri[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        offset = -float(normal @ tri[0])
        inlier = np.abs(points @ normal + offset) < threshold
        score = int(inlier.sum())
        if best is None or score > best[0]:
            best = (score, inlier)
    if best is None:
        return {"stable": False, "reason": "ransac_failed"}
    inlier_points = points[best[1]]
    _, _, vh = np.linalg.svd(inlier_points - inlier_points.mean(0), full_matrices=False)
    normal = vh[-1]
    offset = -float(normal @ inlier_points.mean(0))
    camera_down = T_c2w_raw[0,:3,1]
    alignment = float(abs(normal @ camera_down))
    centers = T_c2w_raw[:, :3, 3]
    heights = np.abs(centers @ normal + offset)
    height = float(np.median(heights))
    height_cv = float(np.std(heights) / max(height, 1e-8))
    inlier_ratio = float(best[0] / len(points))
    stable = bool(inlier_ratio >= 0.25 and alignment >= 0.65 and height > 1e-5 and height_cv <= 0.30)
    return {
        "stable": stable,
        "reason": None if stable else "plane_geometry_or_height_consistency_failed",
        "camera_height_droid_units": height,
        "normal_droid_world": normal.tolist(), "offset_droid_world": offset,
        "inlier_ratio": inlier_ratio, "camera_down_alignment": alignment,
        "camera_height_cv": height_cv, "sample_count": int(len(points)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, default=ROOT / "inputs/idd-ped_inputs")
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--clip_id", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source_root", type=Path, default=None)
    p.add_argument("--weights", type=Path, default=ROOT / "inputs/checkpoints/droid/droid.pth")
    p.add_argument("--calib", type=Path, default=None, help="fx fy cx cy calibration text")
    p.add_argument("--focal_factor", type=float, default=0.9)
    p.add_argument("--dynamic_mask", action="store_true")
    args = p.parse_args()

    manifest_path = args.manifest or args.inputs / "selection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["clips"] if isinstance(manifest, dict) else manifest
    entry = next((x for x in entries if entry_clip_id(x) == args.clip_id), None)
    if entry is None:
        raise KeyError(f"clip_id {args.clip_id} not found in {manifest_path}")
    video, read_start, read_end, has_context = resolve_video(args.inputs, entry, args.source_root)

    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    read_end = min(read_end, total - 1)
    intr = np.loadtxt(args.calib, dtype=np.float32)[:4] if args.calib else default_intrinsics(width, height, args.focal_factor)
    intrinsics_source = "calibration_file" if args.calib else "estimated_pinhole"

    if not args.weights.is_file():
        raise FileNotFoundError(f"DROID weights missing: {args.weights}")
    try:
        from droid import Droid
    except Exception as exc:
        raise RuntimeError("DROID-SLAM extension is not installed in this environment") from exc

    n_read = read_end - read_start + 1
    masker = DynamicMasker(args.dynamic_mask)
    mask_ratios: list[float] = []
    stream = make_stream(video, read_start, read_end, intr, masker, mask_ratios)
    droid = None
    for t, image, K in tqdm(stream, total=n_read, desc=f"DROID {args.clip_id}"):
        if droid is None:
            cfg = droid_args(args.weights, n_read)
            cfg.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(cfg)
        droid.track(t, image, intrinsics=K)
    if droid is None:
        raise RuntimeError(f"No frames decoded from {video}")

    keyframe_count = int(droid.video.counter.value)
    keyframe_times = droid.video.tstamp[:keyframe_count].cpu().numpy().astype(int)
    poses7 = droid.terminate(make_stream(video, read_start, read_end, intr, masker))
    if len(poses7) != n_read:
        raise RuntimeError(f"DROID returned {len(poses7)} poses for {n_read} frames")

    # DROID terminate() returns c2w positions and xyzw quaternions.
    T_c2w_raw = droid_pose7_to_c2w(poses7)
    try:
        road_plane = estimate_road_plane(droid, T_c2w_raw, keyframe_times)
    except Exception as exc:
        road_plane = {"stable": False, "reason": f"road_plane_exception:{type(exc).__name__}:{exc}"}
    nearest = np.min(np.abs(np.arange(n_read)[:, None] - keyframe_times[None]), axis=1)
    confidence = np.exp(-nearest.astype(np.float32) / 8.0)

    clip_start, clip_end = map(int, entry["clip_frames"])
    if has_context:
        lo, hi = clip_start - read_start, clip_end - read_start + 1
        frame_numbers = np.arange(clip_start, clip_end + 1)
    else:
        lo, hi = 0, n_read
        frame_numbers = np.arange(clip_start, clip_start + n_read)
    # Normalize after trimming so the first delivered target frame is exactly
    # the shared world origin, even when SLAM used temporal context.
    T_c2w, T_w2c = normalize_and_y_up(T_c2w_raw[lo:hi])
    confidence = confidence[lo:hi]
    finite = np.isfinite(T_c2w).all(axis=(1, 2))
    step = np.linalg.norm(np.diff(T_c2w[:, :3, 3], axis=0), axis=1)
    med = float(np.median(step))
    mad = float(np.median(np.abs(step - med)))
    jump = step > max(med + 12.0 * max(mad, 1e-6), 20.0 * max(med, 1e-6))
    failed = ~finite
    failed[1:] |= jump
    confidence[1:][jump] = 0.0
    timestamps = frame_numbers / fps
    validation = validate_camera(T_c2w, T_w2c)
    if not validation["passed"]:
        raise RuntimeError(f"Invalid normalized camera trajectory: {validation}")

    metadata = {
        "backend": "DROID-SLAM", "video_used": str(video),
        "source_video_requested": entry_source(entry),
        "temporal_context_available": has_context,
        "read_frame_range_inclusive": [read_start, read_end],
        "target_source_frame_range_inclusive": [clip_start, clip_end],
        "dynamic_mask": args.dynamic_mask,
        "mean_dynamic_mask_ratio": float(np.mean(mask_ratios)) if mask_ratios else 0.0,
        "intrinsics_source": intrinsics_source,
        "tracking_confidence_kind": "distance_to_DROID_keyframe_proxy",
        "tracking_failure_kind": "nonfinite_or_robust_translation_discontinuity",
        "keyframe_count": keyframe_count,
        "road_plane": road_plane,
        "validation": validation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_camera_npz(args.output, T_c2w, T_w2c, intr, frame_numbers, timestamps,
                    confidence, failed, fps, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
