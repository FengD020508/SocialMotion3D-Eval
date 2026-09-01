#!/usr/bin/env python3
"""Fuse DROID camera and GENMO in-camera human results and export Unity data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.idd_ped.camera_geometry import validate_camera  # noqa: E402


def json_dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def as_numpy(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def load_manifest_entry(path: Path, clip_id: str) -> dict:
    manifest = json.loads(path.read_text())
    clips = manifest["clips"] if isinstance(manifest, dict) else manifest
    return next(x for x in clips if str(x.get("clip_id") or x["output"].split("_", 1)[0]) == clip_id)


def optimize_camera_scale(base: np.ndarray, center: np.ndarray, global_root: np.ndarray) -> tuple[float, dict]:
    """Fit monocular camera scale using GENMO root velocity plus smoothness."""
    n = min(len(base), len(center), len(global_root))
    base, center, glob = base[:n], center[:n], global_root[:n]
    if n < 3 or np.linalg.norm(np.diff(center, axis=0)) < 1e-7:
        return 1.0, {"success": False, "reason": "insufficient_camera_translation"}
    vg = np.diff(glob, axis=0)

    def objective(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        world = base + scale * center
        velocity_loss = np.mean((np.diff(world, axis=0) - vg) ** 2)
        accel_loss = np.mean(np.diff(world, n=2, axis=0) ** 2)
        return float(velocity_loss + 0.05 * accel_loss)

    result = minimize_scalar(objective, bounds=(np.log(1e-3), np.log(1e3)), method="bounded")
    return float(np.exp(result.x)), {
        "success": bool(result.success), "objective": float(result.fun),
        "method": "GENMO_global_root_velocity_plus_root_smoothness",
    }


def choose_scale(mode: str, camera_meta: dict, base: np.ndarray, center: np.ndarray,
                 global_root: np.ndarray, camera_height: float) -> tuple[float, dict]:
    road = camera_meta.get("road_plane", {})
    raw_height = float(road.get("camera_height_droid_units", 0.0) or 0.0)
    stable = bool(road.get("stable", False)) and raw_height > 1e-6
    if mode == "relative":
        displacement = np.linalg.norm(center - center[0], axis=1)
        span = float(np.percentile(displacement, 95))
        scale = 1.0 / span if span > 1e-6 else 1.0
        return scale, {
            "mode": mode, "source": "camera_path_p95_normalized_to_one_unit",
            "raw_camera_path_p95": span, "road_plane_stable": stable,
            "assumed_camera_height_m": None,
        }
    if stable:
        return camera_height / raw_height, {
            "mode": mode, "source": "road_plane_camera_height",
            "assumed_camera_height_m": camera_height,
            "camera_height_droid_units": raw_height, "road_plane_stable": True,
        }
    scale, fit = optimize_camera_scale(base, center, global_root)
    return scale, {
        "mode": mode,
        "source": "global_root_optimization" if fit["success"] else "unit_fallback",
        "road_plane_stable": stable,
        "road_plane_fallback_reason": road.get("reason", "road_plane_unavailable"),
        "assumed_camera_height_m": camera_height,
        "optimizer": fit,
    }


def mirror_x_position(p: np.ndarray) -> np.ndarray:
    out = p.copy()
    out[..., 0] *= -1
    return out


def mirror_x_rotation(R: np.ndarray) -> np.ndarray:
    M = np.diag([-1.0, 1.0, 1.0])
    return M[None] @ R @ M[None]


def quat_records(frame_numbers, timestamps, positions, rotations, confidence=None) -> list[dict]:
    quat = Rotation.from_matrix(rotations).as_quat()  # Unity field order x,y,z,w
    records = []
    for i in range(len(positions)):
        row = {
            "index": i, "source_frame": int(frame_numbers[i]), "timestamp": float(timestamps[i]),
            "position": {k: float(v) for k, v in zip("xyz", positions[i])},
            "rotation_xyzw": {k: float(v) for k, v in zip("xyzw", quat[i])},
        }
        if confidence is not None:
            row["tracking_confidence"] = float(confidence[i])
        records.append(row)
    return records


def make_global_preview(path: Path, ego: np.ndarray, human: np.ndarray, fps: float, size=720) -> None:
    all_xz = np.concatenate([ego[:, [0, 2]], human[:, [0, 2]]], axis=0)
    lo, hi = np.nanmin(all_xz, axis=0), np.nanmax(all_xz, axis=0)
    span = np.maximum(hi - lo, 1e-3)
    margin = span * 0.15 + 0.2
    lo, hi = lo - margin, hi + margin
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create {path}")

    def px(points):
        uv = (points[:, [0, 2]] - lo) / (hi - lo)
        uv[:, 1] = 1 - uv[:, 1]
        return np.round(uv * (size - 1)).astype(np.int32)

    ep, hp = px(ego), px(human)
    for i in range(len(ego)):
        canvas = np.full((size, size, 3), 245, np.uint8)
        if i:
            cv2.polylines(canvas, [ep[: i + 1]], False, (200, 80, 30), 3)
            cv2.polylines(canvas, [hp[: i + 1]], False, (60, 120, 220), 3)
        cv2.circle(canvas, tuple(ep[i]), 10, (200, 80, 30), -1)
        cv2.circle(canvas, tuple(hp[i]), 9, (60, 120, 220), -1)
        cv2.putText(canvas, f"frame {i}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (20,20,20), 2)
        cv2.putText(canvas, "ego", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, .65, (200,80,30), 2)
        cv2.putText(canvas, "human", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .65, (60,120,220), 2)
        writer.write(canvas)
    writer.release()


def projection_metrics(incam: dict, K: np.ndarray, preprocess_dir: Path | None,
                       bbx_path: Path | None = None) -> dict:
    """Quantify SMPL reprojection against cached target bbox and COCO limbs."""
    if preprocess_dir is None:
        return {"passed": False, "status": "missing_preprocess_dir"}
    bbx_path = bbx_path or preprocess_dir / "bbx.pt"
    kp_path = preprocess_dir / "vitpose.pt"
    if not bbx_path.is_file() or not kp_path.is_file():
        return {"passed": False, "status": "missing_bbox_or_keypoints"}
    from gem.utils.smplx_utils import make_smplx

    bbx = as_numpy(torch.load(bbx_path, map_location="cpu", weights_only=False))
    kp = as_numpy(torch.load(kp_path, map_location="cpu", weights_only=False))
    model = make_smplx("supermotion").cuda().eval()
    with torch.no_grad():
        result = model(
            body_pose=incam["body_pose"].cuda(), global_orient=incam["global_orient"].cuda(),
            transl=incam["transl"].cuda(), betas=incam["betas"].cuda(),
        )
    verts = result.vertices.detach().cpu().numpy()
    joints = result.joints.detach().cpu().numpy()

    def project(xyz):
        z = np.maximum(xyz[..., 2], 1e-4)
        u = K[:, None, 0, 0] * xyz[..., 0] / z + K[:, None, 0, 2]
        v = K[:, None, 1, 1] * xyz[..., 1] / z + K[:, None, 1, 2]
        return np.stack([u, v], axis=-1)

    uv = project(verts)
    mesh_box = np.stack([uv[..., 0].min(1), uv[..., 1].min(1),
                         uv[..., 0].max(1), uv[..., 1].max(1)], axis=-1)
    half = bbx[:, 2] / 2
    target = np.stack([bbx[:, 0]-half, bbx[:, 1]-half,
                       bbx[:, 0]+half, bbx[:, 1]+half], axis=-1)
    lt = np.maximum(mesh_box[:, :2], target[:, :2])
    rb = np.minimum(mesh_box[:, 2:], target[:, 2:])
    inter = np.maximum(rb-lt, 0).prod(1)
    area_m = np.maximum(mesh_box[:, 2:]-mesh_box[:, :2], 0).prod(1)
    area_t = np.maximum(target[:, 2:]-target[:, :2], 0).prod(1)
    iou = inter / np.maximum(area_m + area_t - inter, 1e-6)
    center_err = np.linalg.norm((mesh_box[:, :2]+mesh_box[:, 2:])/2-bbx[:, :2], axis=1) / np.maximum(bbx[:,2],1)

    # SMPL-X body joint order -> COCO limb joints (face joints intentionally omitted).
    coco_to_smplx = {5:16, 6:17, 7:18, 8:19, 9:20, 10:21,
                     11:1, 12:2, 13:4, 14:5, 15:7, 16:8}
    pred, obs, conf = [], [], []
    joints_uv = project(joints)
    for coco, smplx in coco_to_smplx.items():
        pred.append(joints_uv[:, smplx]); obs.append(kp[:, coco, :2]); conf.append(kp[:, coco, 2])
    pred, obs, conf = np.stack(pred,1), np.stack(obs,1), np.stack(conf,1)
    err = np.linalg.norm(pred-obs,axis=-1) / np.maximum(bbx[:,None,2],1)
    valid = conf > 0.3
    keypoint_nme = float(err[valid].mean()) if valid.any() else float("nan")
    passed = bool(np.median(iou) > 0.10 and np.median(center_err) < 0.35
                  and (not np.isfinite(keypoint_nme) or keypoint_nme < 0.50))
    return {
        "status": "computed", "passed": passed,
        "mesh_bbox_iou_median": float(np.median(iou)),
        "mesh_bbox_iou_p10": float(np.percentile(iou, 10)),
        "mesh_bbox_center_error_normalized_median": float(np.median(center_err)),
        "coco_limb_keypoint_nme": keypoint_nme,
        "valid_keypoint_observations": int(valid.sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip_id", required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--scale_mode", choices=["relative", "camera_height"], default="relative")
    p.add_argument("--camera_height", type=float, default=1.5)
    p.add_argument("--preprocess_dir", type=Path, default=None)
    args = p.parse_args()

    out = args.output_dir
    camera_path, smpl_path = out / "camera_trajectory.npz", out / "smpl_params.pt"
    if not camera_path.is_file() or not smpl_path.is_file():
        raise FileNotFoundError("camera_trajectory.npz and smpl_params.pt must exist before export")
    entry = load_manifest_entry(args.manifest, args.clip_id)
    with np.load(camera_path, allow_pickle=False) as z:
        cam = {k: z[k] for k in z.files}
    camera_meta = json.loads(str(cam["metadata_json"]))
    params = torch.load(smpl_path, map_location="cpu", weights_only=False)
    incam = params["body_params_incam"]
    glob = params["body_params_global"]
    p_incam = as_numpy(incam["transl"]).astype(np.float64)
    global_root = as_numpy(glob["transl"]).astype(np.float64)
    R_c2w = cam["R_c2w"].astype(np.float64)
    center = cam["camera_center"].astype(np.float64)
    n = len(p_incam)
    if not all(len(x) == n for x in (R_c2w, center, global_root, cam["frame_numbers"])):
        raise ValueError("Camera, GENMO and source frame lengths differ")

    base = np.einsum("nij,nj->ni", R_c2w, p_incam)
    scale, scale_report = choose_scale(args.scale_mode, camera_meta, base, center,
                                       global_root, args.camera_height)
    p_world = base + scale * center
    R_body_cam = Rotation.from_rotvec(as_numpy(incam["global_orient"])).as_matrix()
    R_body_world = R_c2w @ R_body_cam

    # Preserve checkpoint outputs and add an explicitly named fused result.
    params["body_params_joint_world"] = {
        "transl": torch.from_numpy(p_world.astype(np.float32)),
        "global_orient": torch.from_numpy(Rotation.from_matrix(R_body_world).as_rotvec().astype(np.float32)),
        "body_pose": incam["body_pose"].clone(), "betas": incam["betas"].clone(),
    }
    params["joint_reconstruction"] = {"camera_scale": scale, "scale_mode": args.scale_mode}
    tmp = smpl_path.with_suffix(".pt.tmp")
    torch.save(params, tmp)
    tmp.replace(smpl_path)

    unity_cam_pos = mirror_x_position(scale * center)
    unity_cam_rot = mirror_x_rotation(R_c2w)
    unity_human_pos = mirror_x_position(p_world)
    unity_human_rot = mirror_x_rotation(R_body_world)
    frames, timestamps = cam["frame_numbers"], cam["timestamps"]
    json_dump(out / "ego_trajectory_unity.json", {
        "coordinate_system": "Unity Y-up; GENMO-to-Unity mirrors X",
        "camera_to_vehicle_extrinsic": {"status": "uncalibrated_identity",
            "position_xyz": [0,0,0], "rotation_xyzw": [0,0,0,1],
            "inspector_adjustable": True,
            "note": "Attach Camera as a fixed child of the vehicle root and tune offset in Inspector."},
        "scale": scale, "frames": quat_records(frames, timestamps, unity_cam_pos,
                                                 unity_cam_rot, cam["tracking_confidence"]),
    })
    json_dump(out / "human_trajectory_unity.json", {
        "coordinate_system": "Unity Y-up; GENMO-to-Unity mirrors X",
        "scale": scale, "frames": quat_records(frames, timestamps, unity_human_pos, unity_human_rot),
    })

    speed = np.r_[0, np.linalg.norm(np.diff(scale * center, axis=0), axis=1) * float(cam["fps"])]
    crossing_frame = entry.get("crossing_frame", entry.get("crossing_point"))
    ego_frame = entry.get("ego_reaction_frame", entry.get("ego_interaction_frame"))
    crossing_local = int(crossing_frame - entry["clip_frames"][0])
    ego_local = int(ego_frame - entry["clip_frames"][0])
    labels = {
        "clip_id": args.clip_id,
        "crossing_type": entry.get("crossing_type", entry.get("crossing_behavior")),
        "traffic_interaction": entry.get("traffic_interaction"),
        "crossing_source_frame": crossing_frame, "crossing_local_frame": crossing_local,
        "ego_reaction_source_frame": ego_frame, "ego_reaction_local_frame": ego_local,
        "ego_action": entry.get("ego_action", entry.get("joint_interaction")),
        "unity_human_lateral_direction": "positive_X" if unity_human_pos[-1,0] > unity_human_pos[0,0] else "negative_X",
        "ego_speed_magnitude": speed.tolist(),
    }
    json_dump(out / "interaction_labels.json", labels)

    recovered = np.einsum("nij,nj->ni", cam["R_w2c"], p_world - scale * center)
    reverse_error = np.linalg.norm(recovered - p_incam, axis=1)
    cap = cv2.VideoCapture(str(args.video))
    video_n, video_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    camera_validation = validate_camera(cam["T_c2w"], cam["T_w2c"])
    projection = projection_metrics(incam, cam["K_fullimg"], args.preprocess_dir)
    report = {
        "clip_id": args.clip_id,
        "passed": bool(video_n == n and abs(video_fps - float(cam["fps"])) < 1e-3
                       and camera_validation["passed"] and reverse_error.max() < 1e-4
                       and projection["passed"]),
        "frame_count": {"video": video_n, "camera": len(center), "smpl": n, "match": video_n == n},
        "fps": {"video": video_fps, "camera": float(cam["fps"]), "match": abs(video_fps-float(cam["fps"])) < 1e-3},
        "camera_validation": camera_validation,
        "reverse_transform_error": {"max_m": float(reverse_error.max()), "mean_m": float(reverse_error.mean())},
        "tracking": {"failed_source_frames": frames[cam["tracking_failed"]].astype(int).tolist(),
                     "confidence_min": float(cam["tracking_confidence"].min()),
                     "confidence_mean": float(cam["tracking_confidence"].mean()),
                     "confidence_kind": camera_meta["tracking_confidence_kind"]},
        "scale_estimation": {**scale_report, "scale": scale},
        "temporal_context_available": camera_meta["temporal_context_available"],
        "intrinsics_source": camera_meta["intrinsics_source"],
        "projection_validation": projection,
    }
    json_dump(out / "quality_report.json", report)
    make_global_preview(out / "preview_global.mp4", unity_cam_pos, unity_human_pos, video_fps)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("Joint reconstruction validation failed; see quality_report.json")


if __name__ == "__main__":
    main()
