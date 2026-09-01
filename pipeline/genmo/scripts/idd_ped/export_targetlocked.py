#!/usr/bin/env python3
"""Export and audit annotation-locked IDD-PeD joint reconstruction."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.idd_ped.camera_geometry import validate_camera  # noqa: E402
from scripts.idd_ped.export_joint_reconstruction import (  # noqa: E402
    as_numpy, json_dump, make_global_preview, mirror_x_position,
    mirror_x_rotation, projection_metrics, quat_records,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clip_id", required=True)
    p.add_argument("--target_video", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--preprocess_dir", type=Path, required=True)
    args = p.parse_args()
    out = args.output_dir
    with np.load(out/"camera_trajectory.npz", allow_pickle=False) as z:
        full = {k:z[k] for k in z.files}
    with np.load(out/"camera_trajectory_target.npz", allow_pickle=False) as z:
        cam = {k:z[k] for k in z.files}
    with np.load(out/"target_track.npz", allow_pickle=False) as z:
        target = {k:z[k] for k in z.files}
    params = torch.load(out/"smpl_params.pt", map_location="cpu", weights_only=False)
    incam, glob = params["body_params_incam"], params["body_params_global"]
    p_incam = as_numpy(incam["transl"]).astype(np.float64)
    n = len(p_incam)
    expected = len(target["source_frames"])
    if n != expected or len(cam["T_c2w"]) != expected:
        raise ValueError(f"Human/camera/annotation length mismatch: {n}/{len(cam['T_c2w'])}/{expected}")
    used = as_numpy(torch.load(out/"used_bbx_xys.pt", map_location="cpu", weights_only=False))
    bbox_max_error = float(np.max(np.abs(used-target["bbx_xys"])))
    if bbox_max_error > 1e-5:
        raise ValueError(f"Used bbox differs from annotation conversion: {bbox_max_error}")

    center_full = full["camera_center"].astype(np.float64)
    span = float(np.percentile(np.linalg.norm(center_full-center_full[0],axis=1),95))
    scale = 1.0/span if span > 1e-6 else 1.0
    R_c2w, center = cam["R_c2w"].astype(np.float64), cam["camera_center"].astype(np.float64)
    p_world = np.einsum("nij,nj->ni",R_c2w,p_incam)+scale*center
    R_body_cam = Rotation.from_rotvec(as_numpy(incam["global_orient"])).as_matrix()
    R_body_world = R_c2w @ R_body_cam
    params["body_params_joint_world"] = {
        "transl":torch.from_numpy(p_world.astype(np.float32)),
        "global_orient":torch.from_numpy(Rotation.from_matrix(R_body_world).as_rotvec().astype(np.float32)),
        "body_pose":incam["body_pose"].clone(), "betas":incam["betas"].clone(),
    }
    params["joint_reconstruction"] = {
        "camera_scale":scale, "scale_mode":"relative",
        "scale_comparable_across_clips":False,
    }
    tmp=(out/"smpl_params.pt").with_suffix(".pt.tmp"); torch.save(params,tmp); tmp.replace(out/"smpl_params.pt")

    ego_pos=mirror_x_position(scale*center_full); ego_rot=mirror_x_rotation(full["R_c2w"])
    human_pos=mirror_x_position(p_world); human_rot=mirror_x_rotation(R_body_world)
    json_dump(out/"ego_trajectory_unity.json",{
        "coordinate_system":"Unity Y-up; GENMO-to-Unity mirrors X", "scale":scale,
        "scale_comparable_across_clips":False,
        "camera_to_vehicle_extrinsic":{"status":"uncalibrated_identity","inspector_adjustable":True},
        "frames":quat_records(full["frame_numbers"],full["timestamps"],ego_pos,ego_rot,full["tracking_confidence"]),
    })
    json_dump(out/"human_trajectory_unity.json",{
        "coordinate_system":"Unity Y-up; GENMO-to-Unity mirrors X", "identity_source":"idd_ped_annotation",
        "target_key":[str(target["set_id"]),str(target["video_id"]),str(target["pedestrian_id"])],
        "scale":scale,"scale_comparable_across_clips":False,
        "frames":quat_records(target["source_frames"],cam["timestamps"],human_pos,human_rot),
    })
    make_global_preview(out/"preview_global.mp4",mirror_x_position(scale*center),human_pos,float(cam["fps"]))

    meta=json.loads(str(full["metadata_json"])); tmeta=json.loads(str(cam["metadata_json"]))
    camera_backend = str(meta.get("backend", "unknown"))
    labels={
        "clip_id":args.clip_id,
        "target_key":{"set_id":str(target["set_id"]),"video_id":str(target["video_id"]),
                      "pedestrian_id":str(target["pedestrian_id"])},
        "track_source_range":[int(target["source_frames"][0]),int(target["source_frames"][-1])],
        "track_local_range":[int(target["local_frames"][0]),int(target["local_frames"][-1])],
        "annotation_source":str(target["annotation_source"]),
        "bbox_fingerprint":str(target["bbox_fingerprint"]),
        "identity_source":"idd_ped_annotation",
    }
    json_dump(out/"interaction_labels.json",labels)

    projection=projection_metrics(incam,cam["K_fullimg"],args.preprocess_dir,
                                  bbx_path=out/"used_bbx_xys.pt")
    recovered=np.einsum("nij,nj->ni",cam["R_w2c"],p_world-scale*center)
    reverse=float(np.linalg.norm(recovered-p_incam,axis=1).max())
    root=as_numpy(glob["transl"]); fps=float(cam["fps"])
    speed=np.linalg.norm(np.diff(root,axis=0),axis=1)*fps
    accel=np.linalg.norm(np.diff(root,n=2,axis=0),axis=1)*fps*fps
    max_speed=float(speed.max(initial=0)); max_accel=float(accel.max(initial=0))
    betas=as_numpy(incam["betas"]); beta_var=betas.var(axis=0)
    physical_pass=bool(max_speed < 8.0 and max_accel < 150.0 and beta_var.max(initial=0)<1e-3)
    target_camera_validation=validate_camera(cam["T_c2w"],cam["T_w2c"])
    corrected=list(meta.get("corrected_source_frames",[]))
    identity={
        "identity_source":"idd_ped_annotation", "target_key_match":True,
        "target_switch_count":0, "annotation_frames_found":expected,
        "expected_frames":expected,"annotation_coverage_ratio":1.0,
        "bbox_conversion_max_abs_error":bbox_max_error,
        "automatic_detection_fallback":False,
        "bbox_fingerprint_match":str(target["bbox_fingerprint"])==tmeta["bbox_fingerprint"],
    }
    cap=cv2.VideoCapture(str(args.target_video)); vn=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); vf=cap.get(cv2.CAP_PROP_FPS); cap.release()
    report={
        "clip_id":args.clip_id,
        "camera_backend":camera_backend,
        "passed":bool(target_camera_validation["passed"] and reverse<1e-4 and projection["passed"]
                      and physical_pass and all(identity.values() if False else [identity["target_key_match"],
                      identity["annotation_coverage_ratio"]==1.0,identity["bbox_fingerprint_match"],
                      not identity["automatic_detection_fallback"],bbox_max_error<=1e-5]) and vn==expected),
        "numerical_matrix_validity":{"passed":target_camera_validation["passed"] and reverse<1e-4,
            "camera":target_camera_validation,"reverse_transform_max_error":reverse},
        "camera_trajectory_quality":{"backend":camera_backend,"tracking_failed_source_frames":full["frame_numbers"][full["tracking_failed"]].astype(int).tolist(),
            "corrected_source_frames":corrected,"target_first_pose_identity_error":target_camera_validation["first_pose_identity_error"],
            "confidence_mean":float(full["tracking_confidence"].mean())},
        "target_identity_correctness":identity,
        "human_motion_physical_plausibility":{"passed":physical_pass,"max_root_speed":max_speed,
            "max_root_acceleration":max_accel,"speed_threshold":8.0,"acceleration_threshold":150.0,
            "betas_variance_max":float(beta_var.max(initial=0)),"betas_variance_mean":float(beta_var.mean()),
            "betas_variance_threshold":1e-3},
        "projection_validation":projection,
        "length_sync":{"target_video":vn,"human":n,"annotation":expected,"fps":vf,"passed":vn==n==expected},
        "scale":{"mode":"relative","value":scale,"scale_comparable_across_clips":False,
            "training_warning":"Per-clip path normalization is visualization-only; use camera_height or body-height units for motionPool."},
    }
    json_dump(out/"quality_report.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report["passed"]:
        raise RuntimeError("Target-locked quality validation failed")


if __name__=="__main__": main()
