#!/usr/bin/env python3
"""Export GEM SMPL-X motion into a documented H36M-like common-17 space."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


COMMON17_NAMES = [
    "pelvis",
    "right_hip",
    "right_knee",
    "right_ankle",
    "left_hip",
    "left_knee",
    "left_ankle",
    "spine",
    "thorax",
    "neck",
    "head",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
]


def coco17_to_common17(joints: torch.Tensor) -> torch.Tensor:
    # COCO: nose, eyes, ears, L/R shoulder, L/R elbow, L/R wrist,
    # L/R hip, L/R knee, L/R ankle.
    pelvis = (joints[..., 11, :] + joints[..., 12, :]) * 0.5
    thorax = (joints[..., 5, :] + joints[..., 6, :]) * 0.5
    spine = (pelvis + thorax) * 0.5
    neck = thorax * 0.75 + joints[..., 0, :] * 0.25
    return torch.stack(
        [
            pelvis,
            joints[..., 12, :],
            joints[..., 14, :],
            joints[..., 16, :],
            joints[..., 11, :],
            joints[..., 13, :],
            joints[..., 15, :],
            spine,
            thorax,
            neck,
            joints[..., 0, :],
            joints[..., 5, :],
            joints[..., 7, :],
            joints[..., 9, :],
            joints[..., 6, :],
            joints[..., 8, :],
            joints[..., 10, :],
        ],
        dim=-2,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_scenes(source_roots: list[Path], requested: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in source_roots:
        if not root.is_dir():
            continue
        for scene in sorted(root.iterdir()):
            if not scene.is_dir() or not (scene / "smpl_params.pt").is_file():
                continue
            clip_id = scene.name.split("_", 1)[0]
            if requested and clip_id not in requested:
                continue
            if clip_id in result:
                raise ValueError(f"duplicate GEM clip id {clip_id}: {result[clip_id]} and {scene}")
            result[clip_id] = scene
    return result


def model_joints(model: torch.nn.Module, params: dict, device: torch.device) -> np.ndarray:
    values = {key: params[key].to(device) for key in ("body_pose", "betas", "global_orient", "transl")}
    with torch.no_grad():
        _, coco17 = model(**values)
        common17 = coco17_to_common17(coco17)
    return common17.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clips", nargs="*", default=[])
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    from gem.utils.smplx_utils import make_smplx

    requested = set(args.clips)
    scenes = find_scenes(args.source_root, requested)
    if requested and set(scenes) != requested:
        raise FileNotFoundError(f"missing GEM clips: {sorted(requested - set(scenes))}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_smplx("supermotion_v437coco17").to(device).eval()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for clip_id, scene in sorted(scenes.items()):
        smpl_path = scene / "smpl_params.pt"
        payload = torch.load(smpl_path, map_location="cpu", weights_only=False)
        incam = model_joints(model, payload["body_params_incam"], device)
        global_joints = model_joints(model, payload["body_params_global"], device)
        track_path = scene / "target_track.npz"
        if not track_path.is_file():
            raise FileNotFoundError(track_path)
        with np.load(track_path, allow_pickle=False) as track:
            local_frames = np.asarray(track["local_frames"], dtype=np.int64)
            source_frames = np.asarray(track["source_frames"], dtype=np.int64)
            valid_track = np.asarray(track["valid_mask"], dtype=bool)
        if len(incam) != len(local_frames):
            raise ValueError(f"{scene.name}: {len(incam)} poses but {len(local_frames)} target frames")
        valid = valid_track & np.isfinite(incam).all(axis=(1, 2)) & np.isfinite(global_joints).all(axis=(1, 2))
        output_dir = args.output_root / scene.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "gem_common17.npz"
        np.savez_compressed(
            output_path,
            joints_incam=incam,
            joints_global=global_joints,
            local_frames=local_frames,
            source_frames=source_frames,
            valid_mask=valid,
            fps=np.asarray(args.fps, dtype=np.float32),
            joint_names=np.asarray(COMMON17_NAMES),
            coordinate_incam=np.asarray("GEM camera coordinates; meters"),
            coordinate_global=np.asarray("GEM global coordinates before OBD metric camera registration"),
        )
        reports.append(
            {
                "clip_id": clip_id,
                "scene": scene.name,
                "source_smpl": str(smpl_path),
                "source_sha256": sha256(smpl_path),
                "output": str(output_path),
                "frames": int(len(incam)),
                "valid_frames": int(valid.sum()),
            }
        )
        print(f"exported {scene.name}: {len(incam)} frames")
    report = {
        "method": "GEM",
        "body_model": "SMPL-X neutral / supermotion_v437coco17",
        "joint_contract": "H36M-like common17 constructed from COCO17 body joints",
        "joint_names": COMMON17_NAMES,
        "device": str(device),
        "scenes": reports,
    }
    (args.output_root / "gem_common17_batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
