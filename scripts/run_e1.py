#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from socialmotion3d_eval.e1 import (
    BONES,
    JOINT_NAMES,
    align_by_local_frame,
    kabsch_align,
    method_disagreement,
    robust_temporal_metrics,
)


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


def resolve_event_artifact(root: Path, scene: str, clip_id: str, relative_path: str) -> Path:
    """Resolve artifacts stored by either descriptive scene name or compact event ID."""
    candidates = [root / scene / relative_path]
    if clip_id and clip_id != scene:
        candidates.append(root / clip_id / relative_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def render_blind_video(
    output: Path,
    scene: str,
    motionbert: np.ndarray,
    gem: np.ndarray,
    valid: np.ndarray,
    fps: float,
    method_a: str,
) -> None:
    aligned_motionbert, normalized_gem = kabsch_align(motionbert, gem, valid)
    method_data = {"MotionBERT": aligned_motionbert, "GEM": normalized_gem}
    method_b = "GEM" if method_a == "MotionBERT" else "MotionBERT"
    data_a, data_b = method_data[method_a], method_data[method_b]
    common = np.concatenate([data_a[valid], data_b[valid]], axis=0)
    radius = float(np.percentile(np.abs(common), 99)) if len(common) else 1.0
    radius = max(radius, 0.25)
    stride = max(int(round(fps / 10.0)), 1)
    frame_indices = np.flatnonzero(valid)[::stride]

    figure = plt.figure(figsize=(8, 4), dpi=120)
    axes = [figure.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    lines = []
    for label, axis in zip(("A", "B"), axes):
        axis.set_title(label)
        axis.set_xlim(-radius, radius)
        axis.set_ylim(-radius, radius)
        axis.set_zlim(-radius, radius)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=12, azim=-70)
        axis.set_axis_off()
        lines.append([axis.plot([], [], [], color="#2563eb", linewidth=2.0)[0] for _ in BONES])
    figure.suptitle(f"{scene} | blinded common-17")
    figure.tight_layout()

    writer = FFMpegWriter(fps=max(int(round(fps / stride)), 1), codec="libx264", bitrate=1800)
    output.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(output), dpi=120):
        for frame_index in frame_indices:
            for motion, panel_lines in ((data_a, lines[0]), (data_b, lines[1])):
                for line, (joint_a, joint_b) in zip(panel_lines, BONES):
                    segment = motion[frame_index, [joint_a, joint_b]]
                    line.set_data_3d(segment[:, 0], segment[:, 2], -segment[:, 1])
            writer.grab_frame()
    plt.close(figure)


def aggregate(rows: list[dict]) -> dict:
    result = {}
    for method in ("MotionBERT", "GEM"):
        subset = [row for row in rows if row["method"] == method]
        method_result = {"clip_count": len(subset)}
        metric_names = sorted(set().union(*(row["metrics"].keys() for row in subset)))
        for name in metric_names:
            values = [row["metrics"].get(name) for row in subset]
            numeric = np.asarray([value for value in values if isinstance(value, (int, float)) and np.isfinite(value)])
            if len(numeric):
                method_result[name] = {"mean": float(np.mean(numeric)), "median": float(np.median(numeric))}
        result[method] = method_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--render-blind", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["manifest"]).read_text(encoding="utf-8"))
    motionbert_root = Path(config["motionbert_root"])
    gem_root = Path(config["gem_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    fps_default = float(config.get("fps", 30.0))
    seed = str(config.get("blind_seed", "e1-batch23-v1"))

    method_rows = []
    pair_rows = []
    blind_key = []
    rating_rows = []
    for clip in manifest["clips"]:
        scene = Path(clip["output"]).stem
        clip_id = str(clip.get("clip_id", ""))
        mb_path = resolve_event_artifact(motionbert_root, scene, clip_id, "motionbert_lifting/X3D.npy")
        mb_meta_path = resolve_event_artifact(motionbert_root, scene, clip_id, "crop_meta.npz")
        gem_path = resolve_event_artifact(gem_root, scene, clip_id, "gem_common17.npz")
        if not mb_path.is_file() or not mb_meta_path.is_file() or not gem_path.is_file():
            raise FileNotFoundError(f"missing E1 input for {scene}: {mb_path}, {mb_meta_path}, {gem_path}")
        motionbert = np.load(mb_path)
        with np.load(mb_meta_path, allow_pickle=False) as meta:
            motionbert_frames = np.asarray(meta["local_frames"], dtype=np.int64)
            fps = float(meta["fps"])
        with np.load(gem_path, allow_pickle=False) as gem_data:
            gem = np.asarray(gem_data["joints_incam"], dtype=np.float64)
            gem_frames = np.asarray(gem_data["local_frames"], dtype=np.int64)
            gem_valid = np.asarray(gem_data["valid_mask"], dtype=bool)
        aligned = align_by_local_frame(motionbert, motionbert_frames, gem, gem_frames, gem_valid)
        valid = aligned.valid_motionbert & aligned.valid_gem
        mb_metrics = robust_temporal_metrics(aligned.motionbert, aligned.valid_motionbert, fps or fps_default)
        gem_metrics = robust_temporal_metrics(aligned.gem, aligned.valid_gem, fps or fps_default)
        method_rows.extend(
            [
                {"scene": scene, "cue": cue_label(clip), "method": "MotionBERT", "metrics": mb_metrics},
                {"scene": scene, "cue": cue_label(clip), "method": "GEM", "metrics": gem_metrics},
            ]
        )
        pair = method_disagreement(aligned.motionbert, aligned.gem, valid)
        pair_rows.append({"scene": scene, "cue": cue_label(clip), **pair})

        selector = int(hashlib.sha256(f"{seed}:{scene}".encode()).hexdigest()[:8], 16) % 2
        method_a = "MotionBERT" if selector == 0 else "GEM"
        method_b = "GEM" if selector == 0 else "MotionBERT"
        blind_video = output_root / "blind_rating" / f"{scene}_blind.mp4"
        blind_key.append({"scene": scene, "A": method_a, "B": method_b, "video": str(blind_video)})
        rating_rows.append(
            {
                "scene": scene,
                "cue": cue_label(clip),
                "video": str(blind_video),
                "A_technical_success_0_1": "",
                "B_technical_success_0_1": "",
                "A_cue_fidelity_1_5": "",
                "B_cue_fidelity_1_5": "",
                "A_onset_fidelity_1_5": "",
                "B_onset_fidelity_1_5": "",
                "preference_A_B_tie": "",
                "notes": "",
            }
        )
        if args.render_blind and not blind_video.is_file():
            print(f"rendering blinded comparison: {scene}")
            render_blind_video(
                blind_video, scene, aligned.motionbert, aligned.gem, valid, fps or fps_default, method_a
            )
        print(f"evaluated {scene}: {int(valid.sum())}/{len(valid)} common valid frames")

    flat_rows = []
    for row in method_rows:
        flat_rows.append({"scene": row["scene"], "cue": row["cue"], "method": row["method"], **row["metrics"]})
    with (output_root / "e1_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(finite_or_none(flat_rows))
    with (output_root / "e1_pair_disagreement.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(finite_or_none(pair_rows))
    with (output_root / "blind_rating_sheet.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rating_rows[0]))
        writer.writeheader()
        writer.writerows(rating_rows)

    capability = {
        "MotionBERT": {
            "native_output": "H36M-17 joints",
            "native_root_trajectory": False,
            "native_parametric_body": False,
            "extra_scene_ready_stages": ["joints-to-SMPL fitting", "trajectory estimation or annotation"],
        },
        "GEM": {
            "native_output": "SMPL-X pose plus camera/global root translation",
            "native_root_trajectory": True,
            "native_parametric_body": True,
            "boundary": "absolute metric world registration still requires external camera scale/ego-motion",
        },
    }
    report = {
        "experiment": "E1 MotionBERT versus GEM pilot",
        "clip_count": len(manifest["clips"]),
        "comparison_boundary": (
            "Automatic metrics are diagnostic consistency measures without 3D ground truth. "
            "They do not establish pose accuracy; social-cue fidelity requires the blinded rating sheet."
        ),
        "joint_contract": JOINT_NAMES,
        "method_aggregate": aggregate(method_rows),
        "pair_disagreement": pair_rows,
        "capability": capability,
        "blind_rating_sheet": str(output_root / "blind_rating_sheet.csv"),
        "blind_key": str(output_root / "blind_key.json"),
    }
    (output_root / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "e1_report.json").write_text(
        json.dumps(finite_or_none(report), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "capability_report.json").write_text(
        json.dumps(capability, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
