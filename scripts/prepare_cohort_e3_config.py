#!/usr/bin/env python3
"""Build an E3 config for the de-duplicated full IDD-PeD scene cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--camera-root", type=Path, required=True)
    parser.add_argument("--obd-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = []
    for entry in manifest["clips"]:
        clip_id = entry["clip_id"]
        clips.append(
            {
                "clip_id": clip_id,
                "fps": float(manifest["fps"]),
                "stratum": entry.get("set_id"),
                "obd_xml": str(args.obd_root / entry["obd_annotation"]),
                "cameras": {
                    method: {
                        "path": str(
                            args.camera_root / method / clip_id / "camera_trajectory.npz"
                        ),
                        "frame_numbers_are_source": True,
                    }
                    for method in ("droid", "megasam")
                },
            }
        )

    config = {
        "output_dir": str(args.result_root),
        "parameters": {
            "calibration_fraction": 0.4,
            "smooth_window_frames": 5,
            "jump_mad_factor": 12.0,
            "orthogonality_tolerance": 0.1,
            "determinant_tolerance": 0.05,
            "wrde_windows_seconds": [1, 2, 3],
            "min_target_distance_m": 0.5,
            "min_calibration_speed_mps": 0.5,
        },
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(clips)} scene-level E3 clips to {args.output}")


if __name__ == "__main__":
    main()
