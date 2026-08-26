#!/usr/bin/env python3
"""Build a private E3 config that combines existing pilot and new outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot-droid-root", required=True)
    parser.add_argument("--pilot-megasam-root", required=True)
    parser.add_argument("--batch-droid-root", required=True)
    parser.add_argument("--batch-megasam-root", required=True)
    parser.add_argument("--pilot-obd-root", required=True)
    parser.add_argument("--batch-input-root", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    clips = []
    for entry in manifest["clips"]:
        scene = Path(entry["output"]).stem
        clip_id = scene.split("_", 1)[0]
        pilot = int(clip_id) <= 5
        droid_root = args.pilot_droid_root if pilot else args.batch_droid_root
        megasam_root = args.pilot_megasam_root if pilot else args.batch_megasam_root
        if pilot:
            source_stem = PurePosixPath(entry["source"]).stem.lower()
            obd_xml = f"{args.pilot_obd_root}/{source_stem}_obd.xml"
        else:
            obd_xml = f"{args.batch_input_root}/{entry['obd_xml']}"
        clips.append(
            {
                "clip_id": scene,
                "fps": float(manifest.get("fps", 30.0)),
                "stratum": entry.get("stratum"),
                "obd_xml": obd_xml,
                "cameras": {
                    "droid": {
                        "path": f"{droid_root}/{scene}/camera_trajectory.npz",
                        "frame_numbers_are_source": True,
                    },
                    "megasam": {
                        "path": f"{megasam_root}/{scene}/camera_trajectory.npz",
                        "frame_numbers_are_source": True,
                    },
                },
            }
        )

    config = {
        "output_dir": args.result_root,
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
    print(f"wrote {len(clips)} clips to {args.output}")


if __name__ == "__main__":
    main()
