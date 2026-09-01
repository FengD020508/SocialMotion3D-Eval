#!/usr/bin/env python3
"""Evaluate fixed-human No-ego/DROID/MegaSAM grounding for every event."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from socialmotion3d_eval.e2a import run_e2a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-manifest", type=Path, required=True)
    parser.add_argument("--scene-manifest", type=Path, required=True)
    parser.add_argument("--gem-root", type=Path, required=True)
    parser.add_argument("--e3-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    events = json.loads(args.event_manifest.read_text(encoding="utf-8"))["clips"]
    scenes = json.loads(args.scene_manifest.read_text(encoding="utf-8"))["clips"]
    scene_by_id = {entry["scene_id"]: entry["clip_id"] for entry in scenes}
    selected = set(args.ids or [])
    entries = [entry for entry in events if not selected or entry["clip_id"] in selected]
    configs = args.output_root / "configs"
    configs.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []

    for index, entry in enumerate(entries, start=1):
        event_id = entry["clip_id"]
        scene_clip_id = scene_by_id[entry["scene_id"]]
        gem_dir = args.gem_root / event_id
        output_dir = args.output_root / event_id
        report_path = output_dir / "e2a_report.json"
        if args.resume and report_path.is_file() and report_path.stat().st_size > 0:
            summary.append({"event_clip_id": event_id, "scene_clip_id": scene_clip_id, "status": "skipped_valid"})
            print(f"[{index}/{len(entries)}] {event_id}: already valid", flush=True)
            continue
        config = {
            # E3 calibrates one scale per de-duplicated camera scene.
            "clip_id": scene_clip_id,
            "e3_report": str(args.e3_report),
            "output_dir": str(output_dir),
            "parameters": {"max_root_speed_mps": 8.0, "jump_mad_factor": 12.0},
            "human_sources": {"fixed_gem": {"path": str(gem_dir / "smpl_params.pt")}},
            "camera_sources": {
                method: {
                    "path": str(gem_dir / f"camera_{method}.npz"),
                    "frame_numbers_are_source": True,
                }
                for method in ("droid", "megasam")
            },
        }
        config_path = configs / f"{event_id}.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            run_e2a(config_path)
            summary.append({"event_clip_id": event_id, "scene_clip_id": scene_clip_id, "status": "ok"})
        except Exception as error:
            summary.append(
                {
                    "event_clip_id": event_id,
                    "scene_clip_id": scene_clip_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            (args.output_root / "run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            raise
        (args.output_root / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[{index}/{len(entries)}] {event_id}: E2a ok", flush=True)


if __name__ == "__main__":
    main()
