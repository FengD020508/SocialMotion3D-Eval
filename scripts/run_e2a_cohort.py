#!/usr/bin/env python3
"""Evaluate fixed-human No-ego/DROID/MegaSAM grounding for every event."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from socialmotion3d_eval.e2a import run_e2a


def _translation_gate(clip_report: dict, methods: tuple[str, ...] = ("droid", "megasam")) -> tuple[bool, dict]:
    details = {}
    for method in methods:
        result = clip_report.get("methods", {}).get(method, {})
        scale = result.get("scale_m_per_raw_unit")
        details[method] = {"status": result.get("status"), "scale_m_per_raw_unit": scale}
    evaluable = all(
        item["status"] == "ok"
        and isinstance(item["scale_m_per_raw_unit"], (int, float))
        and math.isfinite(float(item["scale_m_per_raw_unit"]))
        and float(item["scale_m_per_raw_unit"]) > 0
        for item in details.values()
    )
    return evaluable, details


def _expected_motion_unavailable(error: Exception) -> bool:
    return isinstance(error, ValueError) and "no common valid E2a intervals" in str(error)


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
    e3_report = json.loads(args.e3_report.read_text(encoding="utf-8"))
    e3_by_clip = {entry["clip_id"]: entry for entry in e3_report["clips"]}
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
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            previous_status = previous.get("status", "ok")
            summary.append(
                {
                    "event_clip_id": event_id,
                    "scene_clip_id": scene_clip_id,
                    "status": "skipped_valid" if previous_status == "ok" else previous_status,
                }
            )
            print(f"[{index}/{len(entries)}] {event_id}: already valid", flush=True)
            continue
        if scene_clip_id not in e3_by_clip:
            raise KeyError(f"E3 report is missing scene {scene_clip_id}")
        evaluable, scale_details = _translation_gate(e3_by_clip[scene_clip_id])
        if not evaluable:
            output_dir.mkdir(parents=True, exist_ok=True)
            unavailable = {
                "experiment": "E2a_fixed_human_controlled_ego",
                "status": "translation_not_evaluable",
                "event_clip_id": event_id,
                "clip_id": scene_clip_id,
                "reason": (
                    "E3 did not obtain finite positive OBD scales for both camera methods; "
                    "E2a metric grounding is withheld rather than fabricated."
                ),
                "e3_methods": scale_details,
            }
            report_path.write_text(
                json.dumps(unavailable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            summary.append(
                {
                    "event_clip_id": event_id,
                    "scene_clip_id": scene_clip_id,
                    "status": "translation_not_evaluable",
                }
            )
            (args.output_root / "run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[{index}/{len(entries)}] {event_id}: translation not evaluable", flush=True)
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
            report = run_e2a(config_path)
            report["status"] = "ok"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            summary.append({"event_clip_id": event_id, "scene_clip_id": scene_clip_id, "status": "ok"})
        except Exception as error:
            if _expected_motion_unavailable(error):
                unavailable = {
                    "experiment": "E2a_fixed_human_controlled_ego",
                    "status": "motion_not_evaluable",
                    "event_clip_id": event_id,
                    "clip_id": scene_clip_id,
                    "reason": (
                        "No interval remained valid across No-ego, DROID, and MegaSAM "
                        "under the predeclared robust-jump and 8 m/s root-speed rules; "
                        "metrics are withheld rather than fabricated."
                    ),
                    "error": str(error),
                }
                report_path.write_text(
                    json.dumps(unavailable, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                summary.append(
                    {
                        "event_clip_id": event_id,
                        "scene_clip_id": scene_clip_id,
                        "status": "motion_not_evaluable",
                    }
                )
                (args.output_root / "run_summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                print(f"[{index}/{len(entries)}] {event_id}: motion not evaluable", flush=True)
                continue
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
