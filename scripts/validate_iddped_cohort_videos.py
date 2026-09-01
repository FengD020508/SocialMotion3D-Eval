#!/usr/bin/env python3
"""Decode every extracted cohort video and verify its exact CFR contract."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def _validate_group(root: Path, expected_fps: float) -> list[dict]:
    manifest = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
    failures: list[dict] = []
    for index, clip in enumerate(manifest["clips"], start=1):
        path = root / clip["output"]
        expected_frames = int(clip["clip_frames"][1]) - int(clip["clip_frames"][0]) + 1
        capture = cv2.VideoCapture(str(path))
        metadata_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        decoded_frames = 0
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            decoded_frames += 1
        actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        if (
            metadata_frames != expected_frames
            or decoded_frames != expected_frames
            or abs(actual_fps - expected_fps) > 1e-6
        ):
            failures.append(
                {
                    "clip_id": clip["clip_id"],
                    "path": str(path),
                    "expected_frames": expected_frames,
                    "metadata_frames": metadata_frames,
                    "decoded_frames": decoded_frames,
                    "expected_fps": expected_fps,
                    "actual_fps": actual_fps,
                }
            )
        print(f"{root.name} {index}/{len(manifest['clips'])}: {clip['clip_id']}", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--only", choices=("all", "scenes", "events"), default="all")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    groups = ("scenes", "events") if args.only == "all" else (args.only,)
    failures: list[dict] = []
    totals: dict[str, int] = {}
    expected_fps: float | None = None
    for group in groups:
        root = args.input_root / group
        manifest = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
        group_fps = float(manifest["fps"])
        expected_fps = group_fps if expected_fps is None else expected_fps
        if abs(group_fps - expected_fps) > 1e-9:
            raise RuntimeError(f"Inconsistent manifest fps: {group_fps} versus {expected_fps}")
        totals[group] = len(manifest["clips"])
        failures.extend(_validate_group(root, group_fps))

    report = {"totals": totals, "expected_fps": expected_fps, "failures": failures}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
