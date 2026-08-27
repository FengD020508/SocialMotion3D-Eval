#!/usr/bin/env python3
"""Normalize the original pilot and stratified IDD-PeD manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--extra-manifest", type=Path, required=True)
    parser.add_argument("--supplement-manifest", type=Path)
    parser.add_argument("--expected-count", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pilot = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
    extra = json.loads(args.extra_manifest.read_text(encoding="utf-8"))
    clips = list(pilot["clips"])
    sources = list(extra["clips"])
    if args.supplement_manifest is not None:
        supplement = json.loads(args.supplement_manifest.read_text(encoding="utf-8"))
        sources.extend(supplement["clips"])

    for source in sources:
        clip_start, clip_end = map(int, source["clip_frames"])
        track_start, track_end = map(int, source["track_frames"])
        effective_track = [max(clip_start, track_start), min(clip_end, track_end)]
        if effective_track[0] > effective_track[1]:
            raise ValueError(f"{source['output_video']}: target track does not overlap clip")
        clips.append(
            {
                "output": source["output_video"],
                "source": f"gopro/{source['set_id']}/{source['video_id']}.MP4",
                "pedestrian_id": source["pedestrian_id"],
                "clip_frames": [clip_start, clip_end],
                "track_frames": effective_track,
                "source_track_frames": [track_start, track_end],
                "interaction_frames": source.get("ego_interaction_frames", []),
                "crossing_point": source.get("event_frame"),
                "crossing_behavior": source.get("crossing_behavior", "N/A"),
                "traffic_interaction": source.get("traffic_interaction", "N/A"),
                "social_dynamics": source.get("social_dynamics", []),
                "age": source.get("age"),
                "carrying_object": source.get("carrying_object"),
                "ego_interaction_frame": source.get("event_frame"),
                "joint_interaction": source.get("joint_interaction", "N/A"),
                "stratum": {
                    "kind": source.get("kind"),
                    "time_of_day": source.get("time_of_day"),
                    "dynamic_complexity": source.get("dynamic_complexity"),
                },
                "obd_xml": source.get("output_vehicle_annotation"),
            }
        )

    identifiers = [str(item["output"]).split("_", 1)[0] for item in clips]
    expected_ids = [f"{index:02d}" for index in range(1, args.expected_count + 1)]
    if len(clips) != args.expected_count or identifiers != expected_ids:
        raise ValueError(f"expected clip ids 01..{args.expected_count:02d}, got {identifiers}")
    output = {
        "dataset": "IDD-PeD",
        "fps": float(pilot.get("fps", 30.0)),
        "purpose": extra.get("purpose"),
        "clips": clips,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(clips)} clips to {args.output}")


if __name__ == "__main__":
    main()
