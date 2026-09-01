#!/usr/bin/env python3
"""Extract frame-accurate event and de-duplicated scene videos with ffmpeg."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _without_videos_prefix(value: str) -> str:
    return value[len("videos/") :] if value.startswith("videos/") else value


def _video_info(path: Path, ffprobe: str) -> tuple[int, float]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,avg_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(completed.stdout)["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    return int(stream["nb_frames"]), numerator / denominator


def _extract(
    source: Path,
    output: Path,
    start: int,
    end: int,
    ffmpeg: str,
    ffprobe: str,
    fps: float,
    overwrite: bool,
) -> None:
    expected = end - start + 1
    if output.is_file() and output.stat().st_size > 0 and not overwrite:
        actual, actual_fps = _video_info(output, ffprobe)
        if actual != expected or abs(actual_fps - fps) > 1e-6:
            raise RuntimeError(
                f"{output}: found {actual} frames at {actual_fps:.9g} fps; "
                f"expected {expected} at {fps:.9g} fps"
            )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".part.mp4")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start / fps:.9f}",
        "-i",
        str(source),
        "-frames:v",
        str(expected),
        "-vf",
        f"setpts=N/({fps:.9g}*TB)",
        "-r",
        f"{fps:.9g}",
        "-fps_mode",
        "cfr",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    actual, actual_fps = _video_info(temporary, ffprobe)
    if actual != expected or abs(actual_fps - fps) > 1e-6:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"{output}: encoded {actual} frames at {actual_fps:.9g} fps; "
            f"expected {expected} at {fps:.9g} fps"
        )
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--only", choices=("all", "scenes", "events"), default="all")
    args = parser.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    fps = float(cohort["fps"])
    manifests: dict[str, list[dict]] = {"scenes": [], "events": []}
    if args.only in {"all", "scenes"}:
        for index, scene in enumerate(cohort["scenes"], start=1):
            output_name = f"s{index:04d}_{_safe_name(scene['scene_id'])}.mp4"
            start, end = map(int, scene["clip_frames"])
            source = args.dataset_root / Path(scene["source_video"])
            output = args.output_root / "scenes" / output_name
            _extract(source, output, start, end, args.ffmpeg, args.ffprobe, fps, args.overwrite)
            manifests["scenes"].append(
                {
                    **scene,
                    "clip_id": f"s{index:04d}",
                    "output": output_name,
                    "source": _without_videos_prefix(scene["source_video"]),
                }
            )
            print(f"scene {index}/{len(cohort['scenes'])}: {output_name}", flush=True)

    if args.only in {"all", "events"}:
        for index, event in enumerate(cohort["events"], start=1):
            output_name = f"e{index:04d}_{_safe_name(event['event_id'])}.mp4"
            start, end = map(int, event["clip_frames"])
            source = args.dataset_root / Path(event["source_video"])
            output = args.output_root / "events" / output_name
            _extract(source, output, start, end, args.ffmpeg, args.ffprobe, fps, args.overwrite)
            manifests["events"].append(
                {
                    **event,
                    "clip_id": f"e{index:04d}",
                    "output": output_name,
                    "source": _without_videos_prefix(event["source_video"]),
                    # The target loader expects this field to be the exact
                    # consecutive interval delivered to GEM/MotionBERT.
                    "track_frames": event["effective_target_frames"],
                }
            )
            print(f"event {index}/{len(cohort['events'])}: {output_name}", flush=True)

    args.output_root.mkdir(parents=True, exist_ok=True)
    for kind, entries in manifests.items():
        if not entries:
            continue
        payload = {
            "schema_version": cohort["schema_version"],
            "dataset": cohort["dataset"],
            "fps": cohort["fps"],
            "unit": "scene" if kind == "scenes" else "event",
            "clips": entries,
        }
        (args.output_root / kind / "selection_manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
