#!/usr/bin/env python3
"""Create target-only IDD-PeD crops for a controlled MotionBERT comparison."""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


def target_boxes(annotation: Path, pedestrian_id: str) -> dict[int, np.ndarray]:
    root = ET.parse(annotation).getroot()
    found: dict[int, np.ndarray] = {}
    for track in root.findall("track"):
        for box in track.findall("box"):
            attributes = {
                item.attrib.get("name"): (item.text or "").strip()
                for item in box.findall("attribute")
            }
            if attributes.get("id") != pedestrian_id or box.attrib.get("outside", "0") == "1":
                continue
            frame = int(box.attrib["frame"])
            coords = np.asarray(
                [box.attrib["xtl"], box.attrib["ytl"], box.attrib["xbr"], box.attrib["ybr"]],
                dtype=np.float32,
            )
            previous = found.get(frame)
            if previous is None:
                found[frame] = coords
            else:
                area = max(float(coords[2] - coords[0]), 0.0) * max(float(coords[3] - coords[1]), 0.0)
                old_area = max(float(previous[2] - previous[0]), 0.0) * max(
                    float(previous[3] - previous[1]), 0.0
                )
                if area > old_area:
                    found[frame] = coords
    if not found:
        raise ValueError(f"no boxes for {pedestrian_id} in {annotation}")
    return found


def interpolate_boxes(boxes: dict[int, np.ndarray], frames: np.ndarray) -> np.ndarray:
    known_frames = np.asarray(sorted(boxes), dtype=np.float64)
    known_boxes = np.stack([boxes[int(frame)] for frame in known_frames], axis=0)
    result = np.empty((len(frames), 4), dtype=np.float32)
    for coordinate in range(4):
        result[:, coordinate] = np.interp(frames, known_frames, known_boxes[:, coordinate])
    return result


def square_crop(frame: np.ndarray, box: np.ndarray, padding: float, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = frame.shape[:2]
    center_x = float(box[0] + box[2]) * 0.5
    center_y = float(box[1] + box[3]) * 0.5
    side = max(float(box[2] - box[0]), float(box[3] - box[1])) * (1.0 + 2.0 * padding)
    side = max(side, 32.0)
    x0 = int(np.floor(center_x - side * 0.5))
    y0 = int(np.floor(center_y - side * 0.5))
    x1 = int(np.ceil(center_x + side * 0.5))
    y1 = int(np.ceil(center_y + side * 0.5))

    source_x0, source_y0 = max(x0, 0), max(y0, 0)
    source_x1, source_y1 = min(x1, width), min(y1, height)
    canvas = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    target_x0, target_y0 = source_x0 - x0, source_y0 - y0
    canvas[
        target_y0 : target_y0 + source_y1 - source_y0,
        target_x0 : target_x0 + source_x1 - source_x0,
    ] = frame[source_y0:source_y1, source_x0:source_x1]
    crop = cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return crop, np.asarray([x0, y0, x1, y1], dtype=np.float32)


def prepare_clip(
    clip: dict,
    input_root: Path,
    annotation_root: Path,
    target_track_root: Path | None,
    output_root: Path,
    padding: float,
    output_size: int,
    overwrite: bool,
) -> dict:
    source_name = Path(clip["output"]).name
    scene = Path(source_name).stem
    video_path = input_root / source_name
    annotation_path = annotation_root / f"{scene}.xml"
    scene_root = output_root / scene
    video_output = scene_root / "focus_crop.mp4"
    meta_output = scene_root / "crop_meta.json"
    arrays_output = scene_root / "crop_meta.npz"
    if not overwrite and video_output.is_file() and meta_output.is_file() and arrays_output.is_file():
        return json.loads(meta_output.read_text(encoding="utf-8"))

    if annotation_path.is_file():
        boxes = target_boxes(annotation_path, str(clip["pedestrian_id"]))
        bbox_source = str(annotation_path)
    elif target_track_root is not None:
        target_track_path = target_track_root / scene / "target_track.npz"
        if not target_track_path.is_file():
            raise FileNotFoundError(f"missing annotation and target track: {annotation_path}, {target_track_path}")
        with np.load(target_track_path, allow_pickle=False) as target_track:
            track_frames = np.asarray(target_track["local_frames"], dtype=np.int64)
            track_boxes = np.asarray(target_track["bbox_xyxy"], dtype=np.float32)
        boxes = {int(frame): box for frame, box in zip(track_frames, track_boxes)}
        bbox_source = str(target_track_path)
    else:
        raise FileNotFoundError(annotation_path)
    clip_start, _ = map(int, clip["clip_frames"])
    track_start, track_end = map(int, clip["track_frames"])
    local_start = max(track_start - clip_start, min(boxes))
    local_end = min(track_end - clip_start, max(boxes))
    if local_start > local_end:
        raise ValueError(f"empty target interval for {scene}: {local_start}>{local_end}")
    local_frames = np.arange(local_start, local_end + 1, dtype=np.int64)
    interpolated = interpolate_boxes(boxes, local_frames)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if local_end >= frame_count:
        capture.release()
        raise ValueError(f"target frame {local_end} exceeds {frame_count} frames in {video_path}")

    scene_root.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(video_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_size, output_size)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create {video_output}")

    crop_boxes: list[np.ndarray] = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, local_start)
    try:
        for index, local_frame in enumerate(local_frames):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"failed reading frame {int(local_frame)} from {video_path}")
            crop, crop_box = square_crop(frame, interpolated[index], padding, output_size)
            writer.write(crop)
            crop_boxes.append(crop_box)
    finally:
        writer.release()
        capture.release()

    crop_box_array = np.stack(crop_boxes, axis=0)
    np.savez_compressed(
        arrays_output,
        local_frames=local_frames,
        bbox_xyxy=interpolated,
        crop_xyxy=crop_box_array,
        fps=np.asarray(fps, dtype=np.float32),
        pedestrian_id=np.asarray(str(clip["pedestrian_id"])),
        source_video=np.asarray(str(video_path)),
    )
    report = {
        "scene": scene,
        "pedestrian_id": str(clip["pedestrian_id"]),
        "source_video": str(video_path),
        "bbox_source": bbox_source,
        "focus_video": str(video_output),
        "arrays": str(arrays_output),
        "fps": fps,
        "source_frame_count": frame_count,
        "target_local_frames": [int(local_frames[0]), int(local_frames[-1])],
        "target_frame_count": int(len(local_frames)),
        "padding": padding,
        "output_size": output_size,
        "bbox_interpolation": "linear_between_visible_annotations",
    }
    meta_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--annotation-root", type=Path, required=True)
    parser.add_argument("--target-track-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.35)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--clips", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.clips or [])
    reports = []
    for clip in manifest["clips"]:
        clip_id = Path(clip["output"]).name.split("_", 1)[0]
        if requested and clip_id not in requested:
            continue
        report = prepare_clip(
            clip,
            args.input_root,
            args.annotation_root,
            args.target_track_root,
            args.output_root,
            args.padding,
            args.output_size,
            args.overwrite,
        )
        reports.append(report)
        print(f"prepared {report['scene']}: {report['target_frame_count']} frames")
    (args.output_root / "crop_batch_report.json").write_text(
        json.dumps({"clips": reports}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
