#!/usr/bin/env python3
"""Select the center-locked target from multi-person AlphaPose detections."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np


def frame_number(value) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(str(value or "")).name)
    return int(match.group(1)) if match else None


def keypoints(item: dict) -> np.ndarray | None:
    values = np.asarray(item.get("keypoints", []), dtype=np.float64)
    if values.size == 0 or values.size % 3:
        return None
    return values.reshape(-1, 3)


def candidate_center(item: dict, points: np.ndarray) -> np.ndarray:
    box = item.get("box") or item.get("bbox")
    if box is not None and len(box) >= 4:
        x, y, width, height = map(float, box[:4])
        return np.asarray([x + width * 0.5, y + height * 0.5])
    visible = points[points[:, 2] > 0.05, :2]
    return np.median(visible, axis=0) if len(visible) else np.zeros(2)


def item_box(points: np.ndarray) -> list[float]:
    visible = points[points[:, 2] > 0.05, :2]
    if len(visible) < 2:
        return [0.0, 0.0, 1.0, 1.0]
    low, high = visible.min(axis=0), visible.max(axis=0)
    return [float(low[0]), float(low[1]), float(max(high[0] - low[0], 1.0)), float(max(high[1] - low[1], 1.0))]


def select_track(grouped: dict[int, list[dict]], frame_count: int, width: int, height: int) -> dict[int, dict]:
    diagonal = float(np.hypot(width, height))
    image_center = np.asarray([width * 0.5, height * 0.5])
    frames = sorted(frame for frame in grouped if 1 <= frame <= frame_count)
    if not frames:
        return {}
    candidates = {}
    for frame in frames:
        prepared = []
        for item in grouped[frame]:
            points = keypoints(item)
            if points is None:
                continue
            center = candidate_center(item, points)
            confidence = float(np.mean(np.clip(points[:, 2], 0.0, 1.0)))
            detector_score = float(item.get("score", confidence))
            center_distance = float(np.linalg.norm(center - image_center) / max(diagonal, 1.0))
            emission = detector_score + confidence - 5.0 * center_distance
            prepared.append((item, center, emission))
        if prepared:
            candidates[frame] = prepared
    frames = sorted(candidates)
    if not frames:
        return {}

    scores = [np.asarray([item[2] for item in candidates[frames[0]]], dtype=np.float64)]
    backpointers: list[np.ndarray] = []
    for previous_frame, current_frame in zip(frames[:-1], frames[1:]):
        previous = candidates[previous_frame]
        current = candidates[current_frame]
        gap = max(current_frame - previous_frame, 1)
        transition = np.empty((len(previous), len(current)), dtype=np.float64)
        for i, (_, previous_center, _) in enumerate(previous):
            for j, (_, current_center, _) in enumerate(current):
                displacement = np.linalg.norm(current_center - previous_center) / max(diagonal * gap, 1.0)
                transition[i, j] = -3.0 * displacement
        totals = scores[-1][:, None] + transition
        pointer = np.argmax(totals, axis=0)
        current_emission = np.asarray([item[2] for item in current])
        scores.append(totals[pointer, np.arange(len(current))] + current_emission)
        backpointers.append(pointer)

    state = int(np.argmax(scores[-1]))
    states = [state]
    for pointer in reversed(backpointers):
        state = int(pointer[state])
        states.append(state)
    states.reverse()
    return {frame: candidates[frame][state][0] for frame, state in zip(frames, states)}


def interpolated_points(frame: int, selected: dict[int, dict]) -> tuple[np.ndarray | None, str]:
    frames = sorted(selected)
    previous = [value for value in frames if value < frame]
    following = [value for value in frames if value > frame]
    if previous and following:
        left, right = previous[-1], following[0]
        alpha = (frame - left) / float(right - left)
        result = keypoints(selected[left]) * (1.0 - alpha) + keypoints(selected[right]) * alpha
        result[:, 2] = np.minimum(keypoints(selected[left])[:, 2], keypoints(selected[right])[:, 2]) * 0.5
        return result, "interpolated"
    if previous or following:
        nearest = previous[-1] if previous else following[0]
        result = keypoints(selected[nearest]).copy()
        result[:, 2] *= 0.25
        return result, "nearest_fill"
    return None, "missing"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-in", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {args.video}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    raw_items = json.loads(args.json_in.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict]] = {}
    invalid = 0
    for item in raw_items:
        frame = frame_number(item.get("image_id"))
        if frame is None or keypoints(item) is None:
            invalid += 1
            continue
        grouped.setdefault(frame, []).append(item)
    selected = select_track(grouped, frame_count, width, height)
    clean = []
    source_types = []
    center_distances = []
    for frame in range(1, frame_count + 1):
        item = selected.get(frame)
        if item is None:
            points, source = interpolated_points(frame, selected)
            if points is None:
                points = np.zeros((26, 3), dtype=np.float64)
            box = item_box(points)
            score = float(np.mean(points[:, 2]))
        else:
            points = keypoints(item).copy()
            source = "center_temporal_track"
            box = item.get("box") or item.get("bbox") or item_box(points)
            score = float(item.get("score", np.mean(points[:, 2])))
            center = candidate_center(item, points)
            center_distances.append(float(np.linalg.norm(center - [width * 0.5, height * 0.5]) / np.hypot(width, height)))
        points[:, 2] = np.clip(points[:, 2], 0.0, 1.0)
        source_types.append(source)
        clean.append(
            {
                "image_id": f"{frame:06d}.jpg",
                "category_id": 1,
                "keypoints": points.reshape(-1).astype(float).tolist(),
                "score": score,
                "box": list(map(float, box[:4])),
                "idx": 0.0,
                "demo_pose_source": source,
            }
        )
    args.json_out.write_text(json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "selection": "Viterbi center-lock with temporal center continuity",
        "video_frame_count": frame_count,
        "raw_detections": len(raw_items),
        "valid_detection_frames": len(selected),
        "invalid_items": invalid,
        "duplicate_detections_removed": sum(max(len(items) - 1, 0) for items in grouped.values()),
        "missing_frames_filled": frame_count - len(selected),
        "source_type_counts": {name: source_types.count(name) for name in sorted(set(source_types))},
        "selected_center_distance_normalized_median": (
            float(np.median(center_distances)) if center_distances else None
        ),
        "selected_center_distance_normalized_p95": (
            float(np.percentile(center_distances, 95)) if center_distances else None
        ),
    }
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
