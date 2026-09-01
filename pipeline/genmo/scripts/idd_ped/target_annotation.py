"""Strict IDD-PeD target-track loading and visualization utilities."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def target_key_from_manifest(entry: dict) -> tuple[str, str, str]:
    source = Path(entry["source"])
    video_id = source.stem
    set_id = source.parent.name
    return set_id, video_id, str(entry["pedestrian_id"])


def load_target_track(database: Path, entry: dict) -> dict:
    if not database.is_file():
        raise FileNotFoundError(f"Required annotation database is missing: {database}")
    with database.open("rb") as handle:
        db = pickle.load(handle)
    set_id, video_id, pedestrian_id = target_key_from_manifest(entry)
    try:
        record = db[set_id][video_id]["pedestrian_annotations"][pedestrian_id]
    except KeyError as exc:
        raise KeyError(f"Target key not found: {(set_id, video_id, pedestrian_id)}") from exc
    source_frames = np.asarray(record["frames"], dtype=np.int64)
    bbox = np.asarray(record["bbox"], dtype=np.float32)
    occlusion = np.asarray(record["occlusion"], dtype=np.int64)
    expected = np.arange(int(entry["track_frames"][0]), int(entry["track_frames"][1]) + 1)
    if bbox.shape != (len(source_frames), 4) or len(occlusion) != len(source_frames):
        raise ValueError("Annotation bbox/occlusion length mismatch")
    selection = np.flatnonzero((source_frames >= expected[0]) & (source_frames <= expected[-1]))
    if not np.array_equal(source_frames[selection], expected):
        raise ValueError(f"Annotation frame coverage mismatch for {(set_id, video_id, pedestrian_id)}")
    source_frames = source_frames[selection]
    bbox = bbox[selection]
    occlusion = occlusion[selection]
    if not np.isfinite(bbox).all() or np.any(bbox[:, 2:] < bbox[:, :2]):
        raise ValueError("Annotation contains invalid xyxy boxes")
    clip_start = int(entry["clip_frames"][0])
    local_frames = source_frames - clip_start
    wh = bbox[:, 2:] - bbox[:, :2]
    bbx_xys = np.column_stack(((bbox[:, 0] + bbox[:, 2]) / 2,
                               (bbox[:, 1] + bbox[:, 3]) / 2,
                               1.2 * np.maximum(wh[:, 0], wh[:, 1]))).astype(np.float32)
    digest = hashlib.sha256()
    digest.update("|".join((set_id, video_id, pedestrian_id)).encode())
    for array in (source_frames, local_frames, bbox, occlusion, bbx_xys):
        digest.update(np.ascontiguousarray(array).tobytes())
    fingerprint = digest.hexdigest()
    return {
        "set_id": set_id, "video_id": video_id, "pedestrian_id": pedestrian_id,
        "source_frames": source_frames, "local_frames": local_frames,
        "bbox_xyxy": bbox, "bbx_xys": bbx_xys, "occlusion": occlusion,
        "valid_mask": np.ones(len(expected), dtype=bool),
        "fingerprint": fingerprint, "behavior": record.get("behavior", {}),
        "attributes": record.get("attributes", {}),
        "annotation_source": str(database.resolve()),
    }


def save_target_npz(path: Path, track: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        set_id=np.asarray(track["set_id"]), video_id=np.asarray(track["video_id"]),
        pedestrian_id=np.asarray(track["pedestrian_id"]),
        source_frames=track["source_frames"], local_frames=track["local_frames"],
        bbox_xyxy=track["bbox_xyxy"], bbx_xys=track["bbx_xys"],
        occlusion=track["occlusion"], valid_mask=track["valid_mask"],
        bbox_fingerprint=np.asarray(track["fingerprint"]),
        annotation_source=np.asarray(track["annotation_source"]),
    )


def write_target_video_and_overlay(video: Path, track: dict, target_video: Path,
                                   overlay: Path, fps: float) -> None:
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    target_writer = cv2.VideoWriter(str(target_video), codec, fps, (width, height))
    overlay_writer = cv2.VideoWriter(str(overlay), codec, fps, (width, height))
    if not target_writer.isOpened() or not overlay_writer.isOpened():
        raise RuntimeError("Could not open target video/overlay writer")
    local_to_index = {int(x): i for i, x in enumerate(track["local_frames"])}
    frame_index = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in local_to_index:
            i = local_to_index[frame_index]
            target_writer.write(frame)
            vis = frame.copy()
            x1, y1, x2, y2 = np.round(track["bbox_xyxy"][i]).astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cx, cy, size = track["bbx_xys"][i]
            half = size / 2
            cv2.rectangle(vis, (round(cx-half), round(cy-half)),
                          (round(cx+half), round(cy+half)), (0, 165, 255), 2)
            text = f"{track['pedestrian_id']} source={int(track['source_frames'][i])} occ={int(track['occlusion'][i])}"
            cv2.putText(vis, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            overlay_writer.write(vis)
            written += 1
        frame_index += 1
    cap.release(); target_writer.release(); overlay_writer.release()
    if written != len(track["source_frames"]):
        raise RuntimeError(f"Wrote {written} target frames, expected {len(track['source_frames'])}")
