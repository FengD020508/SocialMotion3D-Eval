"""Build event- and scene-level cohorts from IDD-PeD CVAT annotations.

The evaluation has two statistical units:

* a pedestrian interaction event for E1a/E2a;
* a de-duplicated camera window for E3.

Keeping the two levels explicit prevents several pedestrians observed during the
same camera motion from being counted as independent E3 samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable
import xml.etree.ElementTree as ET


MISSING_VALUES = {"", "-", "n/a", "na", "none", "null"}


def _attributes(box: ET.Element) -> dict[str, str]:
    return {
        str(node.attrib.get("name", "")).strip(): (node.text or "").strip()
        for node in box.findall("attribute")
    }


def _is_interaction(attributes: dict[str, str]) -> bool:
    """Return whether one pedestrian box carries an ego interaction label."""
    joint = attributes.get("Joint Interaction", "").strip().lower()
    ego = attributes.get("ped_ego_veh_interaction", "").strip()
    # Both fields are required. Nine legacy boxes contain only a Joint
    # Interaction value but no linked ego-interaction frame; including them
    # would silently mix incomplete annotations into the evaluation cohort.
    return joint not in MISSING_VALUES and bool(re.fullmatch(r"\d+", ego))


def _runs(frames: Iterable[int]) -> list[tuple[int, int]]:
    ordered = sorted(set(int(frame) for frame in frames))
    if not ordered:
        return []
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for frame in ordered[1:]:
        if frame != previous + 1:
            result.append((start, previous))
            start = frame
        previous = frame
    result.append((start, previous))
    return result


def _posix_relative(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root)).as_posix()


@dataclass(frozen=True)
class ParsedVideo:
    source_id: str
    annotation: Path
    video_frames: int
    events: list[dict]


def parse_annotation(path: Path, annotation_root: Path, context_frames: int) -> ParsedVideo:
    root = ET.parse(path).getroot()
    task_name = root.findtext("./meta/task/name") or path.stem
    # A few CVAT tasks include the source extension in <name>, while most do
    # not. Normalize both forms to the annotation/video stem.
    source_id = Path(task_name).stem
    video_frames = int(root.findtext("./meta/task/size") or 0)
    if video_frames <= 0:
        raise ValueError(f"missing positive video size in {path}")

    relative = path.relative_to(annotation_root)
    set_id = relative.parent.name
    events: list[dict] = []
    occurrence: dict[tuple[str, int, int], int] = {}

    for track in root.findall("track"):
        if track.attrib.get("label", "").strip().lower() != "pedestrian":
            continue
        visible: list[tuple[int, dict[str, str], ET.Element]] = []
        for box in track.findall("box"):
            if box.attrib.get("outside", "0") == "1":
                continue
            visible.append((int(box.attrib["frame"]), _attributes(box), box))
        if not visible:
            continue

        track_start = min(item[0] for item in visible)
        track_end = max(item[0] for item in visible)
        interaction_rows = [item for item in visible if _is_interaction(item[1])]
        if not interaction_rows:
            continue

        pedestrian_ids = [
            attrs.get("id", "").strip()
            for _, attrs, _ in visible
            if attrs.get("id", "").strip().lower() not in MISSING_VALUES
        ]
        pedestrian_id = pedestrian_ids[0] if pedestrian_ids else f"track_{track.attrib.get('id', 'unknown')}"
        by_frame = {frame: attrs for frame, attrs, _ in interaction_rows}

        for interaction_start, interaction_end in _runs(by_frame):
            # The human target must remain observable throughout an E1a/E2a
            # event. Context is therefore clipped to both the source video and
            # the annotated pedestrian track. This also avoids merging camera
            # windows merely through frames in which neither target exists.
            clip_start = max(0, track_start, interaction_start - context_frames)
            clip_end = min(video_frames - 1, track_end, interaction_end + context_frames)
            camera_start = max(0, interaction_start - context_frames)
            camera_end = min(video_frames - 1, interaction_end + context_frames)
            key = (pedestrian_id, interaction_start, interaction_end)
            occurrence[key] = occurrence.get(key, 0) + 1
            suffix = f"_{occurrence[key]:02d}" if occurrence[key] > 1 else ""
            event_id = (
                f"{source_id}__{pedestrian_id}__f{interaction_start:06d}-{interaction_end:06d}{suffix}"
            )
            joint_labels = sorted(
                {
                    by_frame[frame].get("Joint Interaction", "").strip()
                    for frame in range(interaction_start, interaction_end + 1)
                    if by_frame.get(frame, {}).get("Joint Interaction", "").strip().lower()
                    not in MISSING_VALUES
                }
            )
            events.append(
                {
                    "event_id": event_id,
                    "source_id": source_id,
                    "set_id": set_id,
                    "pedestrian_id": pedestrian_id,
                    "cvat_track_id": track.attrib.get("id"),
                    "interaction_frames": [interaction_start, interaction_end],
                    "interaction_frame_count": interaction_end - interaction_start + 1,
                    "clip_frames": [clip_start, clip_end],
                    "camera_clip_frames": [camera_start, camera_end],
                    "track_frames": [track_start, track_end],
                    "effective_target_frames": [max(clip_start, track_start), min(clip_end, track_end)],
                    "joint_interaction": joint_labels,
                    "source_video": f"videos/gopro/{set_id}/{source_id}.MP4",
                    "annotation": _posix_relative(path, annotation_root.parent),
                    "obd_annotation": f"annotations_vehicle/gopro/{set_id}/{source_id}_obd.xml",
                }
            )
    return ParsedVideo(source_id, path, video_frames, events)


def _merge_scenes(events: list[dict]) -> list[dict]:
    """Merge overlapping event context windows within each source video."""
    scenes: list[dict] = []
    by_source: dict[str, list[dict]] = {}
    for event in events:
        by_source.setdefault(event["source_id"], []).append(event)

    for source_id, source_events in sorted(by_source.items()):
        ordered = sorted(
            source_events,
            key=lambda item: (item["camera_clip_frames"], item["event_id"]),
        )
        groups: list[dict] = []
        for event in ordered:
            start, end = event["camera_clip_frames"]
            if not groups or start > groups[-1]["end"]:
                groups.append({"start": start, "end": end, "events": [event]})
            else:
                groups[-1]["end"] = max(groups[-1]["end"], end)
                groups[-1]["events"].append(event)

        for index, group in enumerate(groups, start=1):
            scene_id = f"{source_id}__scene_{index:03d}__f{group['start']:06d}-{group['end']:06d}"
            scene = {
                "scene_id": scene_id,
                "source_id": source_id,
                "set_id": group["events"][0]["set_id"],
                "source_video": group["events"][0]["source_video"],
                "obd_annotation": group["events"][0]["obd_annotation"],
                "clip_frames": [group["start"], group["end"]],
                "frame_count": group["end"] - group["start"] + 1,
                "event_ids": [event["event_id"] for event in group["events"]],
            }
            scenes.append(scene)
            for event in group["events"]:
                event["scene_id"] = scene_id
                event["event_frames_in_scene"] = [
                    event["interaction_frames"][0] - group["start"],
                    event["interaction_frames"][1] - group["start"],
                ]
                event["target_frames_in_scene"] = [
                    event["effective_target_frames"][0] - group["start"],
                    event["effective_target_frames"][1] - group["start"],
                ]
    return scenes


def build_cohort(annotation_root: Path, context_frames: int = 90, fps: float = 30.0) -> dict:
    annotation_root = annotation_root.resolve()
    paths = sorted(annotation_root.rglob("*.xml"))
    parsed = [parse_annotation(path, annotation_root, context_frames) for path in paths]
    events = sorted(
        (event for video in parsed for event in video.events),
        key=lambda item: (item["source_id"], item["interaction_frames"], item["pedestrian_id"]),
    )
    scenes = _merge_scenes(events)
    interaction_frames = sum(event["interaction_frame_count"] for event in events)
    scene_frames = sum(scene["frame_count"] for scene in scenes)
    return {
        "schema_version": "1.0",
        "dataset": "IDD-PeD",
        "fps": float(fps),
        "context_frames": int(context_frames),
        "statistical_units": {
            "E1a": "event",
            "E2a": "event",
            "E3": "scene",
        },
        "summary": {
            "annotation_files": len(paths),
            "source_videos_with_events": len({event["source_id"] for event in events}),
            "pedestrian_tracks_with_events": len(
                {(event["source_id"], event["cvat_track_id"]) for event in events}
            ),
            "events": len(events),
            "interaction_frames": interaction_frames,
            "scenes": len(scenes),
            "deduplicated_scene_frames": scene_frames,
            "deduplicated_scene_seconds": scene_frames / float(fps),
        },
        "scenes": scenes,
        "events": events,
    }


def write_cohort(cohort: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cohort, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
