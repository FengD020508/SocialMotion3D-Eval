from pathlib import Path

from socialmotion3d_eval.iddped_cohort import build_cohort


def _write_xml(path: Path) -> None:
    path.parent.mkdir(parents=True)
    boxes = []
    for frame in range(10, 31):
        if frame in {15, 16, 20}:
            ego = str(frame)
            joint = "Vehicle slows"
        else:
            ego = "-"
            joint = "N/A"
        boxes.append(
            f'''<box frame="{frame}" outside="0" xtl="0" ytl="0" xbr="1" ybr="1">
            <attribute name="id">gp_1</attribute>
            <attribute name="ped_ego_veh_interaction">{ego}</attribute>
            <attribute name="Joint Interaction">{joint}</attribute>
            </box>'''
        )
    path.write_text(
        f'''<annotations><meta><task><name>gp_set_0001_vid_0001</name><size>100</size></task></meta>
        <track id="7" label="pedestrian">{"".join(boxes)}</track></annotations>''',
        encoding="utf-8",
    )


def test_build_cohort_separates_events_and_merges_camera_windows(tmp_path: Path):
    annotation_root = tmp_path / "annotations"
    xml = annotation_root / "gopro" / "gp_set_0001" / "gp_set_0001_vid_0001.xml"
    _write_xml(xml)

    cohort = build_cohort(annotation_root, context_frames=5, fps=30.0)

    assert cohort["summary"]["events"] == 2
    assert cohort["summary"]["interaction_frames"] == 3
    assert cohort["summary"]["scenes"] == 1
    assert cohort["scenes"][0]["clip_frames"] == [10, 25]
    assert len(cohort["scenes"][0]["event_ids"]) == 2
    assert {event["scene_id"] for event in cohort["events"]} == {cohort["scenes"][0]["scene_id"]}
