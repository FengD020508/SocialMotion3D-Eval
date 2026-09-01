#!/usr/bin/env python3
"""Run resumable DROID/MegaSAM camera reconstruction on scene-level clips."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _valid_camera(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            return (
                {"T_c2w", "T_w2c", "frame_numbers"}.issubset(data.files)
                and len(data["T_c2w"]) == len(data["frame_numbers"])
                and len(data["T_c2w"]) > 1
            )
    except Exception:
        return False


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--genmo-root", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("droid", "megasam"), default=["droid", "megasam"])
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    genmo = args.genmo_root.resolve()
    inputs = args.inputs.resolve()
    manifest = json.loads((inputs / "selection_manifest.json").read_text(encoding="utf-8"))
    selected = set(args.ids or [])
    entries = [entry for entry in manifest["clips"] if not selected or entry["clip_id"] in selected]
    env = os.environ.copy()
    rows: list[dict] = []
    summary_path = args.output_root / "run_summary.json"
    for method in args.methods:
        for index, entry in enumerate(entries, start=1):
            clip_id = entry["clip_id"]
            output = args.output_root / method / clip_id / "camera_trajectory.npz"
            row = {
                "method": method,
                "clip_id": clip_id,
                "scene_id": entry["scene_id"],
                "started_at": datetime.now(timezone.utc).isoformat(),
                "output": str(output),
            }
            if args.resume and _valid_camera(output):
                row["status"] = "skipped_valid"
                rows.append(row)
                _write_summary(summary_path, rows)
                print(f"[{method} {index}/{len(entries)}] {clip_id}: already valid", flush=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            if method == "droid":
                command = [
                    sys.executable,
                    "scripts/idd_ped/run_droid_slam.py",
                    "--inputs",
                    str(inputs),
                    "--manifest",
                    str(inputs / "selection_manifest.json"),
                    "--clip_id",
                    clip_id,
                    "--output",
                    str(output),
                ]
            else:
                command = [
                    sys.executable,
                    "scripts/idd_ped/run_megasam.py",
                    "--inputs",
                    str(inputs),
                    "--clip_id",
                    clip_id,
                    "--output",
                    str(output),
                    "--work_root",
                    str(args.work_root),
                    "--resume",
                ]
            print(f"[{method} {index}/{len(entries)}] {clip_id}: starting", flush=True)
            try:
                subprocess.run(command, cwd=genmo, env=env, check=True)
                if not _valid_camera(output):
                    raise RuntimeError(f"camera contract failed validation: {output}")
                row["status"] = "ok"
                if method == "megasam":
                    work = args.work_root / Path(entry["output"]).stem
                    for disposable in (work / "frames", work / "mono_depth", work / "megasam_raw.npz"):
                        if disposable.is_dir():
                            shutil.rmtree(disposable)
                        elif disposable.is_file():
                            disposable.unlink()
            except Exception as error:
                row["status"] = "failed"
                row["error"] = f"{type(error).__name__}: {error}"
                rows.append(row)
                _write_summary(summary_path, rows)
                raise
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            rows.append(row)
            _write_summary(summary_path, rows)
    if "megasam" in args.methods:
        cleanup_rows = []
        for entry in entries:
            clip_id = entry["clip_id"]
            camera = args.output_root / "megasam" / clip_id / "camera_trajectory.npz"
            if not _valid_camera(camera):
                raise RuntimeError(f"refusing final MegaSAM cleanup; invalid camera: {camera}")
            metric_depth = args.work_root / Path(entry["output"]).stem / "metric_depth"
            bytes_removed = _directory_bytes(metric_depth) if metric_depth.is_dir() else 0
            if metric_depth.is_dir():
                shutil.rmtree(metric_depth)
            cleanup_rows.append(
                {
                    "clip_id": clip_id,
                    "metric_depth": str(metric_depth),
                    "bytes_removed": bytes_removed,
                    "status": "removed" if bytes_removed else "already_absent",
                    "cleaned_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        _write_summary(args.output_root / "megasam_cleanup.json", cleanup_rows)
    print(f"completed {len(rows)} camera runs", flush=True)


if __name__ == "__main__":
    main()
