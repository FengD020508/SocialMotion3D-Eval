from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from socialmotion3d_eval.e2a import run_e2a


class E2aTests(unittest.TestCase):
    def test_fixed_human_three_groundings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = np.arange(100, 130, dtype=np.int64)
            timestamps = frames / 30.0
            rotation = np.repeat(np.eye(3, dtype=np.float32)[None], len(frames), axis=0)
            center = np.zeros((len(frames), 3), dtype=np.float32)
            center[:, 0] = np.linspace(0.0, 2.0, len(frames))
            camera_paths = {}
            for method, multiplier in (("droid", 1.0), ("megasam", 0.9)):
                path = root / f"{method}_camera.npz"
                np.savez(
                    path,
                    camera_center=center * multiplier,
                    R_c2w=rotation,
                    frame_numbers=frames,
                    timestamps=timestamps,
                    fps=np.asarray(30.0),
                )
                camera_paths[method] = str(path)

            human_paths = {}
            for method in ("droid", "megasam"):
                path = root / f"{method}_human.pt"
                torch.save(
                    {
                        "body_params_incam": {"transl": torch.zeros((len(frames), 3))},
                        "target_track": {"source_frames": torch.from_numpy(frames)},
                    },
                    path,
                )
                human_paths[method] = str(path)

            e3_path = root / "e3.json"
            e3_path.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "clip_id": "clip",
                                "methods": {
                                    "droid": {"status": "ok", "scale_m_per_raw_unit": 2.0},
                                    "megasam": {"status": "ok", "scale_m_per_raw_unit": 2.0},
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "clip_id": "clip",
                        "e3_report": str(e3_path),
                        "output_dir": str(root / "out"),
                        "human_sources": {method: {"path": path} for method, path in human_paths.items()},
                        "camera_sources": {
                            method: {"path": path, "frame_numbers_are_source": True}
                            for method, path in camera_paths.items()
                        },
                    }
                ),
                encoding="utf-8",
            )
            report = run_e2a(config_path)
            self.assertEqual(set(report["human_sources"]), {"droid", "megasam"})
            for human in report["human_sources"].values():
                self.assertEqual(set(human["conditions"]), {"no_ego", "droid", "megasam"})
                self.assertAlmostEqual(
                    human["conditions"]["droid"]["path_length_on_common_valid_intervals_m"], 4.0, places=5
                )
                self.assertTrue((root / "out" / "e2a_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
