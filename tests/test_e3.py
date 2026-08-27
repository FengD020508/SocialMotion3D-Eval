from __future__ import annotations

import unittest

import numpy as np

from socialmotion3d_eval.e3 import _aggregate, _flat_rows, _translation_not_evaluable_report
from socialmotion3d_eval.metrics import build_motion_series


class E3Tests(unittest.TestCase):
    def test_low_motion_keeps_scale_free_diagnostics(self) -> None:
        n_frames = 31
        fps = 30.0
        camera = {
            "camera_center": np.zeros((n_frames, 3), dtype=np.float64),
            "rotation": np.repeat(np.eye(3)[None], n_frames, axis=0),
            "timestamps": np.arange(n_frames, dtype=np.float64) / fps,
            "frame_numbers": np.arange(n_frames),
            "tracking_failed": None,
        }
        series = build_motion_series(
            camera,
            np.zeros(n_frames, dtype=np.float64),
            smooth_window_frames=1,
            jump_mad_factor=12.0,
            orthogonality_tolerance=0.1,
            determinant_tolerance=0.05,
        )
        result, arrays = _translation_not_evaluable_report(
            series,
            series["interval_valid"],
            calibration_fraction=0.4,
            min_calibration_speed_mps=0.5,
            error=ValueError("fewer than 10 valid calibration intervals"),
        )

        self.assertEqual(result["status"], "translation_not_evaluable")
        self.assertEqual(result["scale_calibration"]["n_samples"], 0)
        self.assertIn("rotation_stability", result)
        self.assertIn("scale_calibration_eligible_mask", arrays)

        report = {"clips": [{"clip_id": "low_motion", "methods": {"droid": result, "megasam": result}}]}
        rows = _flat_rows(report)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["status"] == "translation_not_evaluable" for row in rows))
        summary = _aggregate(report)
        self.assertEqual(summary["methods"]["droid"]["n_translation_clips"], 0)
        self.assertEqual(summary["methods"]["droid"]["n_rotation_clips"], 1)
        self.assertEqual(summary["paired_megasam_minus_droid"]["n_rotation_clips"], 1)


if __name__ == "__main__":
    unittest.main()
