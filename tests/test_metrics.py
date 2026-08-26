from __future__ import annotations

import unittest

import numpy as np

from socialmotion3d_eval.metrics import build_motion_series, evaluate_scaled_series, safe_pearson


class MetricTests(unittest.TestCase):
    def test_held_out_scale_and_accuracy(self) -> None:
        fps = 30.0
        n_frames = 301
        timestamps = np.arange(n_frames, dtype=np.float64) / fps
        obd_frame = 4.0 + np.sin(np.linspace(0.0, 5.0, n_frames))
        obd_interval = 0.5 * (obd_frame[:-1] + obd_frame[1:])
        true_scale = 2.5
        raw_step = obd_interval / true_scale / fps
        center = np.zeros((n_frames, 3), dtype=np.float64)
        center[1:, 0] = np.cumsum(raw_step)
        camera = {
            "camera_center": center,
            "rotation": np.repeat(np.eye(3)[None], n_frames, axis=0),
            "timestamps": timestamps,
            "frame_numbers": np.arange(n_frames),
            "tracking_failed": None,
        }
        series = build_motion_series(
            camera,
            obd_frame,
            smooth_window_frames=1,
            jump_mad_factor=12.0,
            orthogonality_tolerance=0.1,
            determinant_tolerance=0.05,
        )
        report, _ = evaluate_scaled_series(
            series,
            common_mask=series["interval_valid"] & np.isfinite(series["obd_speed"]),
            calibration_fraction=0.4,
            fps=fps,
            windows_seconds=[1, 2, 3],
            min_target_distance_m=0.5,
        )
        self.assertAlmostEqual(report["scale_m_per_raw_unit"], true_scale, places=10)
        self.assertLess(report["accuracy"]["mae_mps"], 1e-10)
        self.assertLess(report["accuracy"]["rmse_mps"], 1e-10)
        self.assertAlmostEqual(report["accuracy"]["pearson_r"], 1.0, places=10)
        self.assertAlmostEqual(report["window_scale_stability"]["1s"]["median"], 1.0, places=10)

    def test_constant_pearson_is_not_applicable(self) -> None:
        self.assertIsNone(safe_pearson(np.ones(10), np.arange(10)))

    def test_scale_estimator_resists_visual_speed_spikes(self) -> None:
        from socialmotion3d_eval.metrics import fit_nonnegative_scale

        visual = np.ones(100)
        visual[:10] = 100.0
        obd = np.full(100, 2.0)
        scale, diagnostics = fit_nonnegative_scale(visual, obd, np.ones(100, dtype=bool))
        self.assertAlmostEqual(scale, 2.0, places=10)
        self.assertLess(diagnostics["ols_scale_diagnostic"], 0.1)


if __name__ == "__main__":
    unittest.main()
