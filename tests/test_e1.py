import unittest

import numpy as np

from socialmotion3d_eval.e1 import align_by_local_frame, method_disagreement, robust_temporal_metrics


class E1Tests(unittest.TestCase):
    def make_motion(self, frames: int = 20) -> np.ndarray:
        rng = np.random.default_rng(4)
        skeleton = rng.normal(size=(17, 3))
        motion = np.repeat(skeleton[None], frames, axis=0)
        motion[:, :, 0] += np.linspace(0.0, 1.0, frames)[:, None]
        return motion

    def test_alignment_uses_common_local_frames(self):
        motion = self.make_motion(5)
        aligned = align_by_local_frame(
            motion,
            np.asarray([10, 11, 12, 13, 14]),
            motion[1:],
            np.asarray([11, 12, 13, 14]),
        )
        np.testing.assert_array_equal(aligned.local_frames, [11, 12, 13, 14])
        self.assertEqual(aligned.motionbert.shape, (4, 17, 3))

    def test_rigidly_equivalent_motions_have_near_zero_disagreement(self):
        motion = self.make_motion()
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = motion @ rotation
        valid = np.ones(len(motion), dtype=bool)
        metrics = method_disagreement(rotated, motion, valid)
        self.assertLess(metrics["normalized_mpjpe_median"], 1e-10)

    def test_constant_local_pose_has_zero_temporal_derivatives(self):
        motion = self.make_motion()
        valid = np.ones(len(motion), dtype=bool)
        metrics = robust_temporal_metrics(motion, valid, 30.0)
        self.assertLess(metrics["joint_velocity_p95_normalized_per_s"], 1e-10)
        self.assertLess(metrics["joint_acceleration_p95_normalized_per_s2"], 1e-10)
        self.assertLess(metrics["joint_jerk_p95_normalized_per_s3"], 1e-10)


if __name__ == "__main__":
    unittest.main()
