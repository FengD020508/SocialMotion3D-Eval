import unittest

import numpy as np

from socialmotion3d_eval.e1 import BONES
from socialmotion3d_eval.e1c import (
    canonicalize_world_up,
    construct_shared_root_variants,
    coupling_metrics,
    desynchronize_articulation,
    recover_camera_from_joint_pairs,
)


def synthetic_skeleton(frames: int = 40) -> np.ndarray:
    base = np.zeros((17, 3), dtype=np.float64)
    base[1] = [0.15, -0.10, 0.0]
    base[2] = [0.15, -0.45, 0.0]
    base[3] = [0.15, -0.90, 0.0]
    base[4] = [-0.15, -0.10, 0.0]
    base[5] = [-0.15, -0.45, 0.0]
    base[6] = [-0.15, -0.90, 0.0]
    base[7] = [0.0, 0.20, 0.0]
    base[8] = [0.0, 0.45, 0.0]
    base[9] = [0.0, 0.60, 0.0]
    base[10] = [0.0, 0.80, 0.0]
    base[11] = [-0.25, 0.42, 0.0]
    base[12] = [-0.45, 0.25, 0.0]
    base[13] = [-0.60, 0.10, 0.0]
    base[14] = [0.25, 0.42, 0.0]
    base[15] = [0.45, 0.25, 0.0]
    base[16] = [0.60, 0.10, 0.0]
    motion = np.repeat(base[None], frames, axis=0)
    phase = np.linspace(0.0, 4.0 * np.pi, frames)
    motion[:, 3, 2] += 0.12 * np.sin(phase)
    motion[:, 6, 2] -= 0.12 * np.sin(phase)
    return motion


class E1CConstructionTests(unittest.TestCase):
    def test_canonicalizes_upside_down_world_gauge_with_proper_rotation(self):
        upright = synthetic_skeleton(12)
        upside_down = upright.copy()
        upside_down[..., 1] *= -1.0
        upside_down[..., 2] *= -1.0
        corrected, changed = canonicalize_world_up(
            upside_down, np.ones(len(upside_down), dtype=bool)
        )
        self.assertTrue(changed)
        self.assertGreater(float(np.median(corrected[:, 10, 1] - corrected[:, 0, 1])), 0.0)
        self.assertLess(float(np.median(corrected[:, [3, 6], 1] - corrected[:, None, 0, 1])), 0.0)
        np.testing.assert_allclose(
            np.linalg.norm(corrected[:, 10] - corrected[:, 0], axis=-1),
            np.linalg.norm(upside_down[:, 10] - upside_down[:, 0], axis=-1),
        )

    def test_recovers_proper_camera_transform_from_paired_joints(self):
        incam = synthetic_skeleton(12)
        angle = np.deg2rad(63.0)
        rotation = np.asarray(
            [
                [np.cos(angle), 0.0, -np.sin(angle)],
                [0.0, 1.0, 0.0],
                [np.sin(angle), 0.0, np.cos(angle)],
            ]
        )
        centers = np.column_stack(
            [np.linspace(1.0, 1.4, len(incam)), np.full(len(incam), 1.5), np.linspace(-2.0, -1.6, len(incam))]
        )
        global_joints = incam @ rotation + centers[:, None, :]
        recovered = recover_camera_from_joint_pairs(incam, global_joints, np.ones(len(incam), dtype=bool))
        np.testing.assert_allclose(recovered.camera_center, centers, atol=1e-10)
        np.testing.assert_allclose(recovered.rotation_row, np.repeat(rotation[None], len(incam), axis=0), atol=1e-10)
        self.assertTrue((np.linalg.det(recovered.rotation_row) > 0.999999).all())
        self.assertLess(float(np.nanmax(recovered.fit_rmse)), 1e-10)

    def test_shared_root_preserves_exact_gem_trajectory(self):
        gem = synthetic_skeleton()
        gem[:, :, 0] += np.linspace(0.0, 2.0, len(gem))[:, None]
        motionbert = synthetic_skeleton() * 1.7
        valid = np.ones(len(gem), dtype=bool)
        variants = construct_shared_root_variants(motionbert, gem, valid)
        np.testing.assert_allclose(variants.motionbert_shared_root[:, 0], gem[:, 0], atol=1e-10)
        for joint_a, joint_b in BONES:
            gem_lengths = np.linalg.norm(gem[:, joint_a] - gem[:, joint_b], axis=-1)
            mb_lengths = np.linalg.norm(
                variants.motionbert_shared_root[:, joint_a] - variants.motionbert_shared_root[:, joint_b],
                axis=-1,
            )
            self.assertTrue(np.isfinite(mb_lengths).all())
            self.assertGreater(np.median(gem_lengths), 0.0)

    def test_desynchronization_preserves_trajectory_and_uses_offset_pose(self):
        motion = synthetic_skeleton(20)
        motion[:, :, 0] += np.arange(20)[:, None]
        motion[:, 13, 1] += np.arange(20)
        shifted = desynchronize_articulation(motion, np.ones(20, dtype=bool), 4)
        np.testing.assert_array_equal(shifted.trajectory_indices, np.arange(16))
        np.testing.assert_array_equal(shifted.pose_indices, np.arange(4, 20))
        np.testing.assert_allclose(shifted.desynchronized[:, 0], shifted.native[:, 0])
        expected_local_wrist = motion[4:, 13] - motion[4:, 0]
        actual_local_wrist = shifted.desynchronized[:, 13] - shifted.desynchronized[:, 0]
        np.testing.assert_allclose(actual_local_wrist, expected_local_wrist)

    def test_negative_offset_crops_without_wrap(self):
        motion = synthetic_skeleton(20)
        shifted = desynchronize_articulation(motion, np.ones(20, dtype=bool), -3)
        np.testing.assert_array_equal(shifted.trajectory_indices, np.arange(3, 20))
        np.testing.assert_array_equal(shifted.pose_indices, np.arange(17))
        self.assertEqual(len(shifted.native), 17)

    def test_coupling_metrics_are_finite_for_moving_sequence(self):
        motion = synthetic_skeleton(60)
        motion[:, :, 0] += np.linspace(0.0, 2.0, len(motion))[:, None]
        metrics = coupling_metrics(motion, np.ones(len(motion), dtype=bool), 30.0)
        self.assertGreater(metrics["valid_frames"], 0)
        self.assertGreater(metrics["contact_samples"], 0)
        self.assertTrue(np.isfinite(metrics["contact_foot_speed_p95_body_scale_per_s"]))


if __name__ == "__main__":
    unittest.main()
