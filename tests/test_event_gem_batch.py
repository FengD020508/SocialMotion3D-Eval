from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_batch_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_event_gem_batch.py"
    spec = importlib.util.spec_from_file_location("run_event_gem_batch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rebase_camera_trajectory_sets_event_origin_and_preserves_motion() -> None:
    module = _load_batch_module()
    trajectory = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    trajectory[0, :3, 3] = [2.0, -1.0, 4.0]
    trajectory[1, :3, 3] = [2.5, -1.0, 4.0]
    trajectory[2, :3, 3] = [3.0, -0.5, 4.0]

    rebased, inverse = module._rebase_camera_trajectory(trajectory)

    np.testing.assert_allclose(rebased[0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(
        rebased, np.linalg.inv(trajectory[0])[None] @ trajectory, atol=1e-6
    )
    identity = np.repeat(np.eye(4)[None], len(rebased), axis=0)
    np.testing.assert_allclose(rebased @ inverse, identity, atol=1e-6)
