from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


def _load_wrapper():
    # The focal validator itself has no OpenCV dependency.  Stub the optional
    # runtime module so this unit test also runs in the lightweight test env.
    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    camera_geometry = types.ModuleType("scripts.idd_ped.camera_geometry")
    camera_geometry.normalize_and_y_up = None
    camera_geometry.save_camera_npz = None
    camera_geometry.validate_camera = None
    sys.modules.setdefault("scripts.idd_ped.camera_geometry", camera_geometry)
    path = (
        Path(__file__).resolve().parents[1]
        / "pipeline/genmo/scripts/idd_ped/run_megasam.py"
    )
    spec = importlib.util.spec_from_file_location("run_megasam", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_valid_focal_accepts_only_finite_positive_focals() -> None:
    wrapper = _load_wrapper()
    positive = np.diag([1200.0, 1200.0, 1.0]).astype(np.float32)
    negative = np.diag([-1200.0, -1200.0, 1.0]).astype(np.float32)
    mixed = np.diag([1200.0, -1200.0, 1.0]).astype(np.float32)
    nonfinite = positive.copy()
    nonfinite[0, 0] = np.nan

    assert wrapper.valid_focal(positive)
    assert not wrapper.valid_focal(negative)
    assert not wrapper.valid_focal(mixed)
    assert not wrapper.valid_focal(nonfinite)
