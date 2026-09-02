from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_e2a_cohort.py"
    spec = importlib.util.spec_from_file_location("run_e2a_cohort", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_translation_gate_requires_positive_scales_for_both_methods() -> None:
    module = _load_module()
    complete = {
        "methods": {
            "droid": {"status": "ok", "scale_m_per_raw_unit": 2.0},
            "megasam": {"status": "ok", "scale_m_per_raw_unit": 1.5},
        }
    }
    unavailable = {
        "methods": {
            "droid": {"status": "translation_not_evaluable"},
            "megasam": {"status": "translation_not_evaluable"},
        }
    }

    assert module._translation_gate(complete)[0]
    assert not module._translation_gate(unavailable)[0]


def test_expected_motion_unavailable_is_narrowly_classified() -> None:
    module = _load_module()

    assert module._expected_motion_unavailable(
        ValueError("fixed_gem: no common valid E2a intervals")
    )
    assert not module._expected_motion_unavailable(ValueError("camera frame mismatch"))
    assert not module._expected_motion_unavailable(RuntimeError("no common valid E2a intervals"))
