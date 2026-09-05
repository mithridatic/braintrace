"""Protect the pinned channel law and isolate the recovery-only patch."""

from pathlib import Path

import pytest

from docs.evidence.h01_l2_prepare_recovery import patch_recovery


def test_pinned_source_changes_only_declared_lines():
    source = (Path(__file__).parents[2] / ".cache/human-pyramidal-l2/source-model/NaTs.mod").read_bytes()
    patched = patch_recovery(source).decode()
    assert "h_recovery_factor = 1" in patched
    assert "if (hInf > h)" in patched
    restored = patched.replace(
        "RANGE gbar, g, ina, h_recovery_factor", "RANGE gbar, g, ina")
    restored = restored.replace("\n\th_recovery_factor = 1", "")
    restored = restored.replace("\n\t\tif (hInf > h) { hTau = hTau * h_recovery_factor }", "")
    assert restored.encode() == source


def test_unexpected_source_is_rejected():
    with pytest.raises(ValueError, match="digest"):
        patch_recovery(b"different mechanism")
