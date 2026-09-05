"""Protect equilibrium, closing, and inactivation laws in the source patch."""

from pathlib import Path

import pytest

from docs.evidence.h01_l2_prepare_activation import patch_activation
from docs.evidence.h01_l2_prepare_recovery import patch_recovery


def test_only_opening_patch_is_added_to_checked_recovery_source():
    source = (Path(__file__).parents[2] /
              ".cache/human-pyramidal-l2/source-model/NaTs.mod").read_bytes()
    result = patch_activation(source)
    text = result.decode()
    assert text.count("if (mInf > m) { mTau = mTau * m_opening_factor }") == 1
    assert text.count("m_opening_factor = 1") == 1
    restored = text.replace(", m_opening_factor", "").replace(
        "\n\tm_opening_factor = 1", "").replace(
        "\n\t\tif (mInf > m) { mTau = mTau * m_opening_factor }", "")
    assert restored.encode() == patch_recovery(source)
    assert patch_activation(source) == result


@pytest.mark.parametrize("source", [b"", b"different mechanism"])
def test_unexpected_original_source_is_rejected(source):
    with pytest.raises(ValueError, match="digest"):
        patch_activation(source)
