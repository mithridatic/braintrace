"""Protect the unmodified Kv3 law and source identity."""

from pathlib import Path

import pytest

from docs.evidence.h01_l2_prepare_kv3_closing import patch_kv3_closing


def test_only_closing_additions_change_pinned_source():
    source = (Path(__file__).parents[2] /
              ".cache/human-pyramidal-l2/source-model/Kv3_1.mod").read_bytes()
    result = patch_kv3_closing(source)
    text = result.decode()
    newline = "\r\n" if "\r\n" in text else "\n"
    assert text.count("if (mInf < m) { mTau = mTau * m_closing_factor }") == 1
    assert text.count("m_closing_factor = 1") == 1
    restored = text.replace(", m_closing_factor", "").replace(
        newline + "\tm_closing_factor = 1", "").replace(
        newline + "\t\tif (mInf < m) { mTau = mTau * m_closing_factor }", "")
    assert restored.encode() == source
    assert patch_kv3_closing(source) == result


@pytest.mark.parametrize("source", [b"", b"different mechanism"])
def test_unknown_source_is_rejected(source):
    with pytest.raises(ValueError, match="digest"):
        patch_kv3_closing(source)
