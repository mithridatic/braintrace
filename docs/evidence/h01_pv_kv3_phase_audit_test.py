"""Preserve the failed cross-build gate when evaluating the phase diagnostic."""

import pytest

from docs.evidence.h01_pv_kv3_phase_audit import main


def test_original_attribution_remains_invalid_after_failed_exact_control():
    """Do not silently substitute the new within-build design for the old test."""
    with pytest.raises(AssertionError, match="exact cross-build control failed"):
        main()
