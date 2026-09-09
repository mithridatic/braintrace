"""Regressions for observation metadata comparisons."""

import pytest
from docs.evidence.h01_cell_state_observation import require_same_metadata


def test_json_tuple_representation_is_equivalent():
    require_same_metadata({'intervals': [(6, .8, 1.)]}, {'intervals': [[6, .8, 1.]]})


def test_changed_geometry_is_rejected():
    with pytest.raises(AssertionError, match='Cell metadata changed'):
        require_same_metadata({'intervals': [(6, .7, 1.)]}, {'intervals': [[6, .8, 1.]]})
