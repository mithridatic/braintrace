"""Check that the import oracle detects reversed and duplicate edges."""

from docs.evidence.h01_population_import_audit import segment_counter


def test_segment_multiset_preserves_direction_and_multiplicity():
    forward = segment_counter([[0, 0, 0]], [[1, 0, 0]])
    reverse = segment_counter([[1, 0, 0]], [[0, 0, 0]])
    twice = segment_counter([[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    assert sum((forward-reverse).values()) == 1
    assert sum((twice-forward).values()) == 1
