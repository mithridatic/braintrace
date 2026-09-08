"""Check that the import oracle detects reversed and duplicate edges."""

from docs.evidence.h01_population_import_audit import segment_counter
from docs.evidence.h01_population_import_audit import audit_soma
from braintrace.datasets.h01_anatomy_test import imported
import pytest


def test_segment_multiset_preserves_direction_and_multiplicity():
    forward = segment_counter([[0, 0, 0]], [[1, 0, 0]])
    reverse = segment_counter([[1, 0, 0]], [[0, 0, 0]])
    twice = segment_counter([[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]])
    assert sum((forward-reverse).values()) == 1
    assert sum((twice-forward).values()) == 1


@pytest.mark.parametrize('polarity', ['E', 'I'])
def test_soma_audit_checks_source_selected_probe_in_electrical_partition(imported, polarity):
    result = audit_soma(imported, polarity)
    assert result['electrical_regions_valid']
    assert result['soma_inside_region']
    assert result['soma_location'] == imported.anatomy().soma_location().evaluate(imported.morphology).points


def test_soma_audit_rejects_probe_outside_soma_without_moving_it(imported, monkeypatch):
    anatomy = imported.anatomy()
    outside = anatomy.location(10)
    monkeypatch.setattr(type(anatomy), 'soma_location', lambda self: outside)
    result = audit_soma(imported, 'E')
    assert result['electrical_regions_valid']
    assert not result['soma_inside_region']
    assert result['soma_location'] == outside.evaluate(imported.morphology).points
