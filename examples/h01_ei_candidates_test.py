"""The shared example map preserves the full component and records inference."""
from braintrace.datasets.h01_anatomy_test import imported
from braintrace.datasets.h01_ei_cell import _validate_regions
from examples.h01_ei_candidates import label_partition


def test_shared_example_map_is_complete_and_explicit(imported):
    regions, basis = label_partition(imported)
    assert _validate_regions(imported.morphology, regions, "I")
    assert "unclassified" in basis and "inferred" in basis
    assert "No apical identity or myelin insulation" in basis
