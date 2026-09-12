"""Preservation of measured cable geometry and source selections after additions."""

from types import SimpleNamespace

import numpy as np

from braincell.filter import AtLocation
from braintrace.biophysics.spines import Spine
from .h01_anatomy import _geometry_signature
from .h01_anatomy_test import imported, _position
from .h01_ei_cell import _validate_regions
from .h01_network import _regions
from .h01_spine_assembly import assemble_spines, assembled_location


def explicit_spine(imported):
    branch, lo, hi = _regions(imported)['dend'].evaluate(imported.morphology).intervals[0]
    return Spine('synthetic_0', int(branch), (lo+hi)/2, .689, .0335, .583, .2915,
                 (0., 1., 0.), 'Synthetic source-anchored fixture; paper dimensions')


def test_added_geometry_keeps_original_soma_regions_and_contact_positions(imported):
    original = _geometry_signature(imported.morphology)
    spine = explicit_spine(imported)
    assembly = assemble_spines(imported, _regions(imported), [spine])
    assert _geometry_signature(imported.morphology) == original
    assert assembly.evidence['source_sha256'] == imported.source_sha256
    assert assembly.evidence['geometry_sha256'] != original
    assert len(assembly.morphology.branches) > len(imported.morphology.branches)
    _validate_regions(assembly.morphology, assembly.regions, 'E')
    target = SimpleNamespace(morphology=assembly.morphology)
    np.testing.assert_allclose(_position(target, assembly.soma),
                               _position(imported, imported.anatomy().soma_location()))
    site = (spine.parent_branch, spine.parent_x)
    mapped = assembled_location(dict(spine_assembly=assembly.evidence), site)
    np.testing.assert_allclose(_position(target, AtLocation(*mapped)), _position(imported, AtLocation(*site)))
    assert assembled_location({}, site) == site
    for part in ('neck', 'head'):
        index = next(i for i, b in enumerate(assembly.morphology.branches) if b.name == 'added_'+spine.identity+'_'+part)
        assert any(b == index and lo == 0 and hi == 1
                   for b, lo, hi in assembly.regions['dend'].evaluate(assembly.morphology).intervals)
