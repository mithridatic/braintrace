"""Check pinned contact identity and geometry against the saved source record."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import brainunit as u
import pytest
from .h01_measured_contact import measured_ie_contact


@pytest.fixture
def components():
    record = json.loads((Path(__file__).parents[2]/'docs/evidence/h01-ie-measured-contact.json').read_text())
    result = {}
    for role, side in (('I', 'pre'), ('E', 'post')):
        site = record[side]
        branch_id, fraction = site['cable_location']
        position = np.array(site['voxel'])*[.008, .008, .033]
        a = position+[-fraction, site['projection_distance_um'], 0.]
        branch = SimpleNamespace(lengths=np.array([1.])*u.um,
            points_proximal=a[None, :]*u.um,
            points_distal=(a+[1., 0., 0.])[None, :]*u.um)
        branches = [None]*(branch_id+1)
        branches[branch_id] = SimpleNamespace(branch=branch)
        result[role] = SimpleNamespace(neuron_id=site['proofread_id'], component_id=0,
            source_sha256=site['source_swc_sha256'], morphology=SimpleNamespace(branches=branches))
    return result


def test_pinned_identity_and_independent_provenance(components):
    _, _, evidence = measured_ie_contact(components)
    assert evidence['annotation_id'] == '8105899'
    assert evidence['pre']['proofread_id'] == '5584343344'
    assert evidence['post']['proofread_id'] == '4157825456'
    evidence['pre']['proofread_id'] = 'changed'
    assert measured_ie_contact(components)[2]['pre']['proofread_id'] == '5584343344'


@pytest.mark.parametrize('field,value', [('neuron_id', 'other'), ('component_id', 1), ('source_sha256', 'bad')])
def test_wrong_source_is_rejected(components, field, value):
    setattr(components['I'], field, value)
    with pytest.raises(ValueError, match='pinned'):
        measured_ie_contact(components)


def test_changed_geometry_is_rejected(components):
    branch = components['E'].morphology.branches[2805].branch
    branch.points_proximal += np.array([[0., 3., 0.]])*u.um
    branch.points_distal += np.array([[0., 3., 0.]])*u.um
    with pytest.raises(ValueError, match='geometry'):
        measured_ie_contact(components)
