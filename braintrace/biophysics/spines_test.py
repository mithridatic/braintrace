"""Cable-preserving attachment, provenance and geometry edge cases."""

from dataclasses import replace

import braincell
import brainunit as u
import numpy as np
import pytest

from .spines import Spine, add_spines


def source():
    branch = braincell.Branch(lengths=np.array([10., 20.])*u.um,
        radii_proximal=np.array([2., 1.])*u.um, radii_distal=np.array([1., .5])*u.um,
        points_proximal=np.array([[0., 0., 0.], [10., 0., 0.]])*u.um,
        points_distal=np.array([[10., 0., 0.], [30., 0., 0.]])*u.um, type='dendrite')
    return braincell.Morphology(root_name='dend', root_branch=branch)


def spine():
    return Spine('synthetic_0', 0, .5, .7, .0335, .6, .3, (0., 1., 0.), 'synthetic test fixture')


def test_interior_spine_splits_taper_without_changing_source():
    original = source()
    result, mapping, heads = add_spines(original, [spine()])
    assert len(original.branches) == 1 and len(result.branches) == 4
    assert mapping[0] == [(0., .5, 'source_0_0'), (.5, 1., 'source_0_1')]
    views = {view.name: view for view in result.branches}
    pieces = [views[name].branch for _, _, name in mapping[0]]
    lengths = np.concatenate([p.lengths.to_decimal(u.um) for p in pieces])
    np.testing.assert_allclose(lengths, [10., 5., 15.])
    np.testing.assert_allclose(pieces[0].radii_distal.to_decimal(u.um), [1., .875])
    np.testing.assert_allclose(pieces[1].radii_proximal.to_decimal(u.um), [.875])
    head = views[heads['synthetic_0']].branch
    np.testing.assert_allclose(head.points_proximal.to_decimal(u.um), [[15., .7, 0.]])
    assert views[heads['synthetic_0']].parent.name == 'added_synthetic_0_neck'


@pytest.mark.parametrize('position', [0., 1.])
def test_endpoint_and_reversed_branch(position):
    original = source()
    branch = original.branches[0].branch
    original.attach(parent='dend', child_name='reverse', child_branch=branch, child_x=1.)
    result, mapping, _ = add_spines(original, [replace(spine(), parent_branch=1, parent_x=position)])
    assert len(mapping) == 2 and len(result.branches) == 4


def test_empty_additions_clone_geometry():
    original = source()
    result, mapping, heads = add_spines(original, [])
    assert not heads and len(mapping) == 1 and result is not original


@pytest.mark.parametrize('kwargs', [dict(identity=''), dict(provenance=''), dict(parent_x=-1.),
    dict(parent_branch=-1), dict(neck_length_um=0.), dict(direction=(1., 1., 1.))])
def test_bad_addition(kwargs):
    with pytest.raises(ValueError):
        replace(spine(), **kwargs)


def test_duplicate_and_outside_parent():
    with pytest.raises(ValueError, match='Duplicate'):
        add_spines(source(), [spine(), spine()])
    with pytest.raises(ValueError, match='outside'):
        add_spines(source(), [replace(spine(), parent_branch=5)])


def test_missing_coordinates_refused():
    branch = braincell.Branch(lengths=np.array([1.])*u.um,
        radii_proximal=np.array([1.])*u.um, radii_distal=np.array([1.])*u.um, type='dendrite')
    with pytest.raises(ValueError, match='xyz'):
        add_spines(braincell.Morphology(root_name='dend', root_branch=branch), [spine()])
