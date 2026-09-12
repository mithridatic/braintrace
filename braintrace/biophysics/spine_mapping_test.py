"""Probe/contact and complete electrical partition preservation across spines."""

from dataclasses import replace

import numpy as np
import pytest

from .spines import add_spines
from .spines_test import source, spine
from .spine_mapping import remap_location, remap_intervals, remap_electrical_regions


def test_probe_and_contact_locations_map_continuously_at_splits():
    _, mapping, _ = add_spines(source(), [spine()])
    assert remap_location(mapping, 0, .25) == ('source_0_0', .5)
    assert remap_location(mapping, 0, .5) == ('source_0_0', 1.)
    assert remap_location(mapping, 0, .75) == ('source_0_1', .5)
    assert remap_intervals(mapping, [(0, .25, .75)]) == (
        ('source_0_0', .5, 1.), ('source_0_1', 0., .5))


def test_regions_cover_new_morphology_and_preserve_parent_family():
    additions = [spine(), replace(spine(), identity='other', parent_x=.25)]
    morphology, mapping, _ = add_spines(source(), additions)
    regions = remap_electrical_regions(mapping, {'soma': [(0, 0., .3)], 'dend': [(0, .3, 1.)]}, additions)
    assert ('added_synthetic_0_head', 0., 1.) in regions['dend']
    assert ('added_other_neck', 0., 1.) in regions['soma']
    covered = {}
    for rows in regions.values():
        for name, lo, hi in rows:
            covered[name] = covered.get(name, 0.)+hi-lo
    assert set(covered) == {branch.name for branch in morphology.branches}
    np.testing.assert_allclose(list(covered.values()), 1.)


@pytest.mark.parametrize('regions', [{'dend': [(0, 0., .9)]}, {'dend': [(0, .1, 1.)]},
    {'dend': [(0, 0., .6), (0, .5, 1.)]}, {'dend': [(1, 0., 1.)]}])
def test_incomplete_or_ambiguous_partition_is_rejected(regions):
    _, mapping, _ = add_spines(source(), [spine()])
    with pytest.raises(ValueError):
        remap_electrical_regions(mapping, regions, [spine()])


def test_bad_locations_intervals_and_duplicate_additions():
    mapping = {0: [(0., .4, 'a'), (.6, 1., 'b')]}
    for branch, x in ((1, .2), (0, -.1), (0, .5)):
        with pytest.raises(ValueError):
            remap_location(mapping, branch, x)
    for interval in ((0, .8, .7), (0, 0., 1.)):
        with pytest.raises(ValueError):
            remap_intervals(mapping, [interval])
    with pytest.raises(ValueError, match='Duplicate'):
        remap_electrical_regions({0: [(0., 1., 'a')]}, {'dend': [(0, 0., 1.)]}, [spine(), spine()])
    with pytest.raises(ValueError, match='unique'):
        remap_electrical_regions({0: [(0., 1., 'a')]}, {'dend': [(0, 0., 1.)]}, [replace(spine(), parent_branch=1)])
