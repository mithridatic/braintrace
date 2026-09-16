import brainstate
import numpy as np
import pytest

from .h01_forest_cell_test import make_cell
from .h01_forest_discretization import canonical_declarations, canonical_key, fuse_discretizations


@pytest.fixture(scope='module')
def discretizations():
    with brainstate.environ.context(precision=64):
        return [make_cell(0, m_open=.5)._discretization, make_cell(1, dendrite_um=25., calcium=False)._discretization]


def test_offsets_and_ids_are_cumulative(discretizations):
    forest, offsets, _ = fuse_discretizations(discretizations)
    n0, p0 = len(discretizations[0].cvs), len(discretizations[0].nodes)
    assert offsets.cv.tolist() == [0, n0, n0+len(discretizations[1].cvs)]
    assert offsets.point.tolist() == [0, p0, p0+len(discretizations[1].nodes)]
    assert offsets.cells == 2
    assert [cv.id for cv in forest.cvs] == list(range(len(forest.cvs)))
    assert [node.id for node in forest.nodes] == list(range(len(forest.nodes)))
    roots = [cv.id for cv in forest.cvs if cv.parent_cv is None]
    assert roots == [0, n0]
    second = discretizations[1].node_tree
    for edge, original in zip(forest.node_tree.edges[len(discretizations[0].node_tree.edges):], second.edges):
        assert edge.parent_node_id == original.parent_node_id+p0 and edge.child_node_id == original.child_node_id+p0
        assert [r.cv_id for r in edge.roles] == [r.cv_id+n0 for r in original.roles]
    np.testing.assert_array_equal(forest.node_tree.cv_to_mid_node_id[n0:], np.asarray(second.cv_to_mid_node_id)+p0)
    assert len(forest.cv_tree.branch_to_cv_ids) == sum(len(d.cv_tree.branch_to_cv_ids) for d in discretizations)


def test_geometry_and_point_mechanisms_are_kept(discretizations):
    forest, offsets, _ = fuse_discretizations(discretizations)
    for index, discretization in enumerate(discretizations):
        for cv, original in zip(forest.cvs[offsets.cv[index]:offsets.cv[index+1]], discretization.cvs):
            assert cv.area == original.area and cv.cm == original.cm and cv.r_axial_prox == original.r_axial_prox
            assert cv.point_mech == original.point_mech


def test_canonical_layouts_carry_the_parameter_union(discretizations):
    canonical = canonical_declarations(discretizations)
    forest, _, _ = fuse_discretizations(discretizations)
    keys = {canonical_key(m) for d in discretizations for cv in d.cvs for m in cv.density_mech}
    assert set(canonical) == keys
    natg = canonical[('channel', 'H01PV_NaTg', 'pv_NaTg')]
    assert 'm_open' in natg.params and 'g_max' in natg.params
    declared = {id(m) for cv in forest.cvs for m in cv.density_mech}
    assert declared <= {id(m) for m in canonical.values()}   # every CV points at a canonical object
    assert all(len({canonical_key(m) for m in cv.density_mech}) == len(cv.density_mech) for cv in forest.cvs)


def test_empty_forest_is_rejected():
    with pytest.raises(ValueError):
        fuse_discretizations([])
