"""Annotation identity, conservative selection, and geometric projection tests."""

from dataclasses import replace

import brainunit as u
import numpy as np
import pytest

from .h01 import H01Archive
from .h01_test import _archive
from .h01_annotations import H01Synapse

SOURCE = (b"10 0 0 0 0 100 -1\n20 3 100 0 0 500 10\n30 3 200 0 0 1000 20\n"
          b"40 1 300 0 0 100 30\n50 1 400 0 0 100 40\n60 -1 300 100 0 100 40\n")


@pytest.fixture
def imported(tmp_path, monkeypatch):
    path, _ = _archive(tmp_path, monkeypatch, [("12.0.swc", SOURCE)])
    return H01Archive(path).load(12, component=0)


def _position(imported, location):
    branch, x = location.evaluate(imported.morphology).points[0]
    b = imported.morphology.branches[branch].branch
    lengths = np.asarray(b.lengths.to_decimal(u.um))
    bounds = np.r_[0., np.cumsum(lengths)]
    points = np.asarray(b.points.to_decimal(u.um))
    return np.array([np.interp(x * bounds[-1], bounds, points[:, i]) for i in range(3)])


def _region_length(imported, region):
    return sum((hi - lo) * float(imported.morphology.branches[b].branch.lengths.sum().to_decimal(u.um))
               for b, lo, hi in region.evaluate(imported.morphology).intervals)


def test_source_identity_and_soma_not_root(imported):
    anatomy = imported.anatomy()
    assert anatomy.label_counts == {"axon": 1, "soma": 2, "dendrite": 2, "unclassified": 1}
    for row in imported.source_rows:
        np.testing.assert_allclose(_position(imported, anatomy.location(int(row[0]))), row[2:5] * [.032, .032, .033])
    np.testing.assert_allclose(_position(imported, anatomy.soma_location()), [6.4, 0, 0])
    assert len(anatomy.locations("soma").evaluate(imported.morphology).points) == 2
    assert not anatomy.locations("cilium").evaluate(imported.morphology).points
    assert "per-sample manual review not supplied" in anatomy.provenance["annotation_verification"]
    with pytest.raises(KeyError):
        anatomy.location(999)
    with pytest.raises(ValueError, match="Unknown"):
        anatomy.locations("somaa")


def test_geodesic_neighborhood_follows_branches_and_cuts_edges(imported):
    anatomy = imported.anatomy()
    assert _region_length(imported, anatomy.cable_neighborhood(30, radius_um=1.)) == pytest.approx(2.)
    assert _region_length(imported, anatomy.cable_neighborhood(30, radius_um=4.8)) == pytest.approx(11.2)
    region = anatomy.cable_neighborhood(30, radius_um=100.)
    assert _region_length(imported, region) == pytest.approx(16.)
    assert "inferred_geodesic" in region.policy
    with pytest.raises(KeyError):
        anatomy.cable_neighborhood(999, radius_um=1.)
    for invalid in (0., -1., np.nan, np.inf):
        with pytest.raises(ValueError, match="positive and finite"):
            anatomy.cable_neighborhood(30, radius_um=invalid)


def test_region_policies_preserve_unknown_and_geometry(imported):
    anatomy = imported.anatomy()
    assert _region_length(imported, anatomy.region("soma")) == pytest.approx(3.2)
    assert _region_length(imported, anatomy.region("dendrite")) == pytest.approx(3.2)
    assert _region_length(imported, anatomy.region("axon")) == 0
    assert _region_length(imported, anatomy.region("soma", policy="sample_neighborhood")) == pytest.approx(6.4)
    assert _region_length(imported, anatomy.region("unclassified", policy="sample_neighborhood")) == pytest.approx(1.6)
    total = sum(_region_length(imported, anatomy.region(label, policy="sample_neighborhood"))
                for label in anatomy.label_counts)
    assert total == pytest.approx(float(imported.morphology.total_length.to_decimal(u.um)))
    with pytest.raises(ValueError, match="policy"):
        anatomy.region("soma", policy="guess")
    with pytest.raises(ValueError, match="different morphology"):
        anatomy.region("soma").evaluate(object())
    with pytest.raises(ValueError, match="different morphology"):
        anatomy.soma_location().evaluate(object())


def test_unknown_labels_and_absent_soma(imported):
    rows = imported.source_rows.copy()
    rows[:, 1] = [1000, 1002, 1005, 5, 7, -1]
    anatomy = replace(imported, source_rows=rows).anatomy()
    assert anatomy.label_counts["myelinated_axon"] == 2
    assert anatomy.label_counts["myelinated_axon_internal_fragment"] == 1
    assert anatomy.label_counts["unknown_7"] == 1
    assert anatomy.locations("unknown_7").evaluate(imported.morphology).points
    with pytest.raises(ValueError, match="no soma"):
        anatomy.soma_location()


def test_projection_mid_segment_junction_rejection_and_ambiguity(imported):
    anatomy = imported.anatomy()
    results = anatomy.project([[11.2, .1, 0], [9.6, 0, 0], [999, 999, 0], [11.2, 1.6, 0]], max_distance_um=2)
    assert [r.status for r in results] == ["projected", "projected", "too_far", "ambiguous"]
    assert results[0].distance_um == pytest.approx(.1)
    np.testing.assert_allclose(_position(imported, results[0].location), [11.2, 0, 0])
    np.testing.assert_allclose(_position(imported, results[1].location), [9.6, 0, 0])
    assert results[2].location is None and results[3].location is None
    assert anatomy.project(np.empty((0, 3)), max_distance_um=0) == ()


@pytest.mark.parametrize("positions,tolerance", [([1, 2, 3], 1), ([[np.nan, 0, 0]], 1),
                                                ([[0, 0, 0]], -1), ([[0, 0, 0]], np.inf)])
def test_invalid_projection_queries(imported, positions, tolerance):
    with pytest.raises(ValueError):
        imported.anatomy().project(positions, max_distance_um=tolerance)


def test_synapse_uses_role_endpoint_and_checks_identity(imported):
    row = H01Synapse("12", 2, "post", (500., 0, 0), (600., 0, 0), (6.4, 0, 0))
    anatomy = imported.anatomy()
    result = anatomy.project_synapses([row], max_distance_um=.1)[0]
    assert result.status == "projected"
    np.testing.assert_allclose(_position(imported, result.location), row.post_um)
    assert anatomy.project_synapses([], max_distance_um=1) == ()
    with pytest.raises(ValueError, match="different H01"):
        anatomy.project_synapses([replace(row, neuron_id="13")], max_distance_um=1)


def test_changed_or_ambiguous_source_geometry_rejected(imported):
    rows = imported.source_rows.copy()
    rows[1, 2:5] = rows[0, 2:5]
    with pytest.raises(ValueError, match="Coincident"):
        replace(imported, source_rows=rows).anatomy()
    rows = imported.source_rows.copy()
    rows[1, 2] += 1
    with pytest.raises(ValueError, match="endpoints"):
        replace(imported, source_rows=rows).anatomy()
    rows = imported.source_rows.copy()
    rows[1, 6] = rows[2, 0]
    with pytest.raises(ValueError, match="edges"):
        replace(imported, source_rows=rows).anatomy()


def test_selections_accept_braincell_clone_but_reject_changed_geometry(imported):
    from braincell import Branch
    from braincell.morph.morphology import clone_morpho

    anatomy = imported.anatomy()
    region, location = anatomy.region("soma"), anatomy.soma_location()
    clone = clone_morpho(imported.morphology)
    assert clone is not imported.morphology
    assert region.evaluate(clone) == region.evaluate(imported.morphology)
    assert location.evaluate(clone) == location.evaluate(imported.morphology)
    clone.attach(parent=clone.root, child_branch=Branch.from_lengths(
        lengths=[5.] * u.um, radii=[1., 1.] * u.um), child_name="extra")
    with pytest.raises(ValueError, match="different morphology"):
        region.evaluate(clone)
    with pytest.raises(ValueError, match="different morphology"):
        location.evaluate(clone)


def test_signature_and_source_mapping_do_not_rebuild_branch_indices(imported, monkeypatch):
    from .h01_anatomy import _geometry_signature
    # Preserve the original serialization, including edge endpoint indices.
    import hashlib
    digest = hashlib.sha256()
    for view in imported.morphology.branches:
        b = view.branch
        digest.update(str((view.index, b.type, b.n_segments)).encode())
        for array in (b.lengths, b.radii_proximal, b.radii_distal, b.points_proximal, b.points_distal):
            digest.update(b'None' if array is None else np.asarray(array.to_decimal(u.um), dtype='<f8').tobytes())
    for edge in imported.morphology.edges:
        digest.update(str((edge.parent.index, edge.child.index, edge.parent_x, edge.child_x)).encode())
    expected = digest.hexdigest()
    prior = imported.anatomy()
    locations = {int(r[0]): prior.location(int(r[0])).evaluate(imported.morphology).points
                 for r in imported.source_rows}
    calls = []
    owner = type(imported.morphology)
    original = owner._branch_index
    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(owner, '_branch_index', counted)
    assert _geometry_signature(imported.morphology) == expected
    after = imported.anatomy()
    for node, points in locations.items():
        assert after.location(node).evaluate(imported.morphology).points == points
    assert not calls, 'Anatomy traversal repeatedly reconstructs the complete branch index map.'
