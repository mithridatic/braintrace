"""Complete fragment selection, malformed forests and exact source geometry."""

import hashlib

import brainunit as u
import numpy as np
import pytest

from .h01_glia import H01GlialSelection, load_glial_fragment


def source(tmp_path, **changes):
    arrays = dict(vertices_nm=np.array([[0.,0,0],[1000,0,0],[2000,1000,0],[2000,-1000,0],
                                      [3000,-1000,0],[10000,0,0],[11000,0,0],[12000,0,0]]),
                  radius_nm=np.arange(8)*10.+100.,
                  edges=np.array([[0,1],[1,2],[1,3],[3,4],[5,6]]))
    arrays.update(changes)
    path=tmp_path/'glia.npz'
    np.savez(path, **arrays)
    document=dict(schema='h01-glial-fragment-v1', source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  identity='synthetic-tree',source='fixture',attribution='Synthetic test geometry',
                  coordinate_frame='h01-c3-nm',anchor_vertex=0,max_vertices=5,selection_basis='Complete fixture component')
    return path, document, arrays


@pytest.mark.parametrize('anchor', [0, 1, 3])
def test_complete_component_preserves_every_tapered_source_edge(tmp_path, anchor):
    path, document, arrays=source(tmp_path)
    selection=H01GlialSelection(dict(document,anchor_vertex=anchor))
    fragment=load_glial_fragment(path, selection)
    record=fragment.evidence
    assert record['selected_vertices']==[0,1,2,3,4]
    assert record['selected_edges']==[0,1,2,3]
    assert record['excluded_components']==[dict(anchor_vertex=5,vertices=2),dict(anchor_vertex=7,vertices=1)]
    seen=[]
    for view in fragment.morphology.branches:
        nodes=record['branch_paths'][view.name]['vertices']
        seen.extend(record['branch_paths'][view.name]['edges'])
        np.testing.assert_array_equal(view.branch.points_proximal.to_decimal(u.um),arrays['vertices_nm'][nodes[:-1]]/1000)
        np.testing.assert_array_equal(view.branch.points_distal.to_decimal(u.um),arrays['vertices_nm'][nodes[1:]]/1000)
        np.testing.assert_array_equal(view.branch.radii_proximal.to_decimal(u.um),arrays['radius_nm'][nodes[:-1]]/1000)
        np.testing.assert_array_equal(view.branch.radii_distal.to_decimal(u.um),arrays['radius_nm'][nodes[1:]]/1000)
    assert sorted(seen)==[0,1,2,3]
    assert load_glial_fragment(path, selection).evidence==record
    document['identity']='changed'
    detached=selection.to_dict()
    detached['identity']='also changed'
    assert selection.to_dict()['identity']=='synthetic-tree'
    record['selected_vertices'].clear()
    assert len(fragment.evidence['selected_vertices'])==5


@pytest.mark.parametrize('change', [dict(schema='unknown'),dict(coordinate_frame='um'),dict(source_sha256='bad'),
    dict(identity=''),dict(anchor_vertex=True),dict(max_vertices=1),dict(extra=1)])
def test_invalid_selection_declarations(tmp_path, change):
    _, document, _=source(tmp_path)
    with pytest.raises(ValueError):
        H01GlialSelection(dict(document,**change))


@pytest.mark.parametrize('change', [dict(edges=np.array([[0,1],[1,0]])),dict(edges=np.array([[0,1],[1,2],[2,0]])),
    dict(edges=np.array([[0,8]])),dict(edges=np.array([[0,0]])),dict(edges=np.array([[0.,1.]])),
    dict(radius_nm=np.zeros(8)),dict(radius_nm=np.full(8,np.nan)),dict(vertices_nm=np.zeros((8,3))),
    dict(vertices_nm=np.zeros((8,2))),dict(extra=np.ones(1))])
def test_malformed_source_geometry_is_rejected(tmp_path, change):
    path, document, _=source(tmp_path,**change)
    with pytest.raises(ValueError):
        load_glial_fragment(path,H01GlialSelection(document))


@pytest.mark.parametrize('change', [dict(anchor_vertex=100),dict(anchor_vertex=7),dict(max_vertices=4)])
def test_anchor_and_size_limit(tmp_path, change):
    path, document, _=source(tmp_path)
    with pytest.raises(ValueError):
        load_glial_fragment(path,H01GlialSelection(dict(document,**change)))


def test_source_tampering(tmp_path):
    path, document, _=source(tmp_path)
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='digest'):
        load_glial_fragment(path,H01GlialSelection(document))


def test_scalar_coordinates_are_rejected_as_invalid_geometry(tmp_path):
    path, document, _=source(tmp_path,vertices_nm=np.asarray(1.))
    with pytest.raises(ValueError,match='coordinates'):
        load_glial_fragment(path,H01GlialSelection(document))
