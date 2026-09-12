"""Topology, source, provenance and contact identity guards for spine manifests."""

import copy
from dataclasses import asdict, FrozenInstanceError

import pytest

from braintrace.biophysics.spines import Spine
from .h01_biology import H01SpatialManifest, topology_digest


def documents():
    topology = dict(active_cells=['a', 'b'], active_contacts=['c'],
        instances={'a': {'source_id': 's'}, 'b': {'source_id': 's'}},
        sources={'s': {'source_sha256': 'a'*64}}, contacts={'c': {'post': 'b', 'post_site': [1, .5]}})
    spine = asdict(Spine('head', 1, .5, .689, .0335, .583, .2915, (0., 1., 0.),
                         'Synthetic fixture using declared paper dimensions'))
    spine['origin'] = 'synthetic'
    document = dict(schema='h01-biology-spines-v1', topology_sha256=topology_digest(topology),
        coordinate_frame='h01-swc-um', cells={'a': dict(source_sha256='a'*64, spines=[]),
        'b': dict(source_sha256='a'*64, spines=[spine])}, contact_heads={'c': 'head'},
        release_probability={'c': .36})
    return document, topology


def test_manifest_freezes_nested_geometry_and_preserves_contact_identity():
    document, topology = documents()
    model = H01SpatialManifest(document, topology)
    identity = model.sha256
    document['cells']['b']['spines'][0]['neck_length_um'] = 9.
    model.to_dict()['cells']['b']['spines'].clear()
    assert model.spines('b')[0].neck_length_um == .689
    assert model.spines('b')[0].direction == (0., 1., 0.)
    assert model.sha256 == identity
    assert not model.spines('a')
    assert H01SpatialManifest(model.to_dict(), topology).sha256 == identity
    with pytest.raises(FrozenInstanceError):
        model._json = '{}'
    reordered = copy.deepcopy(topology)
    reordered['active_cells'].reverse()
    with pytest.raises(ValueError, match='topology digest'):
        H01SpatialManifest(model.to_dict(), reordered)


@pytest.mark.parametrize('change', [
    lambda d: d.update(unrecognized=True), lambda d: d.update(schema='unknown'),
    lambda d: d.update(coordinate_frame='c3-nm'), lambda d: d.update(topology_sha256='b'*64),
    lambda d: d['cells'].pop('a'), lambda d: d['contact_heads'].update(missing='head'),
    lambda d: d['release_probability'].clear(), lambda d: d['release_probability'].update(c=True),
    lambda d: d['release_probability'].update(c=float('nan')), lambda d: d['release_probability'].update(c=1.1),
    lambda d: d['cells']['b'].update(extra=True), lambda d: d['cells']['b'].update(source_sha256='b'*64),
    lambda d: d['cells']['b'].update(spines={}),
    lambda d: d['cells']['b']['spines'][0].pop('provenance'),
    lambda d: d['cells']['b']['spines'][0].update(origin='guessed'),
    lambda d: d['cells']['b']['spines'][0].update(neck_length_um=-1),
    lambda d: d['cells']['b']['spines'].append(copy.deepcopy(d['cells']['b']['spines'][0])),
    lambda d: d['contact_heads'].update(c='unknown'),
    lambda d: d['cells']['b']['spines'][0].update(parent_x=.6),
])
def test_invalid_or_ambiguous_manifest_rejected(change):
    document, topology = documents()
    change(document)
    with pytest.raises(ValueError):
        H01SpatialManifest(document, topology)


def test_contact_cannot_be_duplicated_on_one_spine():
    document, topology = documents()
    topology['active_contacts'].append('second')
    topology['contacts']['second'] = copy.deepcopy(topology['contacts']['c'])
    document['topology_sha256'] = topology_digest(topology)
    document['release_probability']['second'] = .36
    document['contact_heads']['second'] = 'head'
    with pytest.raises(ValueError, match='Multiple contacts'):
        H01SpatialManifest(document, topology)
