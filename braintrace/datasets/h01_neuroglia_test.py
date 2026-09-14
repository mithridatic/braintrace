"""Strict, source-pinned extracellular and release declarations."""

import copy
import pytest

from examples.pp_prop.h01_neuroglial_test import configured, imported, precision
from .h01_neuroglia import H01NeuroglialManifest


@pytest.mark.parametrize('fault',['unknown','schema','basis','polarity','centers','edges','source_rows','source_id',
    'membrane','digest','volume_index','uptake','clock','tonic','calcium','potassium','origin'])
def test_manifest_rejects_ambiguous_or_invalid_placement(imported,tmp_path,fault):
    topology,_,biology,_=configured(imported,tmp_path)
    if fault=='unknown': biology['extra']=1
    elif fault=='schema': biology['schema']='unsupported'
    elif fault=='basis': biology['basis']=''
    elif fault=='polarity': biology['releases']['gaba']=biology['releases']['glutamate']
    elif fault=='centers': biology['extracellular']['centers_um']=[]
    elif fault=='edges': biology['extracellular']['edges']=[[0.1,1.]]
    elif fault=='source_rows': biology['releases']['glutamate']['sources']=[]
    elif fault=='source_id': biology['releases']['glutamate']['sources']=['missing']
    elif fault=='membrane': biology['membranes'].pop('@glia')
    elif fault=='digest': biology['membranes']['12']['geometry_sha256']='invalid'
    elif fault=='volume_index': biology['membranes']['12']['outside_indices']=[999]
    elif fault=='uptake': biology['extracellular']['k']['uptake_per_ms']=[1.,0.]
    elif fault=='clock': biology['dt_ms']=-1.
    elif fault=='tonic': biology['tonic_gaba']['hill']=0.
    elif fault=='calcium': biology['calcium']['substeps']=0
    elif fault=='potassium': biology['potassium']['inside_mm']=0.
    else: biology['extracellular']['origin']='inferred'
    with pytest.raises(ValueError):
        H01NeuroglialManifest(biology,topology.to_dict())


def test_canonical_identity_covers_rates_placements_and_is_detached(imported,tmp_path):
    topology,_,biology,_=configured(imported,tmp_path)
    manifest=H01NeuroglialManifest(biology,topology.to_dict())
    altered=manifest.to_dict()
    altered['releases']['glutamate']['molecules_per_site']=1234.
    assert H01NeuroglialManifest(altered,topology.to_dict()).sha256!=manifest.sha256
    assert manifest.to_dict()==biology
    altered=copy.deepcopy(biology)
    altered['extracellular']['centers_um'][0][0]=2.
    assert H01NeuroglialManifest(altered,topology.to_dict()).sha256!=manifest.sha256
