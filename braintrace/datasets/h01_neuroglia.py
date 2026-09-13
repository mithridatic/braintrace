"""Explicit immutable anatomy, pool and release contracts for H01 neuroglia."""

from dataclasses import dataclass, asdict
import hashlib
import json

import numpy as np

from .h01_biology import H01SpatialManifest
from .h01_glia import H01GlialSelection
from braintrace.biophysics.astrocyte import CalciumParameters
from braintrace.biophysics.gaba import GabaReleaseSites
from braintrace.biophysics.transport import DiffusionGraph


def _fields(value, names, label):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(label+' requires exact fields')


def _basis(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Explicit nonempty modeling basis required')


def _digest(value):
    return isinstance(value, str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def species_graph(extracellular, species, dt_ms):
    """Construct one explicitly declared extracellular transport graph.

    Parameters
    ----------
    extracellular : dict
        Validated accessible volumes, edges and species rates.
    species : str
        k, gaba or glutamate.
    dt_ms : float
        Cable physical interval in ms.

    Returns
    -------
    DiffusionGraph
        Physical graph; no inferred spatial links.
    """
    rates=extracellular[species]
    return DiffusionGraph(extracellular['volumes_um3'],np.asarray(extracellular['edges'],dtype=int).reshape(-1,2),
        rates['conductance_um3_ms'],dt_ms=dt_ms,boundary_conductance=rates['boundary_um3_ms'],
        uptake_per_ms=rates['uptake_per_ms'])


@dataclass(frozen=True, init=False)
class H01NeuroglialManifest:
    """Pin one glial fragment and complete neuronal chemical mappings.

    Parameters
    ----------
    document : dict
        h01-biology-neuroglia-v1: spines, glia, extracellular, membranes,
        releases, potassium, tonic_gaba, calcium, dt_ms and basis.
    topology : dict
        Exact active neuronal topology. Membranes cover all active identities
        and the reserved @glia identity, with explicit geometry SHA256 values.
    """

    _json: str

    def __init__(self, document, topology):
        _fields(document,'schema spines glia extracellular membranes releases potassium tonic_gaba calcium dt_ms basis','Neuroglial manifest')
        if document['schema']!='h01-biology-neuroglia-v1':
            raise ValueError('Unsupported neuroglial schema')
        _basis(document['basis'])
        H01SpatialManifest(document['spines'],topology)
        _fields(document['glia'],'selection electrical','Glial source')
        H01GlialSelection(document['glia']['selection'])
        ex=document['extracellular']
        _fields(ex,'centers_um volumes_um3 edges origin basis k gaba glutamate','Extracellular domain')
        _basis(ex['basis'])
        if ex['origin'] not in ('measured','published_donor','synthetic'):
            raise ValueError('Extracellular origin must be explicit')
        n=len(ex['volumes_um3'])
        centers=np.asarray(ex['centers_um'],dtype=float)
        edges=np.asarray(ex['edges'])
        if centers.shape!=(n,3) or not np.isfinite(centers).all():
            raise ValueError('Extracellular centers must match volumes')
        if edges.size and (edges.ndim!=2 or edges.shape[1]!=2 or not np.issubdtype(edges.dtype,np.integer)):
            raise ValueError('Extracellular edges require integer pairs')
        for species in ('k','gaba','glutamate'):
            _fields(ex[species],'conductance_um3_ms boundary_um3_ms uptake_per_ms','Species transport')
            species_graph(ex,species,document['dt_ms'])
        if np.any(np.asarray(ex['k']['uptake_per_ms'])!=0):
            raise ValueError('K uptake must have an intracellular receiving pool')
        active=topology['active_cells']
        membranes=document['membranes']
        if '@glia' in active or not isinstance(membranes,dict) or set(membranes)!=set(active)|{'@glia'}:
            raise ValueError('Membrane maps must cover every neuron and the glial fragment')
        for row in membranes.values():
            _fields(row,'geometry_sha256 outside_indices','Membrane map')
            ids=np.asarray(row['outside_indices'])
            if (not _digest(row['geometry_sha256']) or ids.ndim!=1 or not len(ids) or
                    not np.issubdtype(ids.dtype,np.integer) or np.any(ids<0) or np.any(ids>=n)):
                raise ValueError('Invalid membrane geometry digest or volume indices')
        _fields(document['releases'],'gaba glutamate','Release maps')
        for species,polarity in [('gaba','I'),('glutamate','E')]:
            row=document['releases'][species]
            if row is None:
                continue
            _fields(row,'sources volume_indices active_sites molecules_per_site seed basis','Release sites')
            _basis(row['basis'])
            sources=row['sources']
            if (not isinstance(sources,list) or any(not isinstance(x,str) or x not in active for x in sources) or
                    len(sources)!=len(set(sources)) or type(row['seed']) is not int):
                raise ValueError('Release sources must be distinct active identities with an integer seed')
            if any(topology['sources'][topology['instances'][x]['source_id']]['polarity']!=polarity for x in sources):
                raise ValueError('Release source polarity differs from neuronal topology')
            release=GabaReleaseSites(row['volume_indices'],n,active_sites=row['active_sites'],
                molecules_per_site=row['molecules_per_site'],seed=row['seed'])
            if release.volume_indices.shape[0]!=len(sources):
                raise ValueError('Release rows must match source identities')
        _fields(document['potassium'],'inside_mm outside_mm temperature_c','Potassium settings')
        k=document['potassium']
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) for v in k.values()) or min(k['inside_mm'],k['outside_mm'])<=0 or k['temperature_c']<=-273.15:
            raise ValueError('Invalid initial potassium pools or temperature')
        _fields(document['tonic_gaba'],'g_max_ms_cm2 reversal_mv ec50_mm hill','Tonic GABA settings')
        tonic=document['tonic_gaba']
        if (any(isinstance(v,bool) or not isinstance(v,(float,int)) or not np.isfinite(v) for v in tonic.values()) or
                tonic['g_max_ms_cm2']<0 or tonic['ec50_mm']<=0 or tonic['hill']<=0):
            raise ValueError('Invalid tonic GABA settings')
        _fields(document['calcium'],'parameters substeps','Calcium settings')
        _fields(document['calcium']['parameters'],' '.join(asdict(CalciumParameters())),'Calcium parameters')
        CalciumParameters(**document['calcium']['parameters'])
        if type(document['calcium']['substeps']) is not int or document['calcium']['substeps']<1:
            raise ValueError('Calcium substeps must be a positive integer')
        object.__setattr__(self,'_json',json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False))

    def to_dict(self):
        """Return a detached canonical manifest document."""
        return json.loads(self._json)

    @property
    def sha256(self):
        """Return the complete immutable biology identity."""
        return hashlib.sha256(self._json.encode()).hexdigest()
