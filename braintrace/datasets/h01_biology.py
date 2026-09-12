"""Immutable, topology-pinned declarations for explicit H01 spine anatomy."""

from dataclasses import dataclass, fields
import hashlib
import json

import numpy as np

from braintrace.biophysics.spines import Spine


def topology_digest(document):
    """Hash the complete canonical topology document.

    Parameters
    ----------
    document : dict
        H01 topology, including ordered active contact and instance identities.

    Returns
    -------
    str
        SHA256 of canonical finite JSON.
    """
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, init=False)
class H01SpatialManifest:
    """Validate and freeze explicit additions against an immutable topology.

    Parameters
    ----------
    document : dict
        ``h01-biology-spines-v1`` with topology_sha256, coordinate_frame, cells,
        contact_heads and release_probability. Cells contain source_sha256 and
        spines; each spine has all Spine fields plus explicit origin.
    topology : dict
        The exact active source topology. Inactive additions are rejected.

    Notes
    -----
    Origin is a provenance declaration, not independent proof of measurement.
    Only source-coordinate attachments are supported; external glial frames
    and unverified transformations are not accepted by this schema.
    """

    _json: str

    def __init__(self, document, topology):
        required = {'schema', 'topology_sha256', 'coordinate_frame', 'cells',
                    'contact_heads', 'release_probability'}
        if not isinstance(document, dict) or set(document) != required:
            raise ValueError('Spatial biology requires the exact manifest fields')
        if document['schema'] != 'h01-biology-spines-v1' or document['coordinate_frame'] != 'h01-swc-um':
            raise ValueError('Unsupported spatial biology schema or coordinate frame')
        if document['topology_sha256'] != topology_digest(topology):
            raise ValueError('Spatial biology topology digest differs')
        cells, contacts = document['cells'], document['contact_heads']
        probabilities = document['release_probability']
        if not isinstance(cells, dict) or set(cells) != set(topology['active_cells']):
            raise ValueError('Spatial cells must cover active instances exactly')
        if not isinstance(contacts, dict) or set(contacts)-set(topology['active_contacts']):
            raise ValueError('Spine targets must name active contacts')
        if not isinstance(probabilities, dict) or set(probabilities) != set(topology['active_contacts']):
            raise ValueError('Release probabilities must cover active contacts exactly')
        for value in probabilities.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Release probabilities must be finite numbers in [0, 1]')
        spine_fields = {field.name for field in fields(Spine)}
        indexed = {}
        for identity, cell in cells.items():
            source = topology['sources'][topology['instances'][identity]['source_id']]
            if not isinstance(cell, dict) or set(cell) != {'source_sha256', 'spines'}:
                raise ValueError('Spatial cell requires source digest and spines')
            if cell['source_sha256'] != source['source_sha256']:
                raise ValueError('Spatial cell source digest differs')
            if not isinstance(cell['spines'], list):
                raise ValueError('Spines must be an explicit list')
            indexed[identity] = {}
            for row in cell['spines']:
                if not isinstance(row, dict) or set(row) != spine_fields | {'origin'}:
                    raise ValueError('Spine requires all geometry and provenance fields')
                if row['origin'] not in ('measured', 'published_donor', 'synthetic'):
                    raise ValueError('Spine origin must be explicit')
                spine = Spine(**{key: row[key] for key in spine_fields})
                if spine.identity in indexed[identity]:
                    raise ValueError('Duplicate spine identity within instance')
                indexed[identity][spine.identity] = spine
        occupied = set()
        for contact, head in contacts.items():
            edge = topology['contacts'][contact]
            if not isinstance(head, str) or head not in indexed[edge['post']]:
                raise ValueError('Contact head must belong to its postsynaptic instance')
            spine = indexed[edge['post']][head]
            if edge['post_site'][0] != spine.parent_branch or not np.isclose(
                    edge['post_site'][1], spine.parent_x, rtol=0, atol=1e-9):
                raise ValueError('Contact source site must match the spine attachment')
            target = (edge['post'], head)
            if target in occupied:
                raise ValueError('Multiple contacts on a spine require a separately supported schema')
            occupied.add(target)
        object.__setattr__(self, '_json', json.dumps(document, sort_keys=True,
                                                   separators=(',', ':'), allow_nan=False))

    def to_dict(self):
        """Return a detached manifest document.

        Returns
        -------
        dict
            Mutating this copy cannot alter the immutable manifest.
        """
        return json.loads(self._json)

    @property
    def sha256(self):
        """Return the canonical manifest SHA256 identity."""
        return hashlib.sha256(self._json.encode()).hexdigest()

    def spines(self, identity):
        """Return fixed spine geometries for one active instance.

        Parameters
        ----------
        identity : str
            Active instance identifier.

        Returns
        -------
        tuple of Spine
            Immutable geometry; provenance origin remains in the manifest.
        """
        return tuple(Spine(**{key: tuple(value) if key == 'direction' else value
                              for key, value in row.items() if key != 'origin'})
                     for row in self.to_dict()['cells'][identity]['spines'])
