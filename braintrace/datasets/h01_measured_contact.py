"""Pinned H01 inhibitory contact with checked endpoint identity and placement."""
from copy import deepcopy
import numpy as np
import brainunit as u
from braincell.filter import AtLocation

_CONTACT = {'annotation_id': '8105899', 'annotation_url': 'https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/precomputed', 'annotation_by_id_payload_hex': 'c0eb7f48c0e14d4800d8864580e97f4800e04d4800d886450100000001000000c3b055e11300000001000000c64ec87e09000000', 'annotation_payload_sha256': '89e775f13c4bf14d6011de50de4c83b10c27f4c745e7671a435c02acc49e6aef', 'type': 1, 'type_meaning': 'inhibitory', 'type_source': 'https://github.com/ashapsoncoe/h01/blob/main/get_neuron_e_to_i_ratios.py', 'pre': {'proofread_id': '5584343344', 'c3_id': '85384868035', 'component': 0, 'source_swc_sha256': '53de2bec08c12096f742ecc01fca1293f24f3a09637558f9bf86ac4edb6a4f92', 'voxel': [262063, 210823, 4315], 'cable_location': [974, 0.5754461212601991], 'projection_distance_um': 0.1556592538234353}, 'post': {'proofread_id': '4157825456', 'c3_id': '40781762246', 'component': 0, 'source_swc_sha256': 'd49c452524cf7c35ded56a1ef55ebf205e00573ea11b2a9204060362777d40ce', 'voxel': [262054, 210816, 4315], 'cable_location': [2805, 0.9324765948060457], 'projection_distance_um': 0.13310219168119006}, 'voxel_size_nm': [8, 8, 33], 'identity_evidence': 'h01-ie-contact-voxel-audit.json', 'source_synapse_verification': 'Released detection; individual manual review not established.', 'electrophysiology': 'No conductance, delay, or receptor kinetics measured for this contact.', 'qualification': 'Anatomical endpoint identity and connected-component cable placement checked. Circuit implementation and delivery validation pending.'}


def measured_ie_contact(components):
    """Validate source components and return the measured contact placements.

    Parameters
    ----------
    components : dict
        E and I imported components from the pinned H01 morphology archive.

    Returns
    -------
    tuple
        Presynaptic location, postsynaptic location, and independent provenance.

    Notes
    -----
    Synapse detection and anatomical placement do not measure conductance,
    transmission delay, or receptor kinetics. Individual manual review of this
    synapse is not established.
    """
    locations = []
    for role, side in (("I", "pre"), ("E", "post")):
        expected = _CONTACT[side]
        cell = components[role]
        if (cell.neuron_id != expected['proofread_id'] or
                cell.component_id != expected['component'] or
                cell.source_sha256 != expected['source_swc_sha256']):
            raise ValueError("Measured H01 contact requires its pinned E/I source components.")
        branch_id, fraction = expected['cable_location']
        branch = cell.morphology.branches[branch_id].branch
        lengths = np.asarray(branch.lengths.to_decimal(u.um))
        bounds = np.r_[0., np.cumsum(lengths)] / lengths.sum()
        index = min(np.searchsorted(bounds, fraction, side='right')-1, len(lengths)-1)
        t = (fraction-bounds[index])/(bounds[index+1]-bounds[index])
        a = np.asarray(branch.points_proximal.to_decimal(u.um))[index]
        b = np.asarray(branch.points_distal.to_decimal(u.um))[index]
        point = a+t*(b-a)
        source = np.asarray(expected['voxel'])*np.asarray(_CONTACT['voxel_size_nm'])/1000
        distance = float(np.linalg.norm(point-source))
        if not np.isclose(distance, expected['projection_distance_um'], rtol=0., atol=1e-6):
            raise ValueError("Measured H01 contact placement no longer matches the source geometry.")
        locations.append(AtLocation(branch_id, fraction))
    evidence = deepcopy(_CONTACT)
    evidence['qualification'] = 'Source endpoint identity and cable placement checked; physiology and event delivery require validation.'
    return *locations, evidence
