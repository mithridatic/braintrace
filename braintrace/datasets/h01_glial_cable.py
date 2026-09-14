"""Explicit electrical construction of preserved glial source fragments."""

import hashlib
import json

import braincell
import brainunit as u
import numpy as np
from braincell.filter import AllRegion
from braincell.mech import Channel, Ion

from braintrace.biophysics.cable_kir import BraincellKir41
from braintrace.biophysics.cable_potassium import EnvironmentPotassium
from braintrace.biophysics.extracellular import nernst_mv
from .h01_anatomy import _geometry_signature


def build_glial_cable(fragment, settings):
    """Construct an uninitialized Kir-only cable with declared sealed ends.

    Parameters
    ----------
    fragment : H01GlialFragment
        Unmodified measured component and reconstruction evidence.
    settings : dict
        Explicit initial_mv, inside_k_mm, outside_k_mm, temperature_c,
        gkir_ms_cm2, cm_uf_cm2, ra_ohm_cm, max_cv_length_um and basis.

    Returns
    -------
    tuple
        BrainCell Cell and canonical geometry/electrical identity record.

    Notes
    -----
    The potassium ion uses declared fixed pools until explicitly bound to a
    shared environment. No uptake or calcium coupling is implied by building
    this cable. Fragment ends are sealed; no synthetic soma is introduced.
    """
    names = {'initial_mv', 'inside_k_mm', 'outside_k_mm', 'temperature_c',
             'gkir_ms_cm2', 'cm_uf_cm2', 'ra_ohm_cm', 'max_cv_length_um', 'basis'}
    if not isinstance(settings, dict) or set(settings) != names:
        raise ValueError('Explicit complete glial electrical settings required')
    if not isinstance(settings['basis'], str) or not settings['basis'].strip():
        raise ValueError('Electrical modeling basis required')
    numeric = names-{'basis'}
    if any(isinstance(settings[k], bool) or not isinstance(settings[k], (int, float)) or
           not np.isfinite(settings[k]) for k in numeric):
        raise ValueError('Finite numerical electrical settings required')
    s = {k: float(settings[k]) for k in numeric}
    if (any(s[k] <= 0 for k in ('inside_k_mm', 'outside_k_mm', 'cm_uf_cm2', 'ra_ohm_cm', 'max_cv_length_um')) or
            s['gkir_ms_cm2'] < 0 or s['temperature_c'] <= -273.15):
        raise ValueError('Electrical settings outside physical domain')
    evidence = fragment.evidence
    if _geometry_signature(fragment.morphology) != evidence['geometry_sha256']:
        raise ValueError('Glial source geometry changed after reconstruction')
    cell = braincell.Cell(fragment.morphology, V_init=s['initial_mv']*u.mV,
        pop_size=(1,), cv_policy=braincell.MaxCVLen(s['max_cv_length_um']*u.um))
    cell.paint(AllRegion(), braincell.CableProperty(resting_potential=s['initial_mv']*u.mV,
        membrane_capacitance=s['cm_uf_cm2']*u.uF/u.cm**2, axial_resistivity=s['ra_ohm_cm']*u.ohm*u.cm))
    reversal = nernst_mv(s['inside_k_mm'], s['outside_k_mm'], temperature_c=s['temperature_c'])
    cell.paint(AllRegion(), Ion('EnvironmentPotassium', name='potassium', E=reversal*u.mV,
                               Ci=s['inside_k_mm']*u.mM, Co=s['outside_k_mm']*u.mM))
    cell.paint(AllRegion(), Channel('BraincellKir41', g_max=s['gkir_ms_cm2']*u.mS/u.cm**2))
    record = dict(fragment=evidence, electrical=dict(s, basis=settings['basis']),
                  boundary_condition='sealed fragment ends', potassium='fixed declared pools until explicit environment binding')
    record['assembly_sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                                         allow_nan=False).encode()).hexdigest()
    return cell, record
