"""Preserve exact AP-command windows and matched human waveform responses."""

import re

import numpy as np


def command_window(voltage_mv, start_ms, end_ms):
    """Select an inclusive source window at the published 125 kHz clock.

    Parameters
    ----------
    voltage_mv : array_like
        Complete original one-dimensional voltage command in mV.
    start_ms, end_ms : float
        Absolute, sample-aligned window endpoints in ms.

    Returns
    -------
    dict
        Source indices and copies of sample times and unchanged voltage values.
    """
    v = np.asarray(voltage_mv, dtype=float)
    endpoints = np.asarray([start_ms,end_ms],dtype=float)*125
    if v.ndim!=1 or v.size==0 or not np.isfinite(v).all() or not np.isfinite(endpoints).all():
        raise ValueError('Nonfinite or invalid command input.')
    if np.any(np.abs(endpoints-np.round(endpoints))>1e-8):
        raise ValueError('Window is not aligned to source samples.')
    first,last=map(int,np.round(endpoints))
    if first<0 or last<first or last>=v.size:
        raise ValueError('Incomplete source window.')
    return dict(first_index=first,last_index=last,time_ms=np.arange(first,last+1)/125,
                voltage_mv=v[first:last+1].copy())


def human_hwide_records(records):
    """Select human sodium Hwide response sequences without replacing missing data.

    Parameters
    ----------
    records : iterable of dict
        Source sodium Fig. 2 workbook records with original header names.

    Returns
    -------
    tuple of list and list
        Selected sequences with identities and metadata, and explicit exclusions.
    """
    selected,excluded,seen=[],[],set()
    for r in records:
        name=r.get('Filename','')
        if not isinstance(name,str) or not re.fullmatch(r'[HM].+\.nwb',name) or name in seen:
            raise ValueError('Unknown or duplicated recording identity.')
        seen.add(name)
        if name.startswith('M'):
            excluded.append(dict(filename=name,reason='nonhuman'))
            continue
        fields={}
        for header,value in r.items():
            match=re.fullmatch(r'relative_amp_Hwide_\s*(\d+)',header)
            if match:
                number=int(match[1])
                if number in fields:
                    raise ValueError('Duplicate normalized response header.')
                fields[number]=value
        if set(fields)!=set(range(1,201)):
            raise ValueError('Incomplete Hwide response header set.')
        values=[fields[i] for i in range(1,201)]
        if any(v is not None and not np.isfinite(v) for v in values):
            raise ValueError('Nonfinite response value.')
        if all(v is None for v in values):
            excluded.append(dict(filename=name,reason='all Hwide responses missing'))
            continue
        selected.append(dict(filename=name,source_excel_row=r['excel_row'],ratios=values,
            metadata={key:r.get(key) for key in ('distance_pia','leakSubt','Rn')},
            waveform='Hwide',physiological_qc_applied=False))
    if not selected:
        raise ValueError('No human Hwide responses available.')
    return selected,excluded
