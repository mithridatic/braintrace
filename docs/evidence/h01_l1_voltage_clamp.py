"""Preserve human L1 nucleated-patch current and reconstruct its voltage command."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from docs.evidence.h01_l1_recording import notebook_value


SOURCE_SHA256 = '367240a152f4304707fc849391109da76898cf29405ac66a6a5b6541f559dc54'
SESSION = '923103553'
SPECIMEN = '923103580'
FAMILIES = {
    **dict.fromkeys(range(98, 107), 'NucVCLS0_DA_0'),
    **dict.fromkeys(range(107, 117), 'NucVCLS1Leak_DA_0'),
    **dict.fromkeys(range(117, 126), 'NucVCprepulse_DA_0'),
    126: 'NucVCQCLS_DA_0',
    **dict.fromkeys(range(127, 132), 'NucVCstails_DA_0'),
    **dict.fromkeys(range(132, 141), 'NucVCSus0_DA_0'),
    **dict.fromkeys(range(141, 151), 'NucVCSus1Leak_DA_0'),
    **dict.fromkeys(range(151, 161), 'NucVCzrecovms5_DA_0'),
    **dict.fromkeys(range(161, 166), 'NucVCzrecovms100_DA_0'),
    **dict.fromkeys(range(166, 170), 'NucVCzrecovs1_DA_0'),
}
PAX6_SESSION = '840043481'
PAX6_SHA256 = '30abbb3cca63b0629242c7ad96f595d6e6ceea2ce7e522fdb5ebf8a3bc2ac944'
# Confirmed against every family in the hash-bound PAX6 acquisition inventory.
PAX6_FAMILIES = {sweep - 28: family for sweep, family in FAMILIES.items()}


def command_segments(voltage, rate):
    """Describe every constant command segment without rounding transitions.

    Parameters
    ----------
    voltage : numpy.ndarray
        Finite one-dimensional command in mV.
    rate : float
        Positive sampling frequency in Hz.

    Returns
    -------
    list of dict
        Half-open sample bounds, times and command level for each segment.
    """
    voltage = np.asarray(voltage)
    if (voltage.ndim != 1 or not voltage.size or not np.isfinite(voltage).all()
            or not np.isfinite(rate) or rate <= 0):
        raise ValueError('Invalid command samples or rate.')
    edges = np.r_[0, np.flatnonzero(np.diff(voltage) != 0) + 1, len(voltage)]
    return [dict(start_index=int(a), stop_index=int(b), start_ms=float(a / rate * 1000),
                 stop_ms=float(b / rate * 1000), command_mv=float(voltage[a]))
            for a, b in zip(edges[:-1], edges[1:])]


def read_sweep(path, sweep, *, session=SESSION):
    """Read a pinned human channel-discovery sweep and its instrument settings.

    Parameters
    ----------
    path : pathlib.Path
        Original human NWB file for the registered session.
    sweep : int
        Registered nucleated-patch sweep; other responses fail before file access.
    session : str, optional
        One of the two pinned discovery/calibration sessions. External validation
        sessions are rejected before opening response data.

    Returns
    -------
    tuple of dict
        Unfiltered arrays and provenance. Current is total amplifier current;
        command voltage is not measured patch voltage. No channel QC is implied.
    """
    if session == SESSION:
        expected_hash, specimen, families = SOURCE_SHA256, SPECIMEN, FAMILIES
    elif session == PAX6_SESSION:
        expected_hash, specimen, families = PAX6_SHA256, '840043506', PAX6_FAMILIES
    else:
        raise ValueError('Session is not registered for discovery/calibration access.')
    if sweep not in families:
        raise ValueError('Only registered nucleated-patch sweeps may be exported.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != expected_hash:
        raise ValueError('NWB source SHA256 mismatch.')
    with h5py.File(path, 'r') as source:
        groups = [source['acquisition'][f'data_{sweep:05d}_AD0'],
                  source['stimulus/presentation'][f'data_{sweep:05d}_DA0']]
        converted, clocks, conversions = [], [], []
        for group, kind, unit, scale in zip(
                groups, ('VoltageClampSeries', 'VoltageClampStimulusSeries'),
                ('amperes', 'volts'), (1e12, 1e3)):
            if (group.attrs['neurodata_type'] != kind
                    or group.attrs['sweep_number'] != sweep
                    or group.attrs['stimulus_description'] != families[sweep]):
                raise ValueError('Clamp mode, sweep identity or protocol family mismatch.')
            data, start = group['data'], group['starting_time']
            if data.ndim != 1 or data.attrs.get('unit') != unit:
                raise ValueError('Unexpected sample units or shape.')
            factor, offset = float(data.attrs['conversion']), float(data.attrs.get('offset', 0))
            values = (np.asarray(data[()], dtype=np.float64) * factor + offset) * scale
            clock = (float(np.asarray(start[()]).item()), float(start.attrs['rate']))
            if (len(values) < 2 or not np.isfinite(values).all()
                    or not np.isfinite(clock).all() or clock[1] <= 0):
                raise ValueError('Invalid recording samples or clock.')
            converted.append(values)
            clocks.append(clock)
            conversions.append(dict(stored_unit=unit, conversion=factor, offset=offset))
        if clocks[0] != clocks[1] or converted[0].shape != converted[1].shape:
            raise ValueError('Response and command clocks or lengths differ.')
        if groups[0]['data'][-1] == 0:
            raise ValueError('Terminal zero current has ambiguous coverage; preserve source for review.')
        nb = source['general/labnotebook/ITC18USB_Dev_0']
        keys = nb['numericalKeys'].asstr()[0].tolist()
        units = nb['numericalKeys'].asstr()[1].tolist()
        values = nb['numericalValues'][()]
        enable = notebook_value(keys, values, 'V-Clamp Holding Enable', sweep, scope=0)
        if enable == 0:
            holding = 0.
        elif enable == 1:
            holding = notebook_value(keys, values, 'V-Clamp Holding Level', sweep, scope=0)
            if holding is None or units[keys.index('V-Clamp Holding Level')] != 'mV':
                raise ValueError('Enabled holding voltage is missing or has wrong units.')
        else:
            raise ValueError('Holding voltage enable state is unavailable.')
        instrument = {key: dict(value=notebook_value(keys, values, key, sweep, scope=0),
                                unit=units[keys.index(key)])
                      for key in ('RsComp Enable', 'RsComp Correction', 'Series Resistance',
                                  'TP Steady State Resistance', 'Fast compensation capacitance',
                                  'Slow compensation capacitance', 'LPF Cutoff',
                                  'Secondary LPF Cutoff', 'Hardware Type',
                                  'Scaled Out Signal', 'Scale Factor Units') if key in keys}
        current, dac = converted
        command = dac + holding
        arrays = dict(time_ms=np.arange(len(current)) / clocks[0][1] * 1000,
                      total_current_pa=current, dac_voltage_mv=dac,
                      holding_voltage_mv=np.float64(holding), command_voltage_mv=command)
        metadata = dict(source_sha256=digest, session_id=session, specimen_id=specimen,
                        sweep=sweep, family=families[sweep], role='channel discovery',
                        absolute_start_s=clocks[0][0], sample_rate_hz=clocks[0][1],
                        samples=len(current), conversions=conversions, instrument=instrument,
                        command_segments=command_segments(command, clocks[0][1]),
                        junction_correction_applied=False, leak_subtraction_applied=False,
                        actual_patch_voltage_measured=False, patch_area_um2=None,
                        channel_kinetics_qualified=False, coverage_qc='pending independent assessment')
    return arrays, metadata


def main(argv=None):
    """Export a new source-bound artifact while preserving earlier evidence.

    Parameters
    ----------
    argv : list of str or None, optional
        NWB path, registered sweep and a new output prefix.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb', type=Path, required=True)
    parser.add_argument('--sweep', type=int, required=True)
    parser.add_argument('--session', default=SESSION)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(args.output.with_suffix(s).exists() for s in ('.npz', '.json')):
        raise FileExistsError('Use a new evidence prefix.')
    arrays, metadata = read_sweep(args.nwb, args.sweep, session=args.session)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f, **arrays)
    metadata['arrays_sha256'] = hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest()
    with args.output.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(sweep=args.sweep, samples=metadata['samples'], output=str(args.output))))


if __name__ == '__main__':
    main()
