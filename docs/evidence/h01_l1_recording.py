"""Prepare calibrated-unit L1 recordings while keeping validation responses sealed."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

SOURCE_SHA256 = '004f27f306180bd610ba2439e600f688c11be24adf5cea2a1663a22ba0336af6'
CALIBRATION = frozenset((*range(4, 13), 14, 16))
HOLDOUT = frozenset((13, 15, *range(17, 35)))


def notebook_value(keys, values, name, sweep, *, scope):
    """Return the last finite numeric entry in one explicit notebook scope.

    Parameters
    ----------
    keys : sequence of str
        Numeric field names, including SweepNum.
    values : numpy.ndarray
        Time-ordered records, fields, and nine headstage/global scopes.
    name : str
        Requested source field.
    sweep : int
        Original sweep identifier.
    scope : int
        Headstage 0-7 or global scope 8. No automatic scope substitution.

    Returns
    -------
    float or None
        Last finite value, or None when no such observation exists.
    """
    if values.ndim != 3 or values.shape[1:] != (len(keys), 9) or scope not in range(9):
        raise ValueError('Invalid notebook shape or explicit scope.')
    if len(set(keys)) != len(keys) or 'SweepNum' not in keys:
        raise ValueError('Notebook keys are ambiguous or lack SweepNum.')
    if name not in keys:
        return None
    sweeps = values[:, keys.index('SweepNum'), scope]
    observed = values[sweeps == sweep, keys.index(name), scope]
    finite = observed[np.isfinite(observed)]
    return None if not len(finite) else float(finite[-1])


def holding_current_pa(keys, values, sweep):
    """Resolve the explicitly enabled amplifier holding current.

    Parameters
    ----------
    keys : sequence of str
        Notebook keys with holding fields in pA.
    values : numpy.ndarray
        Numeric notebook observations.
    sweep : int
        Original sweep identifier, recorded on headstage 0.

    Returns
    -------
    float
        Holding current in pA; absent or inconsistent enable/value data raises.
    """
    enable = notebook_value(keys, values, 'I-Clamp Holding Enable', sweep, scope=0)
    if enable == 0:
        return 0.
    current = notebook_value(keys, values, 'I-Clamp Holding Level', sweep, scope=0)
    if enable != 1 or current is None:
        raise ValueError('Holding current is unavailable; it cannot default to zero.')
    return current


def read_calibration(path, sweep):
    """Read all samples of a registered calibration sweep with source checks.

    Parameters
    ----------
    path : pathlib.Path
        Pinned human NWB recording.
    sweep : int
        A calibration identifier. Validation identifiers fail before file access.

    Returns
    -------
    tuple of dict
        Direct arrays and source/instrument metadata. Voltage remains on the
        recorded reference; no unverified junction or bridge correction is applied.
    """
    if sweep not in CALIBRATION:
        raise ValueError('Response export is restricted to registered calibration sweeps.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError('NWB source SHA256 mismatch.')
    with h5py.File(path, 'r') as source:
        response = source['acquisition'][f'data_{sweep:05d}_AD0']
        stimulus = source['stimulus/presentation'][f'data_{sweep:05d}_DA0']
        if (response.attrs['neurodata_type'] != 'CurrentClampSeries'
                or stimulus.attrs['neurodata_type'] != 'CurrentClampStimulusSeries'
                or response.attrs['sweep_number'] != sweep
                or stimulus.attrs['sweep_number'] != sweep):
            raise ValueError('Clamp mode or sweep identity mismatch.')
        converted, clocks = [], []
        for group, unit, scale in ((response, 'volts', 1e3), (stimulus, 'amperes', 1e12)):
            data, start = group['data'], group['starting_time']
            if data.attrs.get('unit') != unit or data.ndim != 1:
                raise ValueError('Unexpected sample units or shape.')
            samples = (np.asarray(data[()], dtype=np.float64) * float(data.attrs['conversion'])
                       + float(data.attrs.get('offset', 0.))) * scale
            clock = (float(np.asarray(start[()]).item()), float(start.attrs['rate']))
            if (len(samples) < 2 or not np.isfinite(samples).all()
                    or not np.isfinite(clock).all() or clock[1] <= 0):
                raise ValueError('Recording samples or clock are invalid.')
            converted.append(samples)
            clocks.append(clock)
        if len(converted[0]) != len(converted[1]) or clocks[0] != clocks[1]:
            raise ValueError('Response and command clocks do not match.')
        # MIES trailing storage zeros are missing observations (IPFX convention).
        # Detect in stored values, before an SI offset could hide the padding.
        nonzero = np.flatnonzero(response['data'][()])
        if not len(nonzero) or nonzero[-1] < 1:
            raise ValueError('Insufficient recorded response before trailing padding.')
        last_recorded = int(nonzero[-1])
        recorded_sample = np.arange(len(converted[0])) <= last_recorded
        notebook = source['general/labnotebook/ITC18USB_Dev_0']
        keys = notebook['numericalKeys'].asstr()[0].tolist()
        units = notebook['numericalKeys'].asstr()[1].tolist()
        values = notebook['numericalValues'][()]
        if units[keys.index('I-Clamp Holding Level')] != 'pA':
            raise ValueError('Holding current notebook units are not pA.')
        holding = holding_current_pa(keys, values, sweep)
        online_qc = {name: notebook_value(keys, values, name, sweep, scope=8)
                     for name in keys if name.endswith('QC')}
        instrument = {name: dict(value=notebook_value(keys, values, name, sweep, scope=0),
                                unit=units[keys.index(name)])
                      for name in ('Bridge Bal Enable', 'Bridge Bal Value',
                                   'Async AD 1: Bath Temperature') if name in keys}
        metadata = dict(source_sha256=digest, specimen_id='811953283', session_id='811953264',
                        sweep=sweep, role='calibration', sample_rate_hz=clocks[0][1],
                        absolute_start_s=clocks[0][0], samples=len(converted[0]),
                        last_recorded_index=last_recorded,
                        last_recorded_time_ms=last_recorded / clocks[0][1] * 1000,
                        trailing_padding_samples=int((~recorded_sample).sum()),
                        stimulus_description=str(response.attrs['stimulus_description']),
                        holding_current_pa=holding, instrument=instrument,
                        online_global_qc=online_qc, offline_sweep_qc='not evaluated here',
                        voltage_reference='recorded; junction correction not applied',
                        bridge_correction='not reapplied', physiological_qualification=False)
        arrays = dict(time_ms=np.arange(len(converted[0])) / clocks[0][1] * 1000,
                      source_voltage_mv=converted[0], recorded_sample=recorded_sample,
                      recorded_voltage_mv=np.where(recorded_sample, converted[0], np.nan),
                      command_current_pa=converted[1],
                      holding_current_pa=np.float64(holding))
    return arrays, metadata


def main(argv=None):
    """Export a new calibration artifact without overwriting prior evidence.

    Parameters
    ----------
    argv : list of str or None, optional
        CLI arguments specifying NWB path, calibration sweep and output prefix.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb', type=Path, required=True)
    parser.add_argument('--sweep', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if any(args.output.with_suffix(s).exists() for s in ('.npz', '.json')):
        raise FileExistsError('Use a new evidence prefix.')
    arrays, metadata = read_calibration(args.nwb, args.sweep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    metadata['arrays_sha256'] = hashlib.sha256(args.output.with_suffix('.npz').read_bytes()).hexdigest()
    with args.output.with_suffix('.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
