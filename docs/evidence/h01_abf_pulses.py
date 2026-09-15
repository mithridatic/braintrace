"""Explicit ABF command pulses and their recorded presynaptic crossings."""

import numpy as np


def command_from_epochs(epochs):
    """Reconstruct contiguous Step/Pulse epochs, including final partial periods.

    Parameters
    ----------
    epochs : object
        Source epoch waveform with p1s, p2s, types, levels, pulsePeriods and
        pulseWidths arrays as exposed by pyabf's EpochTable.

    Returns
    -------
    ndarray
        Complete reconstructed command on its original sample grid.
    """
    fields = [getattr(epochs, k) for k in ['p1s', 'p2s', 'types', 'levels', 'pulsePeriods', 'pulseWidths']]
    if not len(fields[0]) or any(len(f) != len(fields[0]) for f in fields):
        raise ValueError('Incomplete epoch fields.')
    starts, stops, types, levels, periods, widths = fields
    if starts[0] != 0 or types[0] != 'Step' or not np.isfinite(levels).all():
        raise ValueError('Require a finite initial holding Step at index zero.')
    end = stops[-1]
    if not isinstance(end, (int, np.integer)) or end <= 0:
        raise ValueError('Invalid command extent.')
    command = np.empty(end, dtype=float)
    previous_stop = 0
    previous_level = levels[0]
    for start, stop, kind, level, period, width in zip(*fields):
        if (not isinstance(start, (int, np.integer)) or not isinstance(stop, (int, np.integer))
                or start != previous_stop or stop <= start or stop > end):
            raise ValueError('Epochs must cover contiguous integer sample intervals.')
        if kind == 'Step':
            command[start:stop] = level
        elif kind == 'Pulse':
            if (not isinstance(period, (int, np.integer)) or not isinstance(width, (int, np.integer))
                    or period <= 0 or not 0 < width <= period):
                raise ValueError('Invalid pulse period or width.')
            phase = np.arange(stop-start) % period
            command[start:stop] = np.where(phase < width, level, previous_level)
        else:
            raise ValueError('Unsupported epoch type; only Step and Pulse are implemented.')
        previous_stop, previous_level = stop, level
    return command


def commanded_crossings(voltage_mv, command_pa):
    """Associate one upward voltage crossing with each positive command pulse.

    Parameters
    ----------
    voltage_mv, command_pa : array_like
        Original voltage and source-grid reconstructed current command.

    Returns
    -------
    tuple of ndarray
        Command-associated crossing indices and every unassigned crossing.
    """
    v, c = np.asarray(voltage_mv), np.asarray(command_pa)
    if (v.ndim != 1 or v.size < 2 or v.shape != c.shape
            or not np.isfinite(v).all() or not np.isfinite(c).all()):
        raise ValueError('Require complete finite voltage and command arrays.')
    positive = c > 0
    starts = np.flatnonzero(np.diff(np.r_[False, positive].astype(int)) == 1)
    stops = np.flatnonzero(np.diff(np.r_[positive, False].astype(int)) == -1)+1
    crosses = np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0))+1
    assigned = []
    if not len(starts):
        raise ValueError('No positive command pulses.')
    for start, stop in zip(starts, stops):
        within = crosses[(crosses >= start) & (crosses < stop)]
        if len(within) != 1:
            raise ValueError('A command pulse has missing or ambiguous voltage crossings.')
        assigned.append(within[0])
    assigned = np.asarray(assigned, dtype=int)
    return assigned, crosses[~np.isin(crosses, assigned)]
