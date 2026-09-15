"""Preserve source samples around presynaptic spikes in paired recordings."""

import numpy as np


def electrode_index(names, units, requested):
    """Resolve an explicitly named voltage electrode, ignoring whitespace only.

    Parameters
    ----------
    names, units : sequence of str
        Recorded electrode labels and their physical units.
    requested : str
        Electrode named in the source pairing table.

    Returns
    -------
    int
        Unique voltage-channel index.
    """
    if len(names) != len(units):
        raise ValueError('Channel names and units differ in length.')
    key = ''.join(requested.split())
    matches = [i for i, name in enumerate(names) if ''.join(name.split()) == key]
    if len(matches) != 1 or units[matches[0]] != 'mV':
        raise ValueError('Pairing requires one explicitly named mV channel.')
    return matches[0]


def spike_windows(time_s, pre_mv, post_mv, *, rate_hz, before_ms=10., after_ms=50.):
    """Retain complete paired windows at upward zero-mV source crossings.

    Parameters
    ----------
    time_s, pre_mv, post_mv : array_like
        Original within-sweep timestamps and paired voltage samples.
    rate_hz : float
        Source sampling rate.
    before_ms, after_ms : float, optional
        Inclusive observation extent before and after each crossing.

    Returns
    -------
    dict
        Every crossing index, complete-window indices, omitted boundary events,
        and original timestamps and paired voltages without interpolation.
    """
    arrays = [np.asarray(a, dtype=float) for a in (time_s, pre_mv, post_mv)]
    if any(a.ndim != 1 or a.size < 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Require finite one-dimensional source arrays.')
    t, pre, post = arrays
    if not (t.shape == pre.shape == post.shape):
        raise ValueError('Source arrays have different sample counts.')
    settings = np.asarray([rate_hz, before_ms, after_ms], dtype=float)
    if not np.isfinite(settings).all() or np.any(settings <= 0):
        raise ValueError('Rate and window extents must be positive and finite.')
    if not np.allclose(np.diff(t), 1 / rate_hz, rtol=0, atol=1e-12):
        raise ValueError('Clock does not match the source sampling rate.')
    extents = settings[1:] * rate_hz / 1000
    if not np.allclose(extents, np.rint(extents), rtol=0, atol=1e-9):
        raise ValueError('Window endpoints must lie on the source grid.')
    before, after = np.rint(extents).astype(int)
    crossings = np.flatnonzero((pre[:-1] < 0) & (pre[1:] >= 0)) + 1
    complete = (crossings >= before) & (crossings + after < t.size)
    accepted = crossings[complete]
    indices = accepted[:, None] + np.arange(-before, after + 1)
    return dict(crossing_indices=crossings, accepted_crossings=accepted,
                boundary_crossings=crossings[~complete], indices=indices,
                time_s=t[indices], pre_mv=pre[indices], post_mv=post[indices])
