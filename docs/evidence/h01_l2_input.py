"""Keep recorded command and bias distinct in the layer-2 experiment."""

import numpy as np


def stimulus_current(command, bias, *, include_bias=False):
    """Return source command or command plus the recorded scalar bias.

    Parameters
    ----------
    command : array_like
        Finite one-dimensional command samples in nA.
    bias : float
        Finite recorded scalar bias in nA.
    include_bias : bool, optional
        Add the bias only for the explicitly selected intervention.

    Returns
    -------
    numpy.ndarray
        Independent finite current samples in nA.

    Raises
    ------
    ValueError
        If the input is empty, malformed, or nonfinite.
    """
    command = np.asarray(command, dtype=float)
    bias = np.asarray(bias, dtype=float)
    if command.ndim != 1 or not command.size or not np.isfinite(command).all():
        raise ValueError("Command must be a nonempty finite one-dimensional array.")
    if bias.ndim != 0 or not np.isfinite(bias):
        raise ValueError("Recorded bias must be a finite scalar.")
    if not isinstance(include_bias, bool):
        raise ValueError("Bias selection must be boolean.")
    return command + float(bias) if include_bias else command.copy()


def ramp_command(time_ms, rate_pa_per_ms, *, onset_ms=1020., end_ms=2020., cap_na=2.):
    """A linear current ramp in nA that replaces the recorded command inside the pulse window.

    Parameters
    ----------
    time_ms : array_like
        Finite, nondecreasing sample times of the source waveform.
    rate_pa_per_ms : float
        Positive ramp rate; the current is ``rate x (t - onset)`` inside the window, capped.
    onset_ms, end_ms, cap_na : float, optional
        Window and cap; the recorded long square's window by default.

    Returns
    -------
    numpy.ndarray
        Ramp samples in nA, zero outside the window.
    """
    time_ms = np.asarray(time_ms, dtype=float)
    if time_ms.ndim != 1 or not time_ms.size or not np.isfinite(time_ms).all():
        raise ValueError("Time must be a nonempty finite one-dimensional array.")
    if not np.isfinite(rate_pa_per_ms) or rate_pa_per_ms <= 0.:
        raise ValueError("Ramp rate must be positive and finite.")
    inside = (time_ms >= onset_ms) & (time_ms < end_ms)
    return np.where(inside, np.minimum(rate_pa_per_ms*(time_ms-onset_ms)*1e-3, cap_na), 0.)
