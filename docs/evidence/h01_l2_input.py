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
