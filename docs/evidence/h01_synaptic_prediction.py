"""Human-pair current-kernel prediction through a frozen membrane response."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import exprel


def _inputs(lags_ms, parameters, resistance_mohm, membrane_tau_ms):
    t = np.asarray(lags_ms, dtype=float)
    p = np.asarray(parameters, dtype=float)
    if not np.isfinite(t).all() or p.shape != (4,) or not np.isfinite(p).all():
        raise ValueError('Require finite times and four finite parameters.')
    amplitude, rise, gap, delay = p
    if min(amplitude, rise, gap, resistance_mohm, membrane_tau_ms) <= 0 or delay < 0:
        raise ValueError('Invalid current or membrane parameters.')
    if not np.isfinite([resistance_mohm, membrane_tau_ms]).all():
        raise ValueError('Nonfinite membrane parameters.')
    decay = rise + gap
    peak_time = rise*decay/gap*np.log1p(gap/rise)
    norm = np.exp(-peak_time/decay)-np.exp(-peak_time/rise)
    return np.maximum(t-delay, 0), amplitude, rise, decay, norm


def _filtered_exponential(time_ms, synapse_tau_ms, membrane_tau_ms):
    inverse_difference = abs(1/membrane_tau_ms-1/synapse_tau_ms)
    return (time_ms/membrane_tau_ms * np.exp(-time_ms/max(synapse_tau_ms, membrane_tau_ms))
            * exprel(-inverse_difference*time_ms))


def _voltage(lags_ms, parameters, resistance_mohm, membrane_tau_ms):
    t, amplitude, rise, decay, norm = _inputs(lags_ms, parameters, resistance_mohm, membrane_tau_ms)
    response = (_filtered_exponential(t, decay, membrane_tau_ms)
                - _filtered_exponential(t, rise, membrane_tau_ms))
    return resistance_mohm*amplitude/1000/norm*response


def synaptic_waveforms(lags_ms, parameters, *, resistance_mohm, membrane_tau_ms):
    """Compute an effective current and its exact membrane-filtered voltage.

    Parameters
    ----------
    lags_ms : array_like
        Time since each presynaptic crossing; arbitrary finite array shape.
    parameters : array_like
        Peak current in pA, rise time in ms, decay-minus-rise gap in ms, delay in ms.
    resistance_mohm, membrane_tau_ms : float
        Frozen positive cell-response parameters, without an electrode series term.

    Returns
    -------
    tuple of ndarray
        Inferred current in pA and predicted voltage in mV, with the input shape.
    """
    t, amplitude, rise, decay, norm = _inputs(lags_ms, parameters, resistance_mohm, membrane_tau_ms)
    current = amplitude/norm*(np.exp(-t/decay)-np.exp(-t/rise))
    return current, _voltage(lags_ms, parameters, resistance_mohm, membrane_tau_ms)


def train_prediction(lags_ms, baseline_lags_ms, parameters, *, resistance_mohm, membrane_tau_ms):
    """Sum all spike contributions and apply the same pre-event baseline operator.

    Parameters
    ----------
    lags_ms, baseline_lags_ms : array_like
        Lags from every input spike to scored/baseline samples, shaped
        (sweeps, observed events, samples, input spikes).
    parameters : array_like
        Four shared synaptic-current parameters.
    resistance_mohm, membrane_tau_ms : float
        Frozen cell-response parameters.

    Returns
    -------
    ndarray
        Predicted baseline-centered voltage, shaped (sweeps, events, samples).
    """
    lags, baseline = np.asarray(lags_ms), np.asarray(baseline_lags_ms)
    if (lags.ndim != 4 or baseline.ndim != 4 or 0 in lags.shape or 0 in baseline.shape
            or lags.shape[:2] != baseline.shape[:2] or lags.shape[-1] != baseline.shape[-1]):
        raise ValueError('Require complete score and baseline lag arrays.')
    response = _voltage(lags, parameters, resistance_mohm, membrane_tau_ms).sum(axis=-1)
    before = _voltage(baseline, parameters, resistance_mohm, membrane_tau_ms).sum(axis=-1).mean(axis=-1)
    return response-before[..., None]


def fit_synapse(lags_ms, baseline_lags_ms, centered_mv, *, resistance_mohm, membrane_tau_ms):
    """Fit shared synaptic parameters on the supplied calibration data only.

    Parameters
    ----------
    lags_ms, baseline_lags_ms : array_like
        Original spike-to-sample lags as required by train_prediction.
    centered_mv : array_like
        Calibration voltages, shaped (sweeps, events, scored samples).
    resistance_mohm, membrane_tau_ms : float
        Frozen human cell-response parameters.

    Returns
    -------
    dict
        Fitted parameters, convergence information and all optimizer calls.
    """
    settings = dict(resistance_mohm=resistance_mohm, membrane_tau_ms=membrane_tau_ms)
    initial = np.array([60., .3, 2.7, 1.])
    probe = train_prediction(lags_ms, baseline_lags_ms, initial, **settings)
    y = np.asarray(centered_mv, dtype=float)
    if y.shape != probe.shape or not np.isfinite(y).all():
        raise ValueError('Calibration observations do not match complete prediction windows.')
    history = []

    def residual(log_parameters):
        p = np.r_[np.exp(log_parameters[:3]), log_parameters[3]]
        error = train_prediction(lags_ms, baseline_lags_ms, p, **settings)-y
        history.append(dict(parameters=p.tolist(), rmse_mv=float(np.sqrt(np.mean(error**2)))))
        return error.ravel()

    fit = least_squares(residual, np.r_[np.log(initial[:3]), initial[3]],
                        bounds=(np.r_[np.log([.01, .05, .05]), 0.],
                                np.r_[np.log([2000., 5., 95.]), 5.]), max_nfev=60)
    p = np.r_[np.exp(fit.x[:3]), fit.x[3]]
    return dict(parameters=p.tolist(), success=bool(fit.success), message=fit.message,
                nfev=int(fit.nfev), active_mask=fit.active_mask.tolist(), evaluations=history)
