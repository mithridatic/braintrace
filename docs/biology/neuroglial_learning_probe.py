"""Bounded synthetic full-session finite-window learning diagnostic."""

import hashlib
import json
from pathlib import Path
import tempfile
import time

import brainstate
import braintrace
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace._testing.oracle import chunked_online_param_gradients
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_anatomy_test import SOURCE
from braintrace.datasets.h01_test import _archive
from examples.pp_prop.h01_neuroglial_test import configured
from examples.pp_prop.h01_session import H01Session, numerical_settings
from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter


class _EncoderView(brainstate.nn.Module):
    # Expose the physical model but only the parameter under this diagnostic.
    # The soma objective does not read the ARC readout parameters.
    def __init__(self, model):
        super().__init__()
        self.model = model

    def states(self, *args, **kwargs):
        states = super().states(*args, **kwargs)
        if args == (brainstate.ParamState,):
            return brainstate.util.FlattedDict({('model', 'input_weight'): self.model.input_weight})
        return states

    def update(self, event):
        return self.model(event)


def qualify(session):
    """Measure finite-window encoder gradients on a reset synthetic session.

    Parameters
    ----------
    session : H01Session
        Initialized diagnostic neuroglial session; parameters are restored.

    Returns
    -------
    dict
        Gradient, descent and physical-validity evidence for three events.
    """
    model = session.model
    inputs = jnp.stack((jnp.linspace(.005, .015, 441),
                        jnp.linspace(.03, .01, 441),
                        jnp.where(jnp.arange(441) % 2, .002, .008)))
    base = model.input_weight.value
    chemistry = model.stepper.chemistry
    view = _EncoderView(model)

    def online(decay):
        gradients = chunked_online_param_gradients(lambda: view, inputs,
            algo_factory=lambda m: braintrace.pp_prop.sparse(m, decay),
            chunk_size=1, compiled_scan=True, initialize_model=False,
            after_init=lambda m, a: m.model.reset_episode(a))
        return np.asarray(gradients[('model', 'input_weight')])

    low, high = online(.2), online(.8)

    def objective():
        model.reset_episode()
        return jnp.square(brainstate.transform.for_loop(model.update, inputs)).sum()

    evaluate = brainstate.transform.jit(objective)
    reference = np.asarray(brainstate.transform.jit(brainstate.transform.grad(
        objective, {'encoder': model.input_weight}))()['encoder'])
    before = float(evaluate())
    final_k = np.asarray(chemistry.environment.potassium.value).copy()
    released_glutamate = float(chemistry.glutamate.released_amount.value)
    direction = high/np.linalg.norm(high)
    epsilon = 1e-5
    model.input_weight.value = base+epsilon*jnp.asarray(direction)
    plus = float(evaluate())
    model.input_weight.value = base-epsilon*jnp.asarray(direction)
    minus = float(evaluate())
    finite_difference = (plus-minus)/(2*epsilon)
    projection = float(reference @ direction)
    model.input_weight.value = base-1e-4*jnp.asarray(direction)
    after = float(evaluate())
    valid = bool(chemistry.valid.value)
    ticks = int(model.stepper.tick.value)
    model.input_weight.value = base
    model.reset_episode(session.learner)
    delta_k = float(np.linalg.norm(final_k-np.asarray(chemistry.environment.potassium.value)))
    return dict(events=3, chunk_size=1, physical_ticks=ticks, dt_ms=.005,
        input_rank=int(np.linalg.matrix_rank(np.asarray(inputs))),
        glutamate_released_amount=released_glutamate,
        decay_low=.2, decay_high=.8, low_norm=float(np.linalg.norm(low)),
        high_norm=float(np.linalg.norm(high)), bptt_norm=float(np.linalg.norm(reference)),
        decay_relative_difference=float(np.linalg.norm(high-low)/np.linalg.norm(high)),
        bptt_relative_error=float(np.linalg.norm(high-reference)/np.linalg.norm(reference)),
        cosine=float(high @ reference/(np.linalg.norm(high)*np.linalg.norm(reference))),
        bptt_projection=projection, finite_difference=finite_difference,
        directional_relative_error=abs(projection-finite_difference)/abs(projection),
        objective_before=before, objective_after=after, encoder_update_norm=1e-4,
        extracellular_k_change_norm=delta_k, chemistry_valid=valid,
        gradients_finite=bool(all(np.isfinite(x).all() for x in (low, high, reference))))


def validate(report):
    """Enforce the prespecified bounded diagnostic gates.

    Parameters
    ----------
    report : dict
        Metrics returned by :func:`qualify`.
    """
    assert report['gradients_finite'] and report['chemistry_valid']
    assert min(report['low_norm'], report['high_norm'], report['bptt_norm']) > 1e-8
    assert report['decay_relative_difference'] > 1e-4
    assert report['directional_relative_error'] < 1e-3
    assert report['cosine'] > 0
    assert report['bptt_relative_error'] < 1
    assert report['input_rank'] == 3
    assert report['objective_after'] < report['objective_before']
    assert report['extracellular_k_change_norm'] > 0
    assert report['physical_ticks'] == 60


def main(output=None):
    """Write bounded synthetic learning evidence and pinned session settings.

    Parameters
    ----------
    output : pathlib.Path or None, optional
        Artifact destination, defaulting to the phase evidence directory.
    """
    started = time.perf_counter()
    with (tempfile.TemporaryDirectory() as temporary, pytest.MonkeyPatch.context() as patch,
          brainstate.environ.context(precision=64, dt=.005*u.ms)):
        folder = Path(temporary)
        path, _ = _archive(folder, patch, [('12.0.swc', SOURCE)])
        topology, archive, biology, assets = configured(H01Archive(path).load(12, component=0), folder)
        settings = numerical_settings()
        settings['biology'] = biology
        trainer = Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
        session = H01Session.build(topology, archive, trainer, settings=settings, biology_assets=assets)
        report = qualify(session)
        validate(report)
        report.update(status='SYNTHETIC_NEUROGLIAL_FINITE_WINDOW_PASS',
            elapsed_seconds=time.perf_counter()-started,
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Synthetic encoder descent on soma-voltage SSE; no ARC, recurrent-contact, discrete-release gradient or physiology qualification')
        output = Path(__file__).parent/'neuroglial-learning-phase' if output is None else Path(output)
        output.mkdir(exist_ok=True)
        for name, data in [('learning-probe.json', report), ('settings.json', session.settings)]:
            (output/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
