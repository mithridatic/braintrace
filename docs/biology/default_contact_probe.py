"""Analytic trace diagnosis plus configured-decay neuroglial contact evidence."""

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
from docs.biology import neuroglial_contact_probe as fixture
from docs.biology.delayed_credit_probe import measure, validate


def qualify(session):
    """Measure the configured decay on the unchanged delayed-contact fixture.

    Parameters
    ----------
    session : H01Session
        Initialized synthetic reciprocal-contact session.

    Returns
    -------
    dict
        Fresh finite-window, BPTT, finite-difference and descent metrics.
    """
    model = session.model
    base = model.recurrent_weight.value
    view = fixture._ContactView(model)
    inputs = jnp.arange(1, 10)[:, None]*jnp.linspace(.0005, .0015, 441)[None, :]
    decay = session.settings['decay']

    def trajectory():
        model.reset_episode()
        return brainstate.transform.for_loop(view, inputs)

    evaluate = brainstate.transform.jit(trajectory)
    try:
        gradients = chunked_online_param_gradients(lambda: view, inputs,
            algo_factory=lambda m: braintrace.pp_prop.sparse(m, decay), chunk_size=1,
            compiled_scan=True, initialize_model=False,
            after_init=lambda m, a: m.model.reset_episode(a))
        online = np.asarray(gradients[('model', 'recurrent_weight')])
        reference = np.asarray(brainstate.transform.jit(brainstate.transform.grad(
            lambda: jnp.square(trajectory()).sum(), {'contacts': model.recurrent_weight}))()['contacts'])
        baseline = np.asarray(evaluate())
        direction = online/np.linalg.norm(online)
        model.recurrent_weight.value = base+1e-8*jnp.asarray(direction)
        plus = float(jnp.square(evaluate()).sum())
        model.recurrent_weight.value = base-1e-8*jnp.asarray(direction)
        minus = float(jnp.square(evaluate()).sum())
        projection = float(reference @ direction)
        model.recurrent_weight.value = base-1e-7*jnp.asarray(direction)
        after = float(jnp.square(evaluate()).sum())
        report = dict(decay=decay, chunk_size=1, physical_ticks=int(model.stepper.tick.value),
            online_gradient=online.tolist(), bptt_gradient=reference.tolist(),
            relative_error=float(np.linalg.norm(online-reference)/np.linalg.norm(reference)),
            norm_ratio=float(np.linalg.norm(online)/np.linalg.norm(reference)),
            cosine=float(online @ reference/(np.linalg.norm(online)*np.linalg.norm(reference))),
            bptt_projection=projection, finite_difference=(plus-minus)/2e-8,
            objective_before=float(np.square(baseline).sum()), objective_after=after,
            trajectory_mv=baseline.tolist(), chemistry_valid=bool(model.stepper.chemistry.valid.value))
        assert np.isfinite(online).all() and np.all(np.abs(online) > 1e-8)
        assert report['cosine'] > 0 and report['objective_after'] < report['objective_before']
        assert report['chemistry_valid'] and report['physical_ticks'] == 180
        np.testing.assert_allclose(report['finite_difference'], projection, rtol=1e-3)
        return report
    finally:
        model.recurrent_weight.value = base
        model.reset_episode(session.learner)


def main(output=None):
    """Record the analytic sweep and fresh configured-decay contact run.

    Parameters
    ----------
    output : pathlib.Path or None, optional
        Artifact destination, defaulting to this phase's evidence directory.
    """
    started = time.perf_counter()
    with (tempfile.TemporaryDirectory() as temporary, pytest.MonkeyPatch.context() as patch,
          brainstate.environ.context(precision=64, dt=.005*u.ms)):
        # Independent experiments; every model trajectory uses compiled scans.
        rows = [measure(delay, emission, decay) for delay, emission in [(0, 0), (1, 0), (5, 0), (5, 2)]
                for decay in [0., .2, .8, .99]]
        for row in rows:
            validate(row)
        folder = Path(temporary)
        path, _ = fixture._archive(folder, patch, [('12.0.swc', fixture.SOURCE)])
        topology, archive, biology, assets = fixture.contact_fixture(
            fixture.H01Archive(path).load(12, component=0), folder)
        settings = fixture.numerical_settings()
        settings['biology'] = biology
        trainer = fixture.Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
        session = fixture.H01Session.build(topology, archive, trainer, settings=settings, biology_assets=assets)
        report = qualify(session)
        previous_path = Path(__file__).parent/'neuroglial-contact-phase/contact-probe.json'
        previous = json.loads(previous_path.read_text())
        np.testing.assert_allclose(report['trajectory_mv'], previous['trajectory_mv'], rtol=0, atol=1e-10)
        np.testing.assert_allclose(report['bptt_gradient'], previous['bptt_gradient'], rtol=1e-10)
        assert int(model_tick := session.model.stepper.tick.value) == 0
        report.update(status='TRACE_LAW_VERIFIED_DEFAULT_CONTACT_MAGNITUDE_UNQUALIFIED',
            previous_probe_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            elapsed_seconds=time.perf_counter()-started)
        output = Path(__file__).parent/'delayed-credit-phase' if output is None else Path(output)
        output.mkdir(exist_ok=True)
        for name, data in [('default-contact.json', report), ('analytic-sweep.json', rows),
                           ('settings.json', session.settings)]:
            (output/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'trajectory_mv'}, indent=2))


if __name__ == '__main__':
    main()
