"""Fixed-impulse delayed contact learning in a synthetic neuroglial session."""

import copy
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
from braincell.network.delivery import enqueue_future_events

from braintrace._testing.oracle import chunked_online_param_gradients
from braintrace.biophysics.cable_geometry import CableChemicalGeometry
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_anatomy_test import SOURCE
from braintrace.datasets.h01_test import _archive
from braintrace.datasets.h01_biology import topology_digest
from examples.pp_prop.h01_neuroglial_test import configured
from examples.pp_prop.h01_runtime import build_network
from examples.pp_prop.h01_session import H01Session, numerical_settings
from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter


def contact_fixture(imported, folder):
    """Construct explicit reciprocal contacts and recompute their CV maps.

    Parameters
    ----------
    imported : ImportedH01
        Synthetic source morphology fixture.
    folder : pathlib.Path
        Temporary source asset directory.

    Returns
    -------
    tuple
        Topology, archive, biology manifest and local asset map.
    """
    topology, archive, biology, assets = configured(imported, folder)
    topology = topology.clone('12', stage='synthetic-contact-diagnostic')
    other = topology.to_dict()['active_cells'][-1]
    topology = topology.add_contact('12', other, stage='diagnostic', weight_us=.001)
    topology = topology.add_contact(other, '12', stage='diagnostic', weight_us=.002)
    doc = topology.to_dict()
    spines = biology['spines']
    spines['topology_sha256'] = topology_digest(doc)
    spines['cells'][other] = copy.deepcopy(spines['cells']['12'])
    spines['release_probability'] = {key: 1. for key in doc['active_contacts']}
    network, _ = build_network(topology, archive, environment_potassium=True, biology=spines)
    network.init_state()
    for identity in doc['active_cells']:
        geometry = CableChemicalGeometry(network.populations['cell_'+identity].cell)
        biology['membranes'][identity] = dict(geometry_sha256=geometry.sha256,
            outside_indices=[0]*len(geometry.volume_um3))
    biology['basis'] = 'Synthetic reciprocal contacts; shared diagnostic ECS, fixed presynaptic impulses'
    return topology, archive, biology, assets


class _ContactView(brainstate.nn.Module):
    def __init__(self, model, impulses=True):
        super().__init__()
        self.model, self.impulses = model, impulses

    def states(self, *args, **kwargs):
        if args == (brainstate.ParamState,):
            return brainstate.util.FlattedDict({('model', 'recurrent_weight'): self.model.recurrent_weight})
        return super().states(*args, **kwargs)

    def update(self, event):
        result = self.model(event)
        if self.impulses:
            step = self.model.stepper
            # Fixed impulses at .1 and .2 ms; native 100-tick delay stays intact.
            # Restore native spike state after native weighted queue insertion.
            saved = tuple(cell.spike.value for cell in step.cells)
            for index, cell in enumerate(step.cells):
                cell.spike.value = jnp.full_like(cell.spike.value, step.tick.value == 20*(index+1))
            enqueue_future_events(step.setup.delivery_blocks, step.delivery,
                                  populations=step.network.populations)
            for cell, value in zip(step.cells, saved):
                cell.spike.value = value
        return result


def qualify(session):
    """Measure contact temporal credit and its matched no-impulse control.

    Parameters
    ----------
    session : H01Session
        Initialized two-neuron diagnostic session.

    Returns
    -------
    dict
        Contact gradient, delay, control and descent evidence.
    """
    model = session.model
    base = model.recurrent_weight.value
    inputs = jnp.arange(1, 10)[:, None]*jnp.linspace(.0005, .0015, 441)[None, :]
    view, control = _ContactView(model), _ContactView(model, False)

    def online(decay):
        gradients = chunked_online_param_gradients(lambda: view, inputs,
            algo_factory=lambda m: braintrace.pp_prop.sparse(m, decay),
            chunk_size=1, compiled_scan=True, initialize_model=False,
            after_init=lambda m, a: m.model.reset_episode(a))
        return np.asarray(gradients[('model', 'recurrent_weight')])

    def trajectory(driver):
        model.reset_episode()
        return brainstate.transform.for_loop(driver, inputs)

    def loss():
        return jnp.square(trajectory(view)).sum()

    evaluate = brainstate.transform.jit(lambda: trajectory(view))
    parameters = {'contacts': model.recurrent_weight}
    try:
        low, high = online(.2), online(.8)
        reference = np.asarray(brainstate.transform.jit(brainstate.transform.grad(loss, parameters))()['contacts'])
        control_gradient = np.asarray(brainstate.transform.jit(brainstate.transform.grad(
            lambda: jnp.square(trajectory(control)).sum(), parameters))()['contacts'])
        without = np.asarray(brainstate.transform.jit(lambda: trajectory(control))())
        baseline = np.asarray(evaluate())
        conductances = [float(np.asarray(model.stepper.network.populations['cell_'+row['post']].cell.sample_probes()[
            'syn_'+key+'_g'].to_decimal(u.uS)).reshape(()))
            for key, row in session.topology.to_dict()['contacts'].items()]
        direction = high/np.linalg.norm(high)
        epsilon = 1e-8
        model.recurrent_weight.value = base+epsilon*jnp.asarray(direction)
        plus = float(jnp.square(evaluate()).sum())
        model.recurrent_weight.value = base-epsilon*jnp.asarray(direction)
        minus = float(jnp.square(evaluate()).sum())
        fd = (plus-minus)/(2*epsilon)
        projection = float(reference @ direction)
        model.recurrent_weight.value = base-1e-7*jnp.asarray(direction)
        after = float(jnp.square(evaluate()).sum())
        return dict(events=9, physical_ticks=int(model.stepper.tick.value), dt_ms=.005,
            contact_count=2, contact_delay_ms=.5, imposed_impulse_times_ms=[.1, .2],
            low_gradient=low.tolist(), high_gradient=high.tolist(), bptt_gradient=reference.tolist(),
            no_impulse_gradient=control_gradient.tolist(),
            decay_relative_difference=float(np.linalg.norm(high-low)/np.linalg.norm(high)),
            bptt_relative_error=float(np.linalg.norm(high-reference)/np.linalg.norm(reference)),
            cosine=float(high @ reference/(np.linalg.norm(high)*np.linalg.norm(reference))),
            finite_difference=fd, bptt_projection=projection,
            directional_relative_error=abs(fd-projection)/abs(projection),
            objective_before=float(np.square(baseline).sum()), objective_after=after,
            contact_update_norm_us=1e-7, positive_weights=bool(jnp.all(model.recurrent_weight.value > 0)),
            final_conductances_us=conductances, trajectory_mv=baseline.tolist(),
            no_impulse_trajectory_mv=without.tolist(),
            chemistry_valid=bool(model.stepper.chemistry.valid.value))
    finally:
        model.recurrent_weight.value = base
        model.reset_episode(session.learner)


def validate(report):
    """Enforce active delayed-contact and finite-window diagnostic gates.

    Parameters
    ----------
    report : dict
        Metrics from :func:`qualify`.
    """
    for key in ('low_gradient', 'high_gradient', 'bptt_gradient'):
        value = np.asarray(report[key])
        assert np.isfinite(value).all() and np.all(np.abs(value) > 1e-8)
    assert report['decay_relative_difference'] > 1e-4
    assert report['bptt_relative_error'] < 1 and report['cosine'] > 0
    assert report['directional_relative_error'] < 1e-3
    assert report['objective_after'] < report['objective_before']
    assert report['positive_weights'] and report['chemistry_valid']
    assert report['physical_ticks'] == 180 and min(report['final_conductances_us']) > 0
    np.testing.assert_array_equal(report['no_impulse_gradient'], [0., 0.])
    actual, control = np.asarray(report['trajectory_mv']), np.asarray(report['no_impulse_trajectory_mv'])
    np.testing.assert_array_equal(actual[:6], control[:6])
    assert np.linalg.norm(actual[6:]-control[6:]) > 1e-6


def main(output=None):
    """Write source-pinned fixed-impulse contact diagnostic evidence.

    Parameters
    ----------
    output : pathlib.Path or None, optional
        Artifact directory; defaults to this phase's evidence folder.
    """
    started = time.perf_counter()
    with (tempfile.TemporaryDirectory() as temporary, pytest.MonkeyPatch.context() as patch,
          brainstate.environ.context(precision=64, dt=.005*u.ms)):
        folder = Path(temporary)
        path, _ = _archive(folder, patch, [('12.0.swc', SOURCE)])
        topology, archive, biology, assets = contact_fixture(H01Archive(path).load(12, component=0), folder)
        settings = numerical_settings()
        settings['biology'] = biology
        trainer = Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
        session = H01Session.build(topology, archive, trainer, settings=settings, biology_assets=assets)
        report = qualify(session)
        validate(report)
        assert int(session.model.stepper.tick.value) == 0
        report.update(status='SYNTHETIC_FIXED_IMPULSE_CONTACT_LEARNING_PASS',
            elapsed_seconds=time.perf_counter()-started,
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            qualification='Fixed-impulse synthetic contact-weight credit only; no endogenous recurrent firing, ARC or physiology qualification')
        output = Path(__file__).parent/'neuroglial-contact-phase' if output is None else Path(output)
        output.mkdir(exist_ok=True)
        for name, data in [('contact-probe.json', report), ('settings.json', session.settings),
                           ('topology.json', topology.to_dict())]:
            (output/name).write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
