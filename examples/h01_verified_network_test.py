"""Keep runtime evidence when compilation or returned traces fail."""

from contextlib import nullcontext
import json
from types import SimpleNamespace
import sys

import brainunit as u
import numpy as np
import pytest

from examples import h01_verified_network as example


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    topology = tmp_path/'topology.json'
    topology.write_text(json.dumps(dict(nodes=[], contacts=[], counts={})))
    output = tmp_path/'run'
    result = SimpleNamespace(time=np.arange(4)*.005*u.ms,
        traces={'cell_one':{'voltage':np.full((4,1), -65.)*u.mV,
                            'output_voltage':np.full((4,1), -65.)*u.mV,
                            'syn_E_g':np.zeros((4,1))*u.uS}},
        spikes={'cell_one':np.zeros((4,1), dtype=bool)})
    network = SimpleNamespace(projections=[], run=lambda **kwargs:result)
    evidence = dict(cells={'one':{'n_compartments':1}})
    monkeypatch.setattr(example, 'plan_h01_cells', lambda *args, **kwargs:dict(simulated_cell_ids=['one']))
    monkeypatch.setattr(example, 'make_h01_network', lambda *args, **kwargs:(network,evidence))
    monkeypatch.setattr(example, 'H01Archive', lambda *args:None)
    monkeypatch.setattr(example, 'H01Annotations', lambda *args:None)
    monkeypatch.setattr(example, 'init_h01_network_states', lambda *args, **kwargs:
        dict(init_seconds_by_population={'cell_one':.1}, peak_rss_mb=1.))
    monkeypatch.setattr(example, 'process_rss_mb', lambda **kwargs:2.)
    monkeypatch.setattr(example, 'heartbeat', lambda *args, **kwargs:nullcontext())
    monkeypatch.setattr(sys, 'argv', ['example','--topology',str(topology),'--output',str(output),
                                     '--duration-ms','.02','--dt-ms','.005'])
    return output,result,network


def test_nonfinite_run_retains_all_arrays_and_failed_metadata(run_fixture):
    output,result,_ = run_fixture
    result.traces['cell_one']['output_voltage'] = np.array([[-65.],[-40.],[np.nan],[np.inf]])*u.mV
    with pytest.raises(RuntimeError, match='Nonfinite'):
        example.main()
    with np.load(output.parent/(output.name+'-traces.npz')) as arrays:
        assert len(arrays.files) == 5
        assert np.isnan(arrays['cell_one_output_voltage'][2,0])
        assert arrays['cell_one_events'].dtype == bool
    evidence = json.loads((output.parent/(output.name+'-build.json')).read_text())
    assert evidence['execution'] == 'nonfinite compiled run'
    assert evidence['init_state_seconds'] >= 0
    assert evidence['nonfinite_arrays']['cell_one_output_voltage'] == dict(
        count=2, first_index=[2,0], first_time_ms=.015)


def test_compile_exception_preserves_completed_initialization(run_fixture):
    output,_,network = run_fixture

    def fail(**kwargs):
        raise RuntimeError('compile failed')

    network.run = fail
    with pytest.raises(RuntimeError, match='compile failed'):
        example.main()
    evidence = json.loads((output.parent/(output.name+'-build.json')).read_text())
    assert evidence['init_seconds_by_cell'] == {'cell_one':.1}
    assert evidence['execution'] == 'constructed and initialized; not compiled or simulated'


def test_finite_result_keeps_timing_and_unit_conventions(run_fixture):
    output,_,_ = run_fixture
    example.main()
    evidence = json.loads((output.parent/(output.name+'-build.json')).read_text())
    assert evidence['execution'] == 'finite compiled smoke run'
    assert evidence['nonfinite_arrays'] == {}
    assert evidence['dt_ms'] == .005 and evidence['duration_ms'] == .02
    with np.load(output.parent/(output.name+'-traces.npz')) as arrays:
        np.testing.assert_allclose(arrays['time_ms'], [.005,.01,.015,.02])
        np.testing.assert_array_equal(arrays['cell_one_voltage'], np.full((4,1), -65.))


@pytest.mark.parametrize('arguments', [
    ['--disconnected','--control','e_only'], ['--cells','0'], ['--dt-ms','0'],
    ['--duration-ms','nan'], ['--init-only'], ['--heartbeat-s','0'],
])
def test_invalid_cli_settings_fail_before_execution(run_fixture, monkeypatch, arguments):
    output,_,_ = run_fixture
    monkeypatch.setattr(sys, 'argv', [*sys.argv, *arguments])
    with pytest.raises(SystemExit) as error:
        example.main()
    assert error.value.code == 2
    assert not output.with_suffix('.json').exists()


def test_init_only_keeps_disconnected_alias_and_skips_compile(run_fixture, monkeypatch):
    output,_,network = run_fixture
    monkeypatch.setattr(sys, 'argv', sys.argv[:sys.argv.index('--duration-ms')]+['--init-only','--disconnected'])

    def plan(*args, **kwargs):
        assert kwargs['control'] == 'disconnected'
        return dict(simulated_cell_ids=['one'])

    monkeypatch.setattr(example, 'plan_h01_cells', plan)
    network.run = lambda **kwargs:pytest.fail('init-only compiled a model')
    example.main()
    record = json.loads((output.parent/(output.name+'-build.json')).read_text())
    assert record['execution'] == 'constructed and initialized; not compiled or simulated'
    assert not (output.parent/(output.name+'-traces.npz')).exists()


def test_export_only_preserves_signed_and_constructible_contacts(run_fixture, monkeypatch):
    output,_,_ = run_fixture
    topology = output.parent/'topology.json'
    topology.write_text(json.dumps(dict(nodes=[dict(cell_id='1',dale_sign=1),dict(cell_id='2',dale_sign=-1)],
        contacts=[dict(pre_index=0,post_index=1,dale_sign=1,construction_ready=True),
                  dict(pre_index=1,post_index=0,dale_sign=-1,construction_ready=False)], counts={})))
    monkeypatch.setattr(sys, 'argv', sys.argv[:sys.argv.index('--duration-ms')])
    monkeypatch.setattr(example, 'make_h01_network', lambda *args, **kwargs:pytest.fail('export constructed a model'))
    example.main()
    with np.load(output.with_suffix('.npz')) as arrays:
        assert arrays['assumed_signed_weight_us'][1,0] == -.02
        assert arrays['placed_contact_count'][1,0] == 0
        assert arrays['placed_contact_count'][0,1] == 1


def test_all_nonfinite_arrays_reported_even_when_time_is_invalid(run_fixture):
    output,result,_ = run_fixture
    result.time = np.array([0.,np.nan,.01,.015])*u.ms
    result.traces['cell_one']['voltage'] = np.array([[-65.],[np.inf],[-65.],[-65.]])*u.mV
    result.traces['cell_one']['output_voltage'] = np.full((4,1), np.nan)*u.mV
    result.spikes['cell_one'] = np.full((4,1), np.nan)
    with pytest.raises(RuntimeError, match='Nonfinite'):
        example.main()
    record = json.loads((output.parent/(output.name+'-build.json')).read_text())
    assert set(record['nonfinite_arrays']) == {'time_ms','cell_one_voltage','cell_one_output_voltage','cell_one_events'}
    assert record['nonfinite_arrays']['cell_one_voltage']['first_time_ms'] is None
