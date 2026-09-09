"""Reject changed wheel payloads and exercise the isolated launcher lifecycle."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from docs.evidence import h01_installed_network as launcher


@pytest.fixture
def prepared(tmp_path):
    target = tmp_path/'installed'
    (target/'braintrace').mkdir(parents=True)
    payload = b'VALUE = 104\n'
    (target/'braintrace/__init__.py').write_bytes(payload)
    wheel = tmp_path/'package.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('braintrace/__init__.py', payload)
    audit = dict(status='wheel content and isolated imports passed',
        wheel=dict(path=str(wheel), sha256=hashlib.sha256(wheel.read_bytes()).hexdigest()),
        installed_target=str(target), source_commit='fixture',
        package_files_exactly_match_snapshot_and_install=1)
    audit_path = tmp_path/'audit.json'
    audit_path.write_text(json.dumps(audit))
    return target, wheel, audit, audit_path


def test_verify_payload_and_hashes(prepared):
    target, _, _, audit_path = prepared
    actual, inputs = launcher.verify_install(audit_path)
    assert actual == target
    assert inputs['verified_package_files'] == 1
    assert inputs['audit_sha256'] == hashlib.sha256(audit_path.read_bytes()).hexdigest()


@pytest.mark.parametrize('defect', ['audit', 'wheel', 'payload', 'missing', 'extra', 'inventory', 'escape'])
def test_rejects_changed_install(prepared, defect):
    target, wheel, audit, path = prepared
    if defect == 'audit': audit['status'] = 'failed'
    elif defect == 'wheel': wheel.write_bytes(b'changed')
    elif defect == 'payload': (target/'braintrace/__init__.py').write_text('VALUE = 12')
    elif defect == 'missing': (target/'braintrace/__init__.py').unlink()
    elif defect == 'extra': (target/'braintrace/extra.py').write_text('')
    elif defect == 'inventory': audit['package_files_exactly_match_snapshot_and_install'] = 2
    elif defect == 'escape':
        with zipfile.ZipFile(wheel, 'a') as archive:
            archive.writestr('braintrace/../outside.py', b'')
        audit['wheel']['sha256'] = hashlib.sha256(wheel.read_bytes()).hexdigest()
        audit['package_files_exactly_match_snapshot_and_install'] = 2
    path.write_text(json.dumps(audit))
    with pytest.raises((ValueError, FileNotFoundError)):
        launcher.verify_install(path)


def test_rejects_preloaded_source_module(prepared, monkeypatch):
    target, _, _, _ = prepared
    monkeypatch.setitem(sys.modules, 'braintrace', SimpleNamespace(__file__=__file__))
    with pytest.raises(ValueError, match='outside installed wheel'):
        launcher.production_origins(target)


@pytest.mark.parametrize('scenario', ['success', 'zero_exit', 'exception', 'nonzero_exit', 'contamination', 'not_isolated', 'exists'])
def test_main_restores_paths_and_records_failures(prepared, tmp_path, monkeypatch, scenario):
    target, _, _, audit = prepared
    example, provenance = tmp_path/'model.py', tmp_path/'result.json'
    example.write_text('fixture model')
    if scenario == 'exists': provenance.write_text('preserve')
    arguments = ['launcher', '--audit', str(audit), '--example', str(example),
                 '--provenance', str(provenance), '--', '--cells', '104']
    process = SimpleNamespace(flags=SimpleNamespace(isolated=scenario != 'not_isolated'),
        argv=arguments[:], path=['dependency'], modules={}, executable='fixture python')
    # argparse reads its own reference to the real sys module.
    monkeypatch.setattr(sys, 'argv', arguments[:])
    monkeypatch.setattr(launcher, 'sys', process)

    def load(name):
        process.modules[name] = SimpleNamespace(__file__=str(target/'braintrace/__init__.py'))

    def run(path, run_name):
        assert path == str(example) and run_name == '__main__'
        assert process.argv == [str(example), '--cells', '104']
        assert process.path[0] == str(target)
        if scenario == 'zero_exit': raise SystemExit(0)
        if scenario == 'exception': raise RuntimeError('model failed')
        if scenario == 'nonzero_exit': raise SystemExit(7)
        if scenario == 'contamination': process.modules['braintrace'].__file__ = str(example)

    monkeypatch.setattr(launcher, 'importlib', SimpleNamespace(import_module=load))
    monkeypatch.setattr(launcher, 'runpy', SimpleNamespace(run_path=run))
    if scenario in ('success', 'zero_exit'):
        launcher.main()
    else:
        with pytest.raises((SystemExit, RuntimeError, ValueError, FileExistsError)):
            launcher.main()
    assert process.argv == arguments and process.path == ['dependency']
    if scenario not in ('not_isolated', 'exists'):
        record = json.loads(provenance.read_text())
        assert record['status'] == ('completed' if scenario in ('success', 'zero_exit') else 'failed')
        if scenario in ('exception', 'nonzero_exit'):
            assert record['production_origins']['braintrace'] == str(target/'braintrace/__init__.py')
        if scenario == 'contamination':
            assert 'outside installed wheel' in record['origin_error']


@pytest.mark.parametrize('scenario', ['success', 'zero_exit', 'exception', 'nonzero_exit', 'contamination', 'not_isolated', 'exists'])
def test_launcher_records_real_subprocess_outcome(prepared, tmp_path, scenario):
    _, _, _, audit = prepared
    example, provenance = tmp_path/'example.py', tmp_path/'run.json'
    code = 'import braintrace, sys\nassert braintrace.VALUE == 104\nassert sys.argv[1:] == ["--cells", "104"]\n'
    if scenario == 'zero_exit': code += 'raise SystemExit(0)\n'
    elif scenario == 'exception': code += 'raise RuntimeError("model failed")\n'
    elif scenario == 'nonzero_exit': code += 'raise SystemExit(7)\n'
    elif scenario == 'contamination': code += 'braintrace.__file__ = __file__\n'
    elif scenario == 'exists': provenance.write_text('preserve this run')
    example.write_text(code)
    command = [sys.executable]
    if scenario != 'not_isolated': command.append('-I')
    command.extend([str(Path(launcher.__file__).resolve()), '--audit', str(audit),
        '--example', str(example), '--provenance', str(provenance), '--', '--cells', '104'])
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if scenario in ('not_isolated', 'exists'):
        assert completed.returncode != 0
        assert not provenance.exists() if scenario == 'not_isolated' else provenance.read_text() == 'preserve this run'
        return
    record = json.loads(provenance.read_text())
    expected = 'completed' if scenario in ('success', 'zero_exit') else 'failed'
    assert record['status'] == expected
    assert (completed.returncode == 0) == (expected == 'completed')
    assert record['example_sha256'] == hashlib.sha256(example.read_bytes()).hexdigest()
    assert record['arguments'] == ['--cells', '104']
    assert record['finished_utc'] >= record['started_utc']
    assert record['qualification'].startswith('launcher provenance only')
    if scenario in ('exception', 'nonzero_exit'):
        assert Path(record['production_origins']['braintrace']).is_relative_to(prepared[0])
    if scenario == 'contamination':
        assert 'outside installed wheel' in record['origin_error']
