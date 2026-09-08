"""Run the H01 example with verified installed-wheel imports and provenance."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
from pathlib import Path
import runpy
import sys
import zipfile


def verify_install(audit_path):
    """Verify a prepared wheel and its installed package payload.

    Parameters
    ----------
    audit_path : pathlib.Path
        Wheel audit; its relative paths resolve from the working directory.

    Returns
    -------
    tuple
        Installed target and hash-linked input metadata.
    """
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    if audit.get('status') != 'wheel content and isolated imports passed':
        raise ValueError('Wheel preparation audit did not pass.')
    wheel_path = Path(audit['wheel']['path']).resolve()
    wheel_bytes = wheel_path.read_bytes()
    wheel_hash = hashlib.sha256(wheel_bytes).hexdigest()
    if wheel_hash != audit['wheel']['sha256']:
        raise ValueError('Wheel hash changed.')
    target = Path(audit['installed_target']).resolve()
    package = target/'braintrace'
    with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as wheel:
        names = [n for n in wheel.namelist() if n.startswith('braintrace/') and not n.endswith('/')]
        if ('braintrace/__init__.py' not in names or len(names) != len(set(names)) or
                len(names) != audit['package_files_exactly_match_snapshot_and_install']):
            raise ValueError('Wheel payload inventory differs.')
        for name in names:
            installed = (target/name).resolve()
            if not installed.is_relative_to(package) or installed.read_bytes() != wheel.read(name):
                raise ValueError('Installed wheel payload changed: '+name)
        actual = {p.relative_to(target).as_posix() for p in package.rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts}
        if actual != set(names):
            raise ValueError('Unexpected installed package files.')
    return target, dict(audit_path=str(audit_path.resolve()),
        audit_sha256=hashlib.sha256(audit_bytes).hexdigest(), wheel_path=str(wheel_path),
        wheel_sha256=wheel_hash, installed_target=str(target), source_commit=audit['source_commit'],
        verified_package_files=len(names))


def production_origins(target):
    """Reject loaded BrainTrace modules outside the installed target.

    Parameters
    ----------
    target : pathlib.Path
        Resolved installation root.

    Returns
    -------
    dict
        Loaded module names mapped to their resolved file paths.
    """
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name == 'braintrace' or name.startswith('braintrace.'):
            filename = getattr(module, '__file__', None)
            if not filename or not Path(filename).resolve().is_relative_to(target/'braintrace'):
                raise ValueError('Production import outside installed wheel: '+name)
            origins[name] = str(Path(filename).resolve())
    return origins


def main():
    """Execute the repository example and record library provenance.

    Returns
    -------
    None
        Writes provenance; example failures retain a failed record and propagate.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--example', type=Path, required=True)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error('Run this launcher with Python -I.')
    if args.provenance.exists():
        raise FileExistsError('Provenance already exists; do not overwrite a run.')
    target, inputs = verify_install(args.audit)
    production_origins(target)
    example = args.example.resolve()
    example_hash = hashlib.sha256(example.read_bytes()).hexdigest()
    forwarded = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    record = dict(status='running', inputs=inputs, example_path=str(example),
        example_sha256=example_hash, arguments=forwarded, python=sys.executable,
        started_utc=datetime.now(timezone.utc).isoformat(),
        qualification='launcher provenance only; runtime and physiological gates remain separate')
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(record, indent=2)+'\n')
    previous_argv, previous_path = sys.argv[:], sys.path[:]
    sys.path.insert(0, str(target))
    sys.argv = [str(example), *forwarded]
    try:
        importlib.import_module('braintrace')
        production_origins(target)
        try:
            runpy.run_path(str(example), run_name='__main__')
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        record['production_origins'] = production_origins(target)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record['finished_utc'] = datetime.now(timezone.utc).isoformat()
        args.provenance.write_text(json.dumps(record, indent=2)+'\n')
        sys.argv, sys.path = previous_argv, previous_path


if __name__ == '__main__':
    main()
