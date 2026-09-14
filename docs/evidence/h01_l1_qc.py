"""Run the pinned IPFX sweep-QC pipeline and retain exclusions and criteria."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np


def json_values(obj):
    """Convert numerical containers to JSON while preserving missing readings.

    Parameters
    ----------
    obj : object
        Nested QC output containing NumPy scalars or arrays.

    Returns
    -------
    object
        JSON-compatible values, with nonfinite numbers represented as None.
    """
    if isinstance(obj, dict):
        return {k: json_values(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_values(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_values(obj.tolist())
    if isinstance(obj, np.generic):
        return json_values(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main(argv=None):
    """Evaluate QC with unchanged IPFX criteria and a worktree-local cache.

    Parameters
    ----------
    argv : list of str or None, optional
        NWB, output JSON and cache directory CLI arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nwb', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--cache', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Use a new QC artifact.')
    digest = hashlib.sha256(args.nwb.read_bytes()).hexdigest()
    if digest != '004f27f306180bd610ba2439e600f688c11be24adf5cea2a1663a22ba0336af6':
        raise ValueError('Wrong source recording.')
    distribution = importlib.metadata.distribution('ipfx')
    install = json.loads(distribution.read_text('direct_url.json'))
    if install['vcs_info']['commit_id'] != 'c607a36e25618a6bb8ba5438c1cd945fb429e659':
        raise ValueError('Wrong IPFX source revision.')
    # PyNWB 3.1.2 creates its cache even when caching is disabled. Redirect only
    # its directory lookup during import; restore platformdirs immediately after.
    import platformdirs
    original_dirs = platformdirs.PlatformDirs
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)

    class LocalNWBDirs(original_dirs):
        @property
        def user_cache_dir(self):
            return str(cache)

    platformdirs.PlatformDirs = LocalNWBDirs
    try:
        import pynwb  # noqa: F401
    finally:
        platformdirs.PlatformDirs = original_dirs
    from ipfx.dataset.create import create_ephys_data_set
    from ipfx.qc_feature_extractor import sweep_qc_features
    from ipfx import sweep_props
    from ipfx.qc_feature_evaluator import qc_sweeps, load_default_qc_criteria

    started = time.monotonic()
    dataset = create_ephys_data_set(nwb_file=str(args.nwb))
    features = sweep_qc_features(dataset)
    original = copy.deepcopy(features)
    current_clamp_sweeps = dataset.filtered_sweep_table(
        clamp_mode=dataset.CURRENT_CLAMP)['sweep_number'].tolist()
    sweep_props.drop_tagged_sweeps(features)
    sweep_props.remove_sweep_feature('tags', features)
    criteria = load_default_qc_criteria()
    states = qc_sweeps(dataset.ontology, features, criteria)
    report = json_values(dict(scope='offline recording QC only; no physiology qualification',
                              source_sha256=digest, ipfx_version=distribution.version,
                              ipfx_install=install, criteria=criteria,
                              original_features=original, states=states,
                              not_evaluated_sweeps=sorted(set(current_clamp_sweeps) -
                                  {int(row['sweep_number']) for row in original}),
                              integrity_rejected_sweeps=[row['sweep_number'] for row in original
                                                         if row['tags']],
                              elapsed_seconds=time.monotonic()-started))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(states))


if __name__ == '__main__':
    main()
