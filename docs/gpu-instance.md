# GPU instance (Vast.ai)

Box: Vast 50616476, RTX 4090, 255 CPUs, 128 GB disk. Connect with `ssh braintrace-gpu`
(alias in `~/.ssh/config`; non-interactive use needs `-o BatchMode=yes`; the banner prints
two lines before any output). `/workspace/braintrace` tracks `main` and is never edited in
place; campaign work runs in a worktree (`git -C /workspace/braintrace worktree add
/workspace/braintrace-<name> origin/<branch>`). The pinned simulation environment is
`/workspace/venv314/bin/python` (do not install into it).

## venv-c3 (C3 release data work; cloud-volume, fastavro)

Created 2026-09-16 for the C3 partner-expansion campaign
(`docs/specs/2026-09-16-h01-c3-partner-expansion.md`). Separate from `venv314` because
cloud-volume pins conflict with the simulation stack. `uv` lives at `/root/.local/bin/uv`.

```sh
/root/.local/bin/uv venv /workspace/venv-c3 --python 3.12
/root/.local/bin/uv pip install --python /workspace/venv-c3/bin/python cloud-volume fastavro numpy
```

Installed 2026-09-16: Python 3.12.14, fastavro 1.12.2, numpy 2.5.3. Verification (skeleton
of kept cell 1684504313 from the C3 precomputed volume; expected 28,643 vertices):

```sh
/workspace/venv-c3/bin/python -c "
from cloudvolume import CloudVolume
sk = CloudVolume('precomputed://https://storage.googleapis.com/h01-release/data/20210601/c3',
                 mip=0, use_https=True, cache=False).skeleton.get(1684504313)
print(len(sk.vertices))"
```

`braintrace` itself is not importable from this venv (`braintrace/__init__` imports
brainstate); the C3 evidence scripts load `braintrace/datasets/h01_cell_types.py` by file
path instead. Long jobs: `nohup ... > run.log 2> run.err &` under the worktree's
`var/<campaign>/` receipt directory; poll with ssh; kill by PID only.
