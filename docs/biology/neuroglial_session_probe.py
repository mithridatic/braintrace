"""Executable synthetic fixture for the complete explicit H01 session factory.

Run ``python -m docs.biology.neuroglial_session_probe`` from the checkout root.
The development fixture is deliberately synthetic, not measured H01 anatomy.
"""

import json
from pathlib import Path
import tempfile
import time

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import pytest

from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_anatomy_test import SOURCE
from braintrace.datasets.h01_test import _archive
from braintrace.datasets.h01_neuroglia import H01NeuroglialManifest
from examples.pp_prop.h01_neuroglial_test import configured
from examples.pp_prop.h01_session import H01Session, numerical_settings
from examples.pp_prop.example21_arc_adapter import Example21ArcAdapter
from examples.pp_prop.h01_physical_wait import PhysicalWait
from examples.pp_prop.h01_physical_checkpoint import _pack


def main():
    """Record compiled session construction and exact active-eligibility replay."""
    started=time.perf_counter()
    with (tempfile.TemporaryDirectory() as temporary, pytest.MonkeyPatch.context() as patch,
          brainstate.environ.context(precision=64,dt=.005*u.ms)):
        folder=Path(temporary)
        path,_=_archive(folder,patch,[('12.0.swc',SOURCE)])
        imported=H01Archive(path).load(12,component=0)
        topology,archive,biology,assets=configured(imported,folder)
        settings=numerical_settings()
        settings['biology']=biology
        trainer=Example21ArcAdapter(Path('.'))._model().PPPropEpisodeTrainer
        session=H01Session.build(topology,archive,trainer,settings=settings,biology_assets=assets)
        session.model.reset_episode(session.learner)
        brainstate.transform.jit(session.learner)(jnp.full(441,.01))
        factor_nonzero=sum(int(np.count_nonzero(f)) for f in session.learner.factors.value)
        assert factor_nonzero>0
        wait=PhysicalWait(session.model,.0002,learner=session.learner)
        advance=brainstate.transform.jit(lambda:wait.update(max_events=1))
        assert not advance()
        snapshot=folder/'physical.npz'
        digest=session.save_physical(snapshot,wait=wait)
        assert advance()
        expected,_,_=_pack(dict(model=session.model,learner=session.learner,wait=wait),{})
        fresh=H01Session.build(topology,archive,trainer,settings=settings,biology_assets=assets)
        resumed=PhysicalWait(fresh.model,.0002,learner=fresh.learner)
        fresh.restore_physical(snapshot,wait=resumed,expected_sha256=digest)
        assert brainstate.transform.jit(lambda:resumed.update(max_events=1))()
        actual,_,_=_pack(dict(model=fresh.model,learner=fresh.learner,wait=resumed),{})
        assert expected.keys()==actual.keys()
        assert all(np.isfinite(value).all() and np.array_equal(value,actual[key]) for key,value in expected.items())
        chemistry=session.model.stepper.chemistry
        assert chemistry.valid.value
        report=dict(status='SYNTHETIC_NEUROGLIAL_SESSION_PASS',neurons=session.model.neuron_count,
            neuronal_compartments=sum(cell.n_cv for cell in session.model.stepper.cells),
            glial_compartments=sum(cell.n_cv for cell in chemistry.astrocytes),
            dt_ms=.005,event_ms=.1,total_physical_ticks=int(session.model.stepper.tick.value),
            driven_events=1,wait_events=2,nonzero_eligibility_entries_after_drive=factor_nonzero,
            packed_arrays=len(expected),all_finite=True,chemistry_valid=True,exact_fresh_session_replay=True,
            biology_sha256=H01NeuroglialManifest(biology,topology.to_dict()).sha256,
            elapsed_seconds=time.perf_counter()-started,
            qualification='Synthetic diagnostic assembly, compiler and physical/eligibility replay only; no learning accuracy or physiology qualification')
    output=Path(__file__).parent/'neuroglial-session-phase'
    output.mkdir(exist_ok=True)
    for name,data in [('manifest.json',biology),('session-probe.json',report)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
