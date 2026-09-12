"""Bounded measured-fragment calcium/K geometry and physical replay probe.

Run from the checkout root: ``python -m docs.biology.tapered_chemistry_probe``.
Extracellular pools and the initial calcium pulse are diagnostic assumptions.
"""

import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import tempfile
import time

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_glia import H01GlialSelection, load_glial_fragment
from braintrace.datasets.h01_glial_cable import build_glial_cable
from braintrace.datasets.h01_network_step import H01NetworkStep
from braintrace.biophysics.cable_geometry import CableChemicalGeometry
from braintrace.biophysics.astrocyte_network import AstrocyteCalcium
from braintrace.biophysics.cable_potassium import CablePotassiumBinding, PotassiumCoupling
from braintrace.biophysics.environment import ChemicalEnvironment, MembraneMap
from braintrace.biophysics.transport import DiffusionGraph
from examples.pp_prop.h01_physical_checkpoint import _pack, save_physical_checkpoint, restore_physical_checkpoint


def main():
    """Record exact physical maps, conservative ledgers and snapshot replay."""
    started=time.perf_counter()
    folder=Path(__file__).parent
    assembly=json.loads((folder/'glia-assembly-phase/assembly.json').read_text())
    selection=H01GlialSelection(assembly['fragment']['selection'])
    with brainstate.environ.context(precision=64):
        def construct():
            fragment=load_glial_fragment('.cache/h01-biology/astrocyte-63900941936.npz',selection)
            cell,record=build_glial_cable(fragment,assembly['electrical'])
            network=braincell.Network()
            network.add_population('glia',cell)
            step=H01NetworkStep(network)
            geometry=CableChemicalGeometry(cell)
            n=cell.n_cv
            transport=DiffusionGraph(10*geometry.volume_um3,np.empty((0,2),int),[],dt_ms=.005)
            environment=ChemicalEnvironment(transport,transport,MembraneMap(np.arange(n),geometry.volume_um3,n),
                np.full(n,140.),outside_potassium_mm=10.)
            binding=CablePotassiumBinding(cell,environment)
            step.bind_chemistry(PotassiumCoupling(environment,[binding]))
            calcium=AstrocyteCalcium(geometry=geometry)
            calcium.state.value=calcium.state.value.at[0,0].set(.001)
            return dict(step=step,calcium=calcium),geometry,record
        roots,geometry,record=construct()
        def evolve(owners,count):
            def advance(_):
                owners['step'].update(sample_probes=False)
                owners['calcium'].update(jnp.zeros(len(geometry.volume_um3)))
                return owners['calcium'].valid.value
            return brainstate.transform.for_loop(advance,jnp.arange(count))
        def amounts(owners):
            env=owners['step'].chemistry.environment
            calcium=owners['calcium']
            state=calcium.state.value
            volumes=calcium.calcium_transport.volumes.reshape(-1,4)
            return (float(jnp.sum(env.inside_potassium.value*env.membranes.inside_volumes)+
                          jnp.sum(env.potassium.value*env.k_transport.volumes)),
                    float(jnp.sum(volumes*(state[:,:4]+state[:,8:12]+state[:,16:20]))))
        before=amounts(roots)
        assert np.all(evolve(roots,1))
        manifest=dict(assembly_sha256=record['assembly_sha256'],chemical_geometry_sha256=geometry.sha256,
            dt_ms=.005,calcium_substeps=4,outside_volume_ratio=10.,outside_transport='isolated finite diagnostic pools',
            calcium_parameters=asdict(roots['calcium'].parameters),
            dependencies={name:version(name) for name in ('braincell','brainstate','brainunit','jax','jaxlib','numpy')},
            source_sha256=selection.to_dict()['source_sha256'],
            initial_calcium_pulse=dict(cv=0,shell=0,concentration_mm=.001))
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'physical.npz'
            digest=save_physical_checkpoint(path,roots=roots,optimizer={},manifest=manifest)
            assert np.all(evolve(roots,2))
            expected,_,_=_pack(roots,{})
            fresh,fresh_geometry,fresh_record=construct()
            assert fresh_geometry.sha256==geometry.sha256 and fresh_record==record
            restore_physical_checkpoint(path,roots=fresh,optimizer={},manifest=manifest,expected_sha256=digest)
            assert np.all(evolve(fresh,2))
            restored,_,_=_pack(fresh,{})
            assert expected.keys()==restored.keys()
            assert all(np.array_equal(expected[key],restored[key]) for key in expected)
        after=amounts(roots)
        calcium=roots['calcium']
        env=roots['step'].chemistry.environment
        ca_error=after[1]-before[1]-float(calcium.er_amount.value-calcium.pumped_amount.value)
        assert env.valid.value and abs(after[0]-before[0]) < 1e-8 and abs(ca_error) < 1e-10
        roots['step'].reset_state()
        calcium.reset_state()
        calcium.state.value=calcium.state.value.at[0,0].set(.001)
        assert np.all(evolve(roots,3))
        reset,_,_=_pack(roots,{})
        assert all(np.array_equal(expected[key],reset[key]) for key in expected)
        report=dict(status='MEASURED_TAPERED_CHEMISTRY_PASS',compartments=len(geometry.volume_um3),
            duration_ms=.015,steps=3,dt_ms=.005,total_inside_volume_um3=float(geometry.volume_um3.sum()),
            calcium_shell_k_volume_max_error_um3=float(np.max(np.abs(
                np.asarray(calcium.calcium_transport.volumes).reshape(-1,4).sum(axis=1)-geometry.volume_um3))),
            potassium_balance_error_mm_um3=after[0]-before[0],calcium_balance_error_mm_um3=ca_error,
            all_valid=True,exact_fresh_checkpoint_replay=True,exact_reset_replay=True,
            glial_inside_k_range_mm=[float(env.inside_potassium.value.min()),float(env.inside_potassium.value.max())],
            elapsed_seconds=time.perf_counter()-started,manifest=manifest,
            limitations=['Homothetic finite-volume taper extension, not 3-D or source RxD equivalence',
                'Diagnostic isolated extracellular pools and imposed initial calcium pulse',
                'No neuronal release, measured extracellular anatomy, long wave or session-factory qualification'])
    output=folder/'tapered-chemistry-phase'
    output.mkdir(exist_ok=True)
    for name,data in [('geometry.json',geometry.to_dict()),('measured-probe.json',report)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
