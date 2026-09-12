"""Bounded measured glial-fragment construction and deterministic cable replay.

Run from the checkout root with ``python -m docs.biology.glial_fragment_probe``.
The default asset is the already acquired skeleton; this script never downloads.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import braincell
import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np

from braintrace.datasets.h01_glia import H01GlialSelection, load_glial_fragment
from braintrace.datasets.h01_glial_cable import build_glial_cable
from braintrace.datasets.h01_network_step import H01NetworkStep


def main():
    """Record complete selected-source geometry and a short fixed-pool run."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('.cache/h01-biology/astrocyte-63900941936.npz'))
    args=parser.parse_args()
    started=time.perf_counter()
    folder=Path(__file__).parent
    audit_path=folder/'spine-phase/spatial-frame-audit.json'
    audit=json.loads(audit_path.read_text())
    metadata=audit['glia_skeleton']
    selection=H01GlialSelection(dict(schema='h01-glial-fragment-v1',source_sha256=metadata['artifact_sha256'],
        identity=metadata['identity'],source=metadata['source'],attribution=metadata['attribution'],
        coordinate_frame='h01-c3-nm',anchor_vertex=88562,max_vertices=600,
        selection_basis='Entire component containing vertex 88562 from the pinned prior proximity audit; local diagnostic selection, not biological connectivity'))
    electrical=dict(initial_mv=-100.,inside_k_mm=140.,outside_k_mm=10.,temperature_c=34.,
        gkir_ms_cm2=.4,cm_uf_cm2=1.,ra_ohm_cm=100.,max_cv_length_um=10.,
        basis='Source-paper Kir conductance with diagnostic fixed K, voltage, capacitance and resistivity; sealed fragment, no fitted physiology')
    with brainstate.environ.context(precision=64):
        def construct():
            fragment=load_glial_fragment(args.source,selection)
            cell,record=build_glial_cable(fragment,electrical)
            network=braincell.Network()
            network.add_population('measured_glial_fragment',cell)
            return H01NetworkStep(network),cell,record
        step,cell,record=construct()
        def trajectory(driver,cable):
            def advance(_):
                driver.update(sample_probes=False)
                return cable.V.value.to_decimal(u.mV)
            return brainstate.transform.for_loop(advance,jnp.arange(20))
        voltage=np.asarray(trajectory(step,cell))
        fresh,fresh_cell,fresh_record=construct()
        replay=np.asarray(trajectory(fresh,fresh_cell))
        assert np.isfinite(voltage).all() and np.any(voltage[-1] != electrical['initial_mv'])
        assert record==fresh_record and np.array_equal(voltage,replay)
        report=dict(status='MEASURED_FRAGMENT_CABLE_PASS',identity=metadata['identity'],
            compartments=cell.n_cv,branches=len(record['fragment']['branch_paths']),
            selected_source_vertices=len(record['fragment']['selected_vertices']),
            selected_source_edges=len(record['fragment']['selected_edges']),
            excluded_components=len(record['fragment']['excluded_components']),
            excluded_vertices=sum(c['vertices'] for c in record['fragment']['excluded_components']),
            dt_ms=.005,steps=20,duration_ms=.1,all_finite=True,exact_fresh_reconstruction_replay=True,
            final_voltage_range_mv=[float(voltage[-1].min()),float(voltage[-1].max())],
            elapsed_seconds=time.perf_counter()-started,
            spatial_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
            nearest_neuron_evidence=audit['selected_neuron_proximity'][0],
            assembly_sha256=record['assembly_sha256'],
            limitations=['Component selected for a local diagnostic, not a complete astrocyte',
                        'Proximity does not establish membrane contact or synaptic connectivity',
                        'Fixed K pools; no coupled uptake, tapered calcium mapping or H01 session factory'])
    output=folder/'glia-assembly-phase'
    output.mkdir(exist_ok=True)
    for name,data in [('selection.json',selection.to_dict()),('assembly.json',record),('measured-probe.json',report)]:
        (output/name).write_text(json.dumps(data,indent=2)+'\n',newline='\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
