"""Bounded real-source spine construction and one physical H01 event probe.

Run as a module from the worktree with an external timeout of at most 15 minutes.
The added spine is explicitly synthetic; this is not physiological qualification.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from braintrace.biophysics.spines import Spine
from braintrace.datasets.h01 import H01Archive
from braintrace.datasets.h01_biology import topology_digest
from braintrace.datasets.h01_ei_cell import make_h01_ei_cell
from braintrace.datasets.h01_network import _regions
from braintrace.datasets.h01_network_init import init_h01_network_states
from examples.pp_prop.h01_arc_model import H01ArcModel
from examples.pp_prop.h01_runtime import build_network
from examples.pp_prop.h01_topology import H01Topology


def main():
    """Record verified source identity, explicit additions and finite execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--component', type=int, required=True)
    parser.add_argument('--dt-ms', type=float, default=.000625)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    node = next(row for row in json.loads(args.annotations.read_text())['nodes'] if row['cell_id'] == args.cell)
    begin = time.perf_counter()
    archive = H01Archive(args.archive)
    imported = archive.load(args.cell, component=args.component)
    if len(imported.source_rows) > 5000:
        raise ValueError('This construction probe is bounded to 5000 source nodes')
    source_seconds = time.perf_counter()-begin
    with brainstate.environ.context(precision=64, dt=args.dt_ms*u.ms):
        regions = _regions(imported)
        branch, lo, hi = regions['dend'].evaluate(imported.morphology).intervals[0]
        spine = Spine('synthetic_probe_0', int(branch), (lo+hi)/2, .689, .0335, .583, .2915,
            (0., 1., 0.), 'Synthetic centreline addition using paper mean dimensions; not a measured H01 spine')
        annotations = SimpleNamespace(metadata=lambda _: SimpleNamespace(tags=node['tags']))
        _, source_record = make_h01_ei_cell(imported, annotations, polarity=node['polarity'],
            regions=regions, region_basis='Source-labelled H01 electrical regions',
            solver='h01_staggered_calcium_implicit', pop_size=(1,))
        soma = list(imported.anatomy().soma_location().evaluate(imported.morphology).points[0])
        source = dict(component=args.component, source_sha256=imported.source_sha256,
            polarity=node['polarity'], donor=source_record['donor'], profile=source_record['borrowed_dynamics'],
            source_tags=node['tags'], region_basis='Source-labelled H01 electrical regions',
            soma_site=soma, output_site=soma, archive_sha256=imported.provenance['archive_sha256'])
        doc = dict(schema='h01-topology-v1', sources={args.cell: source},
            instances={args.cell: dict(source_id=args.cell, parent_id=None, synthetic=False, created_stage='source')},
            contacts={}, active_cells=[args.cell], active_contacts=[], next_id=0, blocked_contacts=[], mutations=[])
        topology = H01Topology.from_dict(doc)
        biology = dict(schema='h01-biology-spines-v1', topology_sha256=topology_digest(doc),
            coordinate_frame='h01-swc-um', cells={args.cell: dict(source_sha256=imported.source_sha256,
            spines=[dict(asdict(spine), origin='synthetic')])}, contact_heads={}, release_probability={})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.topology.json').write_text(json.dumps(doc, indent=2)+'\n', newline='\n')
        args.output.with_suffix('.biology.json').write_text(json.dumps(biology, indent=2)+'\n', newline='\n')
        begin = time.perf_counter()
        network, records = build_network(topology, archive, biology=biology, progress=lambda x: print(x, flush=True))
        construction = time.perf_counter()-begin
        begin = time.perf_counter()
        init_h01_network_states(network, progress=lambda x: print(x, flush=True))
        initialization = time.perf_counter()-begin
        model = H01ArcModel(network, [args.cell], release_probability=[], dt_ms=args.dt_ms)
        begin = time.perf_counter()
        soma_result = jax.block_until_ready(brainstate.transform.jit(model.update)(jnp.zeros(441)))
        physical = time.perf_counter()-begin
        voltage = np.asarray(model.stepper.cells[0].V.value.to_decimal(u.mV))
        report = dict(status='PASS' if np.isfinite(voltage).all() and np.isfinite(soma_result).all() else 'FAIL',
            qualification='One real source component with one explicitly synthetic spine; one .1 ms event only',
            source_identity=args.cell, source_component=args.component, source_nodes=len(imported.source_rows),
            component_count=len(archive.components(args.cell)),
            annotation_sha256=hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
            source_loading_seconds=source_seconds, construction_seconds=construction,
            initialization_seconds=initialization, compile_and_event_seconds=physical,
            physical_ms=.1, dt_ms=args.dt_ms, cable_ticks=int(model.stepper.tick.value),
            voltage_range_mv=[float(voltage.min()), float(voltage.max())], soma_mv=np.asarray(soma_result).tolist(),
            record=records[args.cell], full_biology=False, physiological_qualification=False)
        args.output.write_text(json.dumps(report, indent=2)+'\n', newline='\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'record'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
