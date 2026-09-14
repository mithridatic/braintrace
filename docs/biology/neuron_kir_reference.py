"""Independent fixed-pool Kir-only cable reference in NEURON 8.2.7."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from neuron import h, load_mechanisms


def main():
    """Run the pinned original mechanism and record physical voltage/current."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanisms', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args()
    if args.build:
        args.mechanisms.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).parent/'reference/Kir4.mod', args.mechanisms/'Kir4.mod')
        subprocess.run([str(Path(sys.executable).parent/'nrnivmodl')], cwd=args.mechanisms,
                       check=True, timeout=120)
    source_hash = hashlib.sha256((args.mechanisms/'Kir4.mod').read_bytes()).hexdigest()
    manifest = json.loads((Path(__file__).parent/'reference/manifest.json').read_text())
    assert source_hash == next(row['sha256'] for row in manifest['files'] if row['local'] == 'Kir4.mod')
    load_mechanisms(str(args.mechanisms))
    h.load_file('stdrun.hoc')
    h.celsius = 34.
    section = h.Section(name='kir_reference')
    section.L, section.diam, section.nseg, section.cm, section.Ra = 1., 2., 1, 1., 100.
    section.insert('kir4')
    segment = section(.5)
    segment.ki, segment.ko, segment.kir4.gkir = 140., 10., .4
    time = h.Vector().record(h._ref_t)
    voltage = h.Vector().record(segment._ref_v)
    current = h.Vector().record(segment._ref_ik)
    h.CVode().active(0)
    h.dt, h.steps_per_ms, h.secondorder = .005, 200., 0
    h.finitialize(-100.)
    reversal = float(segment.ek)
    h.continuerun(.1)
    report = dict(qualification='Fixed-concentration Kir-only cable reference; not coupled K/calcium physiology',
        neuron=h.nrnversion(), source_sha256=source_hash, dt_ms=.005, duration_ms=.1,
        library_sha256=hashlib.sha256((args.mechanisms/'x86_64/libnrnmech.so').read_bytes()).hexdigest(),
        length_um=1., diameter_um=2., cm_uf_cm2=1., ra_ohm_cm=100., initial_mv=-100.,
        temperature_c=34., inside_mm=140., outside_mm=10., reversal_mv=reversal, gkir_ms_cm2=.4,
        time_ms=np.asarray(time).tolist(), voltage_mv=np.asarray(voltage).tolist(),
        outward_current_ma_cm2=np.asarray(current).tolist())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', newline='\n')
    print(json.dumps({key: value for key, value in report.items() if key not in
                     ('time_ms', 'voltage_mv', 'outward_current_ma_cm2')}, indent=2))


if __name__ == '__main__':
    main()
