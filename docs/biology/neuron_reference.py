"""Run pinned unmodified BRAINCELL mechanisms in an isolated NEURON process."""

import argparse
import json
from pathlib import Path

import numpy as np
from neuron import h, load_mechanisms


def calcium_reference(mechanisms, *, dt_ms=.005, duration_ms=1., stimulus_mm=.1):
    """Record all local calcium states with diffusion and plasma pumping disabled.

    Parameters
    ----------
    mechanisms : str
        Directory containing independently compiled reference mechanisms.
    dt_ms, duration_ms, stimulus_mm : float, optional
        Integration step, duration and glutamate stimulation.

    Returns
    -------
    dict
        Full original-clock trajectories and exact numerical configuration.
    """
    load_mechanisms(mechanisms)
    h.load_file('stdrun.hoc')
    cell = h.Section(name='reference')
    cell.L, cell.diam, cell.nseg = 10., 2., 1
    cell.insert('cadifus')
    mechanism = cell(.5).cadifus
    parameters = dict(DCa=0., DBufm=0., TBufs=.07, TBufm=.0075, cai0=.00005,
                      Ip3init=.00005, gamma=0., modelStim=stimulus_mm, cath=.00002)
    for name, value in parameters.items():
        if hasattr(mechanism, name):
            setattr(mechanism, name, value)
        else:
            setattr(h, name+'_cadifus', value)
    names = [(name, i) for name in ('ca', 'bufs', 'cabufs', 'bufm', 'cabufm', 'ho') for i in range(4)]
    vectors = [h.Vector().record(getattr(mechanism, '_ref_'+name)[i]) for name, i in names]
    vectors.append(h.Vector().record(cell(.5)._ref_ip3i))
    time = h.Vector().record(h._ref_t)
    h.CVode().active(0)
    h.dt, h.steps_per_ms = dt_ms, 1/dt_ms
    h.finitialize(-80.)
    h.continuerun(duration_ms)
    return dict(schema='braincell-local-calcium-reference-v1', neuron=h.nrnversion(),
                dt_ms=dt_ms, duration_ms=duration_ms, parameters=parameters,
                time_ms=np.array(time).tolist(), state=np.stack([np.array(v) for v in vectors], axis=-1).tolist())


def main():
    """Export one independent local-mechanism comparison fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mechanisms', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--spatial', action='store_true')
    args = parser.parse_args()
    reference = spatial_calcium_reference(args.mechanisms) if args.spatial else calcium_reference(args.mechanisms)
    args.output.write_text(json.dumps(reference, indent=2)+'\n')


def spatial_calcium_reference(mechanisms):
    """Record radial and longitudinal calcium spread in three 1 um segments."""
    load_mechanisms(mechanisms)
    h.load_file('stdrun.hoc')
    cell = h.Section(name='spatial_reference')
    cell.L, cell.diam, cell.nseg = 3., 2., 3
    cell.insert('cadifus')
    parameters = dict(DCa=.3, DBufm=.05, TBufs=.07, TBufm=.0075, cai0=.00005,
                      Ip3init=.00005, gamma=0., modelStim=0., cath=.00002, alpha=0.)
    for segment in cell:
        for name, value in parameters.items():
            if hasattr(segment.cadifus, name):
                setattr(segment.cadifus, name, value)
            else:
                setattr(h, name+'_cadifus', value)
    names = [(name, i) for name in ('ca', 'bufs', 'cabufs', 'bufm', 'cabufm', 'ho') for i in range(4)]
    vectors = [[h.Vector().record(getattr(segment.cadifus, '_ref_'+name)[i]) for name, i in names]
               + [h.Vector().record(segment._ref_ip3i)] for segment in cell]
    time = h.Vector().record(h._ref_t)
    h.CVode().active(0)
    h.dt, h.steps_per_ms = .005, 200.
    h.finitialize(-80.)
    cell(1/6).cadifus.ca[0] = .001
    h.fcurrent()
    h.frecord_init()
    h.continuerun(.2)
    readback = {name: float(getattr(cell(.5).cadifus, name) if hasattr(cell(.5).cadifus, name)
                            else getattr(h, name+'_cadifus')) for name in parameters}
    print('Spatial reference parameter readback:', readback)
    return dict(schema='braincell-spatial-calcium-reference-v1', neuron=h.nrnversion(),
        dt_ms=.005, duration_ms=.2, parameters=parameters, parameter_readback=readback,
        time_ms=np.array(time).tolist(),
        state=np.stack([np.stack([np.array(v) for v in row], axis=-1) for row in vectors], axis=1).tolist())


if __name__ == '__main__':
    main()
