"""Acquire an explicitly selected official H01 astrocyte skeleton.

Run with cloud-volume in the isolated reference environment. This downloads
only the selected sharded skeleton and metadata, not the image volume.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from cloudvolume import CloudVolume


def main():
    """Save raw decoded skeleton arrays and acquisition provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = 'https://storage.googleapis.com/h01-release/data/20210601/c3'
    volume = CloudVolume(source, progress=False, cache=False)
    skeleton = volume.skeleton.get(args.identity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # CloudVolume returns physical skeleton vertices/radii in nm for this layer.
    # Keep source units in the artifact; conversion is separately validated.
    np.savez_compressed(args.output, vertices_nm=skeleton.vertices, edges=skeleton.edges,
                        radius_nm=skeleton.radius)
    summary = dict(source=source+'/skeletons', identity=str(args.identity),
        skeleton_metadata=volume.skeleton.meta.info, coordinate_units='nm',
        vertices=len(skeleton.vertices), edges=len(skeleton.edges),
        radius_range_nm=[float(np.min(skeleton.radius)), float(np.max(skeleton.radius))],
        bounds_nm=[np.min(skeleton.vertices, axis=0).tolist(), np.max(skeleton.vertices, axis=0).tolist()],
        artifact_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
        qualification='acquired only; connected-component and coordinate checks pending',
        attribution='H01 Shapson-Coe et al., Science 2024; Harvard Lichtman laboratory and Google; CC BY 4.0')
    args.output.with_suffix('.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
