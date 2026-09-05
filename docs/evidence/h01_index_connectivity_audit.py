"""Check candidate C3 synapse endpoints against the H01 proofread volume.

Candidate query IDs are search keys, not proof of a cell identity. No model
edges are created by this diagnostic.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

ANNOTATIONS = 'https://storage.googleapis.com/h01-release/data/20210729/c3/synapses/precomputed'
VOLUME = 'https://storage.googleapis.com/h01-release/data/20210601/proofread_104'


def decode_related(data, query_id):
    """Decode the release's LINE plus uint32 related-object records.

    Parameters
    ----------
    data : bytes or None
        Decompressed related-object index payload.
    query_id : int
        C3 search key. This is not a validated proofread-cell identity.

    Returns
    -------
    list of dict
        Annotation IDs, endpoint voxels, and released type codes.
    """
    if data is None:
        return []
    if len(data) < 8:
        raise ValueError('Truncated annotation count.')
    count = struct.unpack_from('<Q', data)[0]
    if len(data) != 8+36*count:
        raise ValueError('Annotation payload does not match the release schema.')
    records = []
    for index in range(count):
        values = struct.unpack_from('<6fI', data, 8+28*index)
        identity = struct.unpack_from('<Q', data, 8+28*count+8*index)[0]
        records.append(dict(query_c3_id=str(query_id), annotation_id=str(identity),
                            pre_voxel=list(values[:3]), post_voxel=list(values[3:6]),
                            type=values[6]))
    return records


def main():
    """Query candidate axon keys or check a saved candidate list spatially.

    Returns
    -------
    None
        Write source-backed diagnostic records with explicit search limits.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.cache/h01'))
    parser.add_argument('--check', type=Path, help='Saved candidate JSON to check at both endpoints.')
    args = parser.parse_args()
    os.environ.setdefault('CLOUD_FILES_DIR', str(args.cache.resolve()/'cloudfiles'))
    from cloudvolume import CloudVolume
    volume = CloudVolume('precomputed://'+VOLUME, cache=False, fill_missing=True, progress=False)
    if args.check:
        raw = args.check.read_bytes()
        records = json.loads(raw)
        for offset in range(0, len(records), 100):
            batch = records[offset:offset+100]
            points = [tuple(int(v) for v in row[side+'_voxel'])
                      for row in batch for side in ('pre', 'post')]
            labels = volume.scattered_points(points)
            for row in batch:
                for side in ('pre', 'post'):
                    point = tuple(int(v) for v in row[side+'_voxel'])
                    row[side+'_proofread_label'] = str(int(labels[point]))
            print('checked', min(offset+100, len(records)), flush=True)
        between = [r for r in records if r['pre_proofread_label'] != '0'
                   and r['post_proofread_label'] != '0'
                   and r['pre_proofread_label'] != r['post_proofread_label']]
        report = dict(annotation_url=ANNOTATIONS, volume_url=VOLUME,
            candidate_file=str(args.check), candidate_sha256=hashlib.sha256(raw).hexdigest(),
            resolution_nm=[8, 8, 33], records=records, between_cell_candidates=between,
            scope='Partial candidate search. No claim of absent connections. No circuit edges created.',
            missing_chunks='Background zero; cannot establish endpoint identity.',
            verification='Source detection and endpoint segmentation; manual synapse review not established.')
        output = Path('docs/evidence/h01-axon-index-endpoint-audit.json')
        output.write_text(json.dumps(report, indent=2)+'\n')
        print('between-cell candidates', len(between), 'saved', output, flush=True)
        return
    from cloudvolume.cacheservice import CacheService
    from cloudvolume.datasource.precomputed.sharding import ShardReader, ShardingSpecification
    info = json.loads((args.cache/'c3-annotation-info.json').read_text())
    if info['annotation_type'] != 'LINE' or info['properties'] != [{'id': 'type', 'type': 'uint32'}]:
        raise ValueError('Unsupported annotation schema.')
    cells = json.loads((args.cache/'proofread-base-membership.json').read_text())
    identities = sorted({int(b) for cell in cells for b in cell['regions'].get('axon', [])})
    cache = CacheService(ANNOTATIONS, False, volume.config,
                         meta=SimpleNamespace(cloudpath=ANNOTATIONS))
    relation = next(x for x in info['relationships'] if x['id'] == 'pre_synaptic_cell')
    reader = ShardReader(ANNOTATIONS, cache, ShardingSpecification.from_dict(relation['sharding']))
    records = []
    for offset in range(0, len(identities), 1000):
        payloads = reader.get_data(identities[offset:offset+1000], path=relation['key'])
        for identity, data in payloads.items():
            records.extend(decode_related(data, identity))
        print('queried', min(offset+1000, len(identities)), 'records', len(records), flush=True)
    (args.cache/'c3-axon-index-candidates.json').write_text(json.dumps(records))


if __name__ == '__main__':
    main()
