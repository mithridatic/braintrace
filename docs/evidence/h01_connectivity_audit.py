"""Audit candidate H01 contacts through proofreading base-segment membership.

Run with the isolated h01-inspect Python. This does not modify model defaults.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path


def resolve_identity(witnesses, known_ids):
    """Resolve a cell from consistent nonzero volume labels.

    Parameters
    ----------
    witnesses : list of dict
        Sampled voxel positions and their segmentation labels.
    known_ids : set of str
        Released morphology IDs.

    Returns
    -------
    str or None
        Supported identity, or None when evidence is insufficient or conflicts.
    """
    nonzero = [w['label'] for w in witnesses if w['label'] != '0']
    locations = {tuple(w['voxel']) for w in witnesses if w['label'] != '0'}
    if len(locations) >= 3 and len(set(nonzero)) == 1 and nonzero[0] in known_ids:
        return nonzero[0]
    return None


def main():
    """Read source membership and audit a local export shard.

    Returns
    -------
    None
        Save identity witnesses and candidate contacts, including exclusions.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('.cache/h01'))
    parser.add_argument('--shard', default='export000000000000')
    parser.add_argument('--crosswalk', type=Path,
                        help='Reuse a completed identity audit with the same membership hash.')
    args = parser.parse_args()
    os.environ.setdefault('CLOUD_FILES_DIR', str(args.cache.resolve()/'cloudfiles'))
    from cloudvolume import CloudVolume
    from fastavro import reader

    cells = json.loads((args.cache/'proofread-base-membership.json').read_text())
    volume_url = 'https://storage.googleapis.com/h01-release/data/20210601/proofread_104'
    volume = None if args.crosswalk else CloudVolume(
        'precomputed://'+volume_url, mip=0, progress=False, fill_missing=True, cache=False)
    membership_sha256 = hashlib.sha256(
        (args.cache/'proofread-base-membership.json').read_bytes()).hexdigest()
    reused = None
    if args.crosswalk:
        prior = json.loads(args.crosswalk.read_text())
        if prior.get('membership_sha256') != membership_sha256 or prior['volume'] != volume_url:
            raise ValueError('Crosswalk source mismatch.')
        reused = prior['crosswalk']
        if [(x['file'], x['sha256']) for x in reused] != [(x['file'], x['sha256']) for x in cells]:
            raise ValueError('Crosswalk cell order or membership source mismatch.')
    known_ids = set(json.loads((args.cache/'cell_properties.json').read_text())['inline']['ids'])
    crosswalk = []
    owners = collections.defaultdict(list)
    for index, cell in enumerate(cells):
        points = [tuple(int(v//s) for v, s in zip(w['location_nm'], (8, 8, 33)))
                  for w in cell['witnesses']]
        labels = ({tuple(w['voxel']): int(w['label']) for w in reused[index]['witnesses']}
                  if reused else volume.scattered_points(points))
        witnesses = [dict(w, voxel=list(p), label=str(int(labels[p])))
                     for w, p in zip(cell['witnesses'], points)]
        target = resolve_identity(witnesses, known_ids)
        crosswalk.append(dict(file=cell['file'], sha256=cell['sha256'],
                              released_id=target, witnesses=witnesses,
                              merge_cut_count=len(cell['merge_points'])))
        for region, segments in cell['regions'].items():
            for segment in segments:
                owners[str(segment)].append((index, region))
        if not reused:
            print('identity', index+1, target, flush=True)
    identities = collections.Counter(x['released_id'] for x in crosswalk if x['released_id'])
    report = dict(volume=volume_url, coordinate_resolution_nm=[8, 8, 33],
                  membership_sha256=membership_sha256,
                  crosswalk=crosswalk, contacts=[], counts={},
                  missing_chunks='Read as background zero; zero cannot establish cell identity.',
                  scope='One export shard. Candidate contacts require endpoint validation before use.',
                  source_synapse_verification='Released detection; manual review not established.')
    output = Path('docs/evidence/h01-connectivity-'+args.shard+'-audit.json')
    output.write_text(json.dumps(report, indent=2)+'\n')
    counts = collections.Counter()
    shard = args.cache/(args.shard+'.avro')
    report['export_url'] = ('https://storage.googleapis.com/h01-release/'
                           'data/20210729/c3/synapses/exported/'+args.shard)
    report['export_sha256'] = hashlib.sha256(shard.read_bytes()).hexdigest()
    with shard.open('rb') as stream:
        for row_index, synapse in enumerate(reader(stream)):
            counts['records'] += 1
            pre = owners.get(str(synapse['pre_synaptic_site']['base_neuron_id']), [])
            post = owners.get(str(synapse['post_synaptic_partner']['base_neuron_id']), [])
            if not pre or not post:
                continue
            counts['both_in_archive'] += 1
            if len(pre) != 1 or len(post) != 1:
                counts['ambiguous_base_membership'] += 1
                continue
            a, b = crosswalk[pre[0][0]], crosswalk[post[0][0]]
            exclusions = []
            if not a['released_id'] or not b['released_id']:
                exclusions.append('unresolved_identity')
            elif identities[a['released_id']] != 1 or identities[b['released_id']] != 1:
                exclusions.append('nonunique_identity')
            if a['merge_cut_count'] or b['merge_cut_count']:
                exclusions.append('cell_has_base_merge_cuts')
            if pre[0][1] != 'axon':
                exclusions.append('presynaptic_base_not_axon')
            report['contacts'].append(dict(record_index=row_index,
                pre_released_id=a['released_id'], post_released_id=b['released_id'],
                pre_file=a['file'], post_file=b['file'], pre_region=pre[0][1],
                post_region=post[0][1], exclusions=exclusions, source=synapse))
    report['counts'] = dict(counts)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(dict(counts), 'saved', output, flush=True)


if __name__ == '__main__':
    main()
