"""Extract source membership and identity witnesses from the H01 paper archive."""
import hashlib
import json
from pathlib import Path
import zipfile

ARCHIVE_SHA256 = '364195428726bb09e1674323ffb66d280acf9de28f866b9e21bd2f86f0914881'
SOURCE_URL = ('https://storage.googleapis.com/h01_paper_public_files/'
              'proofread104_neurons_20210511.zip')


def extract(cache):
    """Extract membership from a checksum-verified, already downloaded archive.

    Parameters
    ----------
    cache : pathlib.Path
        Directory containing the published proofreading ZIP.

    Returns
    -------
    pathlib.Path
        Compact source membership file, suitable for the connectivity audit.
    """
    cache = Path(cache)
    archive = cache/'proofread104_neurons_20210511.zip'
    with archive.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != ARCHIVE_SHA256:
            raise ValueError('Proofreading archive checksum mismatch.')
    cells = []
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if not name.endswith('.json'):
                continue
            raw = source.read(name)
            data = json.loads(raw)
            regions, locations = data['base_segments'], data['base_locations']
            anchor = str(data.get('anchor_seg', data['metadata']['main_seg']['base']))
            ordered = list(dict.fromkeys([anchor]+[str(x)
                for region in ('cell_body', 'dendrite', 'axon')
                for x in regions.get(region, [])]))
            witnesses = [dict(base_id=x, location_nm=locations[x])
                         for x in ordered if x in locations][:5]
            cells.append(dict(file=name, sha256=hashlib.sha256(raw).hexdigest(),
                metadata=data['metadata'], regions=regions, witnesses=witnesses,
                merge_points=data['base_seg_merge_points']))
    output = cache/'proofread-base-membership.json'
    output.write_text(json.dumps(cells))
    return output


if __name__ == '__main__':
    print(extract(Path('.cache/h01')))
