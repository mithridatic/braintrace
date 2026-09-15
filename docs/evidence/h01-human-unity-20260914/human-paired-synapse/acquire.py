"""Acquire the prespecified human pair and verify the published checksums."""

from pathlib import Path
import hashlib
import json
import time
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
CACHE = ROOT / '.cache' / 'human-paired-synapse'
CACHE.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
total = 0


def fetch(url):
    """Read a bounded source payload under the shared byte and time limits.

    Parameters
    ----------
    url : str
        Public source endpoint.

    Returns
    -------
    bytes
        Unmodified response payload.
    """
    global total
    pieces = []
    with urllib.request.urlopen(url, timeout=20) as response:
        while True:
            if time.monotonic() - started > 120:
                raise TimeoutError('Acquisition exceeded the 120-second cap.')
            part = response.read(1024 * 1024)
            if not part:
                break
            total += len(part)
            if total > 40_000_000:
                raise ValueError('Acquisition exceeded the 40 MB payload cap.')
            pieces.append(part)
    return b''.join(pieces)


metadata_url = ('https://dataverse.nl/api/datasets/:persistentId/'
                '?persistentId=doi:10.34894/TONOHA')
metadata = fetch(metadata_url)
(OUT / 'release.json').write_bytes(metadata)
version = json.loads(metadata)['data']['latestVersion']
assert (version['versionNumber'], version['versionMinorNumber'],
        version['versionState']) == (1, 0, 'RELEASED')
entries = {f['dataFile']['id']: f for f in version['files']}
records = []
for file_id in [359666, 359652]:
    entry = entries[file_id]
    source = entry['dataFile']
    url = f'https://dataverse.nl/api/access/datafile/{file_id}'
    data = fetch(url)
    checksum = source['checksum']
    assert len(data) == source['filesize']
    assert hashlib.new(checksum['type'].lower().replace('-', ''), data).hexdigest() == checksum['value']
    target = (CACHE if file_id == 359652 else OUT) / source['filename']
    target.write_bytes(data)
    records.append(dict(id=file_id, url=url, directory=entry.get('directoryLabel'),
                        path=str(target.relative_to(ROOT)), bytes=len(data),
                        sha256=hashlib.sha256(data).hexdigest(), checksum=checksum))
paper_url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10569701/fullTextXML'
paper = fetch(paper_url)
(OUT / 'paper.xml').write_bytes(paper)
receipt = dict(metadata_url=metadata_url,
               metadata_sha256=hashlib.sha256(metadata).hexdigest(),
               paper_url=paper_url, paper_sha256=hashlib.sha256(paper).hexdigest(),
               files=records, total_bytes=total, elapsed_seconds=time.monotonic()-started,
               scores_promoted=False)
(OUT / 'acquisition.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
