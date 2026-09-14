"""Reproduce H01 compressed metadata decoding and preserve wire digests."""

import gzip

from .provenance import decode_metadata


def test_plain_and_stored_gzip_metadata():
    plain = b'{"type":"segmentation"}'
    compressed = gzip.compress(plain)
    a, b = decode_metadata(plain), decode_metadata(compressed)
    assert a['metadata'] == b['metadata'] == {'type': 'segmentation'}
    assert a['decoded_sha256'] == b['decoded_sha256']
    assert a['sha256'] != b['sha256']
