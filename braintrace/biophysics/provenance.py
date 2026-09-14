"""Explicit decoding and identity checks for retrieved anatomical metadata."""

import gzip
import hashlib
import json


def decode_metadata(payload):
    """Decode plain or gzip-compressed JSON without changing its wire identity.

    Parameters
    ----------
    payload : bytes
        Retrieved response body; some H01 objects are stored compressed.

    Returns
    -------
    dict
        Wire SHA256, decoded SHA256 and parsed metadata.
    """
    decoded = gzip.decompress(payload) if payload.startswith(b'\x1f\x8b') else payload
    return dict(sha256=hashlib.sha256(payload).hexdigest(),
                decoded_sha256=hashlib.sha256(decoded).hexdigest(), metadata=json.loads(decoded))
