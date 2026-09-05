"""Check the release-specific annotation decoder against independent bytes."""
import struct
import pytest
from docs.evidence.h01_index_connectivity_audit import decode_related


def test_related_preserves_direction_and_large_ids():
    identity = 2**63+17
    payload = struct.pack('<Q6fIQ', 1, 1., 2., 3., 4., 5., 6., 2, identity)
    assert decode_related(payload, 23) == [dict(query_c3_id='23', annotation_id=str(identity),
        pre_voxel=[1., 2., 3.], post_voxel=[4., 5., 6.], type=2)]


@pytest.mark.parametrize('payload', [b'', b'abc', struct.pack('<Q', 1), struct.pack('<Q', 0)+b'x'])
def test_related_rejects_truncation_and_extra_bytes(payload):
    with pytest.raises(ValueError):
        decode_related(payload, 23)


def test_missing_and_empty_are_not_contacts():
    assert decode_related(None, 23) == []
    assert decode_related(struct.pack('<Q', 0), 23) == []
