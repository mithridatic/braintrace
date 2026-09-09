"""Geometry invariants for the H01 attachment specialization."""

import numpy as np
import pytest

from .h01_test import _archive
from .h01 import H01Archive
from docs.evidence.h01_population_import_audit import audit_geometry


@pytest.mark.parametrize('offset', [0, 100000])
@pytest.mark.parametrize('shape', ['leaf', 'fork', 'chain'])
def test_directed_segments_survive_translation_and_branching(tmp_path, monkeypatch, offset, shape):
    rows = [[0, 0, 0, 0, 0, 100, -1], [1, 0, 100, 0, 0, 100, 0],
            [2, 0, 101, 0, 0, 100, 1], [3, 0, 100, 100, 0, 100, 1]]
    if shape == 'fork':
        rows.extend([[4, 0, 201, 0, 0, 100, 2], [5, 0, 101, 100, 0, 100, 2]])
    elif shape == 'chain':
        rows.append([4, 0, 201, 0, 0, 100, 2])
    rows = np.array(rows)
    rows[:, 2:5] += offset
    source = ('\n'.join(' '.join(map(str, row)) for row in rows)+'\n').encode()
    path, _ = _archive(tmp_path, monkeypatch, [('12.0.swc', source)])
    result = audit_geometry(H01Archive(path).load(12, component=0))
    assert result['passed']
    assert result['source_segments'] == len(rows)-1
