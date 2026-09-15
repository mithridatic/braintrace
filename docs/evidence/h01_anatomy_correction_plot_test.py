"""Ensure the visual comparison retains all edges and uses H01 soma codes."""

import zipfile

import numpy as np
import pytest

import h01_anatomy_correction_plot as plot


def test_projection_retains_edges_physical_units_and_source_soma(tmp_path, monkeypatch):
    source = '1 3 1000 2000 3000 2000 -1\n2 1 1100 2100 3100 40 1\n3 0 1200 2200 3200 32 2\n'
    archive = tmp_path / 'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('7.0.swc', source)
    historical = tmp_path / 'old.swc'
    historical.write_text('1 1 32 64 99 2 -1\n2 3 38.4 70.4 105.6 1 1\n')
    captured = {}
    original = plot.plt.subplots

    def capture(*args, **kwargs):
        fig, axes = original(*args, **kwargs)
        captured['axes'] = axes
        return fig, axes

    monkeypatch.setattr(plot.plt, 'subplots', capture)
    image = tmp_path / 'plot.png'
    args = ['--archive', str(archive), '--historical', str(historical),
            '--member', '7.0.swc', '--output', str(image)]
    plot.main(args)
    axes = captured['axes']
    assert len(axes[0, 0].collections[0].get_segments()) == 2
    assert len(axes[0, 1].collections[0].get_segments()) == 1
    np.testing.assert_allclose(axes[0, 0].collections[1].get_offsets(), [[32, 64]])
    np.testing.assert_allclose(axes[1, 0].collections[1].get_offsets(), [[32, 99]])
    assert axes[0, 0].get_xlim() == axes[0, 1].get_xlim()
    assert axes[1, 0].get_ylim() == axes[1, 1].get_ylim()
    assert image.read_bytes().startswith(b'\x89PNG')
    with pytest.raises(FileExistsError):
        plot.main(args)
