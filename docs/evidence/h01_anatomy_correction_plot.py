"""Render all source and historical SWC edges at common physical scales."""

import argparse
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from h01_anatomy_area_audit import read_swc


def main(argv=None):
    """Save orthogonal projections of the measured conversion defect.

    Parameters
    ----------
    argv : list of str or None, optional
        Paths for the archived source, retained bad export and new image.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--historical', type=Path, required=True)
    parser.add_argument('--member', default='955432427.0.swc')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Preserve old images; use a new output.')
    with zipfile.ZipFile(args.archive) as archive:
        source = read_swc(archive.read(args.member).decode())
    source[:, 2:5] *= (.032, .032, .033)
    historical = read_swc(args.historical.read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for column, (rows, title) in enumerate(((source, 'Source / corrected topology'),
                                          (historical, 'Historical malformed topology'))):
        ids = {int(row[0]): i for i, row in enumerate(rows)}
        child = np.flatnonzero(rows[:, 6] != -1)
        parent = np.array([ids[int(p)] for p in rows[child, 6]])
        points = rows[:, 2:5]
        for projection, dims in enumerate(((0, 1), (0, 2))):
            ax = axes[projection, column]
            segments = np.stack((points[parent][:, dims], points[child][:, dims]), axis=1)
            ax.add_collection(LineCollection(segments, colors='#355a78', linewidths=.35, alpha=.65))
            if column == 0:
                soma = rows[:, 1] == 3
                ax.scatter(points[soma, dims[0]], points[soma, dims[1]], s=18,
                           color='#c04428', zorder=5, label='H01 soma samples (code 3)')
            else:
                soma = rows[:, 1] == 1
                ax.scatter(points[soma, dims[0]], points[soma, dims[1]], s=45, marker='x',
                           color='#c04428', zorder=5, label='Inserted donor soma')
            ax.set_xlim(source[:, 2].min()-5, source[:, 2].max()+5)
            ax.set_ylim(source[:, 2+dims[1]].min()-5, source[:, 2+dims[1]].max()+5)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('X (um)')
            ax.set_ylabel(('Y' if projection == 0 else 'Z') + ' (um)')
            ax.set_title(title)
            ax.legend(loc='best', fontsize=8)
    fig.suptitle('H01 955432427: dendrite annotations collapsed into a false soma\n'
                 'All edges retained in these views; geometry evidence, not physiology', fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()
