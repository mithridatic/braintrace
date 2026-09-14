"""Inspect all cells' source/CV observations and selected direct edge projections."""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

root = Path(__file__).resolve().parent
result = json.loads((root/'result.json').read_bytes())
records = result['cells']
fig, axes = plt.subplots(2,3,figsize=(16,9))
positions = np.arange(len(records))
areas, lengths, branch_area_errors = [], [], []
for record in records:
    with np.load(root/(record['cell_id']+'.npz')) as a:
        branches = a['source_branch_area_um2_length_um']
        cvs = a['cv_observations']
        areas.append([branches[:,0].sum(),cvs[:,3].sum()])
        lengths.append([branches[:,1].sum(),cvs[:,4].sum()])
        totals = np.bincount(cvs[:,0].astype(int),weights=cvs[:,3],minlength=len(branches))
        branch_area_errors.append(np.max(np.abs(totals-branches[:,0])/branches[:,0]))
for ax,values,label in zip(axes[0,:2],(areas,lengths),('Lateral area (um2)','Cable length (um)')):
    values = np.array(values)
    ax.plot(positions,values[:,0],'.',color='0.4',label='Source segments')
    ax.plot(positions,values[:,1],'+',color='#0072b2',label='Production CV sum')
    ax.set_xlabel('Cell index in numeric identity order (all 104)')
    ax.set_ylabel(label)
    ax.legend()
axes[0,2].plot(positions,branch_area_errors,'.',color='#0072b2')
axes[0,2].axhline(1e-8,color='#c44e52',label='Registered tolerance')
axes[0,2].set_yscale('symlog',linthresh=1e-16)
axes[0,2].set_ylabel('Largest per-branch relative area discrepancy')
axes[0,2].set_xlabel('Cell index (all 104)')
axes[0,2].legend()
selected = [records[0],max(records,key=lambda r:r['directed_edges']),records[-1]]
view_bounds = []
for ax,record in zip(axes[1],selected):
    with np.load(root/(record['cell_id']+'.npz')) as a:
        for key,color,width,label in [('source_edges_um','0.6',1.2,'Source'),
                                       ('imported_edges_um','#0072b2',.4,'Imported')]:
            edges = a[key]
            segments = np.stack((edges[:,:2],edges[:,3:5]),axis=1)
            ax.add_collection(LineCollection(segments,colors=color,linewidths=width,label=label))
        points = np.concatenate((edges[:,:2],edges[:,3:5]))
        view_bounds.append((points.min(axis=0),points.max(axis=0)))
    ax.autoscale()
    ax.set_aspect('equal',adjustable='datalim')
    ax.set_xlabel('Original X (um)')
    ax.set_ylabel('Original Y (um)')
    ax.set_title(f"{record['cell_id']}.{record['component']}; {record['directed_edges']:,} edges")
    ax.legend(fontsize=8)
span = 1.1*max(float((hi-lo).max()) for lo,hi in view_bounds)
for ax,(lo,hi) in zip(axes[1],view_bounds):
    center = (lo+hi)/2
    ax.set_xlim(center[0]-span/2,center[0]+span/2)
    ax.set_ylim(center[1]-span/2,center[1]+span/2)
    ax.set_aspect('equal',adjustable='box')
fig.suptitle('Source edges and compartment geometry conserved in all 104 human H01 cells\nStrict historical-region comparison failed for one cell; no physiology qualification.\nDirect projections: first, largest edge inventory, and last cell, at a common spatial scale.')
fig.tight_layout(rect=(0,0,1,.94))
fig.savefig(root/'geometry-comparison.png',dpi=150)
plt.close(fig)
