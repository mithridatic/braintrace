"""Preserve source protocols and prepare exact human AP-command observations."""

from pathlib import Path
import gzip
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from scipy.signal import find_peaks

root=Path(__file__).resolve().parent
repo=root.parents[3]
sys.path.insert(0,str(root.parents[1]))
from h01_ap_command import command_window,human_hwide_records

cache=repo/'.cache/human-source-followup/wilbers-protocol'
previous=root.parent/'human-pyramidal-channels'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,v:p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
if (root/'acquisition.json').exists():raise FileExistsError('Preserve existing source receipt.')
metadata=json.loads((previous/'metadata-v3.json').read_bytes())['data']
files=[]
for entry in metadata['files']:
    d=entry['dataFile']
    if d['id'] not in (318541,318542,318544,318538):continue
    name=str(d['id'])+'-'+d['filename'];source=cache/name
    raw=source.read_bytes()
    assert len(raw)==d['filesize']
    assert hashlib.sha1(raw).hexdigest()==d['checksum']['value']
    if d['id']==318541:
        saved=name+'.gz';(root/saved).write_bytes(gzip.compress(raw,mtime=0))
        assert gzip.decompress((root/saved).read_bytes())==raw
    else:
        saved=name;shutil.copyfile(source,root/saved)
    files.append(dict(id=d['id'],source_file=name,stored_file=saved,source_bytes=len(raw),
        published_sha1=d['checksum']['value'],source_sha256=sha(source),stored_sha256=sha(root/saved),
        url='https://dataverse.nl/api/access/datafile/'+str(d['id'])))
assert len(files)==4
epmc=[]
for name in ('paper.xml','sciadv.ade3300_sm.pdf','sciadv.ade3300-f4.jpg','supplement-page-7.png','supplement-page-12.png'):
    shutil.copyfile(cache/name,root/name)
    epmc.append(dict(file=name,bytes=(root/name).stat().st_size,sha256=sha(root/name)))

v=np.loadtxt(cache/'318541-H_wide.csv',delimiter=',',skiprows=1)
peaks,_=find_peaks(v,prominence=50)
five=command_window(v,199,307)
twohundred=command_window(v,0,5182)
five_peaks=peaks[(peaks>=five['first_index'])&(peaks<=five['last_index'])]
train_peaks=peaks[peaks<=twohundred['last_index']]
assert len(five_peaks)==5 and len(train_peaks)==200 and len(peaks)==240
np.savez_compressed(root/'commands.npz',five_time_ms=five['time_ms'],
    five_original_mv=five['voltage_mv'],five_sodium_mv=five['voltage_mv']-10,
    train_time_ms=twohundred['time_ms'],train_original_mv=twohundred['voltage_mv'],
    train_sodium_mv=twohundred['voltage_mv']-10,full_peak_indices=peaks,
    train_peak_indices=train_peaks,five_peak_indices=five_peaks)
write(root/'command.json',dict(full_samples=len(v),full_duration_ms=(len(v)-1)/125,
    full_peaks=len(peaks),sample_rate_hz=125000,source_dtype=str(v.dtype),
    voltage_min_mv=float(v.min()),voltage_max_mv=float(v.max()),
    five_ap_source_indices=[five['first_index'],five['last_index']],
    prospective_200_ap_source_indices=[twohundred['first_index'],twohundred['last_index']],
    five_ap_count=len(five_peaks),prospective_train_ap_count=len(train_peaks),
    peak_200_time_ms=float(peaks[199]/125),peak_201_time_ms=float(peaks[200]/125),
    gap_after_peak200_ms=float((peaks[200]-peaks[199])/125),sodium_command_shift_mv=-10,
    original_recovery_command_acquired=False,original_measured_currents_acquired=False))

ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
workbook=previous/'318668-Na_data_fig2.xlsx'
with zipfile.ZipFile(workbook) as z:
    strings=[''.join(t.text or '' for t in s.findall('.//s:t',ns)) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    rows=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns)
    decoded=[]
    for row in rows:
        cells={}
        for cell in row:
            value=cell.find('s:v',ns)
            if value is None:continue
            kind=cell.get('t','n')
            if kind=='s':item=strings[int(value.text)]
            elif kind=='b':item=bool(int(value.text))
            elif kind=='n':item=float(value.text)
            else:raise ValueError('Unsupported source cell type '+kind)
            cells[re.sub(r'\d','',cell.get('r'))]=item
        decoded.append((int(row.get('r')),cells))
    headers=decoded[0][1]
    assert len(set(headers.values()))==len(headers)
    records=[dict(excel_row=index,**{name:cells.get(column) for column,name in headers.items()})
             for index,cells in decoded[1:] if cells]
selected,excluded=human_hwide_records(records)
write(root/'human-hwide-responses.json',dict(source_file=workbook.relative_to(repo).as_posix(),
    source_sha256=sha(workbook),source_record_count=len(records),selected=selected,excluded=excluded,
    waveform='Hwide',response_kind='published normalized sodium current amplitude',
    physiological_qc_applied=False,blind_holdout=False))
write(root/'acquisition.json',dict(dataverse_doi='10.34894/L5J0SD',version='3.0',
    metadata_sha256=sha(previous/'metadata-v3.json'),files=files,
    europe_pmc_xml_url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10569700/fullTextXML',
    europe_pmc_supplement_url='https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10569700/supplementaryFiles',
    europe_pmc_zip_sha256=sha(cache/'supplementary.zip'),europe_pmc_files=epmc,
    original_source_scripts_executed=False,simulation_run=False,scores_promoted=False,
    selected_human_response_records=len(selected),excluded_records=len(excluded),
    executed_files_sha256={p.relative_to(repo).as_posix():sha(p) for p in
        (Path(__file__).resolve(),repo/'docs/evidence/h01_ap_command.py')}))
print(json.dumps(dict(source_samples=len(v),full_peaks=len(peaks),selected_human_records=len(selected),excluded_records=len(excluded))))
