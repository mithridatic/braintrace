"""Preserve release files and select human rows from the measured-kinetics tables."""

from pathlib import Path
import hashlib
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
import zipfile

root = Path(__file__).resolve().parent
repository = root.parents[3]
sys.path.insert(0,str(root.parents[1]))
from h01_human_channel_source import select_human_records

cache = repository/'.cache/human-source-followup/wilbers-pyramidal'
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write = lambda p,obj:p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8',newline='\n')
if (root/'acquisition.json').exists():raise FileExistsError('Evidence prefix already prepared.')
metadata = json.loads((cache/'metadata-v3.json').read_bytes())['data']
assert metadata['versionNumber']==3 and metadata['versionMinorNumber']==0
ids={318666,318667,318668,318669,326241,326242,326238,326237,318530,318533,318535,318536}
files=[]
for entry in metadata['files']:
    d=entry['dataFile']
    if d['id'] not in ids:continue
    name=str(d['id'])+'-'+d['filename'];source=cache/name
    assert source.stat().st_size==d['filesize']
    assert hashlib.sha1(source.read_bytes()).hexdigest()==d['checksum']['value']
    shutil.copyfile(source,root/name)
    files.append(dict(id=d['id'],path=entry.get('directoryLabel','')+'/'+d['filename'],
        local_file=name,bytes=d['filesize'],published_sha1=d['checksum']['value'],sha256=sha(root/name),
        url='https://dataverse.nl/api/access/datafile/'+str(d['id'])))
assert len(files)==len(ids)
shutil.copyfile(cache/'metadata-v3.json',root/'metadata-v3.json')
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def _read_workbook(path):
    with zipfile.ZipFile(path) as archive:
        shared=[''.join(t.text or '' for t in s.findall('.//s:t',ns))
                for s in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        sheets={}
        for label,index in (('data',1),('variables',2)):
            sheet=ET.fromstring(archive.read(f'xl/worksheets/sheet{index}.xml'))
            rows=[]
            for row in sheet.findall('s:sheetData/s:row',ns):
                values={}
                for cell in row:
                    v=cell.find('s:v',ns)
                    if v is None:continue
                    kind=cell.get('t','n')
                    if kind=='s':value=shared[int(v.text)]
                    elif kind=='b':value=bool(int(v.text))
                    elif kind=='n':value=float(v.text)
                    else:raise ValueError('Unsupported cell type '+kind)
                    values[re.sub(r'\d','',cell.get('r'))]=value
                rows.append(dict(excel_row=int(row.get('r')),cells=values))
            sheets[label]=rows
        header=sheets['data'][0]['cells']
        assert len(set(header.values()))==len(header)
        records=[dict(excel_row=row['excel_row'],**{name:row['cells'].get(col) for col,name in header.items()})
                 for row in sheets['data'][1:] if row['cells']]
        return records,sheets['variables']


counts={}
for channel,name,explicit,temp in (('sodium','318666-data_fig3.xlsx',False,25),
                                    ('potassium','318667-data_fig4.xlsx',True,34)):
    records,variables=_read_workbook(root/name)
    human,selection=select_human_records(records,require_species_column=explicit)
    write(root/(channel+'-human-records.json'),dict(source_file=name,source_sha256=sha(root/name),
        source_table='data',variables=variables,source_record_count=len(records),
        selection=selection,records=human,temperature_c=temp,
        data_kind='published derived experimental measurements; original NWB traces not acquired',
        physiology_qualified=False))
    counts[channel]=selection['human_records']
write(root/'acquisition.json',dict(doi='10.34894/L5J0SD',version='3.0',
    metadata_sha256=sha(root/'metadata-v3.json'),files=files,human_record_counts=counts,
    external_pax6_responses_opened=False,original_nwb_traces_acquired=False,
    upstream_scripts_executed=False,parameters_changed=False,scores_promoted=False))
print(json.dumps(counts))
