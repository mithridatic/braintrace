"""Compare versioned source cells by recording identity without executing source code."""

from pathlib import Path
import ast
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile

out = Path(__file__).resolve().parent
current = out.parent/'human-pyramidal-channels'
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def _read(path):
    with zipfile.ZipFile(path) as z:
        shared = [''.join(t.text or '' for t in s.findall('.//s:t', ns))
                  for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
        sheets = [s.attrib for s in ET.fromstring(z.read('xl/workbook.xml')).find('s:sheets', ns)]
        rows = ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row', ns)
        decoded = []
        for row in rows:
            cells = {}
            for cell in row:
                value = cell.find('s:v', ns)
                if value is None:
                    continue
                kind = cell.get('t', 'n')
                if kind == 's':
                    v = shared[int(value.text)]
                elif kind == 'n':
                    v = float(value.text)
                elif kind == 'b':
                    v = bool(int(value.text))
                else:
                    raise ValueError(f'Unsupported original cell type {kind}')
                column = re.sub(r'\d', '', cell.attrib['r'])
                assert column not in cells
                cells[column] = v
            decoded.append((int(row.attrib['r']), cells))
        headers = decoded[0][1]
        assert len(set(headers.values())) == len(headers)
        records = []
        for index, values in decoded[1:]:
            assert not (values.keys()-headers.keys())
            record = {name: values.get(col) for col, name in headers.items()}
            records.append(dict(excel_row=index, values=record))
        return dict(sheets=sheets, columns=headers, records=records)


old_path = out/'318532-data_fig3.xlsx'
new_path = current/'318666-data_fig3.xlsx'
old, new = _read(old_path), _read(new_path)
old_by_name = {r['values']['Filename']: r for r in old['records']}
new_by_name = {r['values']['Filename']: r for r in new['records']}
assert len(old_by_name) == len(old['records']) == len(new_by_name) == len(new['records']) == 34
assert old_by_name.keys() == new_by_name.keys()
old_fields, new_fields = set(old['columns'].values()), set(new['columns'].values())
common = sorted(old_fields & new_fields)
removed = sorted(old_fields-new_fields)
differences = []
qc = []
for name, old_record in old_by_name.items():
    new_record = new_by_name[name]
    for field in common:
        a, b = old_record['values'][field], new_record['values'][field]
        if a != b:
            differences.append(dict(filename=name, column=field, old=a, new=b))
    if name.startswith('H'):
        qc.append(dict(filename=name, historical_excel_row=old_record['excel_row'],
                       current_excel_row=new_record['excel_row'],
                       recovered_fields={k: old_record['values'][k] for k in removed}))
assert len(qc) == 19
functions = []
for old_name, new_name in [('318547-channel_tools.py', '326242-channel_tools.py'),
                           ('318550-channel_kinetics.py', '326241-channel_kinetics.py'),
                           ('318548-fit_na_hh_cma_human.py', '326238-fit_na_hh_cma_human.py'),
                           ('318546-fit_k_hh_cma_human.py', '326237-fit_k_hh_cma_human.py')]:
    a, b = (out/old_name).read_text(), (current/new_name).read_text()
    # Parsing is read-only. Never import or execute downloaded scientific scripts.
    ta, tb = ast.parse(a), ast.parse(b)
    fa = {n.name: ast.dump(n, include_attributes=False) for n in ta.body if isinstance(n, ast.FunctionDef)}
    fb = {n.name: ast.dump(n, include_attributes=False) for n in tb.body if isinstance(n, ast.FunctionDef)}
    functions.append(dict(old_file=old_name, current_file=new_name, text_identical=a == b,
                          syntax_identical=ast.dump(ta) == ast.dump(tb),
                          changed_functions=[name for name in sorted(fa.keys() | fb.keys()) if fa.get(name) != fb.get(name)],
                          old_sha256=hashlib.sha256((out/old_name).read_bytes()).hexdigest(),
                          current_sha256=hashlib.sha256((current/new_name).read_bytes()).hexdigest()))
versions = json.loads((out/'versions.json').read_text())['data']
inventory = []
for version in versions:
    paths = [f.get('directoryLabel', '')+'/'+f['dataFile']['filename'] for f in version['files']]
    inventory.append(dict(version=f"{version['versionNumber']}.{version['versionMinorNumber']}",
                          files=len(paths), raw_trace_or_archive_paths=[p for p in paths if p.lower().endswith(('.nwb', '.abf', '.h5', '.hdf5', '.mat', '.zip', '.tar', '.gz'))],
                          experimental_mean_paths=[p for p in paths if 'experimental means' in p.lower()]))
result = dict(old_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
              current_sha256=hashlib.sha256(new_path.read_bytes()).hexdigest(),
              old_columns=len(old_fields), current_columns=len(new_fields),
              matched_records=34, matched_human_records=19, compared_common_cells=34*len(common),
              recovered_columns=removed, added_current_columns=sorted(new_fields-old_fields),
              differences=differences, human_recovered_fields=qc,
              scripts=functions, release_history=inventory,
              source_qc_recovered='exclude' in removed,
              original_current_arrays_recovered=False, scores_promoted=False)
(out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({k: v for k, v in result.items() if k not in ['human_recovered_fields', 'differences']}, indent=2))
print('differences', len(differences), differences[:5])
print('human QC', [(r['filename'], r['recovered_fields']['exclude']) for r in qc])

ap_old = _read(out/'318529-Na_data_fig2.xlsx')
ap_new = _read(current/'318668-Na_data_fig2.xlsx')
ap_old_fields, ap_new_fields = set(ap_old['columns'].values()), set(ap_new['columns'].values())
ap_old_rows = {r['values']['Filename']: r for r in ap_old['records']}
ap_new_rows = {r['values']['Filename']: r for r in ap_new['records']}
assert len(ap_old_rows) == len(ap_old['records']) == len(ap_new_rows) == len(ap_new['records']) == 49
ap_differences = []
for name in sorted(ap_old_rows.keys() & ap_new_rows.keys()):
    for field in sorted(ap_old_fields & ap_new_fields):
        a, b = ap_old_rows[name]['values'][field], ap_new_rows[name]['values'][field]
        if a != b:
            ap_differences.append(dict(filename=name, column=field, old=a, new=b))
ap_result = dict(old_columns=len(ap_old_fields), current_columns=len(ap_new_fields),
                 old_records=49, current_records=49,
                 removed_columns=sorted(ap_old_fields-ap_new_fields),
                 added_columns=sorted(ap_new_fields-ap_old_fields), changed_cells=ap_differences,
                 old_only_recordings=sorted(ap_old_rows.keys()-ap_new_rows.keys()),
                 current_only_recordings=sorted(ap_new_rows.keys()-ap_old_rows.keys()))
(out/'ap-workbook-comparison.json').write_text(json.dumps(ap_result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
