"""Run the local human tail extraction under its registered wall cap."""

from pathlib import Path
import json
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
started = time.perf_counter()
receipt = dict(wall_cap_seconds=120, execution='local CPU; no membrane simulation')
with (out / 'stdout.txt').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
    try:
        result = subprocess.run([sys.executable, str(out / 'analyze.py')],
                                stdout=stdout, stderr=stderr, timeout=120)
        receipt.update(exit_code=result.returncode, timeout=False)
    except subprocess.TimeoutExpired:
        receipt.update(exit_code=None, timeout=True)
receipt['wall_seconds'] = time.perf_counter() - started
with (out / 'terminal.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
raise SystemExit(1 if receipt['timeout'] else receipt['exit_code'])
