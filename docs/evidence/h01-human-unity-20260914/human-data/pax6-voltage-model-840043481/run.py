"""Execute one registered local fit under a hard wall-time bound."""

from pathlib import Path
import json
import os
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
started = time.perf_counter()
environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
receipt = dict(wall_cap_seconds=600, execution='local CPU vectorized closed-form model')
with (out / 'stdout.txt').open('xb') as stdout, (out / 'stderr.txt').open('xb') as stderr:
    try:
        result = subprocess.run([sys.executable, str(out / 'fit.py')], env=environment,
                                stdout=stdout, stderr=stderr, timeout=600)
        receipt.update(exit_code=result.returncode, timeout=False)
    except subprocess.TimeoutExpired:
        receipt.update(exit_code=None, timeout=True)
receipt['wall_seconds'] = time.perf_counter() - started
with (out / 'terminal.json').open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
raise SystemExit(1 if receipt['timeout'] else receipt['exit_code'])
