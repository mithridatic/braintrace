"""Enforce a process wall cap and retain terminal output without overwriting."""
from pathlib import Path
import json
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
task = sys.argv[1]
assert task in ('fit', 'review')
cap = 600 if task == 'fit' else 120
began = time.perf_counter()
with (out / f'{task}-stdout.txt').open('x') as stdout, (out / f'{task}-stderr.txt').open('x') as stderr:
    process = subprocess.Popen([sys.executable, str(out / f'{task}.py')], cwd=out.parents[4], stdout=stdout, stderr=stderr)
    timed_out = False
    try:
        code = process.wait(timeout=cap)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        code = process.wait()
result = dict(task=task, wall_cap_seconds=cap, seconds=time.perf_counter()-began,
              timed_out=timed_out, exit_code=code, host='local', device='CPU')
with (out / f'{task}-terminal.json').open('x', encoding='utf-8') as stream:
    stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
