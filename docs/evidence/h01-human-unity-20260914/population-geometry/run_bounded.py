"""Retain a terminal receipt for the capped static population geometry audit."""

from pathlib import Path
import hashlib
import json
import subprocess
import time

root = Path(__file__).resolve().parent
repository = root.parents[3]
primary = repository.parents[1]
command = [str(primary/'.venv/Scripts/python.exe'),str(root/'run_audit.py'),
           '--archive',str(primary/'.cache/h01/proofread104.zip'),
           '--reference',str(primary/'.cache/h01/readiness/104-implicit-construction-reference.json')]
if (root/'run-receipt.json').exists() or (root/'audit.log').exists():
    raise FileExistsError('Attempt already exists.')
started = time.monotonic()
with (root/'audit.log').open('x',encoding='utf-8') as stream:
    process = subprocess.Popen(command,cwd=repository,stdout=stream,stderr=subprocess.STDOUT)
    print('Started audit PID',process.pid,flush=True)
    timed_out = False
    try:
        code = process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        code = process.wait()
receipt = dict(command=command,pid=process.pid,returncode=code,timed_out=timed_out,
      wall_cap_seconds=600,elapsed_seconds=time.monotonic()-started,
      log_sha256=hashlib.sha256((root/'audit.log').read_bytes()).hexdigest())
(root/'run-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(code if code else (1 if timed_out else 0))
