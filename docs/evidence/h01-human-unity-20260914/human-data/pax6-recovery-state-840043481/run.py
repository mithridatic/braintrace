"""Run the local state fit with a terminal receipt and a five-minute wall cap."""

from datetime import datetime,timezone
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
if (root/'run-receipt.json').exists() or (root/'fit.log').exists():
    raise FileExistsError('This bounded attempt already has execution evidence.')
started=datetime.now(timezone.utc).isoformat();timer=time.perf_counter()
command=[sys.executable,str(root/'fit.py')]
print('Starting bounded local CPU fit; wall cap 300 seconds.',flush=True)
timed_out=False;code=None
with (root/'fit.log').open('x',encoding='utf-8') as log:
    try:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=300,check=False)
        code=result.returncode
    except subprocess.TimeoutExpired:
        timed_out=True
receipt=dict(command=command,started_utc=started,ended_utc=datetime.now(timezone.utc).isoformat(),
              elapsed_seconds=time.perf_counter()-timer,wall_cap_seconds=300,
              timed_out=timed_out,returncode=code,execution='local CPU; no neuronal rollout')
(root/'run-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(receipt,indent=2),flush=True)
print((root/'fit.log').read_text(encoding='utf-8'),flush=True)
sys.exit(1 if timed_out or code!=0 else 0)
