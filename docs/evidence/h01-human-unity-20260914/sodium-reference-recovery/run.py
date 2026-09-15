"""Retain terminal state for the bounded local reference comparison."""

from pathlib import Path
import json
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
assert not (out/'terminal.json').exists()
started = time.monotonic()
try:
    p = subprocess.run([sys.executable, str(out/'analyze.py')], capture_output=True, timeout=120)
    (out/'stdout.txt').write_bytes(p.stdout); (out/'stderr.txt').write_bytes(p.stderr)
    status, code = 'completed', p.returncode
except subprocess.TimeoutExpired as exc:
    (out/'stdout.txt').write_bytes(exc.stdout or b''); (out/'stderr.txt').write_bytes(exc.stderr or b'')
    status, code = 'timeout', None
receipt = dict(status=status, exit_code=code, cap_seconds=120,
               elapsed_seconds=time.monotonic()-started, execution='local CPU')
(out/'terminal.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
raise SystemExit(0 if code == 0 else 1)
