"""Run the frozen human synaptic experiment with a 120-second wall cap."""

from pathlib import Path
import json
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
started = time.monotonic()
try:
    result = subprocess.run([sys.executable, str(out/'analyze.py')],
                            capture_output=True, text=True, timeout=120)
    (out/'stdout.txt').write_text(result.stdout, encoding='utf-8')
    (out/'stderr.txt').write_text(result.stderr, encoding='utf-8')
    status, code = 'completed', result.returncode
except subprocess.TimeoutExpired as exc:
    (out/'stdout.txt').write_bytes(exc.stdout or b'')
    (out/'stderr.txt').write_bytes(exc.stderr or b'')
    status, code = 'timeout', None
receipt = dict(status=status, exit_code=code, wall_cap_seconds=120,
               elapsed_seconds=time.monotonic()-started, execution='local CPU closed-form convolution')
(out/'terminal.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps(receipt))
raise SystemExit(0 if code == 0 else 1)
