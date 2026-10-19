"""Wait for the verified paired benchmark controller, then run public models serially."""
import json,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
for handle in json.loads((ROOT/'runs/paired-controller.json').read_text()):
 try:
  proc=psutil.Process(handle['pid'])
  if proc.create_time()!=handle['created']:continue
  print('Waiting for live paired benchmark',handle['pid'],flush=True)
  while proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:time.sleep(1)
 except psutil.NoSuchProcess:pass
print('Starting sequential public campaign',flush=True)
raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts/public_campaign.py'),'--budget','900','--max-experiments','18']))
