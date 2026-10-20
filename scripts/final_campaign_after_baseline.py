"""Run the improved frozen source after the baseline campaign has terminated."""
import json,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
for handle in json.loads((ROOT/'runs/old-campaign-controller.json').read_text()):
 try:
  proc=psutil.Process(handle['pid'])
  if proc.create_time()!=handle['created']:continue
  print('Waiting for verified baseline campaign',handle['pid'],flush=True)
  while proc.is_running() and proc.status()!=psutil.STATUS_ZOMBIE:time.sleep(1)
 except psutil.NoSuchProcess:pass
print('Starting final-source public validation',flush=True)
raise SystemExit(subprocess.call([sys.executable,str(ROOT/'scripts/public_campaign.py'),'--name','public-final','--budget','900','--max-experiments','18','--seeds','42']))
