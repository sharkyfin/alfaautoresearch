"""A scaled 3-hour scheduling simulation; explicitly not a real 3-hour soak."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create
root=ROOT/'runs/deadline';task=create(root/'data','integer_labels',seed=91,n=1000)
scale=1/180;budget=10800*scale # 60 seconds, reserve policy scales with total.
start=time.monotonic()
with (ROOT/'reports/deadline-simulation.log').open('w') as log:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_dev.py'),str(task),'--budget',str(budget),'--threads','8','--max-experiments','60','--mode','mock-agent','--run-dir',str(root/'run')],stdout=log,stderr=log,timeout=budget+10)
elapsed=time.monotonic()-start;summary=json.loads((root/'run/summary.json').read_text())
assert result.returncode==0 and elapsed<budget+3
assert (task.parent/'submission.csv').exists()
report={'nominal_platform_budget_seconds':10800,'time_scale':scale,'actual_budget_seconds':budget,'actual_elapsed_seconds':elapsed,'real_3h_soak':False,'summary':summary}
(ROOT/'reports/deadline-simulation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
