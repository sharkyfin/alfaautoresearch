"""A scaled 3-hour scheduling simulation; explicitly not a real 3-hour soak."""
import json,subprocess,sys,time,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create
root=ROOT/'runs/deadline';task=create(root/'data','integer_labels',seed=91,n=1000)
run_dir=Path(tempfile.mkdtemp(prefix='run-',dir=root))
scale=1/180;budget=10800*scale # 60 seconds, reserve policy scales with total.
start=time.monotonic()
with (ROOT/'reports/deadline-simulation.log').open('w') as log:
    result=subprocess.run([sys.executable,str(ROOT/'scripts/run_dev.py'),str(task),'--budget',str(budget),'--threads','8','--max-experiments','60','--mode','mock-agent','--run-dir',str(run_dir)],stdout=log,stderr=log,timeout=budget+10)
elapsed=time.monotonic()-start;summary=json.loads((run_dir/'summary.json').read_text())
assert result.returncode==0 and elapsed<budget+3
assert (task.parent/'submission.csv').exists()
assert summary['successful_experiments']>0 and not (run_dir/'error.txt').exists()
report={'nominal_platform_budget_seconds':10800,'time_scale':scale,'actual_budget_seconds':budget,'actual_elapsed_seconds':elapsed,'real_3h_soak':False,'summary':summary}
(ROOT/'reports/deadline-simulation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
