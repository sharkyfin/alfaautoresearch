"""Equal-budget sequential deterministic versus mock research ablation (NOT live Qwen)."""
import argparse,json,os,subprocess,sys,time,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.synthetic import create
p=argparse.ArgumentParser();p.add_argument('--budget',type=float,default=40);p.add_argument('--seeds',default='17,43');p.add_argument('--tasks',default='iid,skew,integer_labels,multiclass_imbalance');a=p.parse_args()
root=ROOT/'runs/paired';root.mkdir(parents=True,exist_ok=True);records=[]
snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True)
shutil.copytree(ROOT/'atlas',snapshot/'atlas',dirs_exist_ok=True)
shutil.copy2(ROOT/'solution.py',snapshot/'solution.py')
(snapshot/'scripts').mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts/run_dev.py',snapshot/'scripts/run_dev.py')
for seed in map(int,a.seeds.split(',')):
 for task_name in a.tasks.split(','):
  task=create(root/f'{task_name}_{seed}'/'data',task_name,seed,n=600)
  for mode in ['deterministic','mock-agent']:
   output=root/f'{task_name}_{seed}'/mode
   with (root/f'{task_name}_{seed}_{mode}.log').open('w') as log:
    result=subprocess.run([sys.executable,str(snapshot/'scripts/run_dev.py'),str(task),'--budget',str(a.budget),'--threads','2','--max-experiments','12','--mode',mode,'--seed',str(seed),'--run-dir',str(output)],stdout=log,stderr=log,timeout=a.budget+20)
   summary=json.loads((output/'summary.json').read_text())
   summary.update(task=task_name,seed=seed,returncode=result.returncode)
   records.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
   (ROOT/'reports/paired-ablation.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
