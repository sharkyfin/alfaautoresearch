"""Sequential frozen-source public experiments; no hidden test labels or submission."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--name',default='public-campaign');p.add_argument('--budget',type=float,default=900);p.add_argument('--seeds',default='42,53');p.add_argument('--max-experiments',type=int,default=18);a=p.parse_args()
root=ROOT/'runs'/a.name;root.mkdir(parents=True,exist_ok=True)
snapshot=root/'source_snapshot';snapshot.mkdir(exist_ok=True);shutil.copytree(ROOT/'atlas',snapshot/'atlas',dirs_exist_ok=True);shutil.copy2(ROOT/'solution.py',snapshot/'solution.py');(snapshot/'scripts').mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts/run_dev.py',snapshot/'scripts/run_dev.py')
records=[]
for seed in map(int,a.seeds.split(',')):
 for kind in ['regression','classification']:
  task=ROOT/'artifacts/data'/kind/'task.md';output=root/f'{kind}_{seed}'
  with (root/f'{kind}_{seed}.log').open('w') as log:
   result=subprocess.run([sys.executable,str(snapshot/'scripts/run_dev.py'),str(task),'--budget',str(a.budget),'--threads','8','--max-experiments',str(a.max_experiments),'--mode','deterministic','--seed',str(seed),'--run-dir',str(output)],stdout=log,stderr=log,timeout=a.budget+20)
  summary=json.loads((output/'summary.json').read_text());summary.update(task=kind,seed=seed,returncode=result.returncode)
  records.append(summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
  (ROOT/'reports'/f'{a.name}.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
