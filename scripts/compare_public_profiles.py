"""Confirm final profile-only robustness fix leaves public model inputs identical."""
import hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.contract import read_inputs
from atlas.task import load
snapshot=ROOT/'runs/public-final/source_snapshot';results=[]
for kind in ['regression','classification']:
 task=ROOT/'artifacts/data'/kind/'task.md'
 current_spec=load(read_inputs(task)).spec.json();current_spec.pop('suspicious_leakage_features',None)
 current=json.dumps(current_spec,sort_keys=True,ensure_ascii=False,allow_nan=False)
 code=f'import sys,json;sys.path.insert(0,{str(snapshot)!r});from atlas.contract import read_inputs;from atlas.task import load;print(json.dumps(load(read_inputs({str(task)!r})).spec.json(),sort_keys=True,ensure_ascii=False,allow_nan=False))'
 old=subprocess.check_output([sys.executable,'-I','-c',code],text=True).strip()
 assert current==old,kind
 results.append({'task':kind,'model_task_spec_identical':True,'excluded_new_diagnostic_field':'suspicious_leakage_features','task_spec_sha256':hashlib.sha256(current.encode()).hexdigest()})
(ROOT/'reports/profile-fix-equivalence.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
