"""Legacy synthetic stress suite; real-data evaluation uses benchmark_suite.py."""
from __future__ import annotations
import argparse,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np
from scripts.synthetic import create,SCENARIOS
from atlas.contract import read_inputs
from atlas.task import load
from atlas.validation import design
from atlas.models import fit_predict
from atlas.submit import checkpoint,dummy_prediction
from atlas.metrics import UnsupportedTask
from atlas.judge import diagnostics

p=argparse.ArgumentParser();p.add_argument('--output',default='reports/stress');p.add_argument('--seeds',default='17,43');a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
records=[]
for seed in map(int,a.seeds.split(',')):
 for scenario in SCENARIOS:
  start=time.monotonic();row={'scenario':scenario,'seed':seed}
  try:
   task=create(root/'data'/f'{scenario}_{seed}',scenario,seed);data=load(read_inputs(task));s=design(data,seed=seed)
   cp=dummy_prediction(data.y,data.spec,len(data.test));checkpoint(data,cp)
   if len(s.valid) and len(np.unique(data.y))>1:
    baseline=data.spec.metric.score(data.y[s.valid],dummy_prediction(data.y[s.train],data.spec,len(s.valid)))
    family='linear' if scenario in ['small_high_p','text','high_cardinality'] else 'hist'
    recipe={'model_family':family,'params':{} if family=='linear' else {'iterations':80},'feature_ops':[]}
    pv,pt,info=fit_predict(data,recipe,s.train,s.valid,seed=seed,threads=1)
    score=data.spec.metric.score(data.y[s.valid],pv);checkpoint(data,pt)
    row.update(metric=data.spec.metric.name,baseline=baseline,score=score,gain_utility=data.spec.metric.utility(score)-data.spec.metric.utility(baseline),split=s.json(),family=family)
   row['status']='valid_output'
  except UnsupportedTask as e:
   row.update(status='unsupported',reason=str(e))
   if scenario not in ['conflict','multiple_targets']:row['status']='unexpected_unsupported'
  except Exception as e:row.update(status='failure',exception=type(e).__name__,traceback=traceback.format_exc())
  row['seconds']=time.monotonic()-start;records.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
  (root/'results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps({'runs':len(records),'valid':sum(r['status']=='valid_output' for r in records),'expected_unsupported':sum(r['status']=='unsupported' for r in records),'failures':sum(r['status'] not in ['valid_output','unsupported'] for r in records)},ensure_ascii=False))
