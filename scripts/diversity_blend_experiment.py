"""Confirm a fixed cross-family regression blend on a second predefined split."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np
from atlas.task import load
from atlas.contract import read_inputs
from atlas.validation import design
from atlas.models import fit_predict

data=load(read_inputs(ROOT/'artifacts/data/regression/task.md'));primary=design(data,42);control=design(data,1051,fraction=.25)
root=ROOT/'runs/public-reg-360/experiments';recipes=[];pv=[]
for i in [8,6]:
 directory=root/f'exp_{i:04d}';recipes.append(json.loads((directory/'job.json').read_text())['recipe'])
 with np.load(directory/'predictions.npz') as z:pv.append(z['valid'])
metric=data.spec.metric;primary_scores=[metric.score(data.y[primary.valid],p) for p in pv];weight=.5
preds=[];records=[]
for recipe in recipes:
 start=time.monotonic();v,t,info=fit_predict(data,recipe,control.train,control.valid,seed=1051,threads=1)
 score=metric.score(data.y[control.valid],v);preds.append(v);records.append({'family':recipe['model_family'],'score':score,'seconds':time.monotonic()-start});print(json.dumps(records[-1]),flush=True)
result={'primary_scores':primary_scores,'primary_blend':metric.score(data.y[primary.valid],weight*pv[0]+(1-weight)*pv[1]),'control':records,'control_blend':metric.score(data.y[control.valid],weight*preds[0]+(1-weight)*preds[1]),'weight_fixed_before_control':weight,'primary_split':primary.json(),'control_split':control.json()}
result['confirmed']=result['control_blend']<min(x['score'] for x in records)
(ROOT/'reports/diversity-blend.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
