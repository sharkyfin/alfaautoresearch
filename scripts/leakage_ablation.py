"""Demonstrate optimistic random validation on intentionally grouped/time data."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from atlas.task import load
from atlas.contract import read_inputs
from atlas.validation import design,Split
from atlas.models import fit_predict

root=ROOT/'runs/leakage';root.mkdir(parents=True,exist_ok=True);results=[];rng=np.random.default_rng(41)
for kind in ['group','time']:
 d=root/kind;d.mkdir(exist_ok=True);n=1000;nt=200
 if kind=='group':
  groups=np.repeat(np.arange(240),5);effects=rng.normal(0,5,240);df=pd.DataFrame({'entity_id':groups.astype(str),'noise':rng.normal(size=n+nt),'response':effects[groups]+rng.normal(0,.05,n+nt)})
  extra='validation: group; unseen entities.'
 else:
  t=np.arange(n+nt);df=pd.DataFrame({'event_date':pd.date_range('2020-01-01',periods=n+nt).astype(str),'trend':t,'response':t*.02+np.sin(t*.02)})
  extra='validation: time; forecasting future periods.'
 df.insert(0,'id',np.arange(n+nt));df.iloc[:n].to_csv(d/'train.csv',index=False);df.iloc[n:].drop(columns='response').to_csv(d/'test.csv',index=False)
 pd.DataFrame({'id':df.id.iloc[n:],'response':0.}).to_csv(d/'sample_submission.csv',index=False)
 dic=[]
 for c in df:
  dic.append({'feature':c,'role':'target' if c=='response' else 'id' if c=='id' else 'group' if c=='entity_id' else 'feature','dtype':'datetime' if c=='event_date' else 'category' if c=='entity_id' else 'numeric'})
 pd.DataFrame(dic).to_csv(d/'data_dictionary.csv',index=False);(d/'task.md').write_text('Regression. Primary metric: RMSE\n'+extra+'\n`train.csv` `test.csv` `data_dictionary.csv` `sample_submission.csv`\nWrite `submission.csv`.')
 data=load(read_inputs(d/'task.md'));trusted=design(data,41);a,b=train_test_split(np.arange(n),test_size=.2,random_state=41);random=Split(a,b,'random_unsafe','Deliberate leakage demonstration',41)
 for split in [random,trusted]:
  start=time.monotonic();p,t,info=fit_predict(data,{'model_family':'extra_trees','params':{'iterations':100,'min_samples_leaf':1,'max_features':1.}},split.train,split.valid,threads=1)
  row={'scenario':kind,'validation':split.kind,'metric':'rmse','score':data.spec.metric.score(data.y[split.valid],p),'seconds':time.monotonic()-start,'split':split.json()};results.append(row);print(json.dumps(row),flush=True)
(ROOT/'reports/leakage-ablation.json').write_text(json.dumps(results,indent=2))
