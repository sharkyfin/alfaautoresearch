"""Dev-only controlled hypothesis: stop LightGBM using the task's actual F1."""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='2'
import numpy as np
import lightgbm as lgb
from sklearn.utils.class_weight import compute_sample_weight
from atlas.task import load
from atlas.contract import read_inputs
from atlas.validation import design
from atlas.features import FeatureBuilder,MatrixBuilder

data=load(read_inputs(ROOT/'artifacts/data/classification/task.md'));results=[]
for seed in [42,1051]:
 split=design(data,seed,fraction=.2 if seed==42 else .25)
 a,b=data.train.iloc[split.train],data.train.iloc[split.valid];f=FeatureBuilder(data.spec).fit(a);m=MatrixBuilder('lightgbm',f.cat,f.num).fit(f.transform(a));xa=m.transform(f.transform(a));xv=m.transform(f.transform(b))
 xa.columns=xv.columns=[f'f{i:03}' for i in range(xa.shape[1])]
 y=data.y[split.train];v=data.y[split.valid];classes=np.unique(y)
 for weight in ['none','sqrt']:
  for objective in ['logloss','primary_f1']:
   start=time.monotonic();model=lgb.LGBMClassifier(n_estimators=1000,learning_rate=.045,num_leaves=31,min_child_samples=30,n_jobs=2,verbosity=-1,random_state=42,deterministic=True,force_col_wise=True,**({'metric':'None'} if objective=='primary_f1' else {}))
   def primary(ytrue,pred):
    if pred.ndim==1:pred=np.column_stack([1-pred,pred])
    probs=np.zeros((len(pred),len(data.spec.labels)));probs[:,classes]=pred
    return data.spec.metric.name,data.spec.metric.score(classes[ytrue.astype(int)],probs),True
   sw=compute_sample_weight('balanced',y)**.5 if weight=='sqrt' else None
   kwargs={'eval_metric':primary} if objective=='primary_f1' else {}
   model.fit(xa,y,sample_weight=sw,eval_set=[(xv,v)],callbacks=[lgb.early_stopping(100,first_metric_only=True,verbose=False)],**kwargs)
   p=model.predict_proba(xv);score=data.spec.metric.score(v,p)
   result={'seed':seed,'split':split.json(),'weight':weight,'early_stopping':objective,'metric':'f1_macro','score':score,'iterations':model.best_iteration_,'seconds':time.monotonic()-start};results.append(result);print(json.dumps(result),flush=True)
   (ROOT/'reports/f1-early-stopping.json').write_text(json.dumps(results,indent=2))
