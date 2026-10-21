"""Actual installed API verification against organizer requirements."""
import importlib.metadata,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from atlas.task import APPROVED,load
from atlas.contract import read_inputs
from atlas.validation import design
from atlas.models import fit_predict
from scripts.synthetic import create
records=[]
for scenario in ['iid','integer_labels','binary_imbalance']:
 data=load(read_inputs(create(ROOT/'runs/capabilities'/scenario,scenario)));split=design(data)
 for family in ['linear','hist','extra_trees','lightgbm','xgboost']:
  start=time.monotonic()
  try:
   v,t,info=fit_predict(data,{'model_family':family,'params':{'iterations':25} if family!='linear' else {}},split.train,split.valid,threads=1)
   row={'task':scenario,'family':family,'status':'fit_predict_ok','metric':data.spec.metric.name,'value':data.spec.metric.score(data.y[split.valid],v),'seconds':time.monotonic()-start}
  except Exception as e:row={'task':scenario,'family':family,'status':'error','exception':str(e)}
  records.append(row)
result={'python':sys.version,'approved_pins':APPROVED,'installed':{k:importlib.metadata.version(k) for k in APPROVED},'actual_fits':records}
(ROOT/'reports/capability-matrix.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
