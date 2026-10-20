"""Typed hypothesis validation and recipe-specific executable Python compiler."""
from __future__ import annotations
import hashlib,json,math,re
from .models import FAMILIES

PARAMS={
'linear':{'alpha':(.0001,1e5),'C':(.0001,1e4)},
'hist':{'iterations':(20,3000),'learning_rate':(.005,.3),'max_leaf_nodes':(4,127),'l2_regularization':(0,100),'min_samples_leaf':(2,200)},
'extra_trees':{'iterations':(20,1000),'min_samples_leaf':(1,100),'max_features':(.1,1.),'max_depth':(2,40)},
'catboost':{'iterations':(20,5000),'depth':(3,10),'learning_rate':(.005,.3),'l2_leaf_reg':(.01,100),'random_strength':(0,10),'bagging_temperature':(0,10)},
'lightgbm':{'stopping_metric':('loss','primary'),'iterations':(20,5000),'num_leaves':(7,255),'learning_rate':(.005,.3),'min_child_samples':(5,250),'reg_lambda':(0,100),'colsample_bytree':(.3,1),'subsample':(.5,1)},
'xgboost':{'iterations':(20,3000),'max_depth':(2,12),'learning_rate':(.005,.3),'min_child_weight':(1,100),'reg_lambda':(0,100),'colsample_bytree':(.3,1),'subsample':(.5,1)}}
INTEGER={'iterations','depth','max_depth','num_leaves','min_child_samples','max_leaf_nodes','min_samples_leaf'}
OP_ARITY={'log1p':1,'sqrt':1,'square':1,'ratio':2,'difference':2,'product':2,'group_mean':2,'frequency':-1,'text_stats':-1,'drop':-1,'ip':-1,'clip':-1,'combine':2}
MODEL_PACKAGE={'catboost':'catboost','lightgbm':'lightgbm','xgboost':'xgboost'}

class RecipeError(ValueError):pass

def validate(raw,spec,remaining=10800,seen=None):
    if not isinstance(raw,dict):raise RecipeError('Recipe must be an object')
    allowed={'id','parent_experiment','statement','evidence','operation','model_family','params','feature_ops','target_transform','validation_plan_id','max_runtime_sec','expected_rationale'}
    if set(raw)-allowed:raise RecipeError('Unknown recipe fields')
    family=raw.get('model_family')
    if not isinstance(family,str) or family not in FAMILIES:raise RecipeError('Unknown or forbidden model')
    pkg=MODEL_PACKAGE.get(family,'scikit-learn')
    if pkg not in spec.available_capabilities:raise RecipeError('Unavailable approved library '+pkg)
    if raw.get('operation','train_candidate')!='train_candidate':raise RecipeError('Forbidden operation')
    if raw.get('validation_plan_id',spec.validation_plan.get('id'))!=spec.validation_plan.get('id'):raise RecipeError('Validation plan is immutable')
    if 'id' in raw and not isinstance(raw['id'],str):raise RecipeError('id must be string')
    if raw.get('parent_experiment') is not None and not isinstance(raw['parent_experiment'],str):raise RecipeError('parent must be string or null')
    p=raw.get('params',{})
    if not isinstance(p,dict) or set(p)-set(PARAMS[family])-{'weight'}:raise RecipeError('Unknown model params')
    for k,v in p.items():
        if k=='stopping_metric':
            if not isinstance(v,str) or v not in {'loss','primary'} or v=='primary' and not spec.classification:raise RecipeError('Invalid primary stopping metric')
            continue
        if k=='weight':
            if not isinstance(v,str) or v not in {'none','balanced','sqrt'} or (v!='none' and not spec.classification):raise RecipeError('Invalid class weight')
            continue
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not PARAMS[family][k][0]<=v<=PARAMS[family][k][1]:raise RecipeError('Invalid parameter '+k)
        if k in INTEGER and not isinstance(v,int):raise RecipeError('Integer required '+k)
    ops=raw.get('feature_ops',[])
    if not isinstance(ops,list) or len(ops)>20:raise RecipeError('Too many feature operations')
    total=0
    for op in ops:
        if not isinstance(op,dict) or set(op)!={'op','columns'} or not isinstance(op['op'],str) or op['op'] not in OP_ARITY:raise RecipeError('Invalid feature operation')
        cols=op['columns'];arity=OP_ARITY[op['op']]
        if not isinstance(cols,list) or not 1<=len(cols)<=12 or (arity!=-1 and len(cols)!=arity):raise RecipeError('Invalid feature arity')
        if any(not isinstance(c,str) or c not in spec.features for c in cols):raise RecipeError('Unknown, target or ID column forbidden')
        if op['op'] in {'ratio','difference','product','log1p','sqrt','square','clip'} and any(c not in spec.numeric_features for c in cols):raise RecipeError('Numeric operation on nonnumeric column')
        if op['op']=='frequency' and any(c not in spec.categorical_features for c in cols):raise RecipeError('Frequency needs categorical column')
        if op['op']=='group_mean' and cols[1] not in spec.numeric_features:raise RecipeError('Group mean needs numeric value')
        total+=len(cols)
    if total>60:raise RecipeError('Feature memory/cost bound')
    transform=raw.get('target_transform','none')
    if not isinstance(transform,str) or transform not in {'none','log1p'} or transform!='none' and (spec.classification or min(spec.profile.get('target',{}).get('quantiles',{'0':0}).values())<0):raise RecipeError('Invalid target transform')
    runtime=raw.get('max_runtime_sec',300)
    if isinstance(runtime,bool) or not isinstance(runtime,(int,float)) or not 1<=runtime<=1800:raise RecipeError('Invalid time budget')
    if runtime>remaining:raise RecipeError('Insufficient remaining budget')
    clean={'model_family':family,'params':p,'feature_ops':ops,'target_transform':transform,'max_runtime_sec':float(runtime),'validation_plan_id':spec.validation_plan.get('id')}
    clean['id']=str(raw.get('id','candidate'))[:80]
    clean['statement']=str(raw.get('statement',''))[:1000]
    clean['parent_experiment']=raw.get('parent_experiment')
    digest=recipe_hash(clean)
    if seen is not None and digest in seen:raise RecipeError('Duplicate hypothesis')
    return clean

def recipe_hash(recipe):
    core={k:recipe.get(k) for k in ['model_family','params','feature_ops','target_transform']}
    return hashlib.sha256(json.dumps(core,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def compile_recipe(recipe,job_path,code_root):
    # Column names are Python string literals. No model-supplied source is ever executed.
    lines=['def generated_features(x, learned):','    out = {}']
    for i,op in enumerate(recipe.get('feature_ops',[])):
        kind=op['op'];cols=op['columns'];key=f'__atlas_op_{i}'
        if kind in {'drop','ip','clip','combine'}:continue # trusted specialized parser/drop runs in FeatureBuilder
        if kind=='frequency':
            for c in cols:lines.append(f'    out[{(key+"_"+c)!r}] = x[{c!r}].fillna("__MISSING__").astype(str).map(learned.counts[{c!r}]).fillna(0).astype(float)')
        elif kind=='text_stats':
            for c in cols:
                lines += [f'    out[{(key+"_"+c+"_len")!r}] = x[{c!r}].fillna("").astype(str).str.len().astype(float)',f'    out[{(key+"_"+c+"_digits")!r}] = x[{c!r}].fillna("").astype(str).str.count(r"\\d").astype(float)']
        elif kind=='group_mean':lines.append(f'    out[{key!r}] = x[{cols[0]!r}].fillna("__MISSING__").astype(str).map(learned.group_stats[{i}]).astype(float)')
        else:
            a=f'numeric(x[{cols[0]!r}])';b=f'numeric(x[{cols[1]!r}])' if len(cols)>1 else ''
            expr={'log1p':f'np.sign({a}) * np.log1p(np.abs({a}))','sqrt':f'np.sign({a}) * np.sqrt(np.abs({a}))','square':f'np.clip({a}, -1e100, 1e100)**2','ratio':f'({a} / {b}.where({b}.abs() > 1e-9)).clip(-1e12, 1e12)','difference':f'{a} - {b}','product':f'({a}.clip(-1e100,1e100) * {b}.clip(-1e100,1e100)).clip(-1e200,1e200)'}[kind]
            lines.append(f'    out[{key!r}] = {expr}')
    lines+=['    return out','']
    prefix=f'import sys\nsys.path.insert(0, {str(code_root)!r})\nimport numpy as np\nfrom atlas.features import numeric\n'
    source=prefix+'\n'.join(lines)+f'\nfrom atlas.worker import execute\nexecute({str(job_path)!r}, feature_function=generated_features)\n'
    compile(source,'recipe.py','exec')
    return source
