"""Development-only deterministic profiling; never ships with the ML agent."""
from __future__ import annotations
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--arm',required=True);args=p.parse_args()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='8'
    sys.path.insert(0,str(args.source.resolve()))
    import numpy as np
    from atlas import task
    from atlas.contract import read_inputs
    from atlas.features import FeatureBuilder
    from atlas.budget import Budget
    from atlas.search import SearchController
    from atlas.recipes import recipe_hash
    from atlas.agent import researcher_context,compact_context
    from atlas.utils import code_digest
    rows=[]
    for kind in ['regression','classification']:
        inputs=read_inputs(ROOT/'artifacts/data'/kind/'task.md')
        for repeat in range(3):
            calls=[];original=task.read_csv
            def counted(*a,**kw):calls.append(str(a[0]));return original(*a,**kw)
            kw={'profile_data':False} if 'profile_data' in __import__('inspect').signature(task.load).parameters else {}
            start=time.perf_counter()
            with patch.object(task,'read_csv',counted):data=task.load(inputs,**kw)
            elapsed=time.perf_counter()-start
            digest=hashlib.sha256()
            import pandas as pd
            for frame in [data.train,data.test,data.sample,data.dictionary]:digest.update(pd.util.hash_pandas_object(frame,index=True).to_numpy().tobytes())
            digest.update(data.y.tobytes())
            rows.append({'task':kind,'repeat':repeat,'load_seconds':elapsed,'csv_reads':len(calls),'data_digest':digest.hexdigest(),'profile_requested':not bool(kw)})
        a=data.train.iloc[:min(len(data.train),30000)]
        f=FeatureBuilder(data.spec,[{'op':'ratio','columns':data.spec.numeric_features[:2]}] if len(data.spec.numeric_features)>=2 else [])
        original=FeatureBuilder.transform;calls=[]
        def transformed(self,x):calls.append(len(x));return original(self,x)
        start=time.perf_counter()
        with patch.object(FeatureBuilder,'transform',transformed):
            if hasattr(f,'fit_transform'):z=f.fit_transform(a)
            else:f.fit(a);z=f.transform(a)
        rows.append({'task':kind,'feature_fit_transform_seconds':time.perf_counter()-start,'training_transform_calls':len(calls),'matrix_digest':hashlib.sha256(pd.util.hash_pandas_object(z,index=True).to_numpy().tobytes()).hexdigest()})
    data=task.load(read_inputs(ROOT/'artifacts/data/regression/task.md'))
    replay=[]
    # Controlled scheduler fault injection: both observed models are equally
    # useful, then the faster family crosses the existing two-failure ban.
    for seed in [42,53]:
        c=SearchController(data.spec,Budget(3600),seed=seed);c.queue=[];c.research_index=8
        for family,seconds in [('linear',1.),('lightgbm',10.)]:
            r={'model_family':family,'params':{'alpha':10} if family=='linear' else {'iterations':100,'num_leaves':31},'feature_ops':[],'feature_expressions':[],'target_transform':'none'}
            result={'experiment_id':family,'recipe':r,'standardized_utility':-10.,'metric_value':10.,'runtime_sec':seconds,'status':'ok'}
            c.observe(result)
        champ={**c.family_best['linear'],'metric_value':10.}
        c.failures['linear']=2;families=[]
        for i in range(24):
            proposal=c.next(champ)
            if proposal is None:break
            families.append(proposal['model_family']);c.seen.add(recipe_hash(proposal))
        replay.append({'seed':seed,'proposals':len(families),'already_blocked_proposals':families.count('linear'),'families':families})
    # Measured context retention, not a claim about actual LLM reasoning.
    history=[]
    for i in range(20):
        r={'model_family':'linear','params':{'alpha':1+i},'feature_ops':[],'feature_expressions':[{'name':f'trial_{i}','expression':f'x0 * {i+1}'}],'target_transform':'none'}
        history.append({'experiment_id':f'exp_{i:04d}','origin':'qwen' if i<10 else 'deterministic','recipe':r,'status':'ok','metric_value':10+i,'standardized_utility':-10-i,'runtime_sec':1.,'accepted':False})
    champ={'experiment_id':'champion','recipe':{'model_family':'linear','params':{'alpha':10},'feature_ops':[],'feature_expressions':[],'target_transform':'none'},'metric_value':9.,'standardized_utility':-9.,'diagnostics':{}}
    context=compact_context(researcher_context(data,history,champ,500))
    out={'arm':args.arm,'source_sha':code_digest(),'load_and_features':rows,'scheduler_controlled_replay':replay,'context_bytes':len(json.dumps(context,ensure_ascii=False).encode()),'context_research_evidence':context.get('research_evidence'),'context_history_ids':[r.get('experiment_id') for r in context.get('history',[])],'scope':'Real public input profiling and deterministic controlled scheduler/context probes; not an ML quality benchmark'}
    destination=ROOT/'reports/research_efficiency_20261023'/f'probe-{args.arm}.json'
    assert not destination.exists();destination.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

if __name__=='__main__':main()
