"""Evaluator-only paired metrics and trace audit; no training or policy tuning."""
from __future__ import annotations
import argparse,csv,json,tempfile
from pathlib import Path
import numpy as np,pandas as pd
from scripts.competitive_data import prepare
from atlas.metrics import Metric

def summary_row(r):
    history_path=Path(r['run_dir'])/'experiments.jsonl'
    history=[json.loads(line) for line in history_path.read_text().splitlines()] if history_path.exists() else []
    families={}
    for row in history:
        f=row.get('recipe',{}).get('model_family','unknown');phase=row.get('phase','search')
        cell=families.setdefault(f,{'attempts':0,'errors':0,'seconds':0.,'search_seconds':0.,'confirmation_fits':0,'refit_fits':0})
        cell['attempts']+=1;cell['errors']+=row.get('status')!='ok';cell['seconds']+=row.get('runtime_sec',0.)
        if phase=='search':cell['search_seconds']+=row.get('runtime_sec',0.)
        if phase=='confirmation':cell['confirmation_fits']+=1
        if phase=='refit':cell['refit_fits']+=1
    return {k:r.get(k) for k in ['case','arm','mode','phase','metric','external_metric','budget_seconds','wall_seconds','cpu_seconds','peak_memory_mb','full_refit','status','qwen_calls','qwen_fits','qwen_accepted','archive_sha256','input_sha256','champion_family','champion_recipe','cross_validation','ensemble']}|{'family_trace':families,'worker_errors':sum(c['errors'] for c in families.values())}

def paired(a,b,samples=500):
    assert a['input_sha256']==b['input_sha256'] and a['metric']==b['metric']
    with tempfile.TemporaryDirectory(prefix='atlas-evaluator-analysis-') as tmp:
        meta,truth=prepare(a['case'],Path(tmp)/'input',a['phase'])
    def pred(r):
        output=Path(r['run_dir']).parent/'task'/'submission.csv'
        frame=pd.read_csv(output,dtype={'id':str});assert frame.id.tolist()==truth.id.tolist()
        value=frame.target.to_numpy(dtype=float)
        return np.column_stack((1-value,value)) if meta['labels'] else value
    pa,pb=pred(a),pred(b);y=truth.target.map({v:i for i,v in enumerate(meta['labels'])}).to_numpy() if meta['labels'] else truth.target.to_numpy()
    metric=Metric(a['metric'],len(meta['labels']))
    delta=metric.direction*(metric.score(y,pb)-metric.score(y,pa))
    rng=np.random.default_rng(817);values=[];n=len(y)
    for _ in range(samples):
        if meta['split_regime']=='temporal':
            # Circular moving-block bootstrap, one week/hourly block. Dependence
            # within weeks is retained; few weeks still limits inference.
            width=min(168,n);start=rng.integers(0,n,size=int(np.ceil(n/width)))
            ix=((start[:,None]+np.arange(width))%n).ravel()[:n]
        else:
            ix=np.concatenate([rng.choice(np.flatnonzero(y==c),size=(y==c).sum(),replace=True) for c in np.unique(y)])
        values.append(metric.direction*(metric.score(y[ix],pb[ix])-metric.score(y[ix],pa[ix])))
    return {'before':a['arm'],'after':b['arm'],'case':a['case'],'phase':a['phase'],'oriented_metric_delta':delta,
        'ci95_percentile':np.quantile(values,[.025,.975]).tolist(),'bootstrap_samples':samples,
        'bootstrap':'paired circular weekly blocks' if meta['split_regime']=='temporal' else 'paired stratified rows',
        'scope':'conditional on fitted models and this split; no seed/Qwen/search variability estimate; never used for architecture tuning'}

def main():
    p=argparse.ArgumentParser();p.add_argument('reports',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    runs=[r for f in args.reports for r in json.loads(f.read_text())['runs']]
    rows=[summary_row(r) for r in runs];pairs=[]
    for phase in sorted({r['phase'] for r in runs}):
        for case in sorted({r['case'] for r in runs if r['phase']==phase}):
            group=[r for r in runs if r['phase']==phase and r['case']==case and r['mode']=='live-agent' and r['status']=='ok']
            if len(group)==2:pairs.append(paired(group[0],group[1]))
    result={'rows':rows,'paired_uncertainty':pairs,'limits':'No universal generalization claim from two datasets/one search seed. Runtime data exclude human/tool work.'}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with args.output.with_suffix('.csv').open('w') as f:
        keys=['case','phase','arm','mode','metric','external_metric','wall_seconds','cpu_seconds','peak_memory_mb','full_refit','qwen_calls','qwen_fits','qwen_accepted','worker_errors','archive_sha256']
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'rows':len(rows),'pairs':pairs},indent=2))
if __name__=='__main__':main()
