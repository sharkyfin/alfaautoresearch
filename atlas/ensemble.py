"""Bounded convex blending on common honest holdouts; independent confirmation."""
from __future__ import annotations
import numpy as np
from .judge import meaningful_gain


def controls(value):
    if value is None:return []
    return value if isinstance(value,list) else [value]


def robust_utility(values):
    values=np.asarray(values,dtype=float)
    return float(values.mean()-.25*values.std())


def choose_blend(data,finalists,confirmed):
    if len(finalists)<2:return None
    metric=data.spec.metric;best=None
    common=None
    for candidate in finalists:
        ids={r['split'].id for r in controls(confirmed.get(candidate['experiment_id']))}
        common=ids if common is None else common&ids
    if not common:return None
    evidence={c['experiment_id']:{r['split'].id:r for r in controls(confirmed.get(c['experiment_id'])) if r['split'].id in common} for c in finalists}
    reference=max(robust_utility([c['standardized_utility'],*[r['standardized_utility'] for r in evidence[c['experiment_id']].values()]]) for c in finalists)
    for i,a in enumerate(finalists[:3]):
        for b in finalists[:3][i+1:]:
            ca=evidence[a['experiment_id']];cb=evidence[b['experiment_id']]
            if a['split'].id!=b['split'].id:continue
            # Search only 3 fixed weights; confirmation is evaluated once after choosing.
            base=max(a['standardized_utility'],b['standardized_utility'])
            choices=[]
            for w in [.25,.5,.75]:
                pv=w*a['predictions'][0]+(1-w)*b['predictions'][0]
                s=metric.score(data.y[a['split'].valid],pv)
                choices.append((metric.utility(s),w,s,pv))
            utility,w,score,pv=max(choices,key=lambda t:t[0])
            if not meaningful_gain(utility,base):continue
            confirmation=[]
            for window in sorted(common):
                first,second=ca[window],cb[window]
                cv=w*first['predictions'][0]+(1-w)*second['predictions'][0]
                confirm_score=metric.score(data.y[first['split'].valid],cv)
                confirm_utility=metric.utility(confirm_score)
                if confirm_utility<=max(first['standardized_utility'],second['standardized_utility']):break
                confirmation.append({'split_id':window,'metric_value':confirm_score,'utility':confirm_utility})
            if len(confirmation)!=len(common):continue
            robust=robust_utility([utility,*[c['utility'] for c in confirmation]])
            if not meaningful_gain(robust,reference):continue
            if best is None or robust>best['robust_utility']:
                best={'components':[a,b],'weights':[w,1-w],'predictions':(pv,w*a['predictions'][1]+(1-w)*b['predictions'][1]),'metric_value':score,'confirmation_metric':float(np.mean([c['metric_value'] for c in confirmation])),'confirmation_metrics':confirmation,'robust_utility':robust}
    return best


def adjust_probabilities(p,weights):
    q=p*np.asarray(weights)[None,:];return q/np.maximum(q.sum(1,keepdims=True),1e-15)


def tune_multipliers(data,champion,confirmation=None):
    spec=data.spec
    if not spec.classification or spec.metric.needs_proba or spec.metric.name not in {'f1','f1_macro','balanced_accuracy'}:return None
    y=data.y[champion['split'].train];counts=np.bincount(y,minlength=len(spec.labels))
    if counts.min()<5:return None
    counts=counts/np.max(counts);base=champion['standardized_utility'];best=None
    for power in [-.25,-.1,.1,.25]:
        w=np.clip(counts**power,.5,2.)
        p=adjust_probabilities(champion['predictions'][0],w);score=spec.metric.score(data.y[champion['split'].valid],p)
        if meaningful_gain(spec.metric.utility(score),base):best=(score,w,p);base=spec.metric.utility(score)
    if best is None:return None
    score,w,p=best
    evidence=controls(confirmation)
    if not evidence:return None
    scores=[]
    for control in evidence:
        cp=adjust_probabilities(control['predictions'][0],w)
        cs=spec.metric.score(data.y[control['split'].valid],cp)
        if not meaningful_gain(spec.metric.utility(cs),control['standardized_utility']):return None
        scores.append({'split_id':control['split'].id,'metric_value':cs})
    return {'weights':w.tolist(),'metric_value':score,'confirmation_metric':float(np.mean([s['metric_value'] for s in scores])),'confirmation_metrics':scores,'predictions':(p,adjust_probabilities(champion['predictions'][1],w))}
