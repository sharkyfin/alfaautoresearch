"""Bounded convex blending on common honest holdouts; independent confirmation."""
from __future__ import annotations
import numpy as np
from .judge import meaningful_gain


def choose_blend(data,finalists,confirmed):
    if len(finalists)<2:return None
    metric=data.spec.metric;best=None
    for i,a in enumerate(finalists[:3]):
        for b in finalists[:3][i+1:]:
            if a['experiment_id'] not in confirmed or b['experiment_id'] not in confirmed:continue
            ca=confirmed[a['experiment_id']];cb=confirmed[b['experiment_id']]
            # Search only 3 fixed weights; confirmation is evaluated once after choosing.
            base=max(a['standardized_utility'],b['standardized_utility'])
            choices=[]
            for w in [.25,.5,.75]:
                pv=w*a['predictions'][0]+(1-w)*b['predictions'][0]
                s=metric.score(data.y[a['split'].valid],pv)
                choices.append((metric.utility(s),w,s,pv))
            utility,w,score,pv=max(choices,key=lambda t:t[0])
            if not meaningful_gain(utility,base):continue
            cv=w*ca['predictions'][0]+(1-w)*cb['predictions'][0]
            confirm_score=metric.score(data.y[ca['split'].valid],cv)
            confirm_utility=metric.utility(confirm_score)
            if confirm_utility<=max(ca['standardized_utility'],cb['standardized_utility']):continue
            robust=(utility+confirm_utility)/2
            if best is None or robust>best['robust_utility']:
                best={'components':[a,b],'weights':[w,1-w],'predictions':(pv,w*a['predictions'][1]+(1-w)*b['predictions'][1]),'metric_value':score,'confirmation_metric':confirm_score,'robust_utility':robust}
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
    if confirmation is None:return None
    cp=adjust_probabilities(confirmation['predictions'][0],w)
    cs=spec.metric.score(data.y[confirmation['split'].valid],cp)
    if not meaningful_gain(spec.metric.utility(cs),confirmation['standardized_utility']):return None
    return {'weights':w.tolist(),'metric_value':score,'confirmation_metric':cs,'predictions':(p,adjust_probabilities(champion['predictions'][1],w))}
