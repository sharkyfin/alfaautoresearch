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


def choose_blend(data,finalists,confirmed,strategy='greedy'):
    if strategy=='none':return None
    if strategy not in {'weighted','greedy'}:raise ValueError('Unsupported ensemble strategy')
    if len(finalists)<2:return None
    metric=data.spec.metric
    candidates=finalists[:5]
    primary=candidates[0]['split']
    if any(c['split'].id!=primary.id for c in candidates):return None
    common=None
    for candidate in candidates:
        ids={r['split'].id for r in controls(confirmed.get(candidate['experiment_id']))}
        common=ids if common is None else common&ids
    if not common:return None
    evidence={c['experiment_id']:{r['split'].id:r for r in controls(confirmed.get(c['experiment_id'])) if r['split'].id in common} for c in candidates}
    y=data.y[primary.valid]
    matrices=[c['predictions'][0] for c in candidates]
    choices=[]
    # All weights are selected using primary predictions only. Exactly one
    # selected ensemble is then checked on controls; failed checks cannot start
    # a second weight search against those same control labels.
    for i in range(len(candidates)):
        for j in range(i+1,len(candidates)):
            for weight in [.25,.5,.75]:
                weights=np.zeros(len(candidates));weights[i]=weight;weights[j]=1-weight
                pred=weight*matrices[i]+(1-weight)*matrices[j]
                choices.append((metric.utility(metric.score(y,pred)),weights,pred,'fixed_convex'))
    # Caruana-style selection with replacement, capped at eight steps and three
    # distinct components. A first model alone is never called an ensemble.
    best_index=max(range(len(candidates)),key=lambda i:candidates[i]['standardized_utility'])
    counts=np.zeros(len(candidates));counts[best_index]=1
    running=matrices[best_index].copy()
    for step in range(1,8 if strategy=='greedy' else 1):
        options=[]
        for i,matrix in enumerate(matrices):
            if counts[i]==0 and np.count_nonzero(counts)>=3:continue
            pred=(running+matrix)/(step+1)
            options.append((metric.utility(metric.score(y,pred)),i,pred))
        utility,index,pred=max(options,key=lambda row:row[0])
        counts[index]+=1;running+=matrices[index]
        if np.count_nonzero(counts)>1:choices.append((utility,counts.copy()/(step+1),pred,'greedy_primary'))
    utility,weights,pv,strategy=max(choices,key=lambda row:row[0])
    if not meaningful_gain(utility,max(c['standardized_utility'] for c in candidates)):return None
    selected=[i for i,w in enumerate(weights) if w>0]
    confirmation=[]
    for window in sorted(common):
        rows=[evidence[candidates[i]['experiment_id']][window] for i in selected]
        split=rows[0]['split']
        if any(not np.array_equal(row['split'].valid,split.valid) for row in rows):return None
        pred=sum(weights[i]*evidence[candidates[i]['experiment_id']][window]['predictions'][0] for i in selected)
        score=metric.score(data.y[split.valid],pred)
        value=metric.utility(score)
        reference=max(evidence[c['experiment_id']][window]['standardized_utility'] for c in candidates)
        if not meaningful_gain(value,reference):return None
        confirmation.append({'split_id':window,'metric_value':score,'utility':value})
    robust=robust_utility([utility,*[r['utility'] for r in confirmation]])
    reference=max(robust_utility([c['standardized_utility'],*[r['standardized_utility'] for r in evidence[c['experiment_id']].values()]]) for c in candidates)
    if not meaningful_gain(robust,reference):return None
    residuals=[(matrices[i]-(np.eye(matrices[i].shape[1])[y.astype(int)] if data.spec.classification else y)).reshape(-1) for i in selected]
    correlations=[]
    for a in range(len(selected)):
        for b in range(a+1,len(selected)):
            value=float(np.corrcoef(residuals[a],residuals[b])[0,1]) if np.std(residuals[a])>0 and np.std(residuals[b])>0 else None
            correlations.append({'components':[selected[a],selected[b]],'residual_correlation':value})
    return {'components':[candidates[i] for i in selected],'weights':[float(weights[i]) for i in selected],
        'predictions':(pv,sum(weights[i]*candidates[i]['predictions'][1] for i in selected)),
        'metric_value':metric.direction*utility,'confirmation_metric':float(np.mean([r['metric_value'] for r in confirmation])),
        'confirmation_metrics':confirmation,'robust_utility':robust,'strategy':strategy,
        'weight_selection':'primary legal holdout only; one frozen proposal checked on controls',
        'residual_correlations':correlations}


def store_oof(path,data,candidates,confirmed,max_bytes=128*2**20):
    """Persist row-aligned legal out-of-fold predictions with explicit coverage."""
    arrays={};records=[]
    expected=len(data.y)*(max(len(data.spec.labels),1) if data.spec.classification else 1)*5*len(candidates)
    if expected>max_bytes:
        return {'path':None,'storage':'existing per-fold prediction/split artifacts; aligned matrix exceeds memory bound',
            'estimated_bytes':expected,'memory_bound_bytes':max_bytes,
            'candidates':[{'experiment_id':candidate['experiment_id'],
                'fold_artifacts':[row.get('artifact_path') for row in [candidate,*controls(confirmed.get(candidate['experiment_id']))]]}
                for candidate in candidates]}
    for number,candidate in enumerate(candidates):
        rows=[candidate,*controls(confirmed.get(candidate['experiment_id']))]
        shape=(len(data.y),len(data.spec.labels)) if data.spec.classification else (len(data.y),)
        values=np.full(shape,np.nan,dtype=np.float32);covered=np.zeros(len(data.y),dtype=bool)
        folds=[]
        for row in rows:
            split=row['split']
            if np.intersect1d(split.train,split.valid).size:raise ValueError('OOF training/validation overlap')
            if covered[split.valid].any():raise ValueError('OOF validation rows repeated')
            values[split.valid]=row['predictions'][0];covered[split.valid]=True
            folds.append({'split_id':split.id,'train_rows':len(split.train),'valid_rows':len(split.valid)})
        arrays['prediction_'+str(number)]=values;arrays['covered_'+str(number)]=covered
        records.append({'experiment_id':candidate['experiment_id'],'array':'prediction_'+str(number),
            'covered_rows':int(covered.sum()),'total_rows':len(data.y),'folds':folds})
    np.savez_compressed(path,**arrays)
    return {'path':str(path),'candidates':records,'missing_rows':'NaN with explicit coverage mask; never imputed using targets'}


def adjust_probabilities(p,weights):
    q=p*np.asarray(weights)[None,:];return q/np.maximum(q.sum(1,keepdims=True),1e-15)


def binary_decision_weights(y,probabilities,metric):
    """Select one bounded threshold on the primary split, without control labels."""
    y=np.asarray(y);p=np.asarray(probabilities)[:,1]
    counts=np.bincount(y.astype(int),minlength=2)
    if counts.min()<5:return None
    thresholds=np.unique(np.r_[np.linspace(.05,.95,37),.5,
                                np.clip(np.quantile(p,[.05,.1,.2,.35,.5,.65,.8,.9,.95]),.05,.95)])
    order=np.argsort(p,kind='stable');positive=(y[order]==1).astype(int)
    cumulative=np.r_[0,np.cumsum(positive)]
    cuts=np.searchsorted(p[order],thresholds,side='right')
    fn=cumulative[cuts];tp=counts[1]-fn;tn=cuts-fn;fp=counts[0]-tn
    positive_f1=2*tp/np.maximum(2*tp+fp+fn,1)
    negative_f1=2*tn/np.maximum(2*tn+fp+fn,1)
    if metric.name=='f1_macro':scores=(positive_f1+negative_f1)/2
    elif metric.name=='balanced_accuracy':scores=(tp/counts[1]+tn/counts[0])/2
    else:scores=positive_f1 if metric.positive_index==1 else negative_f1
    maximum=scores.max()
    tied=np.flatnonzero(np.isclose(scores,maximum,rtol=0,atol=1e-12))
    index=tied[np.argmin(np.abs(thresholds[tied]-.5))]
    threshold=float(thresholds[index])
    return np.array([1.,(1-threshold)/threshold]),threshold


def multiclass_decision_weights(y,probabilities,metric):
    """Bounded coordinate search on primary predictions; controls remain sealed."""
    y=np.asarray(y,dtype=int);p=np.asarray(probabilities)
    counts=np.bincount(y,minlength=p.shape[1])
    if p.shape[1]<3 or p.shape[1]>32 or counts.min()<5:return None
    # Favour supported classes with weak recall. At most eight coordinates and
    # two sweeps avoid fitting a free parameter to every rare validation row.
    hard=p.argmax(1)
    recall=np.array([np.mean(hard[y==c]==c) for c in range(p.shape[1])])
    selected=np.argsort(recall,kind='stable')[:8]
    weights=np.ones(p.shape[1]);score=metric.score(y,p)
    for _ in range(2):
        changed=False
        for c in selected:
            best=(score,weights[c])
            for value in [.67,.82,1.,1.22,1.5]:
                trial=weights.copy();trial[c]=value
                measured=metric.score(y,adjust_probabilities(p,trial))
                if measured>best[0]+1e-5:best=(measured,value)
            if best[0]>score+1e-5:
                score=best[0];weights[c]=best[1];changed=True
        if not changed:break
    return weights if not np.allclose(weights,1.) else None


def tune_multipliers(data,champion,confirmation=None):
    spec=data.spec
    if not spec.classification or spec.metric.needs_proba or spec.metric.name not in {'f1','f1_macro','balanced_accuracy'}:return None
    y=data.y[champion['split'].train];counts=np.bincount(y,minlength=len(spec.labels))
    if counts.min()<5:return None
    counts=counts/np.max(counts);base=champion['standardized_utility'];best=None
    proposals=[]
    for power in [-.25,-.1,.1,.25]:
        w=np.clip(counts**power,.5,2.)
        proposals.append((w,{'strategy':'frequency','power':power}))
    if len(spec.labels)==2:
        decision=binary_decision_weights(data.y[champion['split'].valid],champion['predictions'][0],spec.metric)
        if decision is not None:
            weights,threshold=decision
            proposals.append((weights,{'strategy':'binary_threshold','threshold':threshold}))
    elif spec.metric.name=='f1_macro':
        weights=multiclass_decision_weights(data.y[champion['split'].valid],champion['predictions'][0],spec.metric)
        if weights is not None:proposals.append((weights,{'strategy':'bounded_class_coordinates'}))
    for w,detail in proposals:
        p=adjust_probabilities(champion['predictions'][0],w);score=spec.metric.score(data.y[champion['split'].valid],p)
        if meaningful_gain(spec.metric.utility(score),base):best=(score,w,p,detail);base=spec.metric.utility(score)
    if best is None:return None
    score,w,p,detail=best
    evidence=controls(confirmation)
    if not evidence:return None
    scores=[]
    for control in evidence:
        cp=adjust_probabilities(control['predictions'][0],w)
        cs=spec.metric.score(data.y[control['split'].valid],cp)
        if not meaningful_gain(spec.metric.utility(cs),control['standardized_utility']):return None
        scores.append({'split_id':control['split'].id,'metric_value':cs})
    return {'weights':w.tolist(),'metric_value':score,'confirmation_metric':float(np.mean([s['metric_value'] for s in scores])),'confirmation_metrics':scores,**detail,'predictions':(p,adjust_probabilities(champion['predictions'][1],w))}
