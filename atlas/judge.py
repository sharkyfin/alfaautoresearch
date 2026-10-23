"""Independent objective and compact residual/per-class diagnostic summaries."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
import pandas as pd
from .context import error_associations,input_importance


def diagnostics(data,idx,pred):
    y=data.y[idx];spec=data.spec
    if spec.classification:
        hard=pred.argmax(1);labels=list(range(len(spec.labels)))
        report=classification_report(y,hard,labels=labels,target_names=[str(x) for x in spec.labels],zero_division=0,output_dict=True)
        if len(labels)>40:
            from collections import Counter
            confusion={'top_errors':[[int(a),int(b),int(n)] for (a,b),n in Counter(zip(y[y!=hard],hard[y!=hard])).most_common(80)]}
            keys=[str(spec.labels[i]) for i in np.argsort(np.bincount(y,minlength=len(labels)))[:40]]
            report={k:v for k,v in report.items() if k in keys or k in {'macro avg','weighted avg','accuracy'}}
        else:confusion=confusion_matrix(y,hard,labels=labels).tolist()
        confidence=pred.max(1);correct=(hard==y).astype(float);calibration=[];ece=0.
        for lower,upper in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
            mask=(confidence>=lower)&(confidence<=upper if upper==1 else confidence<upper)
            if mask.any():
                accuracy=float(correct[mask].mean());mean_probability=float(confidence[mask].mean())
                ece+=float(mask.mean())*abs(accuracy-mean_probability)
                calibration.append({'support':int(mask.sum()),'confidence':mean_probability,'accuracy':accuracy})
        brier=float(np.mean(np.sum(pred*pred,axis=1)-2*pred[np.arange(len(y)),y.astype(int)]+1))
        return {'per_class':report,'confusion_matrix':confusion,'mean_max_probability':float(confidence.mean()),'calibration_bins':calibration,'expected_calibration_error':ece,'brier_score':max(0.,brier)}
    residual=np.asarray(pred)-y
    bins=[]
    edges=np.unique(np.quantile(y,np.linspace(0,1,11)))
    for i in range(len(edges)-1):
        mask=(y>=edges[i])&(y<=edges[i+1] if i==len(edges)-2 else y<edges[i+1])
        if mask.any():bins.append({'target_range':[float(edges[i]),float(edges[i+1])],'n':int(mask.sum()),'rmse':float(np.sqrt(np.mean(residual[mask]**2))),'mae':float(np.mean(abs(residual[mask]))),'bias':float(np.mean(residual[mask]))})
    groups=[]
    for col in spec.group_candidates[:3]:
        frame=pd.DataFrame({'group':data.train.iloc[idx][col].fillna('__MISSING__').astype(str).to_numpy(),'residual':residual})
        # Expose aggregate errors without sending entity identifiers to the researcher.
        rows=[]
        for _,part in frame.groupby('group',sort=False):
            if len(part)<3:continue
            errors=part['residual'].to_numpy();scale=max(float(abs(errors).max()),1.)
            rows.append({'column':col,'support':len(part),'rmse':float(scale*np.sqrt(np.mean((errors/scale)**2))),'mae':float(np.mean(abs(errors))),'bias':float(np.mean(errors))})
        groups.extend(sorted(rows,key=lambda r:-r['rmse'])[:5])
    return {'residual_bias':float(residual.mean()),'rmse_by_target_quantile':bins,'max_absolute_residual':float(np.max(abs(residual))),'worst_groups':groups,'undershoot_fraction':float(np.mean(residual<0)),'overshoot_fraction':float(np.mean(residual>0)),'residual_quantiles':{str(q):float(v) for q,v in zip([.01,.5,.99],np.quantile(residual,[.01,.5,.99]))}}


def evaluate(data,split,predictions,model_info=None):
    pv,pt=predictions;metric=data.spec.metric
    score=metric.score(data.y[split.valid],pv)
    detail={**(model_info or {}).get('diagnostics',{}),**diagnostics(data,split.valid,pv)}
    importance=input_importance(data.spec,(model_info or {}).get('recipe',{}),detail)
    detail['error_associations']=error_associations(data,split.valid,pv,importance)
    train_metric=(model_info or {}).get('train_sample_metric')
    if train_metric is not None:detail['train_validation_utility_gap']=metric.utility(train_metric)-metric.utility(score)
    result={'metric_name':metric.name,'metric_value':score,'metric_direction':metric.direction,'standardized_utility':metric.utility(score),'diagnostics':detail}
    if data.spec.classification:
        missing=np.setdiff1d(np.arange(len(data.spec.labels)),np.unique(data.y[split.valid])).astype(int).tolist()
        result['metric_feasibility']={'unvalidated_class_indices':missing,
            'all_classes_validated':not missing,
            'macro_fold_aggregation_allowed':not missing or metric.name not in {'f1_macro','balanced_accuracy'}}
    return result


def meaningful_gain(new,old):return new>old+max(1e-5,abs(old)*.0005)
