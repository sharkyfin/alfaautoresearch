"""Independent objective and compact residual/per-class diagnostic summaries."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix


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
        return {'per_class':report,'confusion_matrix':confusion,'mean_max_probability':float(pred.max(1).mean())}
    residual=np.asarray(pred)-y
    bins=[]
    edges=np.unique(np.quantile(y,[0,.25,.5,.75,1]))
    for i in range(len(edges)-1):
        mask=(y>=edges[i])&(y<=edges[i+1] if i==len(edges)-2 else y<edges[i+1])
        if mask.any():bins.append({'target_range':[float(edges[i]),float(edges[i+1])],'n':int(mask.sum()),'rmse':float(np.sqrt(np.mean(residual[mask]**2))),'mae':float(np.mean(abs(residual[mask])))})
    return {'residual_bias':float(residual.mean()),'rmse_by_target_quantile':bins,'max_absolute_residual':float(np.max(abs(residual)))}


def evaluate(data,split,predictions):
    pv,pt=predictions;metric=data.spec.metric
    score=metric.score(data.y[split.valid],pv)
    return {'metric_name':metric.name,'metric_value':score,'metric_direction':metric.direction,'standardized_utility':metric.utility(score),'diagnostics':diagnostics(data,split.valid,pv)}


def meaningful_gain(new,old):return new>old+max(1e-5,abs(old)*.0005)
