"""Verify ensemble/threshold math and independent-control rejection."""
import numpy as np
from atlas.task import load
from atlas.contract import read_inputs
from atlas.validation import design
from atlas.ensemble import choose_blend,tune_multipliers,adjust_probabilities
from atlas.submit import checkpoint
from scripts.synthetic import create


def test_blend_requires_improvement_on_both_splits(tmp_path):
    data=load(read_inputs(create(tmp_path)));s=design(data);c=design(data,111,fraction=.25)
    def candidate(id_,split,sign):
        error=np.where(np.arange(len(split.valid))%2,1.,-1.)*sign
        p=data.y[split.valid]+error
        return {'experiment_id':id_,'split':split,'predictions':(p,np.zeros(len(data.test))+sign),'standardized_utility':-1.}
    a,b=candidate('a',s,1),candidate('b',s,-1);ca,cb=candidate('a',c,1),candidate('b',c,-1)
    result=choose_blend(data,[a,b],{'a':ca,'b':cb})
    assert result and result['metric_value']<1e-12 and result['confirmation_metric']<1e-12
    cb['predictions']=(ca['predictions'][0].copy(),cb['predictions'][1])
    assert choose_blend(data,[a,b],{'a':ca,'b':cb}) is None
    assert choose_blend(data,[a,b],{}) is None


def test_multipliers_need_control_and_preserve_probabilities(tmp_path):
    data=load(read_inputs(create(tmp_path,'binary_imbalance',n=1000)));s=design(data);c=design(data,111,fraction=.25)
    # Controlled probability underprediction of the rare class on two independent splits.
    def candidate(split):
        yy=data.y[split.valid];p=np.stack([np.where(yy==1,.55,.99),np.where(yy==1,.45,.01)],axis=1)
        return {'split':split,'predictions':(p,np.tile([.6,.4],(len(data.test),1))),'standardized_utility':data.spec.metric.score(yy,p)}
    a,b=candidate(s),candidate(c)
    assert tune_multipliers(data,a,None) is None
    result=tune_multipliers(data,a,b)
    assert result is not None
    q=adjust_probabilities(a['predictions'][1],result['weights'])
    assert np.allclose(q.sum(1),1) and (q>=0).all()
    assert result['metric_value']>a['standardized_utility']


def test_probability_headers_preserve_sample_class_order(tmp_path):
    task=create(tmp_path,'logloss');data=load(read_inputs(task))
    data.sample=data.sample[['id','rare','common']]
    p=np.tile([.2,.8],(len(data.test),1));checkpoint(data,p)
    import pandas as pd
    out=pd.read_csv(data.inputs.output)
    assert list(out.columns)==['id','rare','common']
    assert np.allclose(out.rare,.8) and np.allclose(out.common,.2)

def test_finalist_retention_keeps_close_different_family():
    from atlas.search import retain_finalists
    pool=[{'experiment_id':str(i),'recipe':{'model_family':family},'standardized_utility':score} for i,(family,score) in enumerate([('lightgbm',-43.),('lightgbm',-43.1),('lightgbm',-43.2),('xgboost',-43.3),('linear',-90.)])]
    selected=retain_finalists(pool)
    assert len(selected)==3 and selected[0]['standardized_utility']==-43.
    assert selected[2]['recipe']['model_family']=='xgboost'


def test_blend_cannot_replace_a_better_third_single_model():
    from types import SimpleNamespace
    from atlas.metrics import Metric
    from atlas.validation import Split
    metric=Metric('rmse')
    data=SimpleNamespace(y=np.zeros(4),spec=SimpleNamespace(metric=metric))
    primary=Split(np.array([2,3]),np.array([0,1]),'random','test',42)
    control=Split(np.array([0,1]),np.array([2,3]),'random','test',43)
    def candidate(name,errors,split):
        prediction=np.asarray(errors,dtype=float)
        return {'experiment_id':name,'split':split,'predictions':(prediction,prediction),
                'standardized_utility':metric.utility(metric.score(data.y[split.valid],prediction))}
    errors={'best':[.1,.1],'second':[2.,0.],'third':[0.,2.]}
    finalists=[candidate(name,p,primary) for name,p in errors.items()]
    confirmed={name:candidate(name,p,control) for name,p in errors.items()}
    assert choose_blend(data,finalists,confirmed) is None
