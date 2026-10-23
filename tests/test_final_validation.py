"""Sealed-label and fixed-policy correctness; no full benchmark runs."""
import copy,json,sys,types
from pathlib import Path
import numpy as np,pandas as pd,pytest
from atlas.contract import read_inputs
from atlas.task import load
from atlas.validation import research_splits,split_index_digest,confirmation_batches
from scripts.synthetic import create
from atlas import challenger

def fixture(tmp_path,scenario='iid',n=400):
    d=load(read_inputs(create(tmp_path,scenario,n=n)));s,c=research_splits(d)
    d.spec.validation_plan={**s.json(),'confirmation_isolation':'sealed',
        'search_pool_sha256':split_index_digest(np.union1d(s.train,s.valid)),
        'confirmation_train_sha256':split_index_digest(c[0].train)} if c else s.json()
    return d,s,c

@pytest.mark.parametrize('scenario',['iid','integer_labels','group','future','duplicates','missing'])
def test_controls_never_enter_search_and_batch_without_extra_fit(tmp_path,scenario):
    d,s,c=fixture(tmp_path,scenario)
    assert len(c)==2
    sealed=np.concatenate([w.valid for w in c]);pool=np.union1d(s.train,s.valid)
    assert not np.intersect1d(sealed,pool).size
    assert not np.intersect1d(c[0].valid,c[1].valid).size
    assert all(np.array_equal(w.train,c[0].train) for w in c)
    batches=confirmation_batches(c)
    assert len(batches)==1 and len(batches[0][1])==2
    if s.kind=='time':
        dates=pd.to_datetime(d.train[s.time_column],utc=True)
        for w in c:assert dates.iloc[w.train].max()<dates.iloc[w.valid].min()
    for w in [s,*c]:
        if w.kind=='duplicate_group':
            groups=pd.util.hash_pandas_object(d.train[d.spec.features],index=False).to_numpy()
            assert not np.intersect1d(groups[w.train],groups[w.valid]).size
        if w.kind=='group':
            groups=d.train[w.group_column].to_numpy()
            assert not np.intersect1d(groups[w.train],groups[w.valid]).size

@pytest.mark.parametrize('family',['catboost','lightgbm','xgboost'])
def test_confirmation_never_supplies_control_labels_to_early_stopping(tmp_path,monkeypatch,family):
    from atlas.models import fit_predict
    d,s,c=fixture(tmp_path);window=c[0];calls=[]
    class Model:
        def __init__(self,**kw):self.kw=kw;self.tree_count_=kw.get('iterations',23)
        def fit(self,x,y,**kw):
            calls.append((np.array(y),kw));assert 'eval_set' not in kw and 'callbacks' not in kw and 'early_stopping_rounds' not in kw
            return self
        def set_params(self,**kw):self.kw.update(kw)
        def predict(self,x):return np.ones(len(x))
        def get_best_iteration(self):return -1
    if family=='catboost':
        import catboost
        monkeypatch.setattr(catboost,'CatBoostRegressor',Model)
    elif family=='lightgbm':
        import lightgbm
        monkeypatch.setattr(lightgbm,'LGBMRegressor',Model)
    else:
        import xgboost
        monkeypatch.setattr(xgboost,'XGBRegressor',Model)
    r={'model_family':family,'params':{'iterations':100},'feature_ops':[],'target_transform':'none'}
    first=fit_predict(d,r,window.train,window.valid,iterations_override=23,evaluation_only=True)
    altered=copy.deepcopy(d);altered.y[window.valid]+=100000.
    second=fit_predict(altered,r,window.train,window.valid,iterations_override=23,evaluation_only=True)
    assert all(np.array_equal(y,d.y[window.train]) for y,kw in calls)
    np.testing.assert_array_equal(first[1],second[1]);assert first[2]['fit_validation_rows']==0
    assert first[2]['iterations']==23


def install_fake_ag(monkeypatch,spec,calls):
    modules={n:types.ModuleType(n) for n in ['autogluon','autogluon.tabular','autogluon.tabular.models','autogluon.features','autogluon.features.generators','autogluon.core','autogluon.core.metrics']}
    class Model:
        def __init__(self,**kw):self.kw=kw;self.params_trained={};calls.append(('init',kw))
        def fit(self,**kw):
            assert kw['X_val'] is None and kw['y_val'] is None
            calls.append(('fit',kw));return self
        def predict_proba(self,x):return np.full((len(x),len(spec.labels)),1/len(spec.labels)) if len(spec.labels)>2 else np.full(len(x),.25) if spec.classification else np.ones(len(x))
    class Generator:
        feature_metadata={}
        def __init__(self,**kw):pass
        def fit_transform(self,x,y):calls.append(('generator_rows',x.index.tolist()));return x
        def transform(self,x):return x
    modules['autogluon.tabular'].TabularPredictor=object
    for name in ['LGBModel','CatBoostModel','RFModel','XTModel']:setattr(modules['autogluon.tabular.models'],name,Model)
    modules['autogluon.features.generators'].AutoMLPipelineFeatureGenerator=Generator
    modules['autogluon.core.metrics'].make_scorer=lambda *a,**kw:object()
    for name,value in modules.items():monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setattr(challenger,'availability',lambda:True)

@pytest.mark.parametrize('scenario',['iid','integer_labels'])
def test_ag_frozen_fit_has_no_tuning_data_and_only_training_labels(tmp_path,monkeypatch,scenario):
    d,s,c=fixture(tmp_path/'data',scenario);calls=[];install_fake_ag(monkeypatch,d.spec,calls)
    policy={'version':'1.3.1','components':[{'family':'GBM','params':{'num_boost_round':17},'weight':1.,'name':'fixed','source_model':'search_model'}]}
    pv,pt,info=challenger.fit_predict(d,c[0].train,c[0].valid,tmp_path/'run',30,42,8,8192,policy=policy,evaluation_only=True)
    fit=next(kw for name,kw in calls if name=='fit')
    np.testing.assert_array_equal(fit['y'],d.y[c[0].train])
    assert fit['num_cpus']==8 and fit['num_gpus']==0 and info['fit_validation_rows']==0
    assert next(kw for name,kw in calls if name=='init')['hyperparameters']['num_boost_round']==17
    assert len(pv)==len(c[0].valid) and len(pt)==len(d.test)
    assert next(kw for name,kw in calls if name=='generator_rows')==d.train.iloc[c[0].train].index.tolist()


def test_ag_refit_uses_every_known_label_with_fixed_models(tmp_path,monkeypatch):
    d,s,c=fixture(tmp_path/'data');calls=[];install_fake_ag(monkeypatch,d.spec,calls)
    policy={'version':'1.3.1','components':[{'family':'CAT','params':{'iterations':31},'weight':1.,'name':'fixed'}]}
    pv,pt,info=challenger.fit_predict(d,np.arange(len(d.y)),np.empty(0,dtype=int),tmp_path/'refit',30,42,8,8192,policy=policy,refit=True)
    fit=next(kw for name,kw in calls if name=='fit');np.testing.assert_array_equal(fit['y'],d.y)
    assert len(pv)==0 and info['model_fit_rows']==len(d.y)
    assert next(kw for name,kw in calls if name=='init')['hyperparameters']['iterations']==31


def test_positive_class_sentence_punctuation_is_commentary(tmp_path):
    task=create(tmp_path,'integer_labels',n=160)
    d=load(read_inputs(task));positive=str(d.spec.labels[0])
    text=task.read_text();task.write_text(text+'\nPositive class: '+positive+'.\n')
    updated=load(read_inputs(task));assert updated.spec.metric.positive_index==0


def test_rmsle_portfolio_uses_mathematically_matching_log_objective(tmp_path):
    from atlas.search import initial_portfolio
    from atlas.metrics import Metric
    d,s,c=fixture(tmp_path)
    d.spec.metric=Metric('rmsle');assert all(r['target_transform']=='log1p' for r in initial_portfolio(d.spec,600))
    y=np.array([0.,1.,20.,100.]);p=np.array([.1,1.2,18.,110.])
    actual=challenger.CanonicalScore(d.spec.metric,log_target=True)(np.log1p(y),np.log1p(p))
    assert actual==pytest.approx(d.spec.metric.score(y,p))
