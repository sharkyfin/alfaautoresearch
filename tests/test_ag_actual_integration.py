"""Small actual-library integration: reuse frozen policy without control labels."""
import copy
import numpy as np,pandas as pd,pytest
pytest.importorskip('autogluon.tabular')
from scripts.synthetic import create
from atlas.contract import read_inputs
from atlas.task import load
from atlas.validation import research_splits,split_index_digest
from atlas import challenger

def test_actual_ag_search_frozen_confirmation_and_all_label_refit(tmp_path):
    task=create(tmp_path/'data','skew',n=180)
    data=load(read_inputs(task));primary,controls=research_splits(data)
    data.spec.validation_plan={**primary.json(),'confirmation_isolation':'sealed',
        'search_pool_sha256':split_index_digest(np.union1d(primary.train,primary.valid)),
        'confirmation_train_sha256':split_index_digest(controls[0].train)}
    pv,pt,info=challenger.fit_predict(data,primary.train,primary.valid,tmp_path/'search',12,42,1,2048)
    policy=info['challenger_policy'];window=controls[0]
    a=challenger.fit_predict(data,window.train,window.valid,tmp_path/'confirm_a',12,42,1,2048,policy=policy,evaluation_only=True)
    altered=copy.deepcopy(data);altered.y[window.valid]+=100000.
    b=challenger.fit_predict(altered,window.train,window.valid,tmp_path/'confirm_b',12,42,1,2048,policy=policy,evaluation_only=True)
    np.testing.assert_allclose(a[0],b[0],rtol=0,atol=1e-12);np.testing.assert_allclose(a[1],b[1],rtol=0,atol=1e-12)
    full=challenger.fit_predict(data,np.arange(len(data.y)),np.empty(0,dtype=int),tmp_path/'refit',15,42,1,2048,policy=policy,refit=True)
    assert full[2]['model_fit_rows']==len(data.y) and full[2]['fit_validation_rows']==0
    assert len(full[0])==0 and len(full[1])==len(data.test) and np.isfinite(full[1]).all()

@pytest.mark.parametrize('family,params',[('GBM',{'num_boost_round':7}),('CAT',{'iterations':7}),('RF',{'n_estimators':10}),('XT',{'n_estimators':10})])
def test_actual_frozen_leaf_classification_api(tmp_path,family,params):
    data=load(read_inputs(create(tmp_path/'data','integer_labels',n=180)))
    primary,controls=research_splits(data)
    data.spec.validation_plan={**primary.json(),'confirmation_isolation':'sealed',
        'search_pool_sha256':split_index_digest(np.union1d(primary.train,primary.valid)),
        'confirmation_train_sha256':split_index_digest(controls[0].train)}
    policy={'version':'1.3.1','components':[{'family':family,'params':params,'weight':1.,'name':'fixed'}]}
    w=controls[0]
    pv,pt,info=challenger.fit_predict(data,w.train,w.valid,tmp_path/'confirm',10,42,1,2048,policy=policy,evaluation_only=True)
    assert pv.shape==(len(w.valid),len(data.spec.labels)) and pt.shape==(len(data.test),len(data.spec.labels))
    np.testing.assert_allclose(pv.sum(axis=1),1.)
    assert info['fit_validation_rows']==0
