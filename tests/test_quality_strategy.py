"""Bounded correctness checks; estimators are mocked, no benchmark fits here."""
import copy,json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from atlas import challenger,orchestration
from atlas.metrics import Metric
from atlas.contract import read_inputs
from atlas.task import load
from atlas.validation import research_splits as selection_splits,split_index_digest
from atlas.search import SearchController,recipe,resolve_strategy,strategic_options
from atlas.budget import Budget
from atlas.recipes import validate,RecipeError
from scripts.synthetic import create


def fixture_data(tmp_path,scenario='iid'):
    data=load(read_inputs(create(tmp_path,scenario,n=180)))
    split,controls=selection_splits(data);data.spec.validation_plan={**split.json(),
        'confirmation_isolation':'sealed','search_pool_sha256':split_index_digest(__import__('numpy').concatenate((split.train,split.valid))),
        'confirmation_train_sha256':split_index_digest(controls[0].train)}
    base=recipe('linear',{'alpha':10} if not data.spec.classification else {'C':1})
    champion={'recipe':base,'experiment_id':'primary','runtime_sec':2.,'standardized_utility':-1.,'metric_value':1.,'status':'ok'}
    return data,split,controls,champion


def test_challenger_budget_prices_controls_and_refit(tmp_path):
    d,s,c,b=fixture_data(tmp_path)
    p=challenger.plan(d,b,Budget(360),c,available=True)
    assert p['eligible'] and p['worker_seconds']>=20
    assert p['worker_seconds']*(p['confirmation_fits']+2.8)<=p['reserve_seconds']<=108
    assert not challenger.plan(d,b,Budget(120),c,available=True)['eligible']
    assert not challenger.plan(d,None,Budget(360),c,available=True)['eligible']
    assert not challenger.plan(d,b,Budget(360),c,available=False)['eligible']

@pytest.mark.parametrize('kind',['time','group','future_unknown'])
def test_challenger_fails_closed_on_unverified_validation(tmp_path,kind):
    d,s,c,b=fixture_data(tmp_path);d.spec.validation_plan['kind']=kind
    assert not challenger.plan(d,b,Budget(3600),c,available=True)['eligible']


def test_challenger_probabilities_require_all_global_classes():
    spec=SimpleNamespace(classification=True,labels=['a','b','c'])
    p=challenger.aligned_prediction(pd.DataFrame({2:[.6],0:[.1],1:[.3]}),spec,1)
    np.testing.assert_allclose(p,[[.1,.3,.6]])
    for value in [pd.DataFrame({0:[1.]}),pd.DataFrame({0:[.1],1:[-.1],2:[1.]})]:
        with pytest.raises(ValueError):challenger.aligned_prediction(value,spec,1)

@pytest.mark.parametrize('metric,n',[('rmse',0),('rmsle',0),('f1_macro',3),('logloss',2),('f1',2),('pr_auc',2)])
def test_ag_internal_scorer_is_exact_judge_metric(metric,n):
    m=Metric(metric,n)
    y=np.array([0,1,0,1]) if n<=2 else np.array([0,1,2,0])
    p=np.array([.2,.9,.1,.6]) if n==2 else np.array([[.8,.1,.1],[.1,.5,.4],[.1,.2,.7],[.6,.2,.2]]) if n else np.array([.1,1.2,.3,.8])
    canonical=np.column_stack((1-p,p)) if n==2 else p
    assert challenger.CanonicalScore(m)(y,p)==m.score(y,canonical)


def test_selected_portfolio_freezes_weights_and_trained_rounds():
    class Predictor:
        model_best='weighted'
        _trainer=SimpleNamespace(load_model=lambda name:SimpleNamespace(_get_model_weights=lambda:{'gbm':.7,'cat':.3}))
        def model_info(self,name):
            if name=='weighted':return {'model_type':'WeightedEnsembleModel'}
            return {'model_type':'LGBModel' if name=='gbm' else 'CatBoostModel',
                    'hyperparameters':{'num_boost_round':100} if name=='gbm' else {'iterations':100},
                    'hyperparameters_fit':{'num_boost_round':17} if name=='gbm' else {'iterations':31}}
    policy=challenger.freeze_policy(Predictor())
    assert sum(x['weight'] for x in policy['components'])==1
    assert {x['family'] for x in policy['components']}=={'GBM','CAT'}
    assert next(x for x in policy['components'] if x['family']=='GBM')['params']['num_boost_round']==17
    json.dumps(policy,allow_nan=False)

@pytest.mark.parametrize('failure_phase',[None,'search','confirmation','refit'])
def test_challenger_never_overwrites_good_output_on_failure(tmp_path,monkeypatch,failure_phase):
    d,s,c,b=fixture_data(tmp_path/'input')
    monkeypatch.setattr(challenger,'plan',lambda *args,**kw:{'eligible':True,'reason':'mock','reserve_seconds':30.,'worker_seconds':10.})
    seen=[]
    def fake(self,r,split,timeout,seed=42,refit=False,**kw):
        self.counter+=1;ag=r['model_family']=='autogluon';phase='refit' if refit else 'search' if split.id==s.id else 'confirmation'
        seen.append((ag,phase,copy.deepcopy(kw)))
        result={'experiment_id':f'exp_{self.counter:04d}','recipe':r,'status':'error' if ag and phase==failure_phase else 'ok',
                'runtime_sec':1.,'peak_memory_mb':1.,'diagnostics':{},'artifact_path':str(tmp_path/'artifact')}
        if ag:
            result['challenger_policy']={'version':'1.3.1','components':[{'family':'GBM','weight':1.}], 'selection':'search_split_only'}
            if phase!='search':assert kw['challenger_policy']==result['challenger_policy']
        if result['status']!='ok':return result,None
        error=.1 if ag else 1.
        return result,(self.data.y[split.valid]+error,np.full(len(self.data.test),3. if ag else 2.))
    monkeypatch.setattr(orchestration.Executor,'run',fake)
    out=tmp_path/'run'
    assert orchestration.run(d.inputs.task_path,budget_seconds=360,max_experiments=2,mode='deterministic',run_dir=out)==0
    summary=json.loads((out/'summary.json').read_text())
    assert summary['trained_output']
    if failure_phase in {'search','confirmation'}:
        assert summary['champion_recipe']['model_family']!='autogluon'
    if failure_phase=='refit':assert not summary['full_refit_done'] and summary['checkpoint_kind']=='confirmed_champion'
    if failure_phase is None:assert summary['automl_challenger']['selected'] and summary['full_refit_done']
    assert any(ag and phase=='search' for ag,phase,_ in seen)


def test_strategy_direction_compiles_to_existing_dsl(tmp_path):
    d,s,c,b=fixture_data(tmp_path);ctrl=SearchController(d.spec,Budget(360));ctrl.hpo.enabled=False
    action={'strategy':{'direction':'another_family','family':'hist','reason':'different inductive bias','effort':'quick'}}
    raw,info=resolve_strategy(action,ctrl,b,[],recipe('linear',{'alpha':3}),300)
    clean=validate(raw,d.spec,remaining=300)
    assert clean['model_family']=='hist' and info['direction']=='another_family'
    assert clean['max_runtime_sec']<300
    assert ctrl.research_index==0

@pytest.mark.parametrize('action',[{'direction':'target_transform','family':'hist'}, {'direction':'semantic_features'},
                                  {'direction':'another_family','family':'linear'}, {'direction':'arbitrary_code'},
                                  {'direction':'improve_family','effort':[]}, {'direction':'improve_family','shell':'x'}])
def test_invalid_strategies_fail_closed(tmp_path,action):
    d,s,c,b=fixture_data(tmp_path);ctrl=SearchController(d.spec,Budget(360))
    with pytest.raises(RecipeError):resolve_strategy({'strategy':action},ctrl,b,[],recipe('linear'),300)


def test_strategy_quarantine_and_controls_excluded(tmp_path):
    d,s,c,b=fixture_data(tmp_path)
    history=[{**b,'recipe':recipe('hist'),'status':'error'} for _ in range(2)]
    options=strategic_options(d.spec,history,b,300)
    assert 'hist' not in options['families']
    controls=[{**b,'phase':'confirmation','recipe':recipe('extra_trees'),'runtime_sec':999999.}]
    assert strategic_options(d.spec,history+controls,b,300)==options


def test_frozen_ag_adapter_uses_exact_topology_no_ensemble_research(tmp_path,monkeypatch):
    from test_final_validation import install_fake_ag
    calls=[];d,s,c,b=fixture_data(tmp_path/'data');install_fake_ag(monkeypatch,d.spec,calls)
    policy={'version':'1.3.1','components':[
        {'family':'GBM','params':{'num_boost_round':17},'weight':.75,'name':'AtlasFrozen_0'},
        {'family':'CAT','params':{'iterations':31},'weight':.25,'name':'AtlasFrozen_1'}]}
    pv,pt,info=challenger.fit_predict(d,c[0].train,c[0].valid,tmp_path/'worker',30,42,8,8192,policy=policy,evaluation_only=True)
    np.testing.assert_allclose(pv,1.);np.testing.assert_allclose(pt,1.)
    fits=[value for name,value in calls if name=='fit']
    assert len(fits)==2 and all(f['X_val'] is None and f['y_val'] is None for f in fits)
    assert all(f['num_gpus']==0 and f['num_cpus']==8 for f in fits)
    init=[value for name,value in calls if name=='init']
    assert init[0]['hyperparameters']['num_boost_round']==17
    assert init[1]['hyperparameters']['iterations']==31
    assert info['challenger_policy']==policy and info['fit_validation_rows']==0


def test_qwen_strategy_keeps_existing_context_wire_budget(tmp_path):
    from atlas.agent import researcher_context,compact_context,QwenClient
    d,s,c,b=fixture_data(tmp_path)
    context=compact_context(researcher_context(d,[b],b,360))
    assert context['strategic_options']['directions'] and context['research_evidence']
    assert 'family_costs' not in context['strategic_options']
    assert len(json.dumps(context,ensure_ascii=False).encode())<=10000
    client=QwenClient('disabled');assert client.max_calls==8 and client.max_tokens==16000


def test_ag_duplicate_groups_checked_on_each_frozen_split(tmp_path):
    d,s,c,b=fixture_data(tmp_path)
    d.spec.validation_plan['kind']='duplicate_group'
    challenger.verify_split(d,s.train,s.valid)
    d.train.loc[s.valid[0],d.spec.features]=d.train.loc[s.train[0],d.spec.features].to_numpy()
    with pytest.raises(ValueError,match='group_leakage'):challenger.verify_split(d,s.train,s.valid)


@pytest.mark.parametrize('family,key',[('RF','n_jobs'),('XT','n_jobs'),('GBM','num_threads'),('CAT','thread_count')])
@pytest.mark.parametrize('cpus',[2,8,12])
def test_ag_model_defaults_cannot_override_cpu_limit(family,key,cpus):
    source={key:-1}
    result=challenger.bounded_model_params(family,source,cpus)
    assert result[key]==min(cpus,8) and source[key]==-1
    if family=='CAT':assert result['task_type']=='CPU'
    if family=='GBM':assert result['device_type']=='cpu'


def test_challenger_preserves_three_existing_main_finalists(tmp_path,monkeypatch):
    d,s,c,b=fixture_data(tmp_path/'input')
    monkeypatch.setattr(challenger,'plan',lambda *args,**kw:{'eligible':True,'reason':'mock','reserve_seconds':30.,'worker_seconds':10.})
    families=set()
    def fake(self,r,split,timeout,seed=42,refit=False,**kw):
        self.counter+=1;family=r['model_family'];ag=family=='autogluon'
        if split.id!=s.id and not refit:families.add(family)
        result={'experiment_id':f'exp_{self.counter:04d}','recipe':r,'status':'ok','runtime_sec':1.,'peak_memory_mb':1.,'diagnostics':{},'artifact_path':str(tmp_path/'artifact')}
        if ag:result['challenger_policy']={'version':'1.3.1','components':[]}
        err={'linear':1.,'lightgbm':.9,'catboost':.8,'autogluon':.1}[family]
        return result,(self.data.y[split.valid]+err,np.ones(len(self.data.test)))
    monkeypatch.setattr(orchestration.Executor,'run',fake)
    assert orchestration.run(d.inputs.task_path,budget_seconds=360,max_experiments=3,mode='deterministic',run_dir=tmp_path/'run')==0
    assert families=={'linear','lightgbm','catboost','autogluon'}


@pytest.mark.parametrize('setuptools,eligible',[('80.9.0',True),('82.0.0',False),(None,False)])
def test_ag_preflight_requires_official_reload_dependency(monkeypatch,setuptools,eligible):
    def version(name):
        if name=='autogluon.tabular':return '1.3.1'
        if setuptools is None:raise challenger.importlib.metadata.PackageNotFoundError(name)
        return setuptools
    monkeypatch.setattr(challenger.importlib.metadata,'version',version)
    assert challenger.availability()==eligible


def test_ag_refit_uses_frozen_leaves_all_labels_and_bounded_cpu(tmp_path,monkeypatch):
    from test_final_validation import install_fake_ag
    d,s,c,b=fixture_data(tmp_path/'data');calls=[];install_fake_ag(monkeypatch,d.spec,calls)
    policy={'version':'1.3.1','components':[{'family':'GBM','params':{'num_boost_round':37},'weight':1.,'name':'frozen'}]}
    pv,pt,info=challenger.fit_predict(d,np.arange(len(d.y)),np.array([],dtype=int),tmp_path/'refit',30,42,12,8192,policy=policy,refit=True)
    assert not len(pv) and len(pt)==len(d.test)
    fit=next(value for name,value in calls if name=='fit')
    assert fit['num_cpus']==8 and fit['num_gpus']==0 and fit['X_val'] is None and fit['y_val'] is None
    np.testing.assert_array_equal(fit['y'],d.y)
    assert info['model_fit_rows']==len(d.y)
    with pytest.raises(ValueError,match='cover all known'):
        challenger.fit_predict(d,s.train,s.valid,tmp_path/'bad-refit',30,42,8,8192,policy=policy,refit=True)
