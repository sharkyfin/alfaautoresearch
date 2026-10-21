"""Out-of-fold text signals, multi-fold selection, and measured TPE feedback."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from atlas.budget import Budget
from atlas.contract import read_inputs
from atlas.features import TextProjection
from atlas.hpo import AdaptiveTPE, study_key
from atlas.models import fit_predict
from atlas.recipes import recipe_hash, validate
from atlas.search import SearchController
from atlas.task import load
from atlas.validation import selection_splits
from scripts.synthetic import create


def test_text_projection_never_learns_validation_vocabulary_and_is_repeatable():
    train=pd.Series(['bright useful product', 'faulty broken product', 'bright reliable product']*5)
    held=pd.Series(['validationonlytoken', '', None, 'bright product'])
    projection=TextProjection(128,max_rows=10).fit(train)
    first=projection.transform(held)
    assert 'validationonlytoken' not in projection.vectorizer.vocabulary_
    assert projection.fit_rows==10 and 0<projection.components<=len(projection.vectorizer.vocabulary_)
    assert np.isfinite(first).all() and np.array_equal(first,projection.transform(held))
    assert not first[:3].any()
    second=TextProjection(128,max_rows=10).fit(train).transform(held)
    assert np.allclose(first,second)


@pytest.mark.parametrize('documents', [['!!!','---',''],['single','single','single']])
def test_text_projection_handles_empty_or_single_word_and_refit(documents):
    projection=TextProjection(64).fit(pd.Series(['previous vocabulary', 'other terms']))
    projection.fit(pd.Series(documents))
    result=projection.transform(pd.Series(['unseenword','single']))
    assert result.shape[0]==2 and np.isfinite(result).all()
    assert projection.vectorizer is None or 'previous' not in projection.vectorizer.vocabulary_


def test_two_word_vocabulary_keeps_both_distinct_topics():
    projection=TextProjection(64).fit(pd.Series(['bright','faulty']*10))
    vectors=projection.transform(pd.Series(['bright','faulty']))
    assert vectors.shape==(2,2) and not np.allclose(vectors[0],vectors[1])
    assert np.allclose(np.linalg.norm(vectors,axis=1),1)


def test_explicit_text_dictionary_hint_supports_short_messages(tmp_path):
    task=create(tmp_path,'text',n=120)
    path=task.parent/'data_dictionary.csv';frame=pd.read_csv(path)
    frame.loc[frame.feature=='category','dtype']='text';frame.to_csv(path,index=False)
    for filename in ['train.csv','test.csv']:
        path=task.parent/filename;frame=pd.read_csv(path,dtype={'id':'string'})
        frame['category']=['short']*len(frame);frame.to_csv(path,index=False)
    data=load(read_inputs(task))
    assert data.spec.text_candidates==['category']


def test_dense_text_features_improve_a_real_booster_on_unseen_messages(tmp_path):
    task=create(tmp_path,'text',n=600)
    rng=np.random.default_rng(77)
    labels=rng.choice([-5.,5.],660)
    for filename,offset in [('train.csv',0),('test.csv',600)]:
        path=task.parent/filename;frame=pd.read_csv(path,dtype={'id':'string'})
        for c in [f'x{i}' for i in range(5)]:frame[c]=0.
        frame['category']=[f'This is a {"useful" if labels[offset+i]>0 else "faulty"} item and a detailed customer product review number {offset+i:06}' for i in range(len(frame))]
        if offset==0:frame['response']=labels[:600]
        frame.to_csv(path,index=False)
    data=load(read_inputs(task));train=np.arange(450);held=np.arange(450,600)
    scores=[]
    for components in [0,4]:
        recipe={'model_family':'hist','params':{'iterations':60,'max_leaf_nodes':7,'text_svd_components':components}}
        prediction,_,info=fit_predict(data,recipe,train,held,threads=1)
        scores.append(data.spec.metric.score(data.y[held],prediction))
        if components:assert info['diagnostics']['text_projections']['category']['components']==4
    assert scores[1]<scores[0]*.15


@pytest.mark.parametrize('scenario',['iid','binary_imbalance','integer_labels','group','duplicates','singleton'])
def test_three_folds_are_frozen_disjoint_and_respect_group_and_class_boundaries(tmp_path,scenario):
    data=load(read_inputs(create(tmp_path,scenario,n=360)))
    primary,controls=selection_splits(data)
    splits=[primary,*controls]
    assert len(splits)==3
    assert [s.id for s in splits]==[s.id for s in [selection_splits(data)[0],*selection_splits(data)[1]]]
    held=np.concatenate([s.valid for s in splits])
    assert len(held)==len(np.unique(held))
    if scenario!='singleton':assert set(held)==set(range(len(data.y)))
    else:assert 0 not in held
    for split in splits:
        assert not set(split.train)&set(split.valid)
        assert set(split.train)|set(split.valid)==set(range(len(data.y)))
        if data.spec.classification:assert set(data.y[split.train])==set(data.y)
        if scenario=='group':assert not set(data.train.entity_id.iloc[split.train])&set(data.train.entity_id.iloc[split.valid])
        if scenario=='duplicates':
            groups=pd.util.hash_pandas_object(data.train[data.spec.features],index=False)
            assert not set(groups.iloc[split.train])&set(groups.iloc[split.valid])


@pytest.mark.parametrize('incomplete',[False,True])
def test_finalist_must_win_across_all_planned_folds_not_only_search(tmp_path,monkeypatch,incomplete):
    from atlas import orchestration
    from atlas.experiments import Executor
    task=create(tmp_path/'data',n=240);data=load(read_inputs(task))
    primary,controls=selection_splits(data)
    refits=[]
    def fit(self,recipe,split,timeout,seed=42,**kwargs):
        self.counter+=1
        family=recipe['model_family'];is_primary=split.id==primary.id
        result={'experiment_id':f'exp_{self.counter:04d}','recipe':recipe,'status':'ok',
                'runtime_sec':.01,'peak_memory_mb':1.,'diagnostics':{},'iterations':10 if is_primary else 20}
        if kwargs.get('refit'):refits.append((family,kwargs['iterations_override']))
        if incomplete and family=='lightgbm' and split.id==controls[-1].id:
            result['status']='worker_error';return result,None
        error=.1 if family=='lightgbm' and is_primary else 3. if family=='lightgbm' else 1.
        return result,(self.data.y[split.valid]+error,np.zeros(len(self.data.test)))
    monkeypatch.setattr(Executor,'run',fit)
    directory=tmp_path/'run'
    assert orchestration.run(task,budget_seconds=120,max_experiments=2,mode='deterministic',run_dir=directory)==0
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['champion_recipe']['model_family']=='linear'
    assert summary['cross_validation']['planned_folds']==3
    assert summary['cross_validation']['completed_candidates']==(1 if incomplete else 2)
    assert all(len(c['fold_metrics'])==3 for c in summary['cross_validation']['candidates'])
    assert refits==[('linear',20)] and summary['full_refit_published']
    assert not (directory/'error.txt').exists()


def test_tpe_receives_scores_and_keeps_distinct_feature_studies(tmp_path):
    data=load(read_inputs(create(tmp_path,n=200)))
    data.spec.validation_plan=selection_splits(data)[0].json()
    parent={'experiment_id':'base','recipe':validate({'model_family':'hist','params':{'iterations':30}},data.spec)}
    hpo=AdaptiveTPE(data.spec,seed=19)
    hpo.observe({**parent,'status':'ok','standardized_utility':-5.})
    proposed=hpo.propose(parent)
    validate(proposed,data.spec)
    hpo.observe({'recipe':proposed,'status':'ok','standardized_utility':-1.})
    study,_=hpo.get_study(proposed)
    assert sorted(t.value for t in study.trials)==[-5.,-1.]
    assert hpo.completed==1 and not hpo.pending
    other=copy.deepcopy(parent)
    other['recipe']['feature_expressions']=[{'name':'interaction','expression':'x0 * x1'}]
    assert study_key(other['recipe'])!=study_key(parent['recipe'])
    hpo.propose(other)
    second,_=hpo.get_study(other['recipe'])
    assert len(hpo.studies)==2 and all(t.value is None for t in second.trials)
    hpo.abandon()


def test_adaptive_search_revisits_two_families_and_preserves_their_features(tmp_path):
    data=load(read_inputs(create(tmp_path,n=200)))
    data.spec.validation_plan=selection_splits(data)[0].json()
    scheduler=SearchController(data.spec,Budget(900),seed=12)
    scheduler.queue=[];scheduler.research_index=8
    candidates=[]
    for family,score in [('lightgbm',-2.),('xgboost',-3.)]:
        candidate={'experiment_id':family,'recipe':validate({'model_family':family,'params':{'iterations':40},
                    'feature_expressions':[{'name':family+'_feature','expression':'x0 * x1'}]},data.spec),
                   'status':'ok','standardized_utility':score,'runtime_sec':1.}
        scheduler.observe(candidate);candidates.append(candidate)
    proposed=[]
    for _ in range(6):
        raw=scheduler.next(candidates[0]);validate(raw,data.spec)
        assert scheduler.stage=='HYPERPARAMETER_SEARCH'
        family=raw['model_family']
        assert raw['feature_expressions'][0]['name']==family+'_feature'
        scheduler.seen.add(recipe_hash(raw));proposed.append(family)
        scheduler.observe({'recipe':raw,'status':'ok','standardized_utility':-10.,
                           'experiment_id':'trial','runtime_sec':1.})
    assert set(proposed)=={'lightgbm','xgboost'}
    assert not scheduler.should_stop(10)
    assert scheduler.hpo.completed==6


def test_tpe_seed_and_observation_sequence_are_reproducible(tmp_path):
    data=load(read_inputs(create(tmp_path,n=100)))
    data.spec.validation_plan=selection_splits(data)[0].json()
    parent={'experiment_id':'base','recipe':validate({'model_family':'linear'},data.spec)}
    sequences=[]
    for _ in range(2):
        hpo=AdaptiveTPE(data.spec,seed=42);sequence=[]
        for step in range(7):
            raw=hpo.propose(parent);sequence.append(raw['params'])
            score=-abs(np.log10(raw['params']['alpha'])-.5)
            hpo.observe({'recipe':raw,'status':'ok','standardized_utility':score})
        sequences.append(sequence)
    assert sequences[0]==sequences[1]


def test_tpe_skips_seed_values_outside_its_search_grid_without_rejecting_valid_recipe(tmp_path):
    data=load(read_inputs(create(tmp_path,'text',n=100)))
    data.spec.validation_plan=selection_splits(data)[0].json()
    candidate={'experiment_id':'custom','recipe':validate({'model_family':'hist',
               'params':{'text_svd_components':16}},data.spec)}
    hpo=AdaptiveTPE(data.spec)
    hpo.observe({**candidate,'status':'ok','standardized_utility':-1.})
    raw=hpo.propose(candidate)
    assert raw is not None and raw['params']['text_svd_components'] in [0,32,64,128]
    validate(raw,data.spec)
    assert study_key(raw)==study_key(candidate['recipe'])
    hpo.abandon()
