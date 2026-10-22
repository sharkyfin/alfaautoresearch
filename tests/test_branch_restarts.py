import json
import time

import numpy as np
import pytest

from atlas.branches import preprocessing_candidates
from atlas.budget import Budget
from atlas.contract import read_inputs
from atlas.recipes import recipe_hash, validate
from atlas.search import SearchController
from atlas.task import load
from atlas.validation import selection_splits
from scripts.synthetic import create


def setup_scheduler(tmp_path, mode='deterministic'):
    data=load(read_inputs(create(tmp_path/'data',n=160)))
    data.spec.validation_plan=selection_splits(data)[0].json()
    scheduler=SearchController(data.spec,Budget(10500),mode=mode,seed=12)
    scheduler.queue=[];scheduler.research_index=8
    candidates=[]
    for family,score,features in [('lightgbm',-1.,[{'name':'interaction','expression':'x0 * x1'}]),
                                  ('xgboost',-2.,[])]:
        row={'experiment_id':family,'recipe':validate({'model_family':family,
             'params':{'iterations':40},'feature_expressions':features},data.spec),
             'status':'ok','standardized_utility':score,'metric_value':-score,
             'runtime_sec':.01,'accepted':family=='lightgbm'}
        scheduler.observe(row);scheduler.seen.add(recipe_hash(row['recipe']));candidates.append(row)
    return data,scheduler,candidates[0],candidates[1]


def observe_trial(scheduler,raw,score=-5.,iterations=None):
    scheduler.seen.add(recipe_hash(raw))
    result={'experiment_id':'trial_'+str(scheduler.research_index),'recipe':raw,
            'status':'ok','standardized_utility':score,'metric_value':-score,
            'runtime_sec':.01,'search_stage':scheduler.stage,'accepted':False}
    if iterations is not None:result['iterations']=iterations
    scheduler.observe(result)
    return result


def test_plateau_transfers_features_then_tunes_the_new_branch(tmp_path):
    data,scheduler,champion,recipient=setup_scheduler(tmp_path)
    for _ in range(20):
        observe_trial(scheduler,scheduler.next(champion))
    assert scheduler.should_restart(20) and not scheduler.should_stop(20)
    branch=scheduler.restart(champion)
    assert branch['kind']=='feature_transfer'
    raw=scheduler.next(champion)
    validate(raw,data.spec)
    assert raw['model_family']==recipient['recipe']['model_family']
    assert raw['params']==recipient['recipe']['params']
    assert raw['target_transform']==recipient['recipe']['target_transform']
    assert raw['feature_expressions']==champion['recipe']['feature_expressions']
    observe_trial(scheduler,raw,score=-1.5)
    parent=scheduler.parent(champion)
    assert parent['standardized_utility']<champion['standardized_utility']
    raw=scheduler.next(champion)
    assert scheduler.stage=='HYPERPARAMETER_SEARCH'
    assert raw['model_family']==recipient['recipe']['model_family']
    assert raw['feature_expressions']==champion['recipe']['feature_expressions']
    assert champion['standardized_utility']==-1.
    scheduler.hpo.abandon()


def test_local_improvement_keeps_a_branch_alive_below_the_global_champion(tmp_path):
    _,scheduler,champion,_=setup_scheduler(tmp_path)
    scheduler.restart(champion)
    observe_trial(scheduler,scheduler.next(champion),score=-5.)
    for i in range(20):
        observe_trial(scheduler,scheduler.next(champion),score=-4.9+i*.01)
    assert scheduler.branches[-1]['hpo_trials']>=12
    assert scheduler.branches[-1]['no_gain']==0
    assert not scheduler.should_restart(100)
    scheduler.hpo.abandon()


def test_stagnation_closes_only_the_current_representation(tmp_path):
    _,scheduler,champion,_=setup_scheduler(tmp_path)
    first=scheduler.restart(champion)
    observe_trial(scheduler,scheduler.next(champion))
    second=scheduler.restart(champion)
    assert first['representation'] in scheduler.closed_keys
    assert second and second['representation']!=first['representation']
    assert not scheduler.should_stop(100)


def test_short_remaining_budget_does_not_make_all_valid_branches_disappear(tmp_path):
    _,scheduler,champion,_=setup_scheduler(tmp_path)
    budget=scheduler.budget
    budget.started=time.monotonic()-(budget.total-budget.reserve-20)
    branch=scheduler.restart(champion)
    assert branch is not None and scheduler.stop_reason is None
    raw=scheduler.next(champion)
    assert raw['max_runtime_sec']<budget.search_remaining


def test_iteration_growth_uses_the_current_branch_instead_of_the_global_winner(tmp_path):
    _,scheduler,champion,_=setup_scheduler(tmp_path)
    champion['iterations']=10
    scheduler.restart(champion)
    raw=scheduler.next(champion)
    observe_trial(scheduler,raw,score=-1.5,iterations=40)
    observe_trial(scheduler,scheduler.next(champion),score=-2.)
    grown=scheduler.next(champion)
    assert grown['model_family']=='xgboost' and grown['params']['iterations']==80
    assert scheduler.stage=='RESEARCH'
    assert grown['feature_expressions']==champion['recipe']['feature_expressions']


def test_search_stops_when_untested_branches_are_exhausted(tmp_path,monkeypatch):
    _,scheduler,champion,_=setup_scheduler(tmp_path)
    scheduler.nodes={}
    monkeypatch.setattr('atlas.search.preprocessing_candidates',lambda *args:iter(()))
    monkeypatch.setattr('atlas.search.transfer_candidates',lambda *args:iter(()))
    assert scheduler.restart(champion) is None
    assert scheduler.should_stop(100)
    assert 'No untested' in scheduler.stop_reason


def test_preprocessing_candidates_are_valid_bounded_feature_groups(tmp_path):
    data,scheduler,champion,_=setup_scheduler(tmp_path)
    data.spec.profile['structured_columns']={'category':[{
        'separator':'/','median_parts':3,'numeric_parts':[1]}]}
    proposals=list(preprocessing_candidates(data.spec,champion))
    assert any(kind=='structured_parts' for kind,_ in proposals)
    for kind,raw in proposals:
        clean=validate(raw,data.spec)
        assert clean['model_family']==champion['recipe']['model_family']
        assert clean['params']==champion['recipe']['params']
        assert all(c in data.spec.features for op in clean['feature_ops'] for c in op['columns'])
        if kind=='structured_parts':
            assert any('to_number(split_part' in f['expression'] for f in clean['feature_expressions'])


def test_long_run_budgets_allow_periodic_research_beyond_three_calls():
    long=Budget(10500);short=Budget(120)
    assert long.research_call_limit==24 and long.research_token_limit==120000
    assert short.research_token_limit==16000
    assert long.research_trial_limit>512


def test_periodic_qwen_turns_remain_available_after_the_initial_phase(tmp_path):
    _,scheduler,champion,_=setup_scheduler(tmp_path,mode='live-agent')
    scheduler.research_index=20;scheduler.last_research_step=20
    raw=scheduler.next(champion)
    assert scheduler.stage=='HYPERPARAMETER_SEARCH' and not scheduler.request_researcher
    observe_trial(scheduler,raw)
    scheduler.research_index=32
    assert scheduler.next(champion) is not None
    assert scheduler.stage=='RESEARCH' and scheduler.request_researcher
    assert scheduler.parent(champion)['experiment_id']==champion['experiment_id']
    scheduler.hpo.abandon()


@pytest.mark.parametrize('use_final_reserve',[False,True])
def test_orchestrator_uses_late_budget_and_preserves_confirmation_and_refit(tmp_path,monkeypatch,use_final_reserve):
    from atlas import orchestration
    captured=[]
    class CapturedBudget(Budget):
        def __post_init__(self):
            super().__post_init__();captured.append(self)
    monkeypatch.setattr(orchestration,'Budget',CapturedBudget)
    def execute(self,recipe,split,timeout,seed=42,**kwargs):
        self.counter+=1
        if self.counter==2:
            budget=captured[0]
            elapsed=budget.total-budget.reserve*.75 if use_final_reserve else budget.total*.75
            budget.started=time.monotonic()-elapsed
        return ({'experiment_id':f'exp_{self.counter:04d}','recipe':recipe,'status':'ok',
                 'runtime_sec':.01,'peak_memory_mb':1.,'diagnostics':{},'iterations':40},
                (np.zeros(len(split.valid)),np.zeros(len(self.data.test))))
    monkeypatch.setattr(orchestration.Executor,'run',execute)
    monkeypatch.setattr(orchestration,'evaluate',lambda *args:{'metric_value':1.,'standardized_utility':0.})
    directory=tmp_path/'run'
    assert orchestration.run(create(tmp_path/'task',n=120),budget_seconds=10500,threads=1,
                             mode='deterministic',max_experiments=85,run_dir=directory)==0
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['cross_validation']['completed_candidates']>=1
    assert summary['full_refit_published']
    if not use_final_reserve:
        assert summary['experiments']>50
        assert summary['adaptive_search']['branches_started']>=2
    else:
        assert summary['experiments']>=5
