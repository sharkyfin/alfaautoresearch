"""Task-local scheduling effects, with no model training or external labels."""
import copy,time
import pytest
from atlas.budget import Budget
from atlas.search import SearchController,recipe,resolve_strategy,initial_portfolio
from atlas.task import load
from atlas.contract import read_inputs
from scripts.synthetic import create

def controller(tmp_path,mode='deterministic'):
    data=load(read_inputs(create(tmp_path,n=400)));budget=Budget(600)
    ctrl=SearchController(data.spec,budget,mode);ctrl.queue=[]
    candidates=[]
    for family,value,cost in [('hist',.90,15.),('catboost',.89,12.),('extra_trees',.88,20.),('lightgbm',.879,1.)]:
        result={'experiment_id':family,'recipe':recipe(family),'status':'ok','standardized_utility':value,'runtime_sec':cost,'iterations':0,'accepted':False}
        ctrl.observe(result);candidates.append(result)
    ctrl.hpo.family_proposals={'hist':20,'catboost':8,'extra_trees':8,'lightgbm':0}
    return data,ctrl,candidates

def test_cheap_fourth_family_is_not_cut_off_by_initial_ranking(tmp_path,monkeypatch):
    data,ctrl,candidates=controller(tmp_path);ctrl.research_index=8;ctrl.hpo.enabled=True;selected=[]
    def propose(parent,seen):selected.append(parent['recipe']['model_family']);return recipe(selected[-1],{'iterations':50})
    monkeypatch.setattr(ctrl.hpo,'propose',propose)
    raw=ctrl.next(candidates[0]);assert raw['model_family']=='lightgbm' and selected==['lightgbm']

def test_compute_focus_changes_three_choices_then_releases_control(tmp_path):
    data,ctrl,candidates=controller(tmp_path)
    assert ctrl.prioritized_family(candidates)['recipe']['model_family']=='lightgbm'
    ctrl.set_focus('exploit')
    assert [ctrl.prioritized_family(candidates)['recipe']['model_family'] for _ in range(3)]==['hist']*3
    assert ctrl.focus_remaining==0 and ctrl.prioritized_family(candidates)['recipe']['model_family']=='lightgbm'
    ctrl.failures['hist']=2;ctrl.set_focus('exploit')
    assert ctrl.prioritized_family(candidates)['recipe']['model_family']!='hist'

def test_research_calls_wait_for_new_information_and_reach_hpo_phase(tmp_path):
    data,ctrl,candidates=controller(tmp_path,'live-agent');champ=candidates[0]
    assert ctrl.research_due(champ);ctrl.note_research(champ)
    for _ in range(2):ctrl.observe(candidates[1])
    assert not ctrl.research_due(champ)
    ctrl.observe(candidates[1]);assert ctrl.research_due(champ)
    ctrl.note_research(champ)
    for _ in range(4):ctrl.observe(candidates[1])
    assert not ctrl.research_due(champ)
    ctrl.budget.started=time.monotonic()-180
    assert ctrl.research_due(champ)
    ctrl.research_requests=8;assert not ctrl.research_due(champ)


def test_probability_imbalance_has_primary_metric_probe(tmp_path):
    data=load(read_inputs(create(tmp_path,'average_precision',n=400)))
    recipes=initial_portfolio(data.spec,600)
    assert any(r['model_family']=='lightgbm' and r['params'].get('weight')=='sqrt' and r['params'].get('stopping_metric')=='primary' for r in recipes)


def test_direction_evidence_uses_real_measured_gain_only():
    from atlas.agent import research_evidence
    row={'recipe':recipe('hist'),'status':'ok','origin':'qwen','accepted':True,'runtime_sec':2.,
         'standardized_utility':.8,'reference_utility':.7,'research_strategy':{'direction':'semantic_features'}}
    sealed={**row,'phase':'confirmation','standardized_utility':999.}
    result=research_evidence([row,sealed]);assert result['directions']==[{'direction':'semantic_features','attempted':1,'accepted':1,'seconds':2.,'positive_utility_delta':.1}]


def test_long_strategy_comment_is_bounded_without_changing_executable_action(tmp_path):
    from atlas.agent import normalize_annotations
    from atlas.recipes import RecipeError
    data,ctrl,candidates=controller(tmp_path)
    raw={'strategy':{'direction':'compute_allocation','allocation':'exploit','reason':'evidence '*100}}
    clean,changed=normalize_annotations(raw)
    assert changed==['strategy.reason'] and len(clean['strategy']['reason'])==240
    assert clean['strategy']['direction']=='compute_allocation' and clean['strategy']['allocation']=='exploit'
    assert len(raw['strategy']['reason'])>240
    proposal,action=resolve_strategy(clean,ctrl,candidates[0],candidates,recipe('hist'),100.)
    assert action['allocation']=='exploit'
    bad,_=normalize_annotations({'strategy':{'direction':'compute_allocation','reason':{'shell':'denied'}}})
    with pytest.raises(RecipeError):resolve_strategy(bad,ctrl,candidates[0],candidates,recipe('hist'),100.)
