from types import SimpleNamespace
import numpy as np
import pytest
from atlas.budget import Budget
from atlas.ensemble import choose_blend,store_oof
from atlas.metrics import Metric
from atlas.validation import Split
from atlas.search import SearchController,initial_portfolio


def three_components():
    data=SimpleNamespace(y=np.zeros(6),spec=SimpleNamespace(metric=Metric('rmse'),classification=False))
    primary=Split(np.array([3,4,5]),np.arange(3),'random','primary',42)
    control=Split(np.arange(3),np.array([3,4,5]),'random','control',53)
    def candidate(name,split,pred):
        pred=np.asarray(pred,float)
        return {'experiment_id':name,'split':split,'predictions':(pred,pred.copy()),
                'standardized_utility':-float(np.sqrt(np.mean(pred**2)))}
    errors=[[2,-1,-1],[-1,2,-1],[-1,-1,2]]
    candidates=[candidate(str(i),primary,pred) for i,pred in enumerate(errors)]
    controls={str(i):[candidate(str(i),control,pred)] for i,pred in enumerate(errors)}
    return data,candidates,controls


def test_greedy_three_way_blend_uses_common_controls():
    data,candidates,controls=three_components()
    result=choose_blend(data,candidates,controls)
    assert result is not None and len(result['components'])==3
    assert result['metric_value']<1e-10
    assert result['strategy']=='greedy_primary'
    assert np.sum(result['weights'])==pytest.approx(1.)
    assert choose_blend(data,candidates,controls,'none') is None
    for rows in controls.values():
        rows[0]['predictions']=(np.ones(3),np.ones(3))
        rows[0]['standardized_utility']=-1.
    assert choose_blend(data,candidates,controls) is None


def test_oof_alignment_and_illegal_repeated_rows(tmp_path):
    data,candidates,controls=three_components()
    manifest=store_oof(tmp_path/'oof.npz',data,candidates,controls)
    assert all(row['covered_rows']==6 for row in manifest['candidates'])
    with np.load(tmp_path/'oof.npz') as arrays:
        assert arrays['covered_0'].all()
        assert np.array_equal(arrays['prediction_0'][:3],candidates[0]['predictions'][0])
    controls['0']=[candidates[0]]
    with pytest.raises(ValueError,match='repeated'):
        store_oof(tmp_path/'invalid.npz',data,candidates,controls)
    bounded=store_oof(tmp_path/'bounded.npz',data,candidates,{},max_bytes=1)
    assert bounded['path'] is None and not (tmp_path/'bounded.npz').exists()


def test_priority_uses_measured_cost_uncertainty_and_diversity():
    spec=SimpleNamespace(available_capabilities={},text_candidates=[],classification=False)
    scheduler=SearchController(spec,Budget(10500))
    def candidate(family,utility):
        return {'experiment_id':family,'recipe':{'model_family':family},'standardized_utility':utility}
    candidates=[candidate('hist',1.),candidate('extra_trees',.99),candidate('linear',.98)]
    scheduler.observations=[{'family':c['recipe']['model_family'],'ok':True,'seconds':seconds,'normalized_gain':0.}
        for c,seconds in zip(candidates,[100.,1.,10.])]
    chosen=scheduler.prioritized_family(candidates)
    assert chosen['recipe']['model_family']=='extra_trees'
    assert scheduler.priority_events[-1]['considered_families']==3
    assert scheduler.research_limit>48


def test_text_probe_survives_an_early_nontext_champion():
    spec=SimpleNamespace(available_capabilities={'lightgbm'},text_candidates=['message'],
                         classification=True,metric=Metric('f1_macro',2),profile={'target':{}})
    scheduler=SearchController(spec,Budget(120))
    first=scheduler.next(None)
    assert first['model_family']=='linear'
    # A sparse-text opportunity must not depend on the current leader being
    # linear. A later local-search branch alone previously missed this probe.
    leader={'experiment_id':'early','recipe':{'model_family':'hist','params':{}},'standardized_utility':.9}
    second=scheduler.next(leader)
    assert second['model_family']=='linear' and second['params']['text_analyzer']=='word_char'
    spec.text_candidates=[]
    assert all(row['params'].get('text_analyzer')!='word_char' for row in initial_portfolio(spec,120))
