"""Research allocation and preprocessing regressions; no outer evaluation data."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from scripts.synthetic import create
from atlas.contract import read_inputs
from atlas.task import load
from atlas.budget import Budget
from atlas.search import SearchController
from atlas.agent import research_evidence,compact_context,researcher_context

@pytest.mark.parametrize('scenario',['iid','integer_labels','unicode','missing'])
def test_worker_profile_skip_preserves_inputs_labels_and_schema(tmp_path,scenario):
    task=create(tmp_path,scenario,n=80)
    full=load(read_inputs(task));worker=load(read_inputs(task),profile_data=False)
    for name in ['train','test','sample','dictionary']:pd.testing.assert_frame_equal(getattr(full,name),getattr(worker,name))
    assert np.array_equal(full.y,worker.y)
    assert full.spec.features==worker.spec.features and full.spec.labels==worker.spec.labels
    assert full.spec.profile and not worker.spec.profile

@pytest.mark.parametrize('seed',[42,53])
@pytest.mark.parametrize('optuna',[True,False])
def test_failed_family_cannot_starve_healthy_family(tmp_path,seed,optuna):
    data=load(read_inputs(create(tmp_path,n=80)))
    controller=SearchController(data.spec,Budget(3600),seed=seed)
    controller.queue=[];controller.research_index=8;controller.hpo.enabled=optuna
    for family in ['linear','hist']:
        result={'experiment_id':family,'recipe':{'model_family':family,'params':{},'feature_ops':[],
            'feature_expressions':[],'target_transform':'none'},'status':'ok','standardized_utility':-1.,'runtime_sec':1.}
        controller.observe(result)
    champion=controller.family_best['linear'];controller.failures['linear']=2
    for _ in range(12):
        proposal=controller.next(champion)
        assert proposal is not None and proposal['model_family']=='hist'
        assert controller.stage in {'HYPERPARAMETER_SEARCH','MODEL_DIVERSITY'}
    # Its good checkpoint is still retained; only new failing work is excluded.
    assert 'linear' in controller.family_best

def test_aggregate_research_history_excludes_confirmation_labels(tmp_path):
    data=load(read_inputs(create(tmp_path,n=80)))
    history=[]
    for i in range(20):
        history.append({'experiment_id':str(i),'recipe':{'model_family':'linear','params':{},'feature_ops':[],
            'feature_expressions':[{'name':f'formula_{i}','expression':f'x0 * {i+1}'}],'target_transform':'none'},
            'status':'ok','origin':'qwen','metric_value':float(i),'standardized_utility':-float(i),'runtime_sec':1.,'accepted':False})
    confirmation={**history[-1],'experiment_id':'sealed_control','phase':'confirmation','metric_value':999999.}
    champion={**history[0],'diagnostics':{}}
    evidence=research_evidence([*history,confirmation])
    assert len(evidence['negative_hypotheses'])==4
    assert 'sealed_control' not in json.dumps(evidence) and '999999' not in json.dumps(evidence)
    wire=compact_context(researcher_context(data,[*history,confirmation],champion,500))
    assert len(json.dumps(wire,ensure_ascii=False).encode())<=10000
    assert wire['research_evidence']['negative_hypotheses']
