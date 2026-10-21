"""Transport failures and primary-metric decisions must be measurable and controlled."""
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from atlas.agent import QwenClient
from atlas.ensemble import tune_multipliers
from atlas.metrics import Metric
from atlas.validation import Split


def client_with_wire(monkeypatch, handler):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://organizer.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'private-test-key')
    wire = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, 'stream', lambda method, url, **kwargs:
                        wire.stream(method, url, **{k:v for k,v in kwargs.items() if k != 'trust_env'}))
    return QwenClient('live'), wire


def response():
    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop',
        'message': {'content': '{"model_family":"lightgbm"}'}}],
        'usage': {'total_tokens': 100, 'completion_tokens_details': {'reasoning_tokens': 0}}})


def test_qwen_produces_json_without_spending_budget_on_thinking(monkeypatch):
    def handler(request):
        body = json.loads(request.content)
        assert body['reasoning'] == {'effort':'none','exclude':True}
        assert body['max_tokens'] == 1200
        limits = request.extensions['timeout']
        assert limits['connect'] == limits['write'] == 5 and limits['read'] == 30
        return response()
    client, wire = client_with_wire(monkeypatch, handler)
    with wire:
        assert client.propose({}, timeout=30) == {'model_family':'lightgbm'}
    usage = next(e for e in client.events if e['status'] == 'api_usage')
    assert usage['reasoning_tokens'] == 0 and usage['finish_reason'] == 'stop'
    assert usage['elapsed_seconds'] >= 0 and usage['timeout_seconds'] == 30
    assert 'private-test-key' not in json.dumps(client.events)


@pytest.mark.parametrize('phase,exception', [('connect',httpx.ConnectTimeout),('read',httpx.ReadTimeout)])
def test_qwen_failure_distinguishes_connection_and_model_wait(monkeypatch, phase, exception):
    def handler(request):
        raise exception('Do not log this: private-test-key')
    client, wire = client_with_wire(monkeypatch, handler)
    with wire:
        assert client.propose({}) is None
    failure = client.events[-1]
    assert failure['status'] == 'api_failure' and failure['timeout_phase'] == phase
    assert failure['transport_exception'] == exception.__name__
    assert 'private-test-key' not in json.dumps(client.events)


def test_proxy_can_reject_reasoning_control_without_losing_json_mode(monkeypatch):
    seen = []
    def handler(request):
        body = json.loads(request.content);seen.append(body)
        if 'reasoning' in body:
            return httpx.Response(422, json={'error': {'message':'unsupported reasoning parameter'}})
        return response()
    client, wire = client_with_wire(monkeypatch, handler)
    with wire:
        assert client.propose({}) is None
        assert client.propose({}) == {'model_family':'lightgbm'}
    assert 'reasoning' not in seen[1] and 'response_format' in seen[1]
    assert any(e['status'] == 'retry_without_reasoning_control' for e in client.events)


def test_empty_thinking_only_reply_is_reported_and_never_becomes_a_recipe(monkeypatch):
    def handler(request):
        return httpx.Response(200, json={'choices': [{'finish_reason':'length','message':{'content':None}}],
            'usage': {'total_tokens':1200, 'completion_tokens_details':{'reasoning_tokens':1200}}})
    client, wire = client_with_wire(monkeypatch, handler)
    with wire:
        assert client.propose({}) is None
    assert any(e['status']=='empty_action' and e['finish_reason']=='length' for e in client.events)
    assert client.tokens == 1200 and client.failures == 1


@pytest.mark.parametrize('metric,positive', [('f1_macro',1),('balanced_accuracy',1),('f1',0),('f1',1)])
def test_binary_threshold_is_selected_on_primary_and_requires_control(metric, positive):
    train_y = np.tile([0,1],50)
    holdout_y = np.r_[np.zeros(80,dtype=int),np.ones(20,dtype=int)]
    data = SimpleNamespace(y=np.r_[train_y,holdout_y,holdout_y],
                           spec=SimpleNamespace(classification=True,labels=['a','b'],metric=Metric(metric,2,positive)))
    primary = Split(np.arange(100),np.arange(100,200),'random','test',42)
    control = Split(np.arange(100),np.arange(200,300),'random','control',53)
    def candidate(split):
        p1 = np.where(data.y[split.valid] == 1,.35,.1)
        p = np.column_stack([1-p1,p1])
        return {'split':split,'predictions':(p,np.tile([.65,.35],(10,1))),
                'standardized_utility':data.spec.metric.score(data.y[split.valid],p)}
    a,b = candidate(primary),candidate(control)
    assert tune_multipliers(data,a,None) is None
    result = tune_multipliers(data,a,b)
    assert result['strategy'] == 'binary_threshold'
    assert result['metric_value'] == result['confirmation_metric'] == 1
    assert np.allclose(result['predictions'][1].sum(axis=1),1)
    # The selected threshold must be rejected when the independent control gets worse.
    p1 = np.where(holdout_y == 1,.8,.4)
    b['predictions'] = (np.column_stack([1-p1,p1]),b['predictions'][1])
    b['standardized_utility'] = data.spec.metric.score(holdout_y,b['predictions'][0])
    assert tune_multipliers(data,a,b) is None


def test_iteration_growth_keeps_accepted_features_and_obeys_measured_budget(tmp_path):
    from atlas.budget import Budget
    from atlas.contract import read_inputs
    from atlas.recipes import recipe_hash, validate
    from atlas.search import SearchController
    from atlas.task import load
    from atlas.validation import design
    from scripts.synthetic import create
    data = load(read_inputs(create(tmp_path,n=100)))
    data.spec.validation_plan = design(data).json()
    parent = validate({'model_family':'hist','params':{'iterations':40},
                       'feature_expressions':[{'name':'interaction','expression':'x0 * x1'}]},data.spec)
    champion = {'experiment_id':'parent','recipe':parent,'iterations':40,'runtime_sec':1.}
    def proposal(budget, candidate=champion, seen=()):
        controller = SearchController(data.spec,budget)
        controller.queue=[];controller.research_index=2;controller.seen=set(seen)
        return controller.next(candidate)
    grown = proposal(Budget(90))
    assert grown['params']['iterations'] == 80
    assert grown['feature_expressions'] == parent['feature_expressions']
    validate(grown,data.spec)
    # A model stopped by validation, a spent budget and an already tried capacity
    # must each return an ordinary local proposal rather than repeat a costly fit.
    assert proposal(Budget(90),{**champion,'iterations':20})['params']['iterations'] == 40
    assert proposal(Budget(1))['params']['iterations'] == 40
    assert proposal(Budget(90),seen=[recipe_hash(grown)])['params']['iterations'] == 40


@pytest.mark.parametrize('bad_fields',[
    {'unexpected':'invalid proposal metadata'},
    {'model_family':[]},
    {'max_runtime_sec':None},
    {'max_runtime_sec':True},
])
def test_invalid_research_action_preserves_the_planned_local_fit(tmp_path,monkeypatch,bad_fields):
    from atlas import orchestration
    from atlas.experiments import Executor
    from scripts.synthetic import create
    for key,value in {'OPENAI_API_KEY':'private-test-key','OPENAI_BASE_URL':'https://organizer.invalid/v1'}.items():
        monkeypatch.setenv(key,value)
    client = QwenClient('live',max_calls=1,transport=lambda body,timeout:{
        'choices':[{'message':{'content':json.dumps({'model_family':'lightgbm',**bad_fields})}}]})
    calls=[]
    def fit(self,recipe,split,timeout,seed=42,**kwargs):
        self.counter+=1;calls.append(recipe)
        error=1. if recipe['model_family']=='linear' else .5
        result={'experiment_id':f'exp_{self.counter:04d}','recipe':recipe,'status':'ok',
                'runtime_sec':.01,'peak_memory_mb':1.,'diagnostics':{}}
        return result,(self.data.y[split.valid]+error,np.zeros(len(self.data.test)))
    monkeypatch.setattr(Executor,'run',fit)
    directory=tmp_path/'run'
    assert orchestration.run(create(tmp_path/'data',n=120),budget_seconds=90,max_experiments=3,
        mode='live-agent',research_client=client,run_dir=directory)==0
    # The failed API action must not skip local_1 and jump straight to the next family.
    assert calls[2]['id']=='local_1' and calls[2]['model_family']=='lightgbm'
    history=[json.loads(row) for row in (directory/'experiments.jsonl').read_text().splitlines()]
    assert history[2]['origin']=='deterministic'
    errors=[event for event in client.events if event['status']=='recipe_rejected']
    assert errors and ('unexpected' in errors[0]['reason'] if 'unexpected' in bad_fields else True)
    assert not (directory/'error.txt').exists()
