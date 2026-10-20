import json,os,sys
from pathlib import Path
import pytest
from atlas.agent import QwenClient
from scripts.openrouter_client import OpenRouterClient,load_key,MODEL


def test_openrouter_free_request_constraints(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY','organizer-only-not-real')
    seen=[]
    def wire(body,timeout):
        seen.append(body)
        return {'model':MODEL,'choices':[{'message':{'content':'{"model_family":"hist"}'}}],'usage':{'total_tokens':20,'cost':0}}
    client=OpenRouterClient('openrouter-only-not-real',max_calls=1,transport=wire)
    assert client.propose({'untrusted_data':{}})=={'model_family':'hist'}
    assert client.propose({}) is None
    body=seen[0]
    assert body['model']==MODEL and body['model'].endswith(':free')
    assert body['provider']['max_price']=={'prompt':0,'completion':0}
    assert body['reasoning']=={'effort':'low','exclude':True} and body['max_tokens']==4096
    assert 'organizer-only-not-real' not in json.dumps(seen+client.events)
    assert 'openrouter-only-not-real' not in json.dumps(seen+client.events)
    assert client.requests==1 and client.events[-2]['cost_usd']==0


def test_openrouter_rejects_paid_model_and_cost():
    seen=[]
    client=OpenRouterClient('test-only',transport=lambda body,timeout:seen.append(body))
    client.model='minimax/minimax-m3.1-flash'
    assert client.propose({}) is None and not seen
    paid=OpenRouterClient('test-only',transport=lambda body,timeout:{'usage':{'cost':.01}})
    assert paid.propose({}) is None
    assert paid.events[-1]['status']=='api_failure'


def test_key_loader_does_not_eval_or_fall_back(tmp_path,monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY',raising=False)
    monkeypatch.setenv('OPENAI_API_KEY','wrong-provider')
    path=tmp_path/'credentials'
    path.write_text('export OPENROUTER_API_KEY="$(not-executed)"\n')
    assert load_key(path)=='$(not-executed)'
    path.write_text('OPENAI_API_KEY=wrong-provider\n')
    with pytest.raises(ValueError,match='one nonempty'):load_key(path)
    monkeypatch.setenv('OPENROUTER_API_KEY','dedicated-key')
    assert load_key(path)=='dedicated-key'


def test_official_client_does_not_pick_up_development_key(monkeypatch):
    for name in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL']:monkeypatch.delenv(name,raising=False)
    monkeypatch.setenv('OPENROUTER_API_KEY','development-only')
    client=QwenClient('live')
    assert not client.configured and client.open and client.requests==0


def test_injected_backend_provenance_with_actual_fit(tmp_path):
    from atlas.orchestration import run
    from scripts.synthetic import create
    task=create(tmp_path/'data','iid',n=240)
    def wire(body,timeout):
        ctx=json.loads(body['messages'][1]['content'])
        action={'statement':'Test a product interaction from the injected researcher','model_family':'hist','params':{'iterations':30},'feature_ops':[{'op':'product','columns':['x0','x1']}],'validation_plan_id':ctx['objective']['split']['id']}
        return {'model':MODEL,'choices':[{'message':{'content':json.dumps(action)}}],'usage':{'total_tokens':100,'cost':0}}
    client=OpenRouterClient('fake-not-a-credential',max_calls=1,transport=wire)
    assert run(task,budget_seconds=120,threads=1,max_experiments=7,mode='live-agent',run_dir=tmp_path/'run',research_client=client)==0
    summary=json.loads((tmp_path/'run/summary.json').read_text())
    assert summary['researcher_provider']=='openrouter-free' and summary['researcher_model']==MODEL
    assert summary['qwen_calls']==0 and summary['researcher_calls']==1
    history=[json.loads(line) for line in (tmp_path/'run/experiments.jsonl').read_text().splitlines()]
    proposals=[r for r in history if r.get('origin')=='openrouter']
    assert len(proposals)==1 and proposals[0]['status']=='ok' and 'accepted_reason' in proposals[0]


def test_openrouter_key_is_not_forwarded_to_workers(tmp_path,monkeypatch):
    from atlas.budget import run_process
    monkeypatch.setenv('OPENROUTER_API_KEY','fake-parent-only')
    result=run_process([sys.executable,'-c','import os; assert "OPENROUTER_API_KEY" not in os.environ'],tmp_path,timeout=5)
    assert result['status']=='ok'
