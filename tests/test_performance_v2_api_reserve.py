import json
import numpy as np
from atlas.agent import QwenClient
from atlas.budget import Budget
from scripts.synthetic import create


def test_llm_is_not_called_when_confirmation_and_refit_consume_remaining_budget(tmp_path,monkeypatch):
    from atlas import orchestration
    from atlas.experiments import Executor
    count={'fits':0}
    class MeasuredBudget(Budget):
        @property
        def remaining(self):return 30. if count['fits']<2 else 13.
    def fit(self,recipe,split,timeout,seed=42,**kwargs):
        self.counter+=1;count['fits']+=1
        error=1. if recipe['model_family']=='linear' else .5
        return {'experiment_id':f'exp_{self.counter:04d}','recipe':recipe,'status':'ok',
            'runtime_sec':3.,'peak_memory_mb':1.,'diagnostics':{}},(
            self.data.y[split.valid]+error,np.zeros(len(self.data.test)))
    monkeypatch.setattr(orchestration,'Budget',MeasuredBudget)
    monkeypatch.setattr(Executor,'run',fit)
    monkeypatch.setenv('OPENAI_BASE_URL','https://organizer.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY','test-only')
    def transport(*args):raise AssertionError('No API call should be admitted')
    client=QwenClient('live',max_calls=1,transport=transport)
    directory=tmp_path/'run'
    assert orchestration.run(create(tmp_path/'data'),budget_seconds=30,max_experiments=3,
        mode='live-agent',research_client=client,run_dir=directory)==0
    assert client.requests==0 and client.calls==0
    event=next(event for event in client.events if event['status']=='skipped_finalization_reserve')
    assert event['estimated_finalization_seconds']>event['remaining_seconds']
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['trained_output'] and summary['fit_success']>=2
