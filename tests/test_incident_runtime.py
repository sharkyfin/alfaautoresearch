"""Run against the isolated incident candidate, not the preserved zero source."""
import json
from pathlib import Path
from resource import RLIM_INFINITY

import numpy as np
import pytest


@pytest.mark.parametrize('inherited,requested,expected', [
    ((RLIM_INFINITY,RLIM_INFINITY),24576,24576*2**20),
    ((8*2**30,8*2**30),24576,8*2**30),
    ((4*2**30,8*2**30),24576,4*2**30),
    ((8*2**30,16*2**30),4096,4*2**30),
])
def test_inherited_memory_limits_are_respected(monkeypatch,inherited,requested,expected):
    from atlas import worker
    applied=[]
    monkeypatch.setattr(worker.resource,'getrlimit',lambda _:inherited)
    monkeypatch.setattr(worker.resource,'setrlimit',lambda key,value:applied.append((key,value)))
    result=worker.restrict_address_space(requested)
    assert applied==[(worker.resource.RLIMIT_AS,(expected,expected))]
    assert result['applied_bytes']==expected


def test_total_training_failure_has_nonzero_exit_and_retains_valid_csv(tmp_path,monkeypatch):
    from atlas import orchestration
    from atlas.experiments import Executor
    from scripts.synthetic import create
    def fail(self,recipe,split,timeout,seed=42,**kwargs):
        self.counter+=1
        return {'experiment_id':f'exp_{self.counter:04d}','recipe':recipe,'status':'worker_error',
                'runtime_sec':.01,'peak_memory_mb':1.,'exception_class':'RuntimeError',
                'message':"can't start new thread",'diagnostics':{}},None
    monkeypatch.setattr(Executor,'run',fail)
    directory=tmp_path/'run'
    task=create(tmp_path/'task',n=80)
    assert orchestration.run(task,mode='deterministic',budget_seconds=30,
        max_experiments=8,run_dir=directory)==1
    summary=json.loads((directory/'summary.json').read_text())
    assert summary['fit_success']==0 and not summary['trained_output']
    assert summary['status']=='fatal_with_fallback'
    assert summary['fatal_exception']['exception_class']=='NoSuccessfulTraining'
    assert (task.parent/'submission.csv').exists()


def test_intentional_degenerate_task_still_exits_cleanly(tmp_path):
    from atlas.orchestration import run
    from scripts.synthetic import create
    task=create(tmp_path/'task','one_class',n=40)
    assert run(task,mode='deterministic',budget_seconds=30,run_dir=tmp_path/'run')==0
    assert (task.parent/'submission.csv').exists()


def test_controller_limits_idle_pools_before_importing_ml(monkeypatch):
    import os,runpy
    monkeypatch.setattr('sys.argv',['solution.py'])
    namespace=runpy.run_path(str(Path(__file__).resolve().parents[1]/'solution.py'),run_name='not_main')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        assert os.environ[name]=='1'
    # Model workers keep their own separate 8-core allowance; budget.run_process
    # constructs a fresh environment instead of inheriting these controller pools.
    assert namespace['main']
