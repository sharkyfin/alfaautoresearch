import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

from atlas.utils import code_digest
from scripts.benchmark_kaggle import comparisons, research_comparisons, run_one
from scripts.kaggle_data import evaluate_output, frozen_splits, prepare, snapshot_archive


def fixture_data(tmp_path, classification=True):
    rng=np.random.default_rng(15)
    # Repeated feature rows exercise group-aware external splitting.
    base=pd.DataFrame({'x':rng.normal(size=100),'category':np.where(np.arange(100)%2,'a','b')})
    base['label']=np.where(base.x>0,'positive','negative') if classification else base.x*3
    frame=pd.concat([base,base],ignore_index=True)
    archive=tmp_path/'data.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('dataset.csv',frame.to_csv(index=False))
    spec={'target':'label','metric':'f1_macro' if classification else 'rmse',
          'dataset':'test/local-fixture','description':'IID supervised tabular task','columns':{}}
    return frame,archive,spec


@pytest.mark.parametrize('classification',[False,True])
def test_external_splits_are_reproducible_and_keep_duplicate_groups(tmp_path,classification):
    frame,_,_=fixture_data(tmp_path,classification)
    a=frozen_splits(frame,'label',classification,42)
    b=frozen_splits(frame,'label',classification,42)
    assert all(np.array_equal(first,second) for first,second in zip(a,b))
    assert sum(map(len,a))==len(frame)
    groups=pd.util.hash_pandas_object(frame.drop(columns='label'),index=False).to_numpy()
    for i,first in enumerate(a):
        for second in a[i+1:]:assert not set(groups[first])&set(groups[second])


def test_labels_are_outside_agent_inputs_and_final_set_is_reserved(tmp_path):
    _,archive,spec=fixture_data(tmp_path)
    dev=prepare('fixture',spec,archive,tmp_path/'dev')
    final=prepare('fixture',spec,archive,tmp_path/'final',final=True)
    assert dev['split_sha256']==final['split_sha256']
    train=pd.read_csv(tmp_path/'dev/inputs/train.csv')
    test=pd.read_csv(tmp_path/'dev/inputs/test.csv')
    sample=pd.read_csv(tmp_path/'dev/inputs/sample_submission.csv')
    final_train=pd.read_csv(tmp_path/'final/inputs/train.csv')
    final_test=pd.read_csv(tmp_path/'final/inputs/test.csv')
    assert 'label' not in test and sample.label.nunique()==1
    assert not set(final_test.benchmark_id)&set(train.benchmark_id)
    assert not set(final_test.benchmark_id)&set(test.benchmark_id)
    assert set(final_train.benchmark_id)==set(train.benchmark_id)|set(test.benchmark_id)
    assert 'sealed' not in (tmp_path/'dev/inputs/task.md').read_text()
    assert dev['labels_outside_task_directory'] and final['phase']=='final_lockbox'


@pytest.mark.parametrize('classification',[False,True])
def test_evaluator_scores_only_aligned_valid_outputs(tmp_path,classification):
    _,archive,spec=fixture_data(tmp_path,classification)
    metadata=prepare('fixture',spec,archive,tmp_path/'prepared')
    task=tmp_path/'prepared/inputs'
    sealed=tmp_path/'prepared/sealed/evaluation.csv'
    truth=pd.read_csv(sealed)
    truth.to_csv(task/'submission.csv',index=False)
    assert evaluate_output(task,sealed,metadata)==pytest.approx(1. if classification else 0.)
    truth.iloc[::-1].to_csv(task/'submission.csv',index=False)
    with pytest.raises(ValueError,match='ID order'):evaluate_output(task,sealed,metadata)
    truth['label']='unknown-class' if classification else np.nan
    truth.to_csv(task/'submission.csv',index=False)
    with pytest.raises(ValueError):evaluate_output(task,sealed,metadata)


def archive_fixture(path,relative='atlas/__init__.py',tamper=False):
    data=b'"""fixture source"""\n'
    mapping={'entry.py':('solution.py',hashlib.sha256(data).hexdigest()),
             'module.py':(relative,hashlib.sha256(data).hexdigest())}
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('solution.py','SOURCES='+repr(mapping))
        z.writestr('entry.py',data)
        z.writestr('module.py',b'changed' if tamper else data)


def test_snapshot_is_built_from_actual_zip_and_checks_integrity(tmp_path):
    archive=tmp_path/'good.zip';archive_fixture(archive)
    digest=snapshot_archive(archive,tmp_path/'good')
    assert len(digest)==64 and (tmp_path/'good/atlas/__init__.py').is_file()
    archive=tmp_path/'bad.zip';archive_fixture(archive,tamper=True)
    with pytest.raises(ValueError,match='integrity'):snapshot_archive(archive,tmp_path/'bad')
    archive=tmp_path/'traversal.zip';archive_fixture(archive,relative='../escape.py')
    with pytest.raises(ValueError,match='Unsafe'):snapshot_archive(archive,tmp_path/'traversal')
    assert not (tmp_path/'escape.py').exists()


def test_comparisons_preserve_metric_direction_and_reject_mismatched_qwen_runs():
    records=[]
    for case,metric,before,after in [('housing','rmse',100.,80.),('beans','f1_macro',.7,.8)]:
        for mode,value in [('lightgbm',before),('deterministic',before),('live-agent',after)]:
            records.append({'case':case,'seed':42,'metric':metric,'mode':mode,'external_metric':value,
                            'status':'ok','budget_seconds':60,'raw_sha256':'raw','split_sha256':'split','runtime_code_sha256':'source'})
    assert all(r['signed_gain_over_local_lightgbm']>0 for r in comparisons(records) if r['mode']=='live-agent')
    assert all(r['signed_gain']>0 for r in research_comparisons(records))
    records[-1]['budget_seconds']=120
    with pytest.raises(ValueError,match='mismatched'):research_comparisons(records)


def test_real_reference_worker_is_scored_after_completion(tmp_path):
    _,archive,spec=fixture_data(tmp_path)
    metadata=prepare('fixture',spec,archive,tmp_path/'prepared')
    metadata['sealed_labels']=str(tmp_path/'prepared/sealed/evaluation.csv')
    root=Path(__file__).resolve().parents[1]
    args=argparse.Namespace(budget=60,threads=1,memory_mb=2048,max_calls=0)
    result=run_one(root,tmp_path/'prepared/inputs/task.md',tmp_path/'run','lightgbm',args,{},metadata,code_digest())
    assert result['status']=='ok',result
    assert result['external_metric']>.7 and result['full_refit']
    assert result['qwen_calls']==result['qwen_fits']==0
    assert result['runtime_code_sha256']==code_digest()
