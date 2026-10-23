"""The external evaluator and curated split metadata must survive real CSV IO."""
import json

import numpy as np
import pandas as pd
import pytest

from atlas.contract import read_inputs
from atlas.metrics import Metric
from atlas.task import load
from atlas.validation import selection_splits
from scripts import benchmark_data


@pytest.mark.parametrize('regime', ['grouped', 'temporal'])
def test_curated_inputs_keep_split_metadata_and_hide_external_answers(tmp_path, monkeypatch, regime):
    frame = pd.DataFrame({'x': np.arange(12.), 'response': np.arange(12.)})
    task = {}
    problem = 'regression'
    metric = 'rmse'
    if regime == 'grouped':
        frame['patient'] = np.repeat(np.arange(6), 2)
        frame['response'] = frame.patient % 2
        task = {'group_on': 'patient', 'group_labels': 'per_group'}
        problem = 'binary_classification'
        metric = 'roc_auc'
    else:
        frame['date'] = pd.date_range('2020-01-01', periods=12)
        task = {'time_on': 'date'}
    record = {'case': 'fixture', 'source': 'beyondarena', 'dataset': 'fixture',
        'reserved_phase': 'development', 'target': 'response', 'problem_type': problem,
        'metric': metric, 'task_metadata': task, 'license': 'CC BY 4.0', 'split_regime': regime}
    monkeypatch.setattr(benchmark_data, 'source_split', lambda *args: (
        frame, np.arange(8), np.arange(8, 12), {'dataset_sha256': 'fixture', 'official_fold': 0}))
    metadata, truth = benchmark_data.prepare_external(record, tmp_path / 'inputs')
    assert len(truth) == 4 and not list((tmp_path / 'inputs').glob('*evaluation*'))
    assert 'response' not in pd.read_csv(tmp_path / 'inputs/test.csv')
    assert metadata['labels_storage'].startswith('parent evaluator memory')
    data = load(read_inputs(tmp_path / 'inputs/task.md'))
    primary, controls = selection_splits(data)
    if regime == 'grouped':
        assert data.spec.group_candidates == ['patient'] and 'patient' not in data.spec.features
        assert primary.kind == 'group'
        assert not set(data.train.patient.iloc[primary.train]) & set(data.train.patient.iloc[primary.valid])
        assert '__evaluation_group__' in truth
    else:
        assert primary.kind == 'time'
        for split in [primary, *controls]:
            assert data.train.date.iloc[split.train].max() < data.train.date.iloc[split.valid].min()


def test_private_probability_evaluation_aligns_original_numeric_class_names(tmp_path):
    sample = pd.DataFrame({'key': ['a', 'b', 'c'], '3': 0., '7': 0., '42': 0.})
    prediction = np.array([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
    sample.to_csv(tmp_path / 'sample_submission.csv', index=False)
    output = sample.copy()
    output[['3', '7', '42']] = prediction
    output.to_csv(tmp_path / 'submission.csv', index=False)
    truth = pd.DataFrame({'key': ['a', 'b', 'c'], 'response': [3, 7, 42]})
    metadata = {'id_column': 'key', 'target': 'response', 'metric': 'logloss', 'labels': [3, 7, 42]}
    score, _ = benchmark_data.evaluate_memory(tmp_path, truth, metadata)
    assert score == pytest.approx(Metric('logloss', 3).score([0, 1, 2], prediction))
    output.iloc[::-1].to_csv(tmp_path / 'submission.csv', index=False)
    with pytest.raises(ValueError):
        benchmark_data.evaluate_memory(tmp_path, truth, metadata)


def test_group_label_score_does_not_weight_patients_by_row_count(tmp_path):
    sample = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'response': .5})
    sample.to_csv(tmp_path / 'sample_submission.csv', index=False)
    output = sample.copy()
    output['response'] = [.1, .9, .6, .6]
    output.to_csv(tmp_path / 'submission.csv', index=False)
    truth = pd.DataFrame({'key': sample.key, 'response': [0, 0, 1, 1], '__evaluation_group__': ['A', 'A', 'B', 'B']})
    metadata = {'id_column': 'key', 'target': 'response', 'metric': 'roc_auc', 'labels': [0, 1]}
    score, detail = benchmark_data.evaluate_memory(tmp_path, truth, metadata)
    assert score == 1 and detail['row_metric'] == .5 and detail['group_count'] == 2


def test_reserved_final_dataset_cannot_be_opened_as_development(tmp_path):
    with pytest.raises(ValueError, match='reservation'):
        benchmark_data.prepare_external({'reserved_phase': 'final_lockbox'}, tmp_path / 'data')
