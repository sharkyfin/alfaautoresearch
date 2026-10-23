"""Native categorical fitting, honest group boundaries and visible ML failures."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from atlas.contract import read_inputs
from atlas.features import FeatureBuilder, MatrixBuilder
from atlas.hpo import AdaptiveTPE, study_key
from atlas.models import fit_predict
from atlas.recipes import validate
from atlas.task import load
from atlas.validation import selection_splits
from scripts.synthetic import create


@pytest.mark.parametrize('kind', ['group', 'duplicate_group'])
def test_singleton_class_never_splits_its_entity_or_duplicate_group(tmp_path, kind):
    task = create(tmp_path, 'group' if kind == 'group' else 'integer_labels', n=120)
    path = tmp_path / 'train.csv'
    train = pd.read_csv(path, dtype={'id': str})
    train['response'] = np.tile(['common', 'other', 'common', 'other'], 30)
    if kind == 'duplicate_group':
        for column in train.columns.difference(['id', 'response']):
            train.loc[5:7, column] = train.loc[4, column]
    train.loc[4, 'response'] = 'singleton'
    train.to_csv(path, index=False)
    task.write_text(task.read_text().replace('Task: regression', 'Task: classification')
                    .replace('Primary metric: RMSE', 'Primary metric: Macro F1'))
    data = load(read_inputs(task))
    primary, controls = selection_splits(data)
    groups = data.train.entity_id.to_numpy() if kind == 'group' else pd.util.hash_pandas_object(
        data.train[data.spec.features], index=False).to_numpy()
    assert primary.kind == kind and len(controls) == 2
    for split in [primary, *controls]:
        assert not set(groups[split.train]) & set(groups[split.valid])
        assert 4 in split.train and 4 not in split.valid
        assert set(data.y[split.train]) == set(data.y)
        assert split.json()['class_support']['unvalidated_class_indices']


def test_fixed_parameters_define_separate_tpe_objectives(tmp_path):
    data = load(read_inputs(create(tmp_path, 'binary_imbalance')))
    loss = validate({'model_family': 'lightgbm', 'params': {'iterations': 40}}, data.spec)
    primary = copy.deepcopy(loss)
    primary['params']['stopping_metric'] = 'primary'
    explicit_loss = copy.deepcopy(loss)
    explicit_loss['params']['stopping_metric'] = 'loss'
    assert study_key(loss) == study_key(explicit_loss)
    assert study_key(loss) != study_key(primary)
    hpo = AdaptiveTPE(data.spec)
    hpo.observe({'recipe': loss, 'status': 'ok', 'standardized_utility': .7})
    hpo.observe({'recipe': primary, 'status': 'ok', 'standardized_utility': .8})
    assert len(hpo.get_study(loss)[0].trials) == 1
    assert len(hpo.get_study(primary)[0].trials) == 1
    changed = copy.deepcopy(loss)
    changed['params']['num_leaves'] = 63
    assert study_key(changed) == study_key(loss)
    shallow = validate({'model_family': 'extra_trees', 'params': {'max_depth': 3}}, data.spec)
    deep = copy.deepcopy(shallow)
    deep['params']['max_depth'] = 20
    assert study_key(shallow) != study_key(deep)


def test_catboost_preserves_native_unknown_categories_numeric_missingness_and_refits(tmp_path):
    data = load(read_inputs(create(tmp_path, 'unseen')))
    data.train.loc[0, 'x0'] = np.nan
    features = FeatureBuilder(data.spec).fit(data.train.iloc[:160])
    encoded = MatrixBuilder('catboost', features.cat, features.num).fit(features.transform(data.train.iloc[:160]))
    matrix = encoded.transform(features.transform(data.test))
    assert set(matrix.category) == {'new'}
    assert encoded.transform(features.transform(data.train.iloc[:1])).x0.isna().all()
    split, _ = selection_splits(data)
    recipe = validate({'model_family': 'catboost', 'params': {'iterations': 30, 'depth': 4}}, data.spec)
    valid, test, info = fit_predict(data, recipe, split.train, split.valid, threads=1)
    assert np.isfinite(valid).all() and np.isfinite(test).all()
    assert 1 <= info['iterations'] <= 30
    _, full, _ = fit_predict(data, recipe, np.arange(len(data.y)), np.empty(0, dtype=int),
                             threads=1, refit=True, iterations_override=info['iterations'])
    assert full.shape == test.shape and np.isfinite(full).all()


def test_fatal_ml_error_keeps_csv_and_reports_failure(tmp_path, monkeypatch):
    from atlas import orchestration
    from atlas.experiments import Executor
    task = create(tmp_path / 'task')
    def broken(*args, **kwargs):
        raise RuntimeError('injected fit controller failure')
    monkeypatch.setattr(Executor, 'run', broken)
    directory = tmp_path / 'run'
    assert orchestration.run(task, budget_seconds=30, mode='deterministic', run_dir=directory) == 1
    summary = json.loads((directory / 'summary.json').read_text())
    assert summary['status'] == 'fatal_with_fallback'
    assert summary['checkpoint_kind'] == 'valid_dummy'
    assert not summary['trained_output'] and summary['fit_success'] == 0
    assert (task.parent / 'submission.csv').is_file()
    assert summary['fatal_exception']['exception_class'] == 'RuntimeError'


def test_fully_duplicate_inputs_do_not_leak_across_validation(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    for column in data.spec.features:
        data.train[column] = data.train[column].iloc[0]
    primary, controls = selection_splits(data)
    assert primary.kind == 'duplicate_group'
    assert not len(primary.valid) and not controls
    assert data.spec.profile['duplicate_audit']['conflicting_target_groups'] == 1
