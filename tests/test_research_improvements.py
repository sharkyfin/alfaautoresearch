"""Regression coverage for evidence selection, controlled features and time controls."""
import json

import numpy as np
import pandas as pd
import pytest

from atlas.agent import QwenClient, compact_context, researcher_context
from atlas.context import error_associations, input_importance
from atlas.contract import read_inputs
from atlas.ensemble import choose_blend, tune_multipliers
from atlas.expressions import expression_references
from atlas.recipes import RecipeError, recipe_hash, validate
from atlas.research import ablation_options, controlled_hypothesis
from atlas.task import load, profile
from atlas.validation import Split, confirmation_splits, design
from scripts.synthetic import create


def wide_data(tmp_path, n=300):
    data = load(read_inputs(create(tmp_path, n=n)))
    rng = np.random.default_rng(12)
    names = [f'noise_{i}' for i in range(220)] + ['late_signal', 'late_error', 'late_drift', 'late_missing']
    train = pd.DataFrame(rng.normal(size=(n, len(names))), columns=names)
    test = pd.DataFrame(rng.normal(size=(60, len(names))), columns=names)
    test['late_drift'] += 30
    train.loc[:int(n*.75), 'late_missing'] = np.nan
    train['response'] = data.y
    data.train, data.test = train, test
    data.spec.features = names
    data.spec.numeric_features = names
    data.spec.categorical_features = []
    data.spec.text_candidates = []
    data.spec.datetime_candidates = []
    data.spec.profile = profile(train, test, data.spec)
    data.dictionary = pd.DataFrame({'feature': names, 'description_ru': ['meaning of '+c for c in names]})
    return data


def test_late_columns_and_dictionary_are_selected_by_evidence(tmp_path):
    data = wide_data(tmp_path)
    assert 'late_signal' not in data.spec.profile['columns']
    champion = {'experiment_id': 'champ', 'recipe': {'model_family': 'hist', 'feature_expressions': [
        {'name': 'interaction', 'expression': 'late_signal * noise_0'}]},
        'diagnostics': {'feature_importance': {'interaction': 100.},
                        'error_associations': [{'column': 'late_error', 'association': .9}]}}
    context = researcher_context(data, [], champion, 100)
    chosen = context['untrusted_data']
    for name in ['late_signal', 'late_error', 'late_drift', 'late_missing']:
        assert name in chosen['features'][:12]
        assert name in chosen['profile']['columns']
        assert any(row['column'] == name and row['description'] == 'meaning of '+name for row in chosen['dictionary'])
    context['untrusted_data']['description'] *= 100
    compact = compact_context(context)
    assert 'late_signal' in compact['untrusted_data']['features']
    assert len(json.dumps(compact, ensure_ascii=False).encode()) <= 10000
    assert 'response' not in chosen['features'] and 'id' not in chosen['features']


def test_schema_tail_shift_is_found_without_model_importance(tmp_path):
    data = wide_data(tmp_path)
    context = researcher_context(data, [], None, 100)
    assert context['untrusted_data']['features'][0] == 'late_drift'
    assert context['untrusted_data']['profile']['columns']['late_drift']['sampled']


def test_error_associations_are_bounded_aggregates_and_find_tail_feature(tmp_path):
    data = wide_data(tmp_path)
    data.train['late_missing'] = np.linspace(0, 10, len(data.train))
    indices = np.arange(len(data.y))
    result = error_associations(data, indices, data.y + data.train.late_missing.to_numpy(), max_rows=128)
    association = next(row for row in result if row['column'] == 'late_missing')
    assert association['association'] > .99 and association['support'] == 128
    assert set(association) == {'column', 'association', 'kind', 'support'}
    assert error_associations(data, indices, data.y) == []


def test_formula_and_transformed_importance_resolve_to_original_inputs(tmp_path):
    data = load(read_inputs(create(tmp_path, 'future')))
    recipe = {'feature_ops': [{'op': 'product', 'columns': ['x0', 'x1']}],
              'feature_expressions': [{'name': 'inner', 'expression': 'x2 * x3'},
                                      {'name': 'outer', 'expression': 'abs(inner) + col("x4")'}]}
    importance = input_importance(data.spec, recipe, {'feature_importance': {
        'outer': 3., '__atlas_op_0': 2., 'event_date__month': 1.}})
    assert importance == {'x0': 1., 'x1': 1., 'x2': 1., 'x3': 1., 'x4': 1., 'event_date': 1.}
    assert expression_references('abs(x0) + col("space column")') == {'x0', 'space column'}


def parent_candidate(data):
    data.spec.validation_plan = design(data).json()
    recipe = validate({'model_family': 'linear', 'params': {'alpha': .1}}, data.spec)
    return {'experiment_id': 'parent', 'recipe': recipe, 'metric_value': 2., 'standardized_utility': -2.}


def test_feature_trial_freezes_model_params_and_transform_but_hpo_stays_available(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    parent = parent_candidate(data)
    raw = {'model_family': 'hist', 'params': {'iterations': 25},
           'feature_expressions': [{'name': 'rate', 'expression': 'x0 / maximum(abs(x1), 1)'}]}
    candidate, trial = controlled_hypothesis(raw, parent, data.spec, 900)
    assert candidate['model_family'] == 'linear' and candidate['params'] == {'alpha': .1}
    assert candidate['target_transform'] == parent['recipe']['target_transform']
    assert trial['ignored_model_overrides'] == ['model_family', 'params']
    assert raw['model_family'] == 'hist' and parent['recipe']['feature_expressions'] == []
    hpo, trial = controlled_hypothesis({'model_family': 'linear', 'params': {'alpha': 30}}, parent, data.spec, 900)
    assert hpo['params']['alpha'] == 30 and trial is None
    with pytest.raises(RecipeError):
        controlled_hypothesis({**raw, 'operation': 'shell'}, parent, data.spec, 900)
    with pytest.raises(RecipeError, match='at most four'):
        controlled_hypothesis({**raw, 'feature_expressions': [{'name': f'f{i}', 'expression': 'x0 + 1'} for i in range(5)]}, parent, data.spec, 900)


def test_ablation_preserves_dependencies_and_never_retrains_unchanged_parent(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    parent = parent_candidate(data)
    group, _ = controlled_hypothesis({'model_family': 'linear', 'feature_expressions': [
        {'name': 'interaction', 'expression': 'x0 * x1'},
        {'name': 'magnitude', 'expression': 'abs(interaction)'},
        {'name': 'nuisance', 'expression': 'x2 * x3'}]}, parent, data.spec, 900)
    options = ablation_options(group, parent['recipe'])
    assert len(options) == 3
    for candidate, _ in options:
        validate(candidate, data.spec)
        assert recipe_hash(candidate) != recipe_hash(parent['recipe'])
        assert candidate['params'] == group['params']
    without_interaction = next(r for r, detail in options if detail['label'] == 'formula:interaction')
    assert [f['name'] for f in without_interaction['feature_expressions']] == ['nuisance']
    # Rebase the next ablation on the simplified champion; removed formulas stay removed.
    simplified = next(r for r, detail in options if detail['label'] == 'formula:nuisance')
    for candidate, _ in ablation_options(simplified, parent['recipe'], {'formula:nuisance'}):
        assert all(f['name'] != 'nuisance' for f in candidate['feature_expressions'])


def test_ablation_skips_a_cyclic_partial_revert(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    parent = parent_candidate(data)
    parent['recipe'] = validate({**parent['recipe'], 'feature_expressions': [
        {'name': 'a', 'expression': 'x0'}, {'name': 'b', 'expression': 'a + 1'}]}, data.spec)
    candidate = validate({**parent['recipe'], 'feature_expressions': [
        {'name': 'b', 'expression': 'x1'}, {'name': 'a', 'expression': 'b + 1'}]}, data.spec)
    options = ablation_options(candidate, parent['recipe'])
    assert options and all(detail['label'] != 'formula:b' for _, detail in options)
    for recipe, _ in options:validate(recipe, data.spec)


@pytest.mark.parametrize('removal_hurts', [False, True])
def test_controlled_feature_group_and_conditional_ablation_in_full_loop(tmp_path, monkeypatch, removal_hurts):
    from atlas import orchestration
    from atlas.experiments import Executor
    for key, value in {'OPENAI_BASE_URL': 'https://organizer.invalid/v1', 'OPENAI_API_KEY': 'test-only'}.items():
        monkeypatch.setenv(key, value)

    def fit(self, recipe, split, timeout, seed=42, **kwargs):
        self.counter += 1
        names = {f['name'] for f in recipe.get('feature_expressions', [])}
        error = 1. if recipe['model_family'] == 'lightgbm' else 3.
        if 'interaction' in names:
            error = .2 if 'nuisance' in names or not removal_hurts else .4
        result = {'experiment_id': f'exp_{self.counter:04d}', 'recipe': recipe, 'status': 'ok',
                  'runtime_sec': .01, 'peak_memory_mb': 1., 'diagnostics': {}}
        return result, (self.data.y[split.valid]+error, np.zeros(len(self.data.test)))

    def transport(body, timeout):
        proposal = {'model_family': 'hist', 'params': {'iterations': 25}, 'feature_expressions': [
            {'name': 'interaction', 'expression': 'x0 * x1'},
            {'name': 'nuisance', 'expression': 'x2 * x3'}]}
        return {'choices': [{'message': {'content': json.dumps(proposal)}}], 'usage': {'total_tokens': 100}}

    monkeypatch.setattr(Executor, 'run', fit)
    client = QwenClient('live', max_calls=1, transport=transport)
    task = create(tmp_path/'data')
    run_dir = tmp_path/'run'
    assert orchestration.run(task, budget_seconds=120, max_experiments=5, mode='live-agent',
                             research_client=client, threads=1, run_dir=run_dir) == 0
    history = [json.loads(row) for row in (run_dir/'experiments.jsonl').read_text().splitlines()]
    group = next(r for r in history if r.get('origin') == 'qwen')
    baseline = next(r for r in history if r['experiment_id'] == group['feature_trial']['parent_experiment'])
    assert group['recipe']['params'] == baseline['recipe']['params']
    assert group['recipe']['model_family'] == baseline['recipe']['model_family'] == 'lightgbm'
    ablations = [r for r in history if r.get('origin') == 'feature-ablation']
    assert ablations and len(ablations) <= 3
    nuisance = next(r for r in ablations if r['feature_trial']['label'] == 'formula:nuisance')
    assert nuisance['accepted'] == (not removal_hurts)
    assert all(r['recipe']['params'] == baseline['recipe']['params'] for r in ablations)
    summary = json.loads((run_dir/'summary.json').read_text())
    assert summary['full_refit_published'] and summary['feature_group_trials'] == 1
    assert not (run_dir/'error.txt').exists()


def test_researcher_retains_group_and_ablation_feedback_when_compacted(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    group = {'experiment_id': 'group', 'origin': 'qwen', 'status': 'ok', 'metric_value': 1., 'accepted': True}
    ablation = {'experiment_id': 'ablation', 'origin': 'feature-ablation', 'status': 'ok', 'metric_value': 1.,
                'accepted': True, 'feature_trial': {'group_experiment': 'group', 'reverted': ['noise']}}
    history = [group, ablation] + [{'experiment_id': f'base{i}', 'origin': 'deterministic', 'status': 'ok',
                                   'recipe': {'model_family': 'hist'}, 'diagnostics': {'residual_bias': .1}} for i in range(10)]
    context = researcher_context(data, history, None, 100)
    context['untrusted_data']['description'] *= 100
    compact = compact_context(context)
    feedback = {r['experiment_id']: r for r in compact['history']}
    assert {'group', 'ablation'} <= feedback.keys()
    assert feedback['ablation']['feature_trial']['reverted'] == ['noise']


@pytest.mark.parametrize('shuffled', [False, True])
def test_temporal_controls_are_disjoint_forward_windows_with_gap_and_ties(tmp_path, shuffled):
    task = create(tmp_path, 'future', n=300)
    frame = pd.read_csv(task.parent/'train.csv')
    frame['event_date'] = pd.date_range('2022-01-01', periods=100, freq='D').repeat(3).astype(str)
    if shuffled: frame = frame.sample(frac=1, random_state=4)
    frame.to_csv(task.parent/'train.csv', index=False)
    task.write_text(task.read_text() + '\ngap_rows: 7\n')
    data = load(read_inputs(task))
    primary = design(data)
    windows = confirmation_splits(data, primary)
    assert len(windows) == 2
    dates = pd.to_datetime(data.train.event_date)
    for split in [primary, *windows]:
        assert dates.iloc[split.train].max() < dates.iloc[split.valid].min()
        earlier = dates[dates < dates.iloc[split.valid].min()]
        assert len(earlier) - len(split.train) == 7
    assert len({s.id for s in [primary, *windows]}) == 3
    for i, first in enumerate([primary, *windows]):
        for second in [primary, *windows][i+1:]:
            assert not set(first.valid) & set(second.valid)
            assert not set(dates.iloc[first.valid]) & set(dates.iloc[second.valid])
    assert [s.id for s in windows] == [s.id for s in confirmation_splits(data, primary, seed=999)]


def test_time_controls_are_fixed_before_scores_and_used_by_orchestrator(tmp_path, monkeypatch):
    from atlas import orchestration
    from atlas.experiments import Executor
    calls = []

    def fit(self, recipe, split, timeout, seed=42, **kwargs):
        self.counter += 1
        calls.append((recipe['model_family'], split, kwargs.get('refit', False)))
        error = 1. if recipe['model_family'] == 'linear' else 2.
        result = {'experiment_id': f'exp_{self.counter:04d}', 'recipe': recipe, 'status': 'ok',
                  'runtime_sec': .01, 'peak_memory_mb': 1., 'diagnostics': {}}
        return result, (self.data.y[split.valid]+error, np.zeros(len(self.data.test)))

    monkeypatch.setattr(Executor, 'run', fit)
    task = create(tmp_path/'data', 'future', n=300)
    run_dir = tmp_path/'run'
    assert orchestration.run(task, budget_seconds=120, max_experiments=2, mode='deterministic', run_dir=run_dir) == 0
    summary = json.loads((run_dir/'summary.json').read_text())
    assert len(summary['confirmation_plan']) == len(summary['confirmation_metrics']) == 2
    assert summary['full_refit_published'] and not (run_dir/'error.txt').exists()
    for family in ['linear', 'lightgbm']:
        seen = [split for name, split, refit in calls if name == family and not refit]
        assert len(seen) == 3 and len({split.id for split in seen}) == 3


def test_non_temporal_tasks_use_two_controls_and_time_gap_is_never_relaxed(tmp_path):
    data = load(read_inputs(create(tmp_path/'iid')))
    assert len(confirmation_splits(data, design(data))) == 2
    task = create(tmp_path/'time', 'future', n=60)
    task.write_text(task.read_text()+'\ngap_rows: 35\n')
    data = load(read_inputs(task))
    primary = design(data)
    assert primary.kind == 'time'
    assert confirmation_splits(data, primary) == []


def test_blend_is_rejected_if_it_hurts_one_temporal_control(tmp_path):
    data = load(read_inputs(create(tmp_path, 'future')))
    primary = design(data)
    windows = confirmation_splits(data, primary)

    def candidate(name, split, sign):
        error = np.where(np.arange(len(split.valid)) % 2, 1., -1.) * sign
        return {'experiment_id': name, 'split': split,
                'predictions': (data.y[split.valid]+error, np.zeros(len(data.test))), 'standardized_utility': -1.}

    finalists = [candidate('a', primary, 1), candidate('b', primary, -1)]
    evidence = {name: [candidate(name, split, sign) for split in windows] for name, sign in [('a', 1), ('b', -1)]}
    assert choose_blend(data, finalists, evidence) is not None
    evidence['b'][1]['predictions'] = evidence['a'][1]['predictions']
    assert choose_blend(data, finalists, evidence) is None


@pytest.mark.parametrize('scenario', ['iid', 'future'])
def test_real_workers_fit_controlled_features_ablate_and_refit(tmp_path, monkeypatch, scenario):
    from atlas import orchestration, search
    for key, value in {'OPENAI_BASE_URL': 'https://organizer.invalid/v1', 'OPENAI_API_KEY': 'test-only'}.items():
        monkeypatch.setenv(key, value)
    # Keep both measured controls linear so the known interaction can be isolated
    # analytically. The executor and every model fit remain real child processes.
    monkeypatch.setattr(search, 'initial_portfolio', lambda spec, budget: [
        search.recipe('linear', {'alpha': .01}), search.recipe('linear', {'alpha': .02})])
    task = create(tmp_path/'data', scenario, n=600)
    train = pd.read_csv(task.parent/'train.csv')
    train['response'] = 5*train.x0 + 3*train.x0*train.x1
    train.to_csv(task.parent/'train.csv', index=False)

    def transport(body, timeout):
        proposal = {'model_family': 'hist', 'params': {'iterations': 25}, 'feature_expressions': [
            {'name': 'interaction', 'expression': 'x0 * x1'},
            {'name': 'nuisance', 'expression': 'x2 * 0'}]}
        return {'choices': [{'message': {'content': json.dumps(proposal)}}], 'usage': {'total_tokens': 100}}

    client = QwenClient('live', max_calls=1, transport=transport)
    run_dir = tmp_path/'run'
    assert orchestration.run(task, budget_seconds=120, max_experiments=4, mode='live-agent',
                             research_client=client, threads=1, run_dir=run_dir) == 0
    history = [json.loads(row) for row in (run_dir/'experiments.jsonl').read_text().splitlines()]
    group = next(r for r in history if r.get('origin') == 'qwen')
    ablation = next(r for r in history if r.get('origin') == 'feature-ablation')
    assert group['recipe']['model_family'] == ablation['recipe']['model_family'] == 'linear'
    assert group['accepted'] and ablation['accepted'] and group['metric_value'] < .01
    summary = json.loads((run_dir/'summary.json').read_text())
    assert summary['full_refit_published'] and summary['accepted_feature_ablations'] == 1
    assert all(f['name'] != 'nuisance' for f in summary['champion_recipe']['feature_expressions'])
    if scenario == 'future':
        assert len(summary['confirmation_plan']) == len(summary['confirmation_metrics']) == 2
    assert not (run_dir/'error.txt').exists()
    assert np.isfinite(pd.read_csv(task.parent/'submission.csv').response).all()


def test_probability_adjustment_must_improve_every_control(tmp_path):
    data = load(read_inputs(create(tmp_path, 'binary_imbalance', n=1000)))
    primary = design(data)
    splits = [design(data, seed=1), design(data, seed=2)]

    def candidate(split, underpredict):
        y = data.y[split.valid]
        positive = np.where(y == 1, .45 if underpredict else .8, .01)
        p = np.column_stack([1-positive, positive])
        return {'split': split, 'predictions': (p, np.tile([.6, .4], (len(data.test), 1))),
                'standardized_utility': data.spec.metric.score(y, p)}

    champion = candidate(primary, True)
    evidence = [candidate(split, True) for split in splits]
    assert tune_multipliers(data, champion, evidence) is not None
    evidence[1] = candidate(splits[1], False)
    assert tune_multipliers(data, champion, evidence) is None
