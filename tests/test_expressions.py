import json

import numpy as np
import pandas as pd
import pytest

from atlas.agent import QwenClient, compact_context, researcher_context
from atlas.contract import read_inputs
from atlas.experiments import Executor
from atlas.expressions import ExpressionError, ExpressionFeatures, validate_expressions
from atlas.features import FeatureBuilder
from atlas.judge import evaluate
from atlas.models import FAMILIES
from atlas.recipes import RecipeError, merge_hypothesis_features, recipe_hash, validate
from atlas.search import SearchController, initial_portfolio
from atlas.budget import Budget
from atlas.task import load
from atlas.validation import design
from scripts.synthetic import create


def formula(name, expression):
    return {'name': name, 'expression': expression}


def test_nested_formulas_constants_conditions_and_dependencies():
    x = pd.DataFrame({'bytes': [10., 20., 30.], 'packets': [1., 0., 2.]}, index=[7, 2, 9])
    expressions = [formula('rate', 'bytes / (packets + 1)'),
                   formula('flag', '(rate > 9) and (packets < 2)'),
                   formula('adjusted', 'where(flag, log1p(rate), rate ** 2)'),
                   formula('filled', 'fillna(bytes / packets, -1)'),
                   formula('conditional', 'rate if packets > 0 else -1')]
    result = ExpressionFeatures(expressions, x.columns).fit(x).transform(x)
    np.testing.assert_allclose(result['rate'], [5., 20., 10.])
    np.testing.assert_allclose(result['flag'], [0., 1., 0.])
    np.testing.assert_allclose(result['adjusted'], [25., np.log1p(20), 100.])
    np.testing.assert_allclose(result['filled'], [10., -1., 15.])
    np.testing.assert_allclose(result['conditional'], [5., -1., 10.])
    assert result['rate'].index.equals(x.index)


def test_statistics_are_fitted_only_on_training_fold_and_reused():
    train = pd.DataFrame({'value': [1., 3., 8.], 'kind': ['a', 'a', 'b']})
    valid = pd.DataFrame({'value': [1000., -1000., 9999.], 'kind': ['a', 'b', 'new']})
    expressions = [formula('mean_value', 'group_mean(value, kind)'),
                   formula('median_value', 'group_median(value, kind)'),
                   formula('std_value', 'group_std(value, kind)'),
                   formula('count_value', 'group_count(kind)'),
                   formula('frequency_value', 'frequency(kind)'),
                   formula('relative', 'value / maximum(mean_value, 1)')]
    engine = ExpressionFeatures(expressions, train.columns).fit(train)
    state = repr(engine.learned)
    result = engine.transform(valid)
    np.testing.assert_allclose(result['mean_value'], [2., 8., 3.])
    np.testing.assert_allclose(result['median_value'], [2., 8., 3.])
    np.testing.assert_allclose(result['std_value'], [1., 0., 0.])
    np.testing.assert_allclose(result['count_value'], [2., 1., 0.])
    np.testing.assert_allclose(result['frequency_value'], [2/3, 1/3, 0.])
    assert repr(engine.learned) == state


def test_dates_text_and_quoted_column_names_are_vectorized():
    x = pd.DataFrame({'event date': ['2025-01-03', '2026-02-05'],
                      'start': ['2025-01-01', '2026-02-01'], 'text': ['Ab 12', 'CD']})
    expressions = [formula('event_year', 'year(col("event date"))'),
                   formula('elapsed', 'days_between(col("event date"), start)'),
                   formula('tokens', 'word_count(text) + digit_count(text)'),
                   formula('category_pair', 'concat(lower(text), event_year)')]
    result = ExpressionFeatures(expressions, x.columns).fit(x).transform(x)
    np.testing.assert_allclose(result['event_year'], [2025., 2026.])
    np.testing.assert_allclose(result['elapsed'], [2., 4.])
    np.testing.assert_allclose(result['tokens'], [4., 1.])
    assert result['category_pair'].tolist() == ['ab 12::2025.0', 'cd::2026.0']


def test_formula_numeric_strings_and_forbidden_columns():
    x = pd.DataFrame({'value': ['1 000,5', '2,5'], 'target': [99, 99]})
    engine = ExpressionFeatures([formula('scaled', 'value / 2')], x.columns, ['target']).fit(x)
    np.testing.assert_allclose(engine.transform(x)['scaled'], [500.25, 1.25])
    with pytest.raises(ExpressionError):
        validate_expressions([formula('leak', 'target + 1')], x.columns, ['target'])


@pytest.mark.parametrize('source', [
    '__import__("os").system("false")', 'x.__class__', 'x[0]',
    '[v for v in x]', '(lambda: 1)()', 'open("file")',
    'col("target")', 'col("id")', 'target + 1', 'future + 1',
    'x ** x', 'x ** 100', 'x ** -100', 'sum(x)',
    'clip(x, lower=0, upper=1)', 'col(x)', '10' * 200,
])
def test_forbidden_or_unbounded_formulas_are_rejected(source):
    with pytest.raises(ExpressionError):
        validate_expressions([formula('result', source)], ['x'], ['target', 'id'])


def test_formula_dependencies_bounds_and_names():
    for expressions in [
        [formula('a', 'b + 1'), formula('b', 'x + 1')],
        [formula('a', 'x'), formula('a', 'x + 1')],
        [formula('x', 'x + 1')], [formula('target', 'x')],
        [formula('log1p', 'x')], [formula('a', 'abs(' * 15 + 'x' + ')' * 15)],
        [formula('f' + str(i), 'x + 1') for i in range(17)],
    ]:
        with pytest.raises(ExpressionError):
            validate_expressions(expressions, ['x'], ['target'])


def test_named_features_survive_hypothesis_merging_and_local_search(tmp_path):
    parent = {'feature_ops': [{'op': 'square', 'columns': ['x0']}],
              'feature_expressions': [formula('interaction', 'x0 * x1')]}
    raw = {'model_family': 'hist', 'feature_expressions': [formula('offset', 'interaction + 1')]}
    merged = merge_hypothesis_features(raw, parent)
    assert merged['feature_ops'] == parent['feature_ops']
    assert [f['name'] for f in merged['feature_expressions']] == ['interaction', 'offset']
    replaced = merge_hypothesis_features({**raw, 'feature_mode': 'replace'}, parent)
    assert replaced.get('feature_ops', []) == []
    assert parent['feature_expressions'] == [formula('interaction', 'x0 * x1')]
    data = load(read_inputs(create(tmp_path)))
    scheduler = SearchController(data.spec, Budget(120))
    scheduler.queue = []
    champion = {'experiment_id': 'accepted', 'recipe': {
        'model_family': 'hist', 'params': {'iterations': 30}, **parent}}
    for _ in range(8):
        candidate = scheduler.next(champion)
        assert parent['feature_expressions'] == candidate['feature_expressions']
        assert parent['feature_ops'][0] in candidate['feature_ops']


def test_formula_hash_detects_real_changes_but_ignores_explanations():
    base = {'model_family': 'linear', 'feature_expressions': [formula('f', 'x0 * x1')]}
    assert recipe_hash(base) != recipe_hash({**base, 'feature_expressions': [formula('f', 'x0 / x1')]})
    assert recipe_hash(base) == recipe_hash({**base, 'feature_expressions': [{**base['feature_expressions'][0], 'reason': 'same formula'}]})


def test_formulas_reach_real_worker_and_improve_held_out_interaction(tmp_path):
    task = create(tmp_path / 'data', n=600)
    path = task.parent / 'train.csv'
    train = pd.read_csv(path)
    train['response'] = 3 * train.x0 * train.x1
    train.to_csv(path, index=False)
    data = load(read_inputs(task))
    split = design(data)
    data.spec.validation_plan = split.json()
    executor = Executor(data, tmp_path / 'experiments', threads=1)
    raw = {'model_family': 'linear', 'params': {'alpha': .01}, 'max_runtime_sec': 15}
    baseline = validate(raw, data.spec)
    enhanced = validate({**raw, 'feature_expressions': [formula('interaction', 'x0 * x1')]}, data.spec)
    scores = []
    for recipe in [baseline, enhanced]:
        result, predictions = executor.run(recipe, split, timeout=15)
        assert result['status'] == 'ok', result
        scores.append(evaluate(data, split, predictions, result)['metric_value'])
    assert scores[1] < scores[0] * .1


def test_native_catboost_is_capability_gated_in_execution_search_and_qwen(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    assert 'catboost' in FAMILIES
    assert any(r['model_family'] == 'catboost' for r in initial_portfolio(data.spec))
    context = researcher_context(data, [], None, 60)
    assert 'catboost' in context['capabilities']['models']
    assert 'group_mean' in context['capabilities']['feature_formulas']['functions']
    validate({'model_family': 'catboost'}, data.spec)
    data.spec.available_capabilities.pop('catboost')
    assert all(r['model_family'] != 'catboost' for r in initial_portfolio(data.spec))
    assert 'catboost' not in researcher_context(data, [], None, 60)['capabilities']['models']
    with pytest.raises(RecipeError): validate({'model_family': 'catboost'}, data.spec)


def test_explicit_numeric_categorical_hint_is_respected(tmp_path):
    task = create(tmp_path)
    for filename in ['train.csv', 'test.csv']:
        path = task.parent / filename
        frame = pd.read_csv(path, dtype={'id': 'string'})
        frame['category'] = np.arange(len(frame)) % 3
        frame.to_csv(path, index=False)
    path = task.parent / 'data_dictionary.csv'
    dictionary = pd.read_csv(path)
    dictionary.loc[dictionary.feature == 'category', 'dtype'] = 'categorical'
    dictionary.to_csv(path, index=False)
    data = load(read_inputs(task))
    assert 'category' in data.spec.categorical_features
    transformed = FeatureBuilder(data.spec).fit(data.train).transform(data.test)
    assert not pd.api.types.is_numeric_dtype(transformed.category)


def test_context_compaction_preserves_complete_formula_dependencies(tmp_path):
    data = load(read_inputs(create(tmp_path)))
    expressions = [formula('interaction', 'x0 * x1'), formula('nested', 'log1p(abs(interaction))')]
    champion = {'experiment_id': 'champion', 'recipe': {'model_family': 'linear', 'feature_expressions': expressions}}
    context = researcher_context(data, [], champion, 100)
    context['untrusted_data']['description'] *= 1000
    compact = compact_context(context)
    assert compact['champion']['recipe']['feature_expressions'] == expressions
    assert len(json.dumps(compact, ensure_ascii=False).encode()) <= 10000


def test_qwen_protocol_can_execute_named_features(tmp_path, monkeypatch):
    from atlas.orchestration import run
    for key, value in {'OPENAI_BASE_URL': 'https://organizer.invalid/v1', 'OPENAI_API_KEY': 'test-only'}.items():
        monkeypatch.setenv(key, value)
    seen = []
    def transport(body, timeout):
        context = json.loads(body['messages'][1]['content'])
        seen.append(context)
        proposal = {'model_family': 'linear', 'params': {'alpha': .01},
                    'feature_expressions': [formula('interaction', 'x0 * x1'), formula('magnitude', 'abs(interaction)')],
                    'statement': 'Test named interaction formulas', 'max_runtime_sec': 15}
        return {'choices': [{'message': {'content': json.dumps(proposal)}}], 'usage': {'total_tokens': 300}}
    client = QwenClient('live', max_calls=1, transport=transport)
    task = create(tmp_path / 'data', n=240)
    assert run(task, budget_seconds=120, threads=1, max_experiments=6,
               mode='live-agent', research_client=client, run_dir=tmp_path / 'run') == 0
    history = [json.loads(line) for line in (tmp_path / 'run' / 'experiments.jsonl').read_text().splitlines()]
    research = [r for r in history if r.get('origin') == 'qwen']
    assert client.requests == 1 and seen
    assert len(research) == 1 and research[0]['status'] == 'ok'
    assert len(research[0]['recipe']['feature_expressions']) == 2


@pytest.mark.parametrize('family', ['linear', 'lightgbm'])
@pytest.mark.parametrize('scenario', ['iid', 'integer_labels'])
def test_named_numeric_and_categorical_features_support_full_refit(tmp_path, family, scenario):
    from atlas.models import fit_predict
    data = load(read_inputs(create(tmp_path, scenario, n=240)))
    recipe = validate({'model_family': family, 'params': {} if family == 'linear' else {'iterations': 25},
                       'feature_expressions': [formula('interaction', 'x0 * x1'),
                                               formula('composed', 'concat(category, where(x0 > 0, "positive", "negative"))')]}, data.spec)
    valid, test, info = fit_predict(data, recipe, np.arange(len(data.y)), np.empty(0, dtype=int),
                                   threads=1, refit=True, iterations_override=25)
    assert len(valid) == 0 and len(test) == len(data.test) and np.isfinite(test).all()
