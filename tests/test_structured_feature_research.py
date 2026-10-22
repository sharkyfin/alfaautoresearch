import json

import numpy as np
import pandas as pd
import pytest

from atlas.agent import SYSTEM, QwenClient, compact_context, researcher_context
from atlas.context import string_structure
from atlas.contract import read_inputs
from atlas.expressions import ExpressionError, ExpressionFeatures, validate_expressions
from atlas.models import fit_predict
from atlas.recipes import validate
from atlas.task import load
from atlas.validation import design
from scripts.synthetic import create


def formula(name, expression):
    return {'name': name, 'expression': expression}


def test_structured_strings_preserve_index_missing_values_and_numeric_fields():
    frame = pd.DataFrame({'code': ['north/12/left', 'south/3/right', None, 'single',
                                  'north/bad/left']}, index=[9, 3, 17, 8, 1])
    formulas = [formula('region', 'split_part(code, "/", 0)'),
                formula('position', 'to_number(split_part(code, "/", 1))'),
                formula('last', 'split_part(code, "/", -1)'),
                formula('absent', 'to_number(split_part(code, "/", 4))')]
    engine = ExpressionFeatures(formulas, frame.columns).fit(frame)
    result = engine.transform(frame)
    assert result['region'].tolist() == ['north', 'south', '__MISSING__', 'single', 'north']
    np.testing.assert_allclose(result['position'], [12, 3, np.nan, np.nan, np.nan], equal_nan=True)
    assert result['last'].tolist() == ['left', 'right', '__MISSING__', 'single', 'left']
    assert result['absent'].isna().all()
    assert result['position'].index.equals(frame.index)
    assert not len(engine.transform(frame.iloc[:0])['position'])


@pytest.mark.parametrize('expression', [
    'split_part(code, "", 0)', 'split_part(code, code, 0)',
    'split_part(code, "/", 17)', 'split_part(code, "/", -17)',
    'split_part(code, "/", True)', 'split_part(code, "/", 1.5)',
    'split_part(code, "/", code)', 'split_part(code, "/", 1 + 1)',
    'df["target"]', 'df["id"]', 'df[code]', 'df[["code"]]',
    'df["code"][0]', 'df["code"].str.split("/")', 'unknown["code"]',
])
def test_structured_formulas_reject_unsafe_or_unbounded_access(expression):
    with pytest.raises(ExpressionError):
        validate_expressions([formula('parsed', expression)], ['code'], ['target', 'id'])


def test_dataframe_literal_column_spelling_is_normalized_without_python_execution():
    frame = pd.DataFrame({'first value': [2., 3.], 'other': [4., 5.]})
    expressions = [formula('combined', 'df["first value"] * X["other"]'),
                   formula('offset', 'data["other"] + x["first value"]')]
    engine = ExpressionFeatures(expressions, frame.columns).fit(frame)
    assert all('[' not in item['expression'] for item in engine.expressions)
    np.testing.assert_allclose(engine.transform(frame)['combined'], [8, 15])
    np.testing.assert_allclose(engine.transform(frame)['offset'], [6, 8])


def test_separator_padding_repair_is_learned_on_training_and_reported():
    train = pd.DataFrame({'code': ['north/12/left', 'south/3/right', None]})
    expressions = [formula('position', 'to_number(split_part(code, " / ", 1))')]
    engine = ExpressionFeatures(expressions, train.columns).fit(train)
    np.testing.assert_allclose(engine.transform(train)['position'], [12, 3, np.nan], equal_nan=True)
    state = repr(engine.learned)
    result = engine.transform(pd.DataFrame({'code': ['east/40/left', 'west / 51 / right']}))
    np.testing.assert_allclose(result['position'], [40, 51])
    assert repr(engine.learned) == state
    assert engine.repairs[0]['original_separator'] == ' / '
    assert engine.repairs[0]['resolved_separator'] == '/'


def test_existing_exact_separator_takes_precedence_over_shorter_token():
    train = pd.DataFrame({'code': ['north-west - 12', 'south-east - 3']})
    expressions = [formula('region', 'split_part(code, " - ", 0)')]
    engine = ExpressionFeatures(expressions, train.columns).fit(train)
    assert engine.transform(train)['region'].tolist() == ['north-west', 'south-east']
    assert engine.repairs == []


def test_validation_cannot_choose_a_different_separator():
    train = pd.DataFrame({'code': ['north', 'south']})
    expressions = [formula('position', 'to_number(split_part(code, " / ", 1))')]
    engine = ExpressionFeatures(expressions, train.columns).fit(train)
    state = repr(engine.learned)
    assert engine.transform(pd.DataFrame({'code': ['north/12/left']}))['position'].isna().all()
    assert repr(engine.learned) == state and engine.repairs == []


def test_string_structure_is_aggregate_metadata_without_raw_values(tmp_path):
    task = create(tmp_path, n=180)
    for filename in ['train.csv', 'test.csv']:
        path = task.parent / filename
        frame = pd.read_csv(path, dtype={'id': 'string'})
        frame['category'] = [f'private{i}/{i}/secret' for i in range(len(frame))]
        frame.to_csv(path, index=False)
    data = load(read_inputs(task))
    context = compact_context(researcher_context(data, [], None, 100))
    profile = context['untrusted_data']['profile']['columns']['category']
    assert profile['string_structure'] == [
        {'separator': '/', 'coverage': 1., 'median_parts': 3, 'numeric_parts': [1]}]
    assert 'private0/0/secret' not in json.dumps(context)
    assert 'split_part' in context['capabilities']['feature_formulas']['functions']
    assert string_structure(pd.Series([None])) == []


def test_next_qwen_context_shrinks_to_fit_remaining_token_cap(monkeypatch):
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://organizer.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only')
    bodies = []

    def transport(body, timeout):
        bodies.append(body)
        return {'choices': [{'message': {'content': '{"model_family":"hist"}'}}],
                'usage': {'total_tokens': 500}}

    client = QwenClient('live', transport=transport)
    client.tokens = 10_000
    context = {'objective': {'metric': 'accuracy'}, 'capabilities': {'models': {'hist': {}}},
               'untrusted_data': {'description': 'x' * 2000,
                                  'dictionary': [{'column': f'field{i}', 'description': 'd' * 100}
                                                 for i in range(60)]}}
    original_bytes = len(json.dumps(compact_context(context), ensure_ascii=False).encode())
    assert client.propose(context) == {'model_family': 'hist'}
    sent = bodies[0]['messages'][1]['content']
    assert len(sent.encode()) < original_bytes
    assert (len((SYSTEM + sent).encode()) + 1) // 2 + client.output_tokens <= 6000
    assert client.tokens <= client.max_tokens == 16_000
    assert json.loads(sent)['objective']['metric'] == 'accuracy'


def test_parsed_numeric_field_improves_real_held_out_predictions(tmp_path):
    task = create(tmp_path / 'data', n=400)
    for filename in ['train.csv', 'test.csv']:
        path = task.parent / filename
        frame = pd.read_csv(path, dtype={'id': 'string'})
        position = np.arange(len(frame)) + (1000 if filename == 'test.csv' else 0)
        frame['category'] = [f'opaque/{value}/tail' for value in position]
        if filename == 'train.csv':
            frame['response'] = position.astype(float)
        frame.to_csv(path, index=False)
    data = load(read_inputs(task))
    split = design(data)
    baseline = {'model_family': 'linear', 'params': {'alpha': .01}}
    parsed = {**baseline, 'feature_expressions': [
        formula('position', 'to_number(split_part(df["category"], "/", 1))')]}
    errors = []
    for raw in [baseline, parsed]:
        recipe = validate(raw, data.spec)
        valid, test, _ = fit_predict(data, recipe, split.train, split.valid, threads=1)
        assert np.isfinite(test).all()
        errors.append(np.sqrt(np.mean((data.y[split.valid] - valid)**2)))
    assert errors[1] < errors[0] * .1
