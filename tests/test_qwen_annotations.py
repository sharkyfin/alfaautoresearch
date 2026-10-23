import copy
from types import SimpleNamespace

import pytest

from atlas.agent import normalize_annotations
from atlas.recipes import RecipeError,recipe_hash,validate


def fixture():
    spec=SimpleNamespace(available_capabilities={'scikit-learn'},validation_plan={'id':'fixed'},
                         features=['x'],target='y',ids=['id'],classification=False,profile={})
    raw={'model_family':'linear','params':{'alpha':1},'feature_ops':[],'target_transform':'none',
         'feature_expressions':[{'name':'square_x','expression':'x * x','reason':'r'*600}],
         'hypothesis':{'kind':'feature','expected_effect':'e'*500,'uncertainty':.7,'stop_condition':None}}
    return spec,raw


def test_oversized_inert_notes_do_not_discard_an_executable_proposal():
    spec,raw=fixture();original=copy.deepcopy(raw)
    with pytest.raises(RecipeError):validate(raw,spec)
    clean,fields=normalize_annotations(raw)
    result=validate(clean,spec)
    assert raw==original
    assert len(result['hypothesis']['expected_effect'])==240
    assert result['hypothesis']['uncertainty']=='0.7'
    assert 'stop_condition' not in result['hypothesis']
    assert result['feature_expressions'][0]['expression']=='x * x'
    assert len(result['feature_expressions'][0]['reason'])==240
    assert fields
    assert recipe_hash(result)==recipe_hash(original)


@pytest.mark.parametrize('change',[{'model_family':'shell'},
    {'feature_expressions':[{'name':'bad','expression':'__import__("os").system("pwd")','reason':'r'*600}]},
    {'hypothesis':{'unexpected':'not allowed'}}])
def test_note_normalization_cannot_bypass_executable_or_schema_guards(change):
    spec,raw=fixture();raw.update(change)
    clean,_=normalize_annotations(raw)
    with pytest.raises(RecipeError):validate(clean,spec)
