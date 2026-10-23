"""Long-search policy must preserve short budgets and the independent release."""
import ast
import types
import hashlib
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from atlas.budget import Budget
from atlas.search import SearchController
from scripts.build_thorough_variant import PARENT_SHA, extend_search, replace_once


def parent_archive():
    root = Path(__file__).resolve().parents[1]
    for path in [root/'submission_p3j_backup.zip',
                 root/'.bootstrap/performance_v2/p3j/source/submission.zip', root/'submission.zip']:
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == PARENT_SHA:
            return path
    pytest.skip('Frozen parent is required for this optional archive comparison')


def controller(total, extended):
    spec = SimpleNamespace(available_capabilities={'optuna'}, text_candidates=[], classification=False)
    with zipfile.ZipFile(parent_archive()) as archive:
        source = archive.read('atlas_search.py').decode()
    module = types.ModuleType('atlas._thorough_policy_test')
    module.__package__ = 'atlas'
    exec(compile(extend_search(source) if extended else source,
                 'frozen-parent-search.py', 'exec'), module.__dict__)
    cls = module.SearchController
    value = cls(spec, Budget(total))
    value.hpo.completed = 12
    value.family_best = {family: {} for family in ['linear', 'hist', 'extra_trees']}
    return value


@pytest.mark.parametrize('seconds', [30, 120, 900, 1799])
def test_short_search_retains_the_same_stopping_decisions(seconds):
    standard, thorough = controller(seconds, False), controller(seconds, True)
    assert standard.research_limit == thorough.research_limit
    assert [standard.should_stop(i) for i in range(100)] == [thorough.should_stop(i) for i in range(100)]


def test_long_search_can_try_after_the_old_plateau_but_still_stops():
    standard, thorough = controller(10500, False), controller(10500, True)
    assert standard.should_stop(35)
    assert not thorough.should_stop(35)
    assert not thorough.should_stop(70)
    assert thorough.should_stop(71)
    assert thorough.research_limit > standard.research_limit
    # A measured gain resets no_gain in the unchanged orchestrator.
    assert not thorough.should_stop(0)
    assert thorough.budget.reserve == standard.budget.reserve
    assert not thorough.budget.can_run(10500)


def test_missing_optuna_also_has_finite_patience():
    thorough = controller(10500, True)
    thorough.hpo.enabled = False
    assert not thorough.should_stop(10)
    assert thorough.should_stop(20)


def test_patch_refuses_a_different_source():
    with pytest.raises(ValueError):
        extend_search('unexpected code')
    with pytest.raises(ValueError):
        replace_once('twice twice', 'twice', 'patched')


def test_thorough_entry_preserves_the_platform_deadline():
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'solution.py').read_text())
    call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'run')
    assert 'budget_seconds' not in {keyword.arg for keyword in call.keywords}
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(keywords['max_experiments']) == 480


def test_optional_archive_preserves_the_verified_parent_and_only_changes_search():
    root = Path(__file__).resolve().parents[1]
    extended = root / 'submission_thorough.zip'
    if not extended.exists():
        pytest.skip('Optional research archive has not been built')
    # Thorough is a preserved historical archive. A later runtime repair can
    # become primary while this original optional artifact stays immutable.
    from atlas.utils import code_digest
    from scripts.verify_zip import archive_source_digest
    assert code_digest() == archive_source_digest(root/'submission.zip')
    with zipfile.ZipFile(parent_archive()) as original, zipfile.ZipFile(extended) as thorough:
        assert original.namelist() == thorough.namelist()
        changed = {name for name in original.namelist() if original.read(name) != thorough.read(name)}
        assert changed == {'solution.py', '_atlas_entry.py', 'atlas_search.py'}
        assert original.read('requirements.txt') == thorough.read('requirements.txt')
        tree = ast.parse(thorough.read('_atlas_entry.py'))
        call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == 'run')
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        assert ast.literal_eval(keywords['max_experiments']) == 480
        assert 'budget_seconds' not in keywords
