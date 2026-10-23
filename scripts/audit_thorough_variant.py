"""Tie the optional longer-search upload to its actual local QA evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.utils import code_digest
from scripts.build_thorough_variant import PARENT_SHA, extend_search, replace_once
from scripts.verify_zip import archive_source_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    directory = ROOT/'reports/thorough-v1'
    manifest = json.loads((directory/'manifest.json').read_text())
    parent, variant = ROOT/'submission_p3j_backup.zip', ROOT/'submission_thorough.zip'
    if not parent.exists():
        parent = ROOT/'.bootstrap/performance_v2/p3j/source/submission.zip'
    assert sha(parent) == PARENT_SHA
    assert sha(variant) == manifest['sha256']
    assert archive_source_digest(variant) == manifest['runtime_code_sha256']
    assert sha(ROOT/'submission.zip') in {PARENT_SHA, manifest['sha256']}
    assert code_digest() == archive_source_digest(ROOT/'submission.zip')
    assert sha(ROOT/'scripts/build_thorough_variant.py') == manifest['builder_sha256']
    secrets = []
    for filename in ['.env.organizer.local', '.env.openrouter.local']:
        path = ROOT/filename
        if path.exists():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().partition('=')
                value = value.strip().strip('"\'')
                if sep and ('API_KEY' in key or 'TOKEN' in key) and len(value) >= 12:
                    secrets.append(value.encode())
    with zipfile.ZipFile(parent) as old, zipfile.ZipFile(variant) as new:
        assert old.namelist() == new.namelist()
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        assert set(changed) == {'solution.py', '_atlas_entry.py', 'atlas_search.py'}
        assert new.read('atlas_search.py').decode() == extend_search(old.read('atlas_search.py').decode())
        assert new.read('_atlas_entry.py').decode() == replace_once(old.read('_atlas_entry.py').decode(),
            "'task.md', started=STARTED)", "'task.md', started=STARTED, max_experiments=480)")
        assert new.read('requirements.txt') == (ROOT/'artifacts/official/requirements-folder.txt').read_bytes()
        for name in new.namelist():
            payload = new.read(name)
            assert not any(secret in payload for secret in secrets), 'Credential in archive'
        count = len(new.namelist())
    test_log = (directory/'variant-tests.log').read_text()
    test = re.search(r'(\d+) passed, (\d+) warnings in ([\d.]+)s', test_log)
    assert test and int(test[1]) == 304
    assert '9 passed' in (directory/'policy-tests.log').read_text()
    cancellation = json.loads((directory/'exact-cli-cancellation.json').read_text())
    assert cancellation['runtime_code_sha256'] == manifest['runtime_code_sha256']
    assert len(cancellation['tests']) == 5 and all(row['status'] == 'passed' for row in cancellation['tests'])
    plan_path = directory/'synthetic-plan.json'
    plan = json.loads(plan_path.read_text())
    observed = json.loads((directory/'synthetic-comparison.json').read_text())
    assert sha(ROOT/'scripts/compare_thorough_variant.py') == plan['driver_sha256']
    assert observed['plan_sha256'] == sha(plan_path) and observed['status'] == 'completed'
    assert plan['archives'] == {'standard': sha(parent), 'thorough': sha(variant)}
    rows = observed['runs']
    assert len(rows) == 4 and len({(row['task'], row['arm']) for row in rows}) == 4
    for row in rows:
        assert row['archive_sha256'] == plan['archives'][row['arm']]
        expected = archive_source_digest(parent if row['arm'] == 'standard' else variant)
        assert row['runtime_source_sha256'] == expected
        assert row['budget_seconds'] == 10500 and row['qwen_calls'] == 0
        assert row['operational']['status'] == 'ok' and row['operational']['returncode'] == 0
        assert row['valid_output'] and row['full_refit'] and not row['cancelled']
        assert sha(Path(row['run_dir'])/'submission.csv') == row['output_sha256']
        summary = list((Path(row['run_dir'])/'.atlas').glob('*/summary.json'))
        assert len(summary) == 1
        for path in summary[0].parent.glob('experiments/exp_*/job.json'):
            job = json.loads(path.read_text())
            assert job['threads'] <= 8 and job['memory_mb'] <= 24576
        for name, expected in row['inputs'].items():
            assert sha(Path(row['run_dir'])/name) == expected
    for task in {row['task'] for row in rows}:
        pair = [row for row in rows if row['task'] == task]
        assert pair[0]['inputs'] == pair[1]['inputs']
    result = {'status': 'LOCAL_OPERATIONAL_QA_PASSED',
        'audited_at_utc': datetime.now(timezone.utc).isoformat(),
        'standard_archive_sha256': sha(parent), 'thorough_archive_sha256': sha(variant),
        'thorough_source_sha256': manifest['runtime_code_sha256'], 'root_files': count,
        'variant_runtime_tests_passed': int(test[1]), 'variant_runtime_test_seconds': float(test[3]),
        'policy_and_packaging_tests_passed': 9, 'exact_cli_cancellation_checks_passed': 5,
        'natural_exact_cli_full_refit_runs': len(rows),
        'actual_secret_values_checked_without_logging': len(secrets),
        'generalization_advantage_established': False,
        'promoted_over_standard_release': sha(ROOT/'submission.zip') == sha(variant),
        'platform_uploaded': False,
        'limitations': ['Synthetic 2-task/1-seed comparison only; no independent real-data generalization panel',
            'Longer-search archive not separately verified in Linux/full organizer image',
            'No continuous three-hour soak or live organizer API test for this variant',
            'No guarantee of improving unknown private tasks']}
    if result['promoted_over_standard_release']:
        current = json.loads((ROOT/'reports/test-state.json').read_text())
        assert current['status'] == 'passed'
        assert current['source_sha256_before'] == current['source_sha256_after'] == manifest['runtime_code_sha256']
        assert current['passed'] >= 313
        result['current_worktree_test_gate'] = current
    (directory/'release-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
