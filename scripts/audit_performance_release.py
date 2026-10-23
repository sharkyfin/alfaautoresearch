"""Read-only release audit; never retrain, alter the frozen panel, or upload."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.utils import code_digest


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    directory = ROOT / 'reports/performance_v2'
    freeze_path = ROOT / 'benchmarks/performance_v2/final_freeze.json'
    freeze = read(freeze_path)
    final = read(directory / 'final-lockbox-live.json')
    tests = read(ROOT / 'reports/test-state.json')
    archive = ROOT / 'submission.zip'
    source = freeze['sources']['candidate']
    require(code_digest() == source, 'Working runtime differs from frozen candidate')
    def numerical_functions(path):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef)}
    archived_reporting = numerical_functions(ROOT / '.bootstrap/performance_v2/p3j/source/scripts/report_performance_v2.py')
    current_reporting = numerical_functions(ROOT / 'scripts/report_performance_v2.py')
    for name in ['effect', 'paired_effects', 'stats', 'resource_stats']:
        require(current_reporting[name] == archived_reporting[name], 'Numerical reporting function changed after freeze: ' + name)
    require(sha(archive) == freeze['archives']['candidate'], 'Root ZIP differs from frozen candidate')
    for path, expected in {**freeze['evaluator_fingerprints'], **freeze['gates']}.items():
        require(sha(ROOT / path) == expected, 'Frozen proof or scorer changed: ' + path)
    require(read(freeze_path.with_name('final_freeze.opened.json'))['freeze_sha256'] == sha(freeze_path),
            'Opening record differs from original freeze')
    require(sha(ROOT / 'benchmarks/manifest_v2.json') == freeze['manifest_sha256'],
            'Original frozen manifest changed')
    require(final['status'] == 'completed', 'Final panel is incomplete')
    protocol = freeze['protocol']
    planned = {(case, seed, arm, mode) for case in freeze['cases'] for seed in protocol['seeds']
               for arm in freeze['archives'] for mode in protocol['modes']}
    observed = [(row['case'], row['seed'], row['arm'], row['mode']) for row in final['runs']]
    require(len(observed) == len(set(observed)) and set(observed) == planned,
            'Missing, extra or duplicated final observations')
    verified_outputs = 0
    for row in final['runs']:
        require(row['source_sha'] == freeze['sources'][row['arm']] and
                row['zip_sha'] == freeze['archives'][row['arm']], 'Final row source differs')
        require(row['budget_seconds'] == protocol['budget_seconds'] and
                row['threads'] == protocol['threads'] and row['memory_mb'] == protocol['memory_mb'],
                'Final row resource protocol differs')
        if row.get('output_sha256'):
            path = Path(row['run_dir']).parent / 'task/submission.csv'
            require(sha(path) == row['output_sha256'], 'Retained prediction changed: ' + str(path))
            verified_outputs += 1
    preserved = 0
    for audit in final.get('resumptions', []):
        snapshot = Path(audit['preserved_snapshot'])
        require(sha(snapshot) == audit['preserved_snapshot_sha256'], 'Interrupted snapshot changed')
        previous = read(snapshot)['runs']
        require(final['runs'][:len(previous)] == previous, 'Previously completed row changed')
        require(sha(ROOT / 'scripts/resume_frozen_final.py') == audit['resume_driver_sha256'],
                'Resume driver changed after execution')
        preserved = max(preserved, len(previous))
    require(tests['status'] == 'passed' and tests['source_sha256_before'] == source and
            tests['source_sha256_after'] == source and tests['passed'] >= 304,
            'Complete current test gate missing')
    queue = read(ROOT / 'benchmarks/performance_v2/queue13.state.json')
    require(queue.get('status') == 'completed' and len(queue['jobs']) == 3 and
            all(job.get('returncode') == 0 for job in queue['jobs']),
            'Final queue did not complete')
    additional = None
    additional_freeze = ROOT / 'benchmarks/performance_v2/openml_followup_freeze.json'
    if additional_freeze.exists():
        specification = read(additional_freeze)
        require(sha(ROOT / 'benchmarks/performance_v2/openml_followup_manifest.json') == specification['manifest_sha256'],
                'Additional OpenML manifest changed after its freeze')
        require(specification['sources'] == freeze['sources'] and specification['archives'] == freeze['archives'],
                'Additional OpenML uses a different candidate')
        additional = read(directory / 'openml-restored-120.json')
        require(additional['status'] == 'completed' and
                additional['evaluator_fingerprints'] == freeze['evaluator_fingerprints'],
                'Additional OpenML is incomplete or uses a different evaluator')
        keys = [(r['case'], r['seed'], r['arm'], r['mode']) for r in additional['runs']]
        expected = {(case, seed, arm, 'deterministic') for case in specification['cases']
                    for seed in [42, 53] for arm in freeze['archives']}
        require(len(keys) == len(set(keys)) == 24 and set(keys) == expected,
                'Missing, extra or duplicated additional OpenML observations')
        for row in additional['runs']:
            require(row['source_sha'] == freeze['sources'][row['arm']] and
                    row['zip_sha'] == freeze['archives'][row['arm']] and row['budget_seconds'] == 120,
                    'Additional OpenML source/resource mismatch')
            if row.get('output_sha256'):
                require(sha(Path(row['run_dir']).parent / 'task/submission.csv') == row['output_sha256'],
                        'Additional OpenML retained prediction changed')
                verified_outputs += 1
        additional_queue = read(ROOT / 'benchmarks/performance_v2/queue14.state.json')
        require(additional_queue.get('status') == 'completed' and len(additional_queue['jobs']) == 2 and
                all(job.get('returncode') == 0 for job in additional_queue['jobs']), 'Additional OpenML/operational QA queue failed')
        text_qa = read(directory / 'large_text_runtime.json')
        text_plan_path = ROOT / 'benchmarks/performance_v2/large_text_runtime_plan.json'
        text_plan = read(text_plan_path)
        require(sha(text_plan_path) == text_qa['plan_sha256'] and
                sha(ROOT / 'scripts/verify_large_text_runtime.py') == text_plan['driver_sha256'],
                'Mixed-text operational plan or driver changed after execution')
        require(text_qa['status'] == 'passed' and text_qa['source_sha256'] == source and
                text_qa['archive_sha256'] == freeze['archives']['candidate'] and text_qa['full_refit'] and
                text_qa['qwen_calls'] == 0 and text_qa['external_metric'] is None and
                text_qa['external_evaluator_called'] is False,
                'Longer mixed-text operational QA failed or improperly scored the outer answers')
        require(sha(Path(text_qa['run_root']) / 'trial/task/submission.csv') == text_qa['output_sha256'],
                'Operational QA retained prediction changed')
        verified_outputs += 1
    manifest = read(ROOT / 'reports/release-manifest.json')
    require(manifest['sha256'] == sha(archive) and manifest['runtime_code_sha256'] == source,
            'Release manifest differs')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) == 27, 'Unexpected ZIP contents')
        require(all(Path(name).name == name and (name.endswith('.py') or name == 'requirements.txt')
                    for name in names), 'Non-root code or forbidden payload in ZIP')
        require(z.read('requirements.txt') == (ROOT / 'artifacts/official/requirements-folder.txt').read_bytes(),
                'Official requirements changed')
        for item in manifest['files']:
            require(hashlib.sha256(z.read(item['path'])).hexdigest() == item['sha256'],
                    'ZIP member differs from manifest')
        zip_contents = [z.read(name) for name in names]

    # Match actual local credential values without ever logging their contents.
    secrets = []
    secret_files = [ROOT / '.env.organizer.local', ROOT / '.env.openrouter.local']
    for path in secret_files:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            key, separator, value = line.partition('=')
            value = value.strip().strip('\"\'')
            if separator and ('API_KEY' in key or 'TOKEN' in key) and len(value) >= 12:
                secrets.append(value.encode())
        require(path.stat().st_mode & 0o077 == 0, 'Credential permissions are too broad')
        require(subprocess.run(['git', 'check-ignore', '-q', str(path)], cwd=ROOT).returncode == 0,
                'Credential file is not ignored')
    scanned = 0
    text_paths = set()
    for base in ['atlas', 'scripts', 'tests', 'docs', 'artifacts/official', 'reports/performance_v2', 'benchmarks']:
        for path in (ROOT / base).rglob('*'):
            text_paths.add(path)
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    text_paths.update(ROOT / name for name in tracked if name)
    for path in text_paths:
        if not path.is_file() or path.suffix not in {'.py', '.md', '.json', '.jsonl', '.csv', '.log', '.txt', '.toml', '.yaml', '.yml'}:
            continue
        content = path.read_bytes()
        require(not any(secret in content for secret in secrets), 'Credential found in ' + str(path.relative_to(ROOT)))
        scanned += 1
    require(not any(secret in content for secret in secrets for content in zip_contents), 'Credential found in ZIP')

    checked_links = 0
    for path in [ROOT / 'README.md', ROOT / 'CONTRACT.md', directory / 'FINAL.md',
                 directory / 'final_lockbox.md', directory / 'before_after.md', directory / 'P4.md',
                 directory / 'experiment_coverage.md', directory / 'final_error_review.md',
                 directory / 'openml_external_validation.md']:
        for raw in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = raw.strip().split('#', 1)[0]
            if not target or re.match(r'[a-z]+://', target):
                continue
            destination = path.parent / unquote(target.strip('<>'))
            require(destination.exists(), 'Broken local link in ' + str(path.relative_to(ROOT)) + ': ' + target)
            checked_links += 1
    result = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Artifact integrity and execution gates; this does not certify a quality improvement.',
              'universal_quality_improvement_confirmed': False,
              'numerical_reporting_matches_preopening_reference': True,
              'runtime_sha256': source, 'zip_sha256': sha(archive), 'zip_bytes': archive.stat().st_size,
              'root_files': len(names), 'final_rows': len(observed),
              'final_status_counts': dict(Counter(row['status'] for row in final['runs'])),
              'additional_openml_rows': len(additional['runs']) if additional else 0,
              'additional_openml_status_counts': dict(Counter(row['status'] for row in additional['runs'])) if additional else {},
              'retained_prediction_hashes_checked': verified_outputs, 'preserved_completed_rows': preserved,
              'tests_passed': tests['passed'], 'secret_values_checked_without_logging': len(secrets),
              'text_files_checked': scanned, 'local_links_checked': checked_links,
              'platform_uploads': 0}
    (directory / 'release_integrity.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
