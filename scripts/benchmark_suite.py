"""Frozen-ZIP paired benchmarks on curated sources; evaluator labels stay private.

Example: python scripts/benchmark_suite.py --archives baseline=old.zip,candidate=new.zip
  --cases concrete_strength,beyond_airfoil_self_noise --budget 120 --seeds 42
Final evaluation requires a predeclared code/dataset freeze and records its opening.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EVALUATOR_FILES = ['atlas/metrics.py', 'scripts/benchmark_suite.py',
    'scripts/benchmark_data.py', 'scripts/benchmark_kaggle.py', 'scripts/kaggle_data.py',
    'scripts/source_adapters.py']


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def final_protocol(args, modes):
    return {'seeds': [int(value) for value in args.seeds.split(',')], 'modes': modes,
        'budget_seconds': args.budget, 'threads': args.threads, 'memory_mb': args.memory_mb,
        'max_qwen_calls': args.max_calls, 'exclude_families': args.exclude_families,
        'exclude_by_arm': args.exclude_by_arm, 'recipe_by_arm': args.recipe_by_arm,
        'blend_by_arm': args.blend_by_arm, 'finalization_by_arm': args.finalization_by_arm}


def validate_final_freeze(freeze, cases, archives, protocol, manifest_sha, evaluator_hashes):
    """Reject protocol or scorer drift before any reserved data are read."""
    checks = {'cases': sorted(cases), 'archives': archives, 'protocol': protocol,
        'manifest_sha256': manifest_sha, 'evaluator_fingerprints': evaluator_hashes}
    for key, expected in checks.items():
        actual = sorted(freeze.get(key, [])) if key == 'cases' else freeze.get(key)
        if actual != expected:
            raise ValueError('Final freeze differs: ' + key)


def prepare(record, destination, seed, stage):
    from scripts.benchmark_data import prepare_external
    if record['source'] != 'kaggle':
        return prepare_external(record, destination, seed, stage)
    import pandas as pd
    from scripts.kaggle_data import prepare as prepare_kaggle
    if stage != 'development':
        raise ValueError('Historical Kaggle lockboxes have been inspected; development-only')
    specification = json.loads((ROOT / 'benchmarks/kaggle_cases.json').read_text())[record['dataset_key']]
    metadata = prepare_kaggle(record['dataset_key'], specification,
        ROOT / 'artifacts/data/kaggle/raw' / (record['dataset_key'] + '.zip'), destination, seed)
    labels = destination / 'sealed/evaluation.csv'
    truth = pd.read_csv(labels, dtype={metadata['id_column']: str})
    labels.unlink()
    training=pd.read_csv(destination/'inputs/train.csv')
    features=training.drop(columns=[metadata['id_column'],metadata['target']])
    support=training[metadata['target']].value_counts() if metadata['labels'] else None
    metadata.update(source='kaggle', source_url=metadata['source'], license=record['license'],
        n_classes=len(metadata['labels']), evaluation_unit='row',
        missingness=float(features.isna().to_numpy().mean()),
        imbalance_ratio=float(support.max()/support.min()) if support is not None else None,
        train_class_support={str(key):int(value) for key,value in support.items()} if support is not None else None,
        labels_storage='parent evaluator memory; former sealed label file unlinked')
    return metadata, truth


def sandbox_command(trial, runtime,allow_network=False):
    """Deny raw caches, credentials, other runs and external score reports."""
    binary = shutil.which('sandbox-exec')
    if not binary:
        return [], 'private evaluator memory; OS filesystem isolation unavailable'
    profile = trial / 'agent.sb'
    q = lambda value: json.dumps(str(Path(value).resolve()))
    forbidden = [ROOT / 'artifacts/data', ROOT / 'reports', ROOT / '.bootstrap/colleague-review',
        ROOT / '.bootstrap/performance_v2/baseline/source/runs', ROOT / '.bootstrap/performance_v2/p0/source/runs',
        ROOT / '.bootstrap/performance_v2/research', ROOT / '.env.organizer.local', ROOT / '.env.openrouter.local']
    rules = ['(version 1)', '(allow default)']
    if not allow_network:rules.append('(deny network*)')
    for path in forbidden:
        rules.append('(deny file-read* (subpath ' + q(path) + '))')
    rules.append('(deny file-read* (require-all (subpath ' + q(ROOT / 'runs') + ') ' + ' '.join(
        '(require-not (subpath ' + q(path) + '))' for path in [trial / 'task', trial / 'run', runtime]) + '))')
    profile.write_text('\n'.join(rules) + '\n')
    return [binary, '-f', str(profile)], 'inherited macOS sandbox denies raw caches, secrets, other runs and external scores'


def summarize(rows):
    pairs = []
    for case, seed, mode in sorted({(row['case'], row['seed'], row['mode']) for row in rows}):
        arms = {row['arm']: row for row in rows if (row['case'], row['seed'], row['mode']) == (case, seed, mode)}
        reference = arms.get('baseline')
        if reference is None:
            continue
        for arm, candidate in arms.items():
            if arm == 'baseline':
                continue
            pair = {'case': case, 'seed': seed, 'mode': mode, 'arm': arm,
                    'reference_status': reference['status'], 'candidate_status': candidate['status']}
            if reference['status'] == candidate['status'] == 'trained_success':
                assert reference['input_sha256'] == candidate['input_sha256']
                assert reference['budget_seconds'] == candidate['budget_seconds']
                from atlas.metrics import Metric
                metric = Metric(candidate['metric'], candidate.get('n_classes', 0))
                before, after = reference['external_metric'], candidate['external_metric']
                gain = metric.direction * (after - before)
                relative = gain / max(abs(before), 1e-12)
                pair.update(metric=metric.name, reference_score=before, candidate_score=after,
                    oriented_gain=gain, relative_gain=relative,
                    outcome='win' if relative > .001 else 'loss' if relative < -.001 else 'tie',
                    reference_seconds=reference['wall_seconds'], candidate_seconds=candidate['wall_seconds'])
            else:
                pair['outcome'] = 'failure_or_unavailable'
            pairs.append(pair)
    values = [row['relative_gain'] for row in pairs if 'relative_gain' in row]
    import numpy as np
    return {'pairs': pairs, 'win_tie_loss': {name: sum(row['outcome'] == name for row in pairs) for name in
        ['win', 'tie', 'loss', 'failure_or_unavailable']},
        'median_relative_gain': float(np.median(values)) if values else None,
        'worst_quartile_relative_gain': float(np.quantile(values, .25)) if values else None,
        'trained_success': sum(row['status'] == 'trained_success' for row in rows),
        'planned_rows': len(rows), 'normalization': 'direction*(new-old)/abs(old), per-task; no raw metric averaging'}


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report['summary'] = summarize(report['runs'])
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    fields = ['case', 'source', 'dataset', 'license', 'seed', 'official_repeat', 'official_fold',
        'arm', 'mode', 'source_sha', 'zip_sha', 'split_sha256', 'training_rows', 'features', 'n_classes',
        'missingness', 'imbalance_ratio', 'metric', 'external_metric', 'status', 'wall_seconds',
        'cpu_seconds', 'peak_memory_mb', 'qwen_calls', 'llm_valid_recipes', 'qwen_fits', 'qwen_accepted',
        'full_refit', 'fallback_kind', 'champion_family', 'output_sha256', 'error']
    with path.with_suffix('.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(report['runs'])
    lines = ['**Frozen-source paired benchmark**', '',
        '| Dataset | Seed | Arm | Mode | Metric | External | Seconds | Status |',
        '|---|---:|---|---|---|---:|---:|---|']
    for row in report['runs']:
        value = f'{row["external_metric"]:.6g}' if row.get('external_metric') is not None else '—'
        seconds = f'{row["wall_seconds"]:.1f}' if row.get('wall_seconds') is not None else '—'
        lines.append(f'| {row["case"]} | {row["seed"]} | {row["arm"]} | {row["mode"]} | {row.get("metric", "—")} | {value} | {seconds} | {row["status"]} |')
    lines += ['', 'Stage: ' + report['stage'] + '. Evaluation labels remain in the independent parent process.',
        'Original outer indices and source metrics are preserved; explicit metric-study overrides are recorded.',
        'Group-labelled tasks report row scores and equally weighted group scores.',
        'Unavailable/unrun cells stay empty with a reason. These offline metrics are not Boosters scores.',
        '', 'Summary: ' + json.dumps(report['summary']['win_tie_loss'])]
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'benchmarks/manifest_v2.json')
    parser.add_argument('--archives', required=True, help='label=zip,label=zip; baseline denotes the reference')
    parser.add_argument('--cases', default='development')
    parser.add_argument('--modes', default='deterministic')
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--budget', type=float, default=120)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-mb', type=int, default=8192)
    parser.add_argument('--max-calls', type=int, default=2)
    parser.add_argument('--exclude-families', default='')
    parser.add_argument('--exclude-by-arm', default='', help='label=family+family; controlled family ablations')
    parser.add_argument('--recipe-by-arm',default='',help='label=recipe.json; one fixed typed recipe per arm')
    parser.add_argument('--blend-by-arm',default='',help='label=none|weighted|greedy')
    parser.add_argument('--finalization-by-arm',default='',help='label=refit|fold_average')
    parser.add_argument('--autogluon-python',type=Path,default=ROOT/'.bootstrap/performance_v2/backends/autogluon/bin/python')
    parser.add_argument('--key-file', type=Path, default=ROOT / '.env.organizer.local')
    parser.add_argument('--stage', choices=['development', 'final_lockbox'], default='development')
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--resume',action='store_true',help='Continue only missing development trials from an interrupted report')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/performance_v2/benchmark_matrix.json')
    args = parser.parse_args()
    if args.resume and args.stage!='development':parser.error('Final lockbox resumption needs an explicit separate opening audit')
    if args.report.exists() and not args.resume:parser.error('Report already exists; preserve it or explicitly resume')
    if not 1 <= args.threads <= 8 or not 15 <= args.budget <= 10500:
        parser.error('Require 1–8 threads and 15–10500 seconds')
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = str(args.threads)
    manifest = json.loads(args.manifest.read_text())
    cases = {row['case']: row for row in manifest['cases']}
    selected = [name for name, row in cases.items() if row.get('reserved_phase') == args.stage] if args.cases in {'development', 'final_lockbox'} else args.cases.split(',')
    if any(name not in cases for name in selected):
        parser.error('Unknown dataset case')
    modes = args.modes.split(',')
    if any(mode not in {'deterministic', 'live-agent', 'fixed-allocation', 'lightgbm','autogluon','flaml'} for mode in modes):
        parser.error('Unknown model/search mode')
    archives = {}
    for item in args.archives.split(','):
        label, path = item.split('=', 1)
        if not re.fullmatch('[a-zA-Z0-9_-]+', label) or label in archives:
            parser.error('Archive labels must be unique safe names')
        archives[label] = Path(path).resolve()
    per_arm_exclusions={}
    for item in filter(None,args.exclude_by_arm.split(',')):
        label,excluded=item.split('=',1)
        if label not in archives:parser.error('Unknown intervention arm')
        per_arm_exclusions[label]=excluded.replace('+',',')
    default_exclusions=args.exclude_families
    policies={}
    for key,entries,allowed in [('ensemble_strategy',args.blend_by_arm,{'none','weighted','greedy'}),
                                ('finalization',args.finalization_by_arm,{'refit','fold_average'})]:
        for entry in filter(None,entries.split(',')):
            label,value=entry.split('=',1)
            if label not in archives or value not in allowed:parser.error('Invalid experiment policy')
            policies.setdefault(label,{})[key]=value
    per_arm_recipes={}
    for item in filter(None,args.recipe_by_arm.split(',')):
        label,path=item.split('=',1)
        if label not in archives:parser.error('Unknown fixed recipe arm')
        path=Path(path).resolve()
        value=json.loads(path.read_text())
        if not isinstance(value,dict):parser.error('Fixed recipes must be objects')
        per_arm_recipes[label]=value
    settings = {}
    if 'live-agent' in modes:
        from scripts.run_organizer import load_environment
        settings = load_environment(args.key_file)
    parent = ROOT / 'runs'
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='performance-v2-', dir=parent))
    from scripts.kaggle_data import snapshot_archive, INPUT_FILES
    from scripts.benchmark_kaggle import run_one
    import scripts.benchmark_data
    args.hard_deadline=True
    versions = {}
    for label, archive in archives.items():
        runtime = root / ('runtime-' + label)
        versions[label] = {'source_sha': snapshot_archive(archive, runtime), 'zip_sha': file_hash(archive), 'runtime': str(runtime)}
    evaluator_hashes = {name: file_hash(ROOT / name) for name in EVALUATOR_FILES}
    if args.stage == 'final_lockbox':
        if not args.freeze:
            parser.error('Final opening requires a source/dataset freeze manifest')
        freeze = json.loads(args.freeze.read_text())
        try:
            validate_final_freeze(freeze, selected, {label: row['zip_sha'] for label, row in versions.items()},
                final_protocol(args, modes), file_hash(args.manifest), evaluator_hashes)
        except ValueError as error:
            parser.error(str(error))
        opening = args.freeze.with_name(args.freeze.stem + '.opened.json')
        if opening.exists():
            parser.error('This lockbox has been opened; do not silently reuse it')
        opening.write_text(json.dumps({'opened_at_utc': datetime.now(timezone.utc).isoformat(),
            'run_root': str(root), 'freeze_sha256': file_hash(args.freeze),
            'protocol': final_protocol(args, modes)}, indent=2))
    report = {'status': 'running', 'stage': args.stage, 'run_root': str(root), 'versions': versions,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'budget_seconds': args.budget,
        'manifest_sha256': file_hash(args.manifest), 'runs': [], 'platform_submissions': 0}
    report['evaluator_fingerprints'] = evaluator_hashes
    if args.resume:
        prior=json.loads(args.report.read_text())
        if prior.get('status')!='running':parser.error('Only interrupted running campaigns may be resumed')
        if prior.get('manifest_sha256')!=report['manifest_sha256'] or prior.get('budget_seconds')!=args.budget:
            parser.error('Interrupted manifest or budget differs')
        if {label:row['zip_sha'] for label,row in prior['versions'].items()}!={label:row['zip_sha'] for label,row in versions.items()}:
            parser.error('Interrupted source archive identity differs')
        planned={(name,seed,label,mode) for name in selected for seed in map(int,args.seeds.split(',')) for label in versions for mode in modes}
        if any((row['case'],row['seed'],row['arm'],row['mode']) not in planned for row in prior['runs']):
            parser.error('Existing observed trials differ from the selected plan')
        snapshot=ROOT/'.bootstrap/performance_v2/interrupted'/f'{args.report.stem}-{int(datetime.now(timezone.utc).timestamp())}.json'
        snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.report,snapshot)
        prior.setdefault('resumptions',[]).append({'resumed_at_utc':datetime.now(timezone.utc).isoformat(),
            'new_run_root':str(root),'preserved_snapshot':str(snapshot),'evaluator_fingerprints':report['evaluator_fingerprints'],
            'policy':'Preserve completed observations; execute only missing trial keys; frozen source unchanged'})
        report=prior
    completed={(row['case'],row['seed'],row['arm'],row['mode']) for row in report['runs']}
    save_report(args.report, report)
    for name in selected:
        record = cases[name]
        for seed in map(int, args.seeds.split(',')):
            if all((name,seed,label,mode) in completed for label in versions for mode in modes):continue
            prepared = root / f'inputs-{name}-{seed}'
            try:
                if record.get('status') != 'metadata_available':
                    raise ValueError(record.get('status', 'missing_data') + ': ' + record.get('error', 'Source unavailable'))
                if str(record.get('license', '')).lower() in {'', 'none', 'unknown'}:
                    raise ValueError('dataset_not_licensed: source license declaration missing')
                metadata, truth = prepare(record, prepared, seed, args.stage)
                input_root = prepared / 'inputs' if record['source'] == 'kaggle' else prepared
                prep_error = None
                previous=[row for row in report['runs'] if row['case']==name and row['seed']==seed and row.get('input_sha256')]
                if any(row['input_sha256']!=metadata['input_sha256'] for row in previous):
                    raise RuntimeError('Resume input identity differs from completed observations')
            except Exception as error:
                prep_error = str(error)[:500]
                metadata = {**record, 'seed': seed, 'metric': record.get('metric')}
            for label, version in versions.items():
                for mode in modes:
                    if (name,seed,label,mode) in completed:continue
                    row = {**{key: metadata.get(key) for key in ['case', 'source', 'dataset', 'license', 'seed', 'metric', 'original_source_metric', 'metric_override', 'split_regime', 'split_sha256', 'input_sha256', 'training_rows', 'evaluation_rows', 'features', 'n_classes', 'missingness', 'imbalance_ratio', 'train_class_support', 'official_repeat', 'official_fold', 'evaluation_unit']},
                        'arm': label, 'mode': mode, 'source_sha': version['source_sha'], 'zip_sha': version['zip_sha'], 'budget_seconds': args.budget}
                    args.exclude_families=per_arm_exclusions.get(label,default_exclusions)
                    row['intervention']={'excluded_families':args.exclude_families,
                        'dev_worker_sha256':file_hash(Path(version['runtime'])/'scripts/kaggle_worker.py')}
                    if prep_error:
                        status = next((kind for kind in ['dataset_not_licensed', 'unsupported_metric', 'download_blocked', 'missing_data'] if prep_error.startswith(kind)), 'preparation_failed')
                        row.update(status=status, error=prep_error)
                    else:
                        trial = root / f'{name}-{seed}-{label}-{mode}'
                        trial.mkdir()
                        task_root = trial / 'task'
                        task_root.mkdir()
                        args.fixed_recipe=None
                        args.ensemble_strategy=policies.get(label,{}).get('ensemble_strategy')
                        args.finalization=policies.get(label,{}).get('finalization')
                        row['intervention'].update(policies.get(label,{}))
                        if label in per_arm_recipes:
                            args.fixed_recipe=task_root/'fixed-recipe.json'
                            args.fixed_recipe.write_text(json.dumps(per_arm_recipes[label],sort_keys=True))
                            row['intervention']['fixed_recipe']=per_arm_recipes[label]
                        for filename in INPUT_FILES:
                            shutil.copy2(input_root / filename, task_root / filename)
                        args.sandbox_prefix, isolation = sandbox_command(trial, Path(version['runtime']),mode=='live-agent')
                        # Resolving a venv's Python symlink loses pyvenv.cfg and
                        # silently selects the base interpreter without packages.
                        args.worker_python=str(args.autogluon_python.absolute()) if mode=='autogluon' else sys.executable
                        print(json.dumps({'event': 'case_start', 'case': name, 'seed': seed, 'arm': label, 'mode': mode}), flush=True)
                        result = run_one(Path(version['runtime']), task_root / 'task.md', trial / 'run', mode, args, settings, metadata, version['source_sha'], evaluation_truth=truth)
                        row.update(result, isolation=isolation)
                        row['status'] = {'ok': 'trained_success', 'constant_fallback': 'fall_back_dummy', 'agent_error': 'runtime_failed'}.get(result['status'], result['status'])
                    report['runs'].append(row)
                    save_report(args.report, report)
                    print(json.dumps({key: row.get(key) for key in ['case', 'seed', 'arm', 'mode', 'status', 'external_metric', 'wall_seconds', 'error']}), flush=True)
    report['status'] = 'completed'
    save_report(args.report, report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
