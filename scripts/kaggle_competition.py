"""Execute the unchanged upload on original competition train/test files.

This development adapter prepares task metadata, verifies CSV identities and
exports predictions for manual Kaggle upload. It never submits or reads hidden
test labels, and it does not alter the agent's budget, models or search settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd

from scripts.kaggle_data import INPUT_FILES, snapshot_archive
from scripts.run_kaggle_full import install_flat, sha256, stop_group
from scripts.run_organizer import load_environment

ROOT = Path(__file__).resolve().parents[1]
RAW_FILES = ('train.csv', 'test.csv', 'sample_submission.csv')


def prepare_competition(raw, destination, spec):
    raw, destination = Path(raw), Path(destination)
    identifier, target = spec['id_column'], spec['target']
    frames = {name: pd.read_csv(raw/name, dtype={identifier: 'string'}) for name in RAW_FILES}
    train, test, sample = [frames[name] for name in RAW_FILES]
    if list(train) != list(spec['columns']) or list(test) != [c for c in train if c != target]:
        raise ValueError('Competition input columns differ from the original schema')
    if len(train) != spec['training_rows'] or len(test) != spec['test_rows']:
        raise ValueError('Competition row counts differ from the original train/test split')
    for name, frame in frames.items():
        if frame[identifier].isna().any() or frame[identifier].duplicated().any():
            raise ValueError('Missing or duplicate competition IDs in '+name)
    if set(train[identifier]) & set(test[identifier]):
        raise ValueError('Competition train/test IDs overlap')
    if list(sample) != [identifier, target] or not sample[identifier].equals(test[identifier]):
        raise ValueError('Sample submission does not match the original test IDs and order')
    if train[target].isna().any() or set(train[target].astype(str)) != set(spec['labels']):
        raise ValueError('Training labels do not match the competition')
    destination.mkdir(parents=True, exist_ok=False)
    for name in RAW_FILES:
        shutil.copyfile(raw/name, destination/name)
    dictionary = []
    for column in train:
        role = 'id' if column == identifier else 'target' if column == target else 'feature'
        numeric = pd.api.types.is_numeric_dtype(train[column]) and not pd.api.types.is_bool_dtype(train[column])
        dictionary.append({'feature': column, 'role': role,
                           'dtype': 'string' if role == 'id' else 'category' if role == 'target' or not numeric else 'numeric',
                           'description_ru': spec['columns'][column]})
    pd.DataFrame(dictionary).to_csv(destination/'data_dictionary.csv', index=False)
    text = (f"Task: binary classification\nPrimary metric: {spec['metric']}\n"
            f"{spec['description']}\nCompetition: {spec['source']}\n"
            'Inputs: `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv`.\n'
            'Choose and validate the model autonomously using only supplied training labels.\n'
            'Write `submission.csv` next to task.md; preserve sample IDs, row order and column names.\n')
    (destination/'task.md').write_text(text, encoding='utf-8')
    provenance_path = raw.parent/'data-provenance.json'
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else None
    raw_hashes = {name: sha256(raw/name) for name in RAW_FILES}
    if provenance and any(provenance['files'][name]['sha256'] != raw_hashes[name] for name in RAW_FILES):
        raise ValueError('Raw competition data differs from the recorded provenance')
    return {'competition': spec['slug'], 'source': spec['source'], 'metric': spec['metric'],
            'target': target, 'id_column': identifier, 'training_rows': len(train),
            'test_rows': len(test), 'features': len(test.columns)-1,
            'original_split_preserved': True, 'test_labels_available': False,
            'raw_sha256': raw_hashes, 'input_sha256': {name: sha256(destination/name) for name in INPUT_FILES},
            'provenance': provenance}


def validate_competition_output(output, sample_path, spec):
    sample, frame = [pd.read_csv(path, dtype=str) for path in [sample_path, output]]
    identifier, target = spec['id_column'], spec['target']
    if list(frame) != list(sample) or len(frame) != len(sample):
        raise ValueError('Competition submission schema or row count is invalid')
    if frame.isna().any().any() or frame[identifier].duplicated().any():
        raise ValueError('Competition submission contains missing values or duplicate IDs')
    if not frame[identifier].equals(sample[identifier]):
        raise ValueError('Competition submission IDs or row order changed')
    if not set(frame[target]) <= set(spec['labels']):
        raise ValueError('Competition submission contains invalid labels')
    return {'rows': len(frame), 'columns': list(frame),
            'label_counts': {str(k): int(v) for k, v in frame[target].value_counts().items()}}


def write_competition_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if report['status'] == 'running':
        return
    data, summary = report['data'], report.get('agent_summary', {})
    lines = ['# Kaggle competition: manual submission', '',
             f"Competition: [{report['title']}]({data['source']}).",
             f"Status: **{report['status']}**. Wall time: {report['wall_seconds']} seconds.",
             f"Original train: {data['training_rows']} rows. Original private-label test: {data['test_rows']} rows.",
             'The original CSV files were copied byte for byte; no new external train/test split was made.',
             'The frozen upload ran as `python3 solution.py task.md` with no agent default overrides.',
             f"Agent budget: {summary.get('budget_seconds')} seconds. Experiments: {summary.get('experiments')}.",
             f"Qwen requests: {summary.get('qwen_calls')}; completed Qwen trials: {report.get('qwen_fits')}.",
             f"Full-data refit: {summary.get('full_refit_published')}.", '',
             '**Official Kaggle score and leaderboard rank are pending manual upload.**',
             'Internal validation metrics below are not Kaggle leaderboard scores.', '']
    cv = summary.get('cross_validation', {})
    for candidate in cv.get('candidates', []):
        scores=', '.join(f'{value:.6f}' for value in candidate['fold_metrics'])
        lines.append(f"- {candidate['model_family']}: internal fold accuracy {scores}; mean {candidate['mean_metric']:.6f}.")
    lines += ['', 'Upload the generated CSV at: '+report['manual_submission_url'],
              'Check the official score and rank at: '+report['leaderboard_url'],
              'No Kaggle submission was made by this script.', '',
              'Output CSV: `'+report.get('exported_submission', report['submission'])+'`.',
              'Output SHA256: `'+str(report.get('output_sha256'))+'`.',
              'Archive SHA256: `'+report['archive_sha256']+'`.',
              'Runtime SHA256: `'+report['runtime_code_sha256']+'`.',
              'Process log: `'+report['log']+'`.',
              'Local execution used the existing macOS environment, not the organizer Docker image.']
    rejected = [event for event in summary.get('qwen_events', []) if event.get('status') == 'recipe_rejected']
    if rejected:
        lines += ['', f"Qwen recipe rejections: {len(rejected)}; completed Qwen trials: {report.get('qwen_fits', 0)}.",
                  'Reasons: '+ '; '.join(sorted({event['reason'] for event in rejected}))+'.',
                  'When these proposals were rejected, the existing model portfolio and TPE search continued.']
    if report.get('delivery_file'):
        lines += ['', 'Convenient manual-upload copy: `'+report['delivery_file']+'`.']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def run_competition(args):
    registry = json.loads((ROOT/'benchmarks/kaggle_competitions.json').read_text())
    if args.competition not in registry:
        raise ValueError('Unknown Kaggle competition')
    spec = registry[args.competition]
    parent = ROOT/'runs'; parent.mkdir(exist_ok=True)
    root = args.run_root.resolve() if args.run_root else Path(tempfile.mkdtemp(prefix='kaggle-competition-', dir=parent))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError('run-root must be new or empty')
    digest = snapshot_archive(args.zip, root/'source-integrity-check')
    task = root/'inputs'
    metadata = prepare_competition(args.raw_dir, task, spec)
    copied = install_flat(args.zip, task)
    report_path = args.report
    if report_path == ROOT/'reports/kaggle-full-concrete.json':
        report_path = ROOT/'reports'/('kaggle-competition-'+spec['slug']+'.json')
    log_path = root/'pipeline.log'
    report = {'status': 'running', 'title': spec['title'], 'data': metadata,
              'entrypoint': ['python3', 'solution.py', 'task.md'], 'agent_default_overrides': {},
              'archive_sha256': sha256(args.zip), 'runtime_code_sha256': digest,
              'copied_root_files': copied, 'supervisor_seconds': args.supervisor_seconds,
              'run_root': str(root), 'submission': str(task/'submission.csv'), 'log': str(log_path),
              'manual_submission_url': spec['source']+'/submit', 'leaderboard_url': spec['source']+'/leaderboard',
              'official_public_score': None, 'leaderboard_rank': None, 'submission_mode': 'manual',
              'kaggle_submissions': 0, 'platform_submissions': 0}
    settings = load_environment(args.key_file)
    env = {'PATH': str(Path(sys.executable).parent)+os.pathsep+os.defpath,
           'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
           'MPLCONFIGDIR': str(root/'mpl-cache'), **settings}
    write_competition_report(report_path, report)
    print(json.dumps({'event': 'competition_cli_start', 'competition': spec['slug'],
                      'report': str(report_path), 'run_root': str(root)}), flush=True)
    started = time.monotonic(); timed_out = False
    with log_path.open('w') as log:
        process = subprocess.Popen(report['entrypoint'], cwd=task, env=env, stdout=log,
                                   stderr=log, start_new_session=True)
        try:
            process.wait(timeout=args.supervisor_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            stop_group(process)
    report.update(returncode=process.returncode, wall_seconds=round(time.monotonic()-started, 3), timed_out=timed_out)
    try:
        summaries = list((task/'.atlas').glob('*/summary.json'))
        if len(summaries) != 1:
            raise ValueError('Expected one complete agent summary')
        run_dir = summaries[0].parent
        summary = json.loads(summaries[0].read_text()); report['agent_summary'] = summary
        if summary['runtime_code_sha256'] != digest or sha256(args.zip) != report['archive_sha256']:
            raise ValueError('Executed agent or frozen archive changed')
        report['inputs_unchanged'] = all(sha256(task/name) == metadata['input_sha256'][name] for name in INPUT_FILES)
        if not report['inputs_unchanged']:
            raise ValueError('Competition input files changed during execution')
        history = [json.loads(line) for line in (run_dir/'experiments.jsonl').read_text().splitlines()]
        report['experiments'] = [{key: row[key] for key in
                                 ['experiment_id', 'origin', 'phase', 'status', 'metric_value', 'accepted', 'recipe']
                                 if key in row} for row in history]
        report['qwen_fits'] = sum(row.get('origin') == 'qwen' and row['status'] == 'ok' for row in history)
        report['final_recipe'] = json.loads((run_dir/'final_recipe.json').read_text())
        report['output_validation'] = validate_competition_output(task/'submission.csv', task/'sample_submission.csv', spec)
        report['output_sha256'] = sha256(task/'submission.csv')
        if report['output_sha256'] != summary['output_sha256']:
            raise ValueError('Submission hash differs from the agent summary')
        artifacts = [log_path, *run_dir.glob('*.json'), *run_dir.glob('*.jsonl'), task/'submission.csv']
        report['credential_absent_from_artifacts'] = all(settings['OPENAI_API_KEY'].encode() not in p.read_bytes() for p in artifacts)
        if not report['credential_absent_from_artifacts']:
            raise ValueError('Credential found in an output artifact')
        if timed_out:
            report['status'] = 'timeout'
        elif process.returncode or (run_dir/'error.txt').exists():
            report['status'] = 'agent_error'
        elif not summary.get('champion_recipe') or not summary.get('full_refit_published'):
            report['status'] = 'incomplete_refit'
        else:
            export = ROOT/'artifacts/data/kaggle/competitions'/spec['slug']/(spec['slug']+'_submission.csv')
            export.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(task/'submission.csv', export)
            report.update(status='completed', exported_submission=str(export))
    except (ValueError, OSError, KeyError, TypeError) as error:
        report.update(status='evaluation_error', error=str(error))
    write_competition_report(report_path, report)
    print(json.dumps({key: report[key] for key in
                      ['status', 'wall_seconds', 'qwen_fits', 'exported_submission', 'output_validation', 'error'] if key in report}), flush=True)
    return 0 if report['status'] == 'completed' else 1
