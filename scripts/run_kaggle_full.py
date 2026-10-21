"""Run an actual flat solution ZIP through the platform CLI on public Kaggle data.

Unlike benchmark_kaggle.py this does not inject a budget, model, experiment cap
or research client. External labels are scored only after the process exits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_flat(archive, task):
    """Match the platform's root-only copying rule and reject unsafe uploads."""
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)) or not {'solution.py', 'requirements.txt'} <= set(names):
            raise ValueError('Upload must contain unique root files and both entry files')
        if any(Path(name).name != name or not (name.endswith('.py') or name == 'requirements.txt') for name in names):
            raise ValueError('Only root Python files and requirements.txt are allowed')
        if any((task / name).exists() for name in names):
            raise ValueError('Upload would replace task input files')
        for name in names:
            (task / name).write_bytes(zipped.read(name))
    return names


def stop_group(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=12)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    except ProcessLookupError:
        process.wait(timeout=5)


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if report['status'] == 'running':
        return
    summary = report.get('agent_summary', {})
    metadata = report['data']
    lines = ['# Full Kaggle pipeline run', '',
             'Dataset: [' + metadata['dataset'] + '](' + metadata['source'] + ').',
             'Executed the flat upload as `python3 solution.py task.md`; no agent defaults were overridden.',
             'Local execution uses the existing virtual environment, not a fresh platform Docker image.',
             'The unseen external split was fixed before launch; labels stayed outside the task folder.',
             'Exact duplicate feature groups cannot cross the external train/test boundary.', '',
             f"Status: **{report['status']}**. Exit code: {report.get('returncode')}. Wall time: {report.get('wall_seconds')} seconds.",
             f"Training rows: {metadata['training_rows']}; external test rows: {metadata['evaluation_rows']}; original features: {metadata['features']}.",
             f"Agent budget: {summary.get('budget_seconds')} seconds; experiments: {summary.get('experiments')}; Qwen requests: {summary.get('qwen_calls')}.",
             f"Completed Qwen trials: {report.get('qwen_fits')}; primary-holdout accepted Qwen trials: {report.get('qwen_accepted')}; full refit: {summary.get('full_refit_published')}."]
    if 'external_metric' in report:
        lines += ['', f"External {metadata['metric']}: **{report['external_metric']:.6f}**."]
    if 'regression_metrics' in report:
        metrics = report['regression_metrics']
        lines += [f"MAE: {metrics['mae']:.6f}; R²: {metrics['r2']:.6f}; training-mean baseline RMSE: {metrics['training_mean_rmse']:.6f}."]
    hpo=summary.get('adaptive_search', {})
    if hpo:lines += ['', f"TPE: {hpo.get('completed_trials', 0)} completed trials; family proposals: {hpo.get('family_proposals', {})}."]
    cv=summary.get('cross_validation', {})
    if cv:
        lines += [f"Frozen CV: {cv.get('planned_folds')} planned folds; {cv.get('completed_candidates')} fully checked candidates.",
                  '', '| Candidate | Model | Fold scores | Mean | Standard deviation |',
                  '|---|---|---|---:|---:|']
        for row in cv.get('candidates', []):
            scores=', '.join(f'{score:.6f}' for score in row['fold_metrics'])
            lines.append(f"| {row['experiment_id']} | {row['model_family']} | {scores} | {row['mean_metric']:.6f} | {row['metric_std']:.6f} |")
    lines += ['', '| Trial | Phase | Origin | Model | Internal score | Accepted |',
              '|---|---|---|---|---:|---|']
    for row in report.get('experiments', []):
        score = row.get('metric_value')
        formatted = f'{score:.6f}' if score is not None else '—'
        lines.append(f"| {row['experiment_id']} | {row.get('phase', 'search')} | {row.get('origin', '')} | {row['recipe']['model_family']} | {formatted} | {row.get('accepted', False)} |")
    lines += ['', 'Primary validation acceptance does not guarantee that an external test improves.',
              'This single run is an integration and quality check; it is not a hackathon leaderboard score.',
              'No external metric was supplied to the agent and no post-test model changes were made.',
              'No Kaggle or hackathon submissions were made.', '',
              'Archive SHA256: `' + report['archive_sha256'] + '`.',
              'Runtime SHA256: `' + report['runtime_code_sha256'] + '`.',
              'Submission: `' + report['submission'] + '`.',
              'Full process log: `' + report['log'] + '`.',
              'Run directory: `' + report['run_root'] + '`.']
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='concrete_strength')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--zip', type=Path, default=ROOT / 'submission.zip')
    parser.add_argument('--raw-dir', type=Path, default=ROOT / 'artifacts/data/kaggle/raw')
    parser.add_argument('--key-file', type=Path, default=ROOT / '.env.organizer.local')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/kaggle-full-concrete.json')
    parser.add_argument('--supervisor-seconds', type=float, default=960,
                        help='Outer failure guard only; does not alter the agent budget')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'benchmarks/kaggle_cases.json').read_text())
    if args.case not in registry:
        parser.error('Unknown dataset')
    archive = args.raw_dir / (args.case + '.zip')
    if not archive.is_file():
        parser.error('Download the public dataset first with download_kaggle.py')
    if args.supervisor_seconds < 930:
        parser.error('Allow the default 900-second agent budget plus shutdown time')
    from scripts.run_organizer import load_environment
    settings = load_environment(args.key_file)
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '8'
    from scripts.kaggle_data import prepare, snapshot_archive, evaluate_output, INPUT_FILES
    parent = ROOT / 'runs'
    parent.mkdir(exist_ok=True)
    root = args.run_root.resolve() if args.run_root else Path(tempfile.mkdtemp(prefix='kaggle-full-', dir=parent))
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        parser.error('run-root must be new or empty')
    # Reuse the tested ZIP integrity/fingerprint checker. This source snapshot is
    # never executed: the child runs solution.py extracted verbatim below.
    digest = snapshot_archive(args.zip, root / 'source-integrity-check')
    metadata = prepare(args.case, registry[args.case], archive, root / 'prepared', args.seed, final=True)
    task = root / 'prepared/inputs'
    copied = install_flat(args.zip, task)
    log_path = root / 'pipeline.log'
    report = {'status': 'running', 'entrypoint': ['python3', 'solution.py', 'task.md'],
              'agent_default_overrides': {}, 'supervisor_seconds': args.supervisor_seconds,
              'archive_sha256': sha256(args.zip), 'runtime_code_sha256': digest,
              'copied_root_files': copied, 'run_root': str(root), 'log': str(log_path),
              'submission': str(task / 'submission.csv'), 'data': metadata,
              'packages': {name: importlib.metadata.version(name) for name in
                           ['numpy', 'pandas', 'scikit-learn', 'scipy', 'lightgbm', 'xgboost']},
              'platform_submissions': 0, 'kaggle_submissions': 0}
    write_report(args.report, report)
    env = {'PATH': str(Path(sys.executable).parent) + os.pathsep + os.defpath,
           'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
           'MPLCONFIGDIR': str(root / 'mpl-cache'), **settings}
    print(json.dumps({'event': 'platform_cli_start', 'case': args.case,
                      'command': report['entrypoint'], 'report': str(args.report),
                      'run_root': str(root)}, ensure_ascii=False), flush=True)
    started = time.monotonic()
    timed_out = False
    with log_path.open('w') as log:
        process = subprocess.Popen(report['entrypoint'], cwd=task, env=env, stdout=log,
                                   stderr=log, start_new_session=True)
        try:
            process.wait(timeout=args.supervisor_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            stop_group(process)
    report.update(returncode=process.returncode, timed_out=timed_out,
                  wall_seconds=round(time.monotonic() - started, 3))
    try:
        summaries = list((task / '.atlas').glob('*/summary.json'))
        if len(summaries) != 1:
            raise ValueError('Expected exactly one completed run summary')
        run_dir = summaries[0].parent
        summary = json.loads(summaries[0].read_text())
        report['agent_summary'] = summary
        if summary['runtime_code_sha256'] != digest:
            raise ValueError('Executed runtime differs from the frozen ZIP fingerprint')
        if sha256(args.zip) != report['archive_sha256']:
            raise ValueError('Upload archive changed during the run')
        report['inputs_unchanged'] = all(sha256(task / name) == metadata['input_sha256'][name] for name in INPUT_FILES)
        if not report['inputs_unchanged']:
            raise ValueError('Agent modified task input material')
        history = [json.loads(line) for line in (run_dir / 'experiments.jsonl').read_text().splitlines()]
        report['experiments'] = [{key: row[key] for key in
                                 ['experiment_id', 'origin', 'phase', 'status', 'metric_value', 'accepted', 'recipe', 'feature_trial', 'runtime_sec', 'confirmation_metric']
                                 if key in row} for row in history]
        research = [row for row in history if row.get('origin') == 'qwen']
        report['qwen_fits'] = sum(row['status'] == 'ok' for row in research)
        report['qwen_accepted'] = sum(bool(row.get('accepted')) for row in research)
        report['final_recipe'] = json.loads((run_dir / 'final_recipe.json').read_text())
        report['run_dir'] = str(run_dir)
        report['output_sha256'] = sha256(task / 'submission.csv')
        if report['output_sha256'] != summary['output_sha256']:
            raise ValueError('Published submission differs from the final summary')
        artifact_paths = [log_path, *run_dir.glob('*.json'), *run_dir.glob('*.jsonl'), task / 'submission.csv']
        report['credential_absent_from_artifacts'] = all(
            settings['OPENAI_API_KEY'].encode() not in path.read_bytes() for path in artifact_paths)
        if not report['credential_absent_from_artifacts']:
            raise ValueError('Credential found in a generated artifact')
        # External labels are opened only here, after the actual CLI has exited.
        sealed = root / 'prepared/sealed/evaluation.csv'
        report['external_metric'] = evaluate_output(task, sealed, metadata)
        report['output_schema_valid'] = True
        if metadata['metric'] == 'rmse':
            import numpy as np
            import pandas as pd
            from sklearn.metrics import mean_absolute_error, r2_score
            target = metadata['target']
            truth = pd.read_csv(sealed)[target].to_numpy()
            prediction = pd.read_csv(task / 'submission.csv')[target].to_numpy()
            training_mean = pd.read_csv(task / 'train.csv')[target].mean()
            report['regression_metrics'] = {'mae': float(mean_absolute_error(truth, prediction)),
                                            'r2': float(r2_score(truth, prediction)),
                                            'training_mean_rmse': float(np.sqrt(np.mean((truth - training_mean) ** 2)))}
        if timed_out:
            report['status'] = 'timeout'
        elif process.returncode or (run_dir / 'error.txt').exists():
            report['status'] = 'agent_error'
        elif not summary.get('champion_recipe'):
            report['status'] = 'constant_fallback'
        elif not summary.get('full_refit_published'):
            report['status'] = 'incomplete_refit'
        else:
            report['status'] = 'completed'
    except (ValueError, OSError, KeyError, TypeError) as error:
        report.update(status='evaluation_error', error=str(error))
    write_report(args.report, report)
    print(json.dumps({key: report[key] for key in
                      ['status', 'wall_seconds', 'external_metric', 'regression_metrics', 'qwen_fits', 'qwen_accepted', 'submission', 'error']
                      if key in report}, ensure_ascii=False), flush=True)
    return 0 if report['status'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
