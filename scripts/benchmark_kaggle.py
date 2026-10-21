"""Compare the uploaded agent on real Kaggle datasets without platform submits.

Run download_kaggle.py first. Quick runs score only an external development set;
--final exposes the reserved lockbox after a single mode/source has been chosen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODES = ('lightgbm', 'deterministic', 'live-agent')


def stop_child(process):
    if process.poll() is not None:return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        import psutil
        try: children = psutil.Process(process.pid).children(recursive=True)
        except psutil.NoSuchProcess: children = []
        for child in reversed(children):
            try:child.kill()
            except psutil.NoSuchProcess:pass
        process.kill(); process.wait(timeout=5)


def run_one(snapshot, task, run_dir, mode, args, settings, metadata, source_digest):
    from scripts.kaggle_data import evaluate_output, INPUT_FILES
    env = {'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
           'MPLCONFIGDIR': str(task.parent/'mpl-cache')}
    if mode == 'live-agent':env.update(settings)
    command = [sys.executable, str(snapshot/'scripts/kaggle_worker.py'), str(task),
               '--mode', mode, '--budget', str(args.budget), '--threads', str(args.threads),
               '--memory-mb', str(args.memory_mb), '--max-calls', str(args.max_calls),
               '--seed', str(metadata['seed']), '--run-dir', str(run_dir)]
    log_path = task.parent.parent/(mode+'.log')
    started = time.monotonic(); timed_out = False
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=task.parent, env=env, stdout=log, stderr=log, start_new_session=True)
        try:
            process.wait(timeout=args.budget+30)
        except subprocess.TimeoutExpired:
            timed_out = True; stop_child(process)
        finally:
            if process.poll() is None:stop_child(process)
    summary_path = run_dir/'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    record = {'case': metadata['case'], 'mode': mode, 'seed': metadata['seed'], 'phase': metadata['phase'],
              'metric': metadata['metric'], 'budget_seconds': args.budget,
              'wall_seconds': round(time.monotonic()-started, 3), 'threads': args.threads,
              'memory_mb': args.memory_mb, 'runtime_code_sha256': source_digest,
              'raw_sha256': metadata['raw_sha256'], 'split_sha256': metadata['split_sha256'],
              'returncode': process.returncode, 'log': str(log_path), 'run_dir': str(run_dir),
              'qwen_calls': summary.get('qwen_calls', 0),
              'qwen_tokens': summary.get('qwen_tokens', 0),
              'internal_metric': summary.get('champion_metric'),
              'experiments': summary.get('experiments', 0),
              'full_refit': summary.get('full_refit_published', False),
              'champion_family': summary.get('champion_recipe', {}).get('model_family') if summary.get('champion_recipe') else None,
              'decision_adjustment': summary.get('multipliers'),
              'ensemble': summary.get('ensemble'),
              'adaptive_search': summary.get('adaptive_search'),
              'cross_validation': summary.get('cross_validation'),
              'output_sha256': summary.get('output_sha256'),
              'feature_ablation_checks': summary.get('feature_ablation_checks', 0),
              'accepted_feature_ablations': summary.get('accepted_feature_ablations', 0)}
    try:
        if summary.get('runtime_code_sha256') != source_digest:
            raise ValueError('Worker source fingerprint differs from the frozen uploaded runtime')
        if any(hashlib.sha256((task.parent/name).read_bytes()).hexdigest() != metadata['input_sha256'][name] for name in INPUT_FILES):
            raise ValueError('Agent modified benchmark input material')
        # No evaluation labels are read until the child has completed.
        record['external_metric'] = evaluate_output(task.parent, Path(metadata['sealed_labels']), metadata)
        if timed_out:record['status'] = 'timeout'
        elif process.returncode or (run_dir/'error.txt').exists():record['status'] = 'agent_error'
        elif not summary.get('champion_recipe'):record['status'] = 'constant_fallback'
        else:record['status'] = 'ok'
        history = [json.loads(line) for line in (run_dir/'experiments.jsonl').read_text().splitlines()] if (run_dir/'experiments.jsonl').exists() else []
        research = [row for row in history if row.get('origin') == 'qwen']
        record['qwen_fits'] = sum(row['status'] == 'ok' for row in research)
        record['qwen_accepted'] = sum(row.get('accepted', False) for row in research)
        record['research_experiments'] = [{key: row[key] for key in ['experiment_id','status','metric_value','accepted','recipe','feature_trial'] if key in row} for row in research]
        record['champion_recipe'] = summary.get('champion_recipe')
        record['qwen_events'] = summary.get('qwen_events', [])
    except (ValueError, OSError, KeyError, TypeError) as error:
        record.update(status='evaluation_error', error=str(error))
    return record


def comparisons(records):
    pairs = []
    for case, seed in sorted({(r['case'], r['seed']) for r in records}):
        rows = {r['mode']: r for r in records if r['case']==case and r['seed']==seed and r['status']=='ok'}
        reference = rows.get('lightgbm')
        for mode in ['deterministic', 'live-agent']:
            row = rows.get(mode)
            if row is None or reference is None:continue
            direction = -1 if row['metric']=='rmse' else 1
            gain = direction*(row['external_metric']-reference['external_metric'])
            pairs.append({'case':case,'seed':seed,'mode':mode,'metric':row['metric'],
                          'external_metric':row['external_metric'],'reference_metric':reference['external_metric'],
                          'signed_gain_over_local_lightgbm':gain,
                          'relative_metric_gain':gain/max(abs(reference['external_metric']),1e-12)})
    return pairs


def research_comparisons(records):
    pairs=[]
    for case,seed in sorted({(r['case'],r['seed']) for r in records}):
        rows={r['mode']:r for r in records if r['case']==case and r['seed']==seed and r['status']=='ok'}
        before,after=rows.get('deterministic'),rows.get('live-agent')
        if before is None or after is None:continue
        if any(before.get(key)!=after.get(key) for key in ['raw_sha256','split_sha256','budget_seconds','runtime_code_sha256']):
            raise ValueError('Cannot compare Qwen on mismatched datasets, budgets or source versions')
        gain=(-1 if before['metric']=='rmse' else 1)*(after['external_metric']-before['external_metric'])
        pairs.append({'case':case,'seed':seed,'metric':before['metric'],
                      'without_qwen':before['external_metric'],'with_qwen':after['external_metric'],
                      'signed_gain':gain,'qwen_fits':after.get('qwen_fits',0),
                      'qwen_accepted':after.get('qwen_accepted',0)})
    return pairs


def save_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    report['comparisons'] = comparisons(report['runs'])
    report['research_comparisons'] = research_comparisons(report['runs'])
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Kaggle development benchmark', '',
             'External development metrics; no hackathon leaderboard score is estimated.', '',
             '| Dataset | Mode | Metric | External | Internal | Seconds | Qwen fits | Status |',
             '|---|---|---|---:|---:|---:|---:|---|']
    for row in report['runs']:
        external = f"{row['external_metric']:.6g}" if 'external_metric' in row else '—'
        internal = f"{row['internal_metric']:.6g}" if row.get('internal_metric') is not None else '—'
        lines.append(f"| {row['case']} | {row['mode']} | {row['metric']} | {external} | {internal} | {row['wall_seconds']:.1f} | {row.get('qwen_fits',0)} | {row['status']} |")
    lines += ['', 'RMSE: lower is better. Macro F1: higher is better. Raw metrics are never averaged.',
              '', 'Source SHA256: `'+report['runtime_code_sha256']+'`.',
              'The same train/evaluation splits and time cap are used for all modes within a dataset/seed.',
              'Qwen is not shown the external labels or the external metrics during its run.',
              'The 20% final lockbox remains unused unless --final is explicitly selected.',
              'These data and development scripts are excluded from the submitted solution ZIP.']
    path.with_suffix('.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='all')
    parser.add_argument('--modes', default=','.join(MODES))
    parser.add_argument('--budget', type=float, default=90)
    parser.add_argument('--seeds', default='42')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-mb', type=int, default=8192)
    parser.add_argument('--max-calls', type=int, default=2)
    parser.add_argument('--zip', type=Path, default=ROOT/'submission.zip')
    parser.add_argument('--raw-dir', type=Path, default=ROOT/'artifacts/data/kaggle/raw')
    parser.add_argument('--key-file', type=Path, default=ROOT/'.env.organizer.local')
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--report', type=Path, default=ROOT/'reports/kaggle-benchmark.json')
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    registry = json.loads((ROOT/'benchmarks/kaggle_cases.json').read_text())
    cases = list(registry) if args.cases=='all' else args.cases.split(',')
    modes = args.modes.split(','); seeds = list(map(int,args.seeds.split(',')))
    if not cases or any(name not in registry for name in cases):parser.error('Unknown dataset')
    if not modes or any(mode not in MODES for mode in modes):parser.error('Unknown mode')
    if not 1<=args.threads<=8 or not 15<=args.budget<=10500:parser.error('Use 1–8 threads and a 15–10500 second budget')
    if args.final and len(modes)!=1:parser.error('Choose a single already selected mode for the final lockbox')
    missing = [case for case in cases if not (args.raw_dir/(case+'.zip')).is_file()]
    if missing:parser.error('Download datasets first: python scripts/download_kaggle.py --cases '+','.join(missing))
    settings = {}
    if 'live-agent' in modes:
        from scripts.run_organizer import load_environment
        settings = load_environment(args.key_file) if args.key_file.exists() else {k:os.environ[k] for k in ['OPENAI_API_KEY','OPENAI_BASE_URL','OPENAI_MODEL'] if os.environ.get(k)}
        if not all(settings.get(k) for k in ['OPENAI_API_KEY','OPENAI_BASE_URL']):parser.error('Configure organizer Qwen or omit live-agent mode')
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]=str(args.threads)
    from scripts.kaggle_data import prepare, snapshot_archive, INPUT_FILES
    parent = ROOT/'runs'; parent.mkdir(exist_ok=True)
    root = args.run_root.resolve() if args.run_root else Path(tempfile.mkdtemp(prefix='kaggle-benchmark-',dir=parent))
    if root.exists() and any(root.iterdir()):parser.error('run-root must be new or empty')
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root/'runtime'
    digest = snapshot_archive(args.zip, snapshot)
    packages = {name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','scipy','lightgbm','xgboost']}
    report = {'status':'running','runtime_code_sha256':digest,
              'archive_sha256':hashlib.sha256(args.zip.read_bytes()).hexdigest(),
              'budget_seconds_per_task':args.budget,'threads':args.threads,'memory_mb':args.memory_mb,
              'phase':'final_lockbox' if args.final else 'development','run_root':str(root),
              'packages':packages,'runs':[],'cases':[],
              'platform_submissions':0,'kaggle_submissions':0,'final_lockbox_used':args.final}
    print(json.dumps({'event':'benchmark_start','source_sha256':digest,'datasets':cases,'modes':modes,
                      'maximum_model_minutes':len(cases)*len(modes)*len(seeds)*args.budget/60,'report':str(args.report)}),flush=True)
    save_report(args.report, report)
    for case in cases:
        for seed in seeds:
            prepared = root/f'{case}_{seed}'
            metadata = prepare(case,registry[case],args.raw_dir/(case+'.zip'),prepared,seed,args.final)
            report['cases'].append(metadata)
            metadata['sealed_labels'] = str(prepared/'sealed/evaluation.csv')
            for mode in modes:
                trial = prepared/mode; trial.mkdir()
                task_root = trial/'task'; task_root.mkdir()
                for name in INPUT_FILES:shutil.copy2(prepared/'inputs'/name,task_root/name)
                print(json.dumps({'event':'case_start','case':case,'mode':mode,'seed':seed,'budget':args.budget}),flush=True)
                row = run_one(snapshot,task_root/'task.md',trial/'run',mode,args,settings,metadata,digest)
                report['runs'].append(row)
                save_report(args.report,report)
                print(json.dumps({key:row[key] for key in ['case','mode','status','external_metric','internal_metric','wall_seconds','qwen_calls','qwen_fits'] if key in row}),flush=True)
    report['status'] = 'completed' if all(row['status']=='ok' for row in report['runs']) else 'completed_with_failures'
    save_report(args.report,report)
    print(json.dumps({'event':'benchmark_done','status':report['status'],'runs':len(report['runs']),'report':str(args.report)}),flush=True)
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':raise SystemExit(main())
