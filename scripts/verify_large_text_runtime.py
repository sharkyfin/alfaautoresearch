"""Operational mixed-text QA with more fit headroom, never a second outer score."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_suite import sandbox_command
from scripts.benchmark_kaggle import stop_child
from scripts.kaggle_data import INPUT_FILES, snapshot_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget', type=int, default=900)
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/performance_v2/large_text_runtime.json')
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError('Preserve existing operational QA observations')
    plan_path = ROOT / 'benchmarks/performance_v2/large_text_runtime_plan.json'
    plan = json.loads(plan_path.read_text())
    assert args.budget == plan['budget_seconds']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == plan['driver_sha256']
    freeze = json.loads((ROOT / 'benchmarks/performance_v2/final_freeze.json').read_text())
    observations = json.loads((ROOT / 'reports/performance_v2/final-lockbox-live.json').read_text())
    original = next(row for row in observations['runs'] if
                    (row['case'], row['seed'], row['arm']) == ('beyond_emscad', 42, 'candidate'))
    root = Path(tempfile.mkdtemp(prefix='large-text-runtime-qa-', dir=ROOT / 'runs'))
    runtime = root / 'runtime'
    archive = ROOT / '.bootstrap/performance_v2/p3j/source/submission.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == freeze['archives']['candidate']
    source = snapshot_archive(archive, runtime)
    assert source == freeze['sources']['candidate']
    trial = root / 'trial'; task = trial / 'task'; task.mkdir(parents=True)
    original_task = Path(original['run_dir']).parent / 'task'
    hashes = {}
    for name in INPUT_FILES:
        shutil.copy2(original_task / name, task / name)
        hashes[name] = hashlib.sha256((task / name).read_bytes()).hexdigest()
    assert hashes == original['input_sha256']
    prefix, isolation = sandbox_command(trial, runtime, False)
    command = prefix + [sys.executable, str(runtime / 'scripts/kaggle_worker.py'), str(task / 'task.md'),
                        '--mode', 'deterministic', '--budget', str(args.budget), '--threads', '8',
                        '--memory-mb', '8192', '--max-calls', '0', '--seed', '42', '--run-dir', str(trial / 'run')]
    record = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Operational fit/checkpoint/refit QA only. Reuses supplied task inputs, never reads or scores outer answers. Does not replace any original final observation or establish new blind quality.',
              'source_sha256': source, 'archive_sha256': freeze['archives']['candidate'],
              'budget_seconds': args.budget, 'threads': 8, 'memory_mb': 8192,
              'source_case': 'beyond_emscad', 'seed': 42, 'input_sha256': hashes,
              'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              'run_root': str(root), 'command': command, 'isolation': isolation,
              'external_metric': None, 'external_evaluator_called': False}
    args.report.write_text(json.dumps(record, indent=2) + '\n')
    started = time.monotonic(); peak = 0.; failure = None
    with (trial / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, cwd=task, env={'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8',
                                   'PYTHONUNBUFFERED': '1'}, stdout=log, stderr=log, start_new_session=True)
        import psutil
        while process.poll() is None:
            try:
                parent = psutil.Process(process.pid)
                peak = max(peak, sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)] if p.is_running()) / 2**20)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            if peak > 8192 or time.monotonic() - started > args.budget + 10:
                failure = 'memory_limit' if peak > 8192 else 'timeout'
                stop_child(process)
                break
            time.sleep(.2)
    record.update(wall_seconds=round(time.monotonic() - started, 3), peak_memory_mb=peak, returncode=process.returncode)
    try:
        import numpy as np
        import pandas as pd
        summary = json.loads((trial / 'run/summary.json').read_text())
        assert summary['runtime_code_sha256'] == source
        sample = pd.read_csv(task / 'sample_submission.csv', dtype=str)
        output = pd.read_csv(task / 'submission.csv', dtype=str)
        assert list(sample.columns) == list(output.columns) and len(sample) == len(output)
        assert sample.iloc[:, 0].equals(output.iloc[:, 0])
        probabilities = output.iloc[:, 1].astype(float).to_numpy()
        assert np.isfinite(probabilities).all() and ((0 <= probabilities) & (probabilities <= 1)).all()
        assert all(hashlib.sha256((task / name).read_bytes()).hexdigest() == expected for name, expected in hashes.items())
        history = [json.loads(line) for line in (trial / 'run/experiments.jsonl').read_text().splitlines()]
        record.update(output_schema_valid=True, output_rows=len(output),
                      output_sha256=hashlib.sha256((task / 'submission.csv').read_bytes()).hexdigest(),
                      checkpoint_kind=summary.get('checkpoint_kind'), fit_success=summary.get('fit_success'),
                      full_refit=summary.get('full_refit_published', False), qwen_calls=summary.get('qwen_calls', 0),
                      experiment_statuses=dict(Counter(row['status'] for row in history)),
                      champion_family=(summary.get('champion_recipe') or {}).get('model_family'))
        assert process.returncode == 0 and failure is None and summary.get('champion_recipe') and summary.get('fit_success', 0) > 0
        assert summary.get('full_refit_published') is True and summary.get('qwen_calls', 0) == 0
        record['status'] = 'passed'
    except (OSError, ValueError, KeyError, AssertionError) as error:
        record.update(status='failed', error=failure or type(error).__name__ + ': ' + str(error))
    args.report.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))
    return 0 if record['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
