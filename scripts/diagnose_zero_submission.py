"""Retain exact-upload public-task execution evidence for the zero-score incident.

This deliberately stops after the initial portfolio or a diagnostic wall guard.
It is not a full-budget quality benchmark and has no official test targets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.budget import stop_process
from atlas.contract import read_inputs
from atlas.submit import validate_frame
from atlas.task import load, read_csv
from scripts.verify_zip import archive_source_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['zero', 'feature-discovery', 'patched'], default='zero')
    parser.add_argument('--fits', type=int, default=6)
    parser.add_argument('--guard', type=float, default=700)
    args = parser.parse_args()
    private = ROOT/'.bootstrap/incident-zero-20261023'
    archive = private/('platform-'+args.arm+'.zip')
    report_dir = ROOT/'reports/incident_zero_20261023'
    report_dir.mkdir(exist_ok=True)
    report = report_dir/('public-exact-'+args.arm+'.json')
    if report.exists():
        raise RuntimeError('Preserve the original attempt; choose a new run label before retries')
    runroot = ROOT/'runs'/('incident-zero-'+args.arm+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    rows = []
    for kind in ['regression', 'classification']:
        inputs = read_inputs(ROOT/'artifacts/data'/kind/'task.md')
        task = runroot/kind
        task.mkdir(parents=True)
        shutil.copy2(inputs.task_path, task/'task.md')
        for source in [inputs.train, inputs.test, inputs.dictionary, inputs.sample]:
            target = task/source.relative_to(inputs.task_path.parent)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        with zipfile.ZipFile(archive) as payload:
            payload.extractall(task)
        env = {'PATH': os.defpath, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
               'MPLCONFIGDIR': str(task/'mpl-cache')}
        log = task/'controller.log'
        start = time.monotonic()
        termination = 'natural'
        print(json.dumps({'event': 'start', 'arm': args.arm, 'task': kind, 'directory': str(task)}), flush=True)
        with log.open('w') as stream:
            proc = subprocess.Popen([sys.executable, 'solution.py', 'task.md'], cwd=task, env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while proc.poll() is None:
                    text = log.read_text()
                    if text.count('"event": "experiment"') >= args.fits:
                        termination = 'diagnostic_portfolio_SIGTERM'
                        proc.terminate()
                        break
                    if time.monotonic()-start >= args.guard:
                        termination = 'diagnostic_guard_SIGTERM'
                        proc.terminate()
                        break
                    time.sleep(.2)
                proc.wait(timeout=15)
            finally:
                if proc.poll() is None:
                    stop_process(proc)
        summaries = list((task/'.atlas').glob('*/summary.json'))
        summary = json.loads(summaries[0].read_text()) if len(summaries) == 1 else {}
        events = []
        if summaries:
            history = summaries[0].parent/'experiments.jsonl'
            if history.exists():
                events = [json.loads(line) for line in history.read_text().splitlines()]
        output = task/'submission.csv'
        valid = False
        validation_error = None
        if output.exists():
            try:
                data = load(read_inputs(task/'task.md'), strict=True)
                validate_frame(data, read_csv(output, dtype={name: 'string' for name in data.spec.ids}))
                valid = True
            except Exception as error:
                validation_error = {'class': type(error).__name__, 'message': str(error)}
        record = {'arm': args.arm, 'task': kind, 'run_dir': str(task),
            'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'expected_runtime_source_sha256': archive_source_digest(archive),
            'wall_seconds': time.monotonic()-start, 'returncode': proc.returncode,
            'termination': termination, 'valid_output': valid, 'validation_error': validation_error,
            'summary': summary, 'experiments': events, 'real_qwen_calls': 0}
        rows.append(record)
        report.write_text(json.dumps({'status': 'running', 'runs': rows}, indent=2)+'\n')
        print(json.dumps({'event': 'finished', 'task': kind, 'returncode': proc.returncode,
            'termination': termination, 'fits': summary.get('fit_success'),
            'trained': summary.get('trained_output'), 'valid_output': valid}), flush=True)
    report.write_text(json.dumps({'status': 'completed', 'runs': rows,
        'scope': 'Exact-upload initial-portfolio diagnosis; no outer labels, no official score, deliberate SIGTERM'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
