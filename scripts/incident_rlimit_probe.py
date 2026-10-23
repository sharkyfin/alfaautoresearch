"""Reproduce worker failures under an inherited finite Linux address-space cap.

This is a controlled hypothesis test, not evidence of the actual platform limit.
All lowered limits belong to short-lived child processes, never this controller.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.budget import stop_process
from scripts.synthetic import create


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arms', default='zero,feature-discovery')
    args = parser.parse_args()
    if sys.platform != 'linux':
        raise RuntimeError('Only lower real RLIMIT_AS in isolated Linux subprocesses')
    report = ROOT/'reports/incident_zero_20261023'/('rlimit-probe-'+args.arms.replace(',', '-')+'.json')
    report.parent.mkdir(exist_ok=True)
    assert not report.exists(), 'Keep prior observations'
    rows = []
    for arm in args.arms.split(','):
        if arm not in {'zero', 'feature-discovery', 'patched'}:
            raise ValueError('Unknown archived arm')
        archive = ROOT/'.bootstrap/incident-zero-20261023'/('platform-'+arm+'.zip')
        task = ROOT/'runs'/('inherited-address-limit-'+arm)
        assert not task.exists()
        create(task, n=120)
        with zipfile.ZipFile(archive) as payload:
            payload.extractall(task)
        def inherited_limit():
            resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
        env = {'PATH': os.defpath, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8',
               'MPLCONFIGDIR': str(task/'mpl-cache')}
        start = time.monotonic()
        log = task/'controller.log'
        termination = 'natural'
        with log.open('w') as stream:
            proc = subprocess.Popen([sys.executable, 'solution.py', 'task.md'], cwd=task,
                env=env, preexec_fn=inherited_limit, start_new_session=True,
                stdout=stream, stderr=subprocess.STDOUT)
            try:
                while proc.poll() is None:
                    text = log.read_text()
                    if text.count('"accepted": true') >= 2 or text.count('"event": "experiment"') >= 2 and arm == 'patched':
                        termination = 'diagnostic_two_fits_SIGTERM'
                        proc.terminate()
                        break
                    if time.monotonic()-start > 120:
                        termination = 'diagnostic_guard_SIGTERM'
                        proc.terminate()
                        break
                    time.sleep(.1)
                proc.wait(timeout=10)
            finally:
                if proc.poll() is None:
                    stop_process(proc)
        summaries = list((task/'.atlas').glob('*/summary.json'))
        summary = json.loads(summaries[0].read_text()) if len(summaries) == 1 else {}
        events = []
        if summaries and (summaries[0].parent/'experiments.jsonl').exists():
            events = [json.loads(line) for line in (summaries[0].parent/'experiments.jsonl').read_text().splitlines()]
        result = {'arm': arm, 'inherited_rlimit_as_bytes': 8*2**30,
            'worker_requested_memory_mb': 24576, 'returncode': proc.returncode,
            'termination': termination, 'wall_seconds': time.monotonic()-start,
            'output_exists': (task/'submission.csv').exists(),
            'fit_success': sum(event['status'] == 'ok' for event in events),
            'trained_champion_exists': bool(summary.get('champion_recipe')),
            'errors': [{'status': event['status'], 'exception_class': event.get('exception_class'),
                        'message': event.get('message'), 'traceback_excerpt': event.get('traceback_excerpt')}
                       for event in events if event['status'] != 'ok'],
            'summary': summary, 'run_dir': str(task)}
        rows.append(result)
        report.write_text(json.dumps({'status': 'running', 'runs': rows}, indent=2)+'\n')
        print(json.dumps({k: result[k] for k in ['arm', 'returncode', 'fit_success', 'trained_champion_exists', 'output_exists', 'termination']}), flush=True)
    report.write_text(json.dumps({'status': 'completed', 'runs': rows,
        'actual_platform_rlimit_known': False,
        'scope': 'Controlled finite-address-space-limit reproduction, not asserted actual platform root cause'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
