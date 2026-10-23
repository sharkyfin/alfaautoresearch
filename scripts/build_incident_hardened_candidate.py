"""Freeze a narrowly hardened candidate from the exact zero-scoring upload.

No model, validation, metric or search-quality changes. The primary archive is
preserved while the incident candidate is tested independently.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ZERO_SHA = '469c8c90430d4ac25c9a125ed125c110a503772a1fba496afcb8da996dff15d2'


def once(text, before, after):
    if text.count(before) != 1:
        raise ValueError('Unexpected archived source; do not apply an ambiguous patch')
    return text.replace(before, after, 1)


def main():
    private = ROOT/'.bootstrap/incident-zero-20261023'
    archive = private/'platform-zero.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == ZERO_SHA
    source = private/'hardened/source'
    assert not source.exists(), 'Keep frozen diagnostic versions immutable'
    source.mkdir(parents=True)
    with zipfile.ZipFile(archive) as payload:
        assignment = next(node.value for node in ast.parse(payload.read('solution.py')).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id=='SOURCES' for target in node.targets))
        for name, (relative, expected) in ast.literal_eval(assignment).items():
            content = payload.read(name)
            assert hashlib.sha256(content).hexdigest() == expected
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    entry = source/'solution.py'
    entry.write_text(once(entry.read_text(),
        'os.environ[variable] = str(min(8, os.cpu_count() or 1))',
        "# The controller evaluates and schedules; model subprocesses receive\n"
        "    # their own full thread allowance in budget.run_process. Idle BLAS\n"
        "    # pools here must not exhaust the container's PID/thread quota.\n"
        "    os.environ[variable] = '1'"))
    worker = source/'atlas/worker.py'
    text = worker.read_text()
    text = once(text, '\n\ndef execute(job_path,feature_function=None):', '''

def restrict_address_space(memory_mb):
    """Respect inherited finite limits; never attempt to raise a hard limit."""
    requested = int(memory_mb)*2**20
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    finite = [value for value in (soft, hard) if value != resource.RLIM_INFINITY]
    applied = min([requested, *finite])
    resource.setrlimit(resource.RLIMIT_AS, (applied, applied))
    return {'requested_bytes': requested, 'applied_bytes': applied,
            'inherited_soft_bytes': soft, 'inherited_hard_bytes': hard}


def execute(job_path,feature_function=None):''')
    text = once(text, "    watch_parent(job.get('controller_pid'))\n    try:\n", "    try:\n        watch_parent(job.get('controller_pid'))\n")
    text = once(text,
        "        if os.sys.platform=='linux':resource.setrlimit(resource.RLIMIT_AS,(job['memory_mb']*2**20,job['memory_mb']*2**20))",
        "        limits=restrict_address_space(job['memory_mb']) if os.sys.platform=='linux' else {}")
    text = once(text, "        info['warnings']=[str(w.message)[:500] for w in caught[:20]]",
        "        info['resource_limits']=limits\n        info['warnings']=[str(w.message)[:500] for w in caught[:20]]")
    worker.write_text(text)
    orchestration = source/'atlas/orchestration.py'
    orchestration.write_text(once(orchestration.read_text(),
        "        return 0\n    except Exception as error:\n        # The valid atomic output exists.",
        "        if not champion and not stop['requested'] and not any(row['status']=='ok' for row in history):\n"
        "            fatal_error={'exception_class':'NoSuccessfulTraining','message':'No worker completed training; the retained CSV is only an emergency constant fallback'}\n"
        "            emit('training_unavailable',failed_experiments=len(history),status='fatal_with_fallback')\n"
        "            return 1\n"
        "        return 0\n    except Exception as error:\n        # The valid atomic output exists."))
    for name in ['scripts', 'tests']:
        shutil.copytree(ROOT/name, source/name, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'scripts/incident_guard_tests.py', source/'tests/test_incident_runtime.py')
    for name in ['requirements-folder.txt', 'requirements.txt']:
        target = source/'artifacts/official'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'artifacts/official'/name, target)
    (source/'reports').mkdir()
    report = ROOT/'reports/incident_zero_20261023'
    with (report/'hardened-package.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/package_submission.py'], cwd=source,
            stdout=log, stderr=log, check=True)
    manifest = json.loads((source/'reports/release-manifest.json').read_text())
    output = ROOT/'submission_runtime_hardened.zip'
    assert not output.exists()
    shutil.copy2(source/'submission.zip', output)
    manifest.update(parent_archive_sha256=ZERO_SHA, source_directory=str(source),
        archive=str(output), status='FROZEN_FOR_INCIDENT_QA; actual platform cause unconfirmed',
        changed_files=['solution.py', 'atlas/worker.py', 'atlas/orchestration.py'],
        model_worker_thread_budget_unchanged=8, controller_blas_threads=1,
        actual_platform_limits_known=False, quality_advantage_claimed=False)
    (report/'hardened-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({key: manifest[key] for key in ['archive','sha256','runtime_code_sha256','bytes','source_directory']}, indent=2))


if __name__ == '__main__':
    main()
