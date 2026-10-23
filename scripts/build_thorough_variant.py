"""Build an opt-in longer-search archive without changing the verified release.

The frozen P3j source is the parent. Only its search patience/trial bounds and
entrypoint experiment cap change. This is a research candidate, not a promotion.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PARENT_SHA = 'daa1087f405e21a698375e81e1960c422bd29b6363d2e7c2dc8f36e807226b8b'
DESTINATION = ROOT / '.bootstrap/thorough-v1/source'


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Frozen source does not match the reviewed patch')
    return source.replace(before, after, 1)


def extend_search(source):
    source = replace_once(source,
        'self.research_limit=min(192,max(48,int(budget.total/30)))',
        'self.research_limit=(min(384,max(96,int(budget.total/15))) if budget.total>=1800 '
        'else min(192,max(48,int(budget.total/30))))')
    source = replace_once(source,
        'limit=18 if self.budget.total<1800 else min(48,24+int(self.budget.total/900))',
        'limit=18 if self.budget.total<1800 else min(96,48+int(self.budget.total/450))')
    return replace_once(source,
        'if not self.hpo.enabled and no_gain>=10:',
        'if not self.hpo.enabled and no_gain>=(20 if self.budget.total>=1800 else 10):')


def main():
    parent = ROOT / '.bootstrap/performance_v2/p3j/source'
    archive = parent / 'submission.zip'
    if hashlib.sha256(archive.read_bytes()).hexdigest() != PARENT_SHA:
        raise RuntimeError('Unexpected parent archive')
    if (ROOT / 'submission.zip').read_bytes() != archive.read_bytes():
        raise RuntimeError('Keep the verified root release unchanged')
    output = ROOT / 'submission_thorough.zip'
    if output.exists():
        raise RuntimeError('Do not replace an existing experimental archive')
    if DESTINATION.exists():
        raise RuntimeError('Variant already frozen; do not overwrite its evidence')
    runtime_files = [parent / 'solution.py', *sorted((parent / 'atlas').glob('*.py'))]
    with zipfile.ZipFile(archive) as frozen:
        for path in runtime_files:
            name = '_atlas_entry.py' if path.name == 'solution.py' else 'atlas_' + path.name
            if frozen.read(name) != path.read_bytes():
                raise RuntimeError('Frozen runtime differs from its archive: ' + path.name)
    DESTINATION.mkdir(parents=True)
    for path in runtime_files:
        target = DESTINATION / path.relative_to(parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    search = DESTINATION / 'atlas/search.py'
    search.write_text(extend_search(search.read_text()))
    entry = DESTINATION / 'solution.py'
    entry.write_text(replace_once(entry.read_text(),
        "'task.md', started=STARTED)", "'task.md', started=STARTED, max_experiments=480)"))
    # Dev helpers and tests stay outside the uploaded archive.
    for directory in ['scripts', 'tests']:
        shutil.copytree(ROOT / directory, DESTINATION / directory,
                        ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['requirements-folder.txt', 'requirements.txt']:
        target = DESTINATION / 'artifacts/official' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'artifacts/official' / name, target)
    (DESTINATION / 'reports').mkdir()
    with (DESTINATION.parent / 'package.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/package_submission.py'], cwd=DESTINATION,
                       stdout=log, stderr=log, check=True)
    manifest = json.loads((DESTINATION / 'reports/release-manifest.json').read_text())
    shutil.copy2(DESTINATION / 'submission.zip', output)
    manifest.update(parent_zip_sha256=PARENT_SHA, variant='thorough-v1',
        status='FROZEN_FOR_TESTING; generalization benefit unverified',
        changed_runtime_files=['solution.py', 'atlas/search.py'],
        max_experiments=480, platform_budget_seconds=10500, platform_margin_seconds=300,
        search_steps_at_10500=384, stagnation_patience_at_10500=71,
        unchanged=['validation', 'models', 'metrics', 'finalization reserves',
                   'Qwen call limits', 'worker timeout and memory limits'],
        source_directory=str(DESTINATION), archive=str(output),
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report = ROOT / 'reports/thorough-v1'
    report.mkdir(exist_ok=True)
    (report / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({k: manifest[k] for k in ['variant', 'sha256', 'runtime_code_sha256',
                                             'bytes', 'source_directory', 'archive']}, indent=2))


if __name__ == '__main__':
    main()
