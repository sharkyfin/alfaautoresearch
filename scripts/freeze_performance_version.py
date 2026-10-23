"""Preserve an immutable research version and build its flat ZIP in isolation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from atlas.utils import code_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]+', args.label):
        parser.error('Use a simple unique version label')
    destination = ROOT / '.bootstrap/performance_v2' / args.label / 'source'
    if destination.exists():
        parser.error('Version already exists; never overwrite an experimental baseline')
    before = code_digest()
    destination.mkdir(parents=True)
    files = [ROOT / 'solution.py']
    for directory in ['atlas', 'scripts', 'tests', 'benchmarks']:
        files.extend(path for path in (ROOT / directory).glob('*') if path.is_file() and path.suffix in {'.py', '.json'})
    files.extend(ROOT / 'artifacts/official' / name for name in ['requirements-folder.txt', 'requirements.txt'])
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / 'reports').mkdir()
    with (destination.parent / 'package.log').open('w') as log:
        subprocess.run([sys.executable, 'scripts/package_submission.py'], cwd=destination,
            stdout=log, stderr=log, check=True)
    manifest = json.loads((destination / 'reports/release-manifest.json').read_text())
    after = code_digest()
    assert before == after == manifest['runtime_code_sha256'], 'Runtime changed during freeze'
    manifest.update(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        version_label=args.label, project_copy=str(destination),
        dev_script_hashes={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (destination / 'scripts').glob('*.py')})
    report = ROOT / 'reports/performance_v2' / (args.label + '-source-manifest.json')
    report.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({key: manifest[key] for key in ['version_label', 'runtime_code_sha256', 'sha256', 'bytes', 'project_copy']}, indent=2))


if __name__ == '__main__':
    main()
