"""Download public Kaggle datasets for development, outside the submitted agent.

No login, upload, submission or Kaggle credentials are used. Dataset archives are
cached with hashes. Raw benchmark data is ignored by git and excluded from ZIP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / 'benchmarks/kaggle_cases.json'


def download(case, spec, directory):
    import httpx
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / (case + '.zip')
    record_path = directory / (case + '.json')
    if archive.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Cached archive hash mismatch: ' + str(archive))
        return record
    url = 'https://www.kaggle.com/api/v1/datasets/download/' + spec['dataset']
    temporary = archive.with_suffix('.part')
    started = time.monotonic()
    try:
        size = 0
        with httpx.stream('GET', url, follow_redirects=True, timeout=60) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > 64_000_000:
                        raise ValueError('Development archive exceeds 64 MB bound')
                    output.write(chunk)
        if not zipfile.is_zipfile(temporary):
            raise ValueError('Kaggle returned a non-ZIP response; download the public dataset manually')
        with zipfile.ZipFile(temporary) as z:
            if sum(item.file_size for item in z.infolist()) > 128_000_000:
                raise ValueError('Uncompressed dataset exceeds 128 MB bound')
            files = [{'name': item.filename, 'bytes': item.file_size} for item in z.infolist() if not item.is_dir()]
        temporary.replace(archive)
    finally:
        temporary.unlink(missing_ok=True)
    record = {'case': case, 'dataset': spec['dataset'],
              'source': 'https://www.kaggle.com/datasets/' + spec['dataset'],
              'download_url': url, 'archive': archive.name,
              'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'bytes': archive.stat().st_size, 'files': files,
              'download_seconds': round(time.monotonic()-started, 3)}
    record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default='all')
    parser.add_argument('--directory', type=Path, default=ROOT/'artifacts/data/kaggle/raw')
    args = parser.parse_args()
    registry = json.loads(CASES.read_text())
    selected = list(registry) if args.cases == 'all' else args.cases.split(',')
    if any(name not in registry for name in selected):
        parser.error('Unknown case; choose from ' + ', '.join(registry))
    for name in selected:
        record = download(name, registry[name], args.directory)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
