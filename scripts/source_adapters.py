"""Development-only bounded source downloads; never imported by the agent.

Metadata is data: download descriptions and embedded commands are never executed.
BeyondArena revision and original outer split files are retained verbatim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'artifacts/data/benchmark_v2'
BEYOND_REVISION = '2ecfe882ccfb814fc27c4de10a64ceefd5d7655c'


def download(url, path, max_bytes=64_000_000):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record_path = path.with_name(path.name + '.provenance.json')
    if path.exists() and record_path.exists():
        record = json.loads(record_path.read_text())
        if record['url'] != url or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Cached source identity mismatch: ' + str(path))
        return record
    temporary = path.with_name(path.name + '.part')
    size = 0
    start = time.monotonic()
    try:
        with httpx.stream('GET', url, follow_redirects=True, timeout=20,
                          headers={'User-Agent': 'ATLAS-offline-benchmark/2'}) as response:
            response.raise_for_status()
            with temporary.open('wb') as output:
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > max_bytes or time.monotonic() - start > 90:
                        raise ValueError('Source download exceeded its size/time bound')
                    output.write(chunk)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    record = {'url': url, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'bytes': path.stat().st_size, 'download_seconds': round(time.monotonic() - start, 3)}
    record_path.write_text(json.dumps(record, indent=2) + '\n')
    return record


def json_download(url, path, max_bytes=8_000_000):
    provenance = download(url, path, max_bytes)
    return json.loads(Path(path).read_text()), provenance


def beyond_metadata(name):
    if not name.replace('_', '').isalnum():
        raise ValueError('Invalid curated dataset name')
    directory = CACHE / 'beyondarena' / name
    tree, _ = json_download(
        f'https://huggingface.co/api/datasets/TabArena/BeyondArena/tree/{BEYOND_REVISION}/{name}?recursive=true&limit=1000',
        directory / 'tree.json')
    containers = sorted(row['path'] for row in tree if row['path'].endswith('/container_metadata.json'))
    if not containers:
        raise ValueError('No curated container found: ' + name)
    prefix = containers[-1].rsplit('/', 1)[0]
    files = [row for row in tree if row['type'] == 'file' and row['path'].rsplit('/', 1)[0] == prefix]
    metadata = {}
    provenance = {}
    for row in files:
        filename = row['path'].rsplit('/', 1)[-1]
        if not filename.endswith('.json'):
            continue
        value, record = json_download(
            f'https://huggingface.co/datasets/TabArena/BeyondArena/resolve/{BEYOND_REVISION}/{row["path"]}',
            directory / filename,max_bytes=64_000_000 if filename.startswith('experiment_metadata.') else 8_000_000)
        metadata[filename] = value
        provenance[filename] = record
    dataset_key = next(key for key in metadata if key.startswith('dataset_metadata.'))
    task_key = next(key for key in metadata if key.startswith('task_metadata.'))
    experiment_key = next(key for key in metadata if key.startswith('experiment_metadata.'))
    dataset = metadata[dataset_key]
    task = metadata[task_key]
    parquet = next(row for row in files if row['path'].endswith('/dataset.parquet'))
    result = {'case': 'beyond_' + name, 'source': 'beyondarena', 'dataset': name,
              'revision': BEYOND_REVISION, 'container': prefix,
              'source_url': 'https://huggingface.co/datasets/TabArena/BeyondArena',
              'original_source': dataset.get('original_dataset_source_download_link'),
              'license': dataset.get('license'), 'target': task['target_column_name'],
              'problem_type': task['problem_type'], 'metric': task['objective_metric_name'],
              'split_regime': 'temporal' if task.get('time_on') else 'grouped' if task.get('group_on') else 'iid',
              'task_metadata': task, 'experiment_file': experiment_key,
              'parquet_bytes': parquet['size'], 'parquet_path': parquet['path'],
              'cache_directory': str(directory), 'metadata_provenance': provenance}
    (directory / 'adapter.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def beyond_data(record):
    if str(record.get('license','')).strip().lower() in {'','none','unknown','unspecified'}:
        raise ValueError('dataset_not_licensed: curated source has no license metadata')
    directory = Path(record['cache_directory'])
    provenance = download(
        f'https://huggingface.co/datasets/TabArena/BeyondArena/resolve/{record["revision"]}/{record["parquet_path"]}',
        directory / 'dataset.parquet')
    return directory / 'dataset.parquet', provenance


def openml_metadata(task_id):
    task_id = int(task_id)
    directory = CACHE / 'openml' / str(task_id)
    task, task_record = json_download(f'https://www.openml.org/api/v1/json/task/{task_id}', directory / 'task.json')
    task = task['task']
    source = next(item['data_set'] for item in task['input'] if item['name'] == 'source_data')
    data_id = int(source['data_set_id'])
    dataset, data_record = json_download(f'https://www.openml.org/api/v1/json/data/{data_id}', directory / 'dataset.json')
    dataset = dataset['data_set_description']
    procedure = next(item['estimation_procedure'] for item in task['input'] if item['name'] == 'estimation_procedure')
    result = {'case': 'openml_' + str(task_id), 'source': 'openml', 'task_id': task_id,
              'dataset': dataset['name'], 'dataset_id': data_id, 'version': dataset.get('version'),
              'source_url': 'https://www.openml.org/t/' + str(task_id), 'license': dataset.get('licence'),
              'target': source['target_feature'], 'problem_type': 'classification', 'metric': 'f1_macro',
              'metric_policy': 'Explicit Macro F1 study variant; preserves original OpenML outer splits',
              'split_regime': 'iid', 'data_url': dataset['url'], 'split_url': procedure['data_splits_url'],
              'cache_directory': str(directory), 'metadata_provenance': {'task': task_record, 'dataset': data_record}}
    (directory / 'adapter.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def openml_data(record):
    if str(record.get('license','')).strip().lower() in {'','none','unknown','unspecified'}:
        raise ValueError('dataset_not_licensed: OpenML source has no license metadata')
    directory = Path(record['cache_directory'])
    data_record = download(record['data_url'], directory / 'dataset.arff')
    split_record = download(record['split_url'], directory / 'splits.arff', max_bytes=128_000_000)
    return directory / 'dataset.arff', directory / 'splits.arff', {'dataset': data_record, 'splits': split_record}
