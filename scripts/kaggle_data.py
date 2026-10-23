"""Prepare reproducible development tasks and sealed labels from public data."""
from __future__ import annotations

import ast
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = ('train.csv', 'test.csv', 'data_dictionary.csv', 'sample_submission.csv', 'task.md')


def read_dataset(archive, target):
    with zipfile.ZipFile(archive) as z:
        candidates = sorted((name for name in z.namelist() if Path(name).suffix.lower() in {'.csv', '.arff'}),
                            key=lambda name: (Path(name).suffix.lower() != '.csv', name))
        for name in candidates:
            content = z.read(name)
            if name.lower().endswith('.arff'):
                from scipy.io.arff import loadarff
                rows, _ = loadarff(io.StringIO(content.decode('utf-8-sig')))
                frame = pd.DataFrame(rows)
                for col in frame.select_dtypes(include='object'):
                    frame[col] = frame[col].map(lambda value: value.decode() if isinstance(value, bytes) else value)
            else:
                frame = pd.read_csv(io.BytesIO(content), skipinitialspace=True, na_values=['?', ' ?'])
            frame.columns = [str(col).strip().lstrip('\ufeff') for col in frame]
            match = next((col for col in frame if col.casefold() == target.casefold()), None)
            if match is None:
                continue
            if match != target:
                frame = frame.rename(columns={match: target})
            for col in frame.select_dtypes(include='object'):
                frame[col] = frame[col].str.strip().replace('?', np.nan)
            frame = frame.loc[frame[target].notna()].reset_index(drop=True)
            if len(frame) < 50 or frame.columns.duplicated().any():
                raise ValueError('Dataset has insufficient rows or duplicate column names')
            return frame, name
    raise ValueError('No CSV/ARFF file with target ' + target + ' in ' + str(archive))


def frozen_splits(frame, target, classification, seed):
    """60/20/20 by feature groups; no exact duplicate crosses a boundary."""
    indices = np.arange(len(frame))
    groups = pd.util.hash_pandas_object(frame.drop(columns=target), index=False).to_numpy()
    unique = len(np.unique(groups)) == len(frame)
    y = frame[target].astype(str).to_numpy() if classification else None
    if classification and frame[target].value_counts().min() < 5:
        raise ValueError('At least five examples per class are required for this benchmark')
    if unique:
        fit_dev, lockbox = train_test_split(indices, test_size=.2, random_state=seed, stratify=y)
        train, dev = train_test_split(fit_dev, test_size=.25, random_state=seed+1,
                                    stratify=y[fit_dev] if classification else None)
    elif classification:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_dev, lockbox = next(splitter.split(indices, y, groups))
        splitter = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed+1)
        a, b = next(splitter.split(fit_dev, y[fit_dev], groups[fit_dev]))
        train, dev = fit_dev[a], fit_dev[b]
    else:
        fit_dev, lockbox = next(GroupShuffleSplit(test_size=.2, random_state=seed).split(indices, groups=groups))
        a, b = next(GroupShuffleSplit(test_size=.25, random_state=seed+1).split(fit_dev, groups=groups[fit_dev]))
        train, dev = fit_dev[a], fit_dev[b]
    for a, b in [(train, dev), (train, lockbox), (dev, lockbox)]:
        if set(groups[a]) & set(groups[b]):
            raise ValueError('Duplicate group crosses an external split')
    if classification:
        if set(y[train]) != set(y) or set(y[dev]) != set(y) or set(y[lockbox]) != set(y):
            raise ValueError('An external split lost a class')
    return train, dev, lockbox


def prepare(case, spec, archive, destination, seed=42, final=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    frame, member = read_dataset(archive, spec['target'])
    target = spec['target']
    classification = spec['metric'] == 'f1_macro'
    train, dev, lockbox = frozen_splits(frame, target, classification, seed)
    fit, test = (np.concatenate([train, dev]), lockbox) if final else (train, dev)
    id_column = 'benchmark_id'
    while id_column in frame:
        id_column = '_' + id_column
    ids = pd.Series([f'{case}:{i:08d}' for i in range(len(frame))])
    inputs = destination/'inputs'
    sealed = destination/'sealed'
    inputs.mkdir(); sealed.mkdir()
    train_frame = frame.iloc[fit].copy()
    test_frame = frame.iloc[test].drop(columns=target).copy()
    train_frame.insert(0, id_column, ids.iloc[fit].to_numpy())
    test_frame.insert(0, id_column, ids.iloc[test].to_numpy())
    train_frame.to_csv(inputs/'train.csv', index=False)
    test_frame.to_csv(inputs/'test.csv', index=False)
    sample = pd.DataFrame({id_column: ids.iloc[test].to_numpy(),
                           target: str(frame[target].iloc[train[0]]) if classification else 0.})
    sample.to_csv(inputs/'sample_submission.csv', index=False)
    dictionary = []
    for column in train_frame:
        role = 'id' if column == id_column else 'target' if column == target else 'feature'
        dtype = 'numeric' if pd.api.types.is_numeric_dtype(train_frame[column]) else 'category'
        dictionary.append({'feature': column, 'role': role, 'dtype': dtype,
                           'description_ru': spec.get('columns', {}).get(column, column)})
    pd.DataFrame(dictionary).to_csv(inputs/'data_dictionary.csv', index=False)
    metric = 'Macro F1' if classification else 'RMSE'
    description = f"Task: {'classification' if classification else 'regression'}\nPrimary metric: {metric}\n{spec['description']}\n"
    description += 'Inputs: `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv`.\n'
    description += 'Write `submission.csv` next to task.md, preserving sample IDs, row order and column names.\n'
    (inputs/'task.md').write_text(description, encoding='utf-8')
    pd.DataFrame({id_column: ids.iloc[test].to_numpy(), target: frame[target].iloc[test].to_numpy()}).to_csv(sealed/'evaluation.csv', index=False)
    # The reserved final set is not exposed in quick runs, including its feature rows.
    np.savez(sealed/'split_indices.npz', train=train, development=dev, lockbox=lockbox)
    split_hash = hashlib.sha256(b''.join(np.asarray(part, dtype=np.int64).tobytes() for part in [train, dev, lockbox])).hexdigest()
    metadata = {'case': case, 'dataset': spec['dataset'], 'source': 'https://www.kaggle.com/datasets/'+spec['dataset'],
                'raw_sha256': hashlib.sha256(Path(archive).read_bytes()).hexdigest(), 'member': member,
                'target': target, 'id_column': id_column, 'metric': spec['metric'],
                'seed': seed, 'phase': 'final_lockbox' if final else 'development',
                'rows': len(frame), 'training_rows': len(fit), 'evaluation_rows': len(test),
                'reserved_final_rows': len(lockbox), 'features': len(frame.columns)-1,
                'split_sha256': split_hash,
                'input_sha256': {name: hashlib.sha256((inputs/name).read_bytes()).hexdigest() for name in INPUT_FILES},
                'labels': sorted(frame[target].astype(str).unique().tolist()) if classification else [],
                'labels_outside_task_directory': True, 'feature_duplicate_groups_kept_together': True}
    (destination/'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    return metadata


def snapshot_archive(archive, destination):
    """Materialize the exact hashed Python files from the flat submitted ZIP."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        tree = ast.parse(z.read('solution.py').decode())
        assignment = next((node for node in tree.body if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets)), None)
        if assignment is None:
            raise ValueError('Expected a current flat upload with a SOURCES integrity manifest')
        mapping = ast.literal_eval(assignment.value)
        for filename, (relative, expected) in mapping.items():
            path = PurePosixPath(relative)
            if path.is_absolute() or '..' in path.parts or path.suffix != '.py' or Path(filename).name != filename:
                raise ValueError('Unsafe source mapping in ZIP')
            content = z.read(filename)
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Archive source integrity mismatch: '+filename)
            output = destination.joinpath(*path.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
    digest = hashlib.sha256()
    for source in [destination/'solution.py', *sorted((destination/'atlas').glob('*.py'))]:
        digest.update(source.name.encode()); digest.update(source.read_bytes())
    (destination/'scripts').mkdir()
    shutil.copy2(ROOT/'scripts/kaggle_worker.py', destination/'scripts/kaggle_worker.py')
    if (ROOT/'scripts/backend_worker.py').exists():shutil.copy2(ROOT/'scripts/backend_worker.py', destination/'scripts/backend_worker.py')
    return digest.hexdigest()


def evaluate_output(task_root, sealed_path, metadata):
    """Only the evaluator reads these labels, after the agent process has exited."""
    from sklearn.metrics import f1_score, mean_squared_error
    sample = pd.read_csv(Path(task_root)/'sample_submission.csv', dtype=str)
    output = pd.read_csv(Path(task_root)/'submission.csv', dtype=str)
    truth = pd.read_csv(sealed_path, dtype=str)
    id_column, target = metadata['id_column'], metadata['target']
    if list(output) != list(sample) or len(output) != len(sample) or not output[id_column].equals(sample[id_column]):
        raise ValueError('Invalid output schema, row count or ID order')
    if not truth[id_column].equals(output[id_column]) or output.isna().any().any():
        raise ValueError('Labels and output IDs differ or predictions are missing')
    if metadata['metric'] == 'rmse':
        y, prediction = pd.to_numeric(truth[target]).to_numpy(), pd.to_numeric(output[target]).to_numpy()
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite predictions')
        return float(np.sqrt(mean_squared_error(y, prediction)))
    if not set(output[target]) <= set(metadata['labels']):
        raise ValueError('Output contains unknown class labels')
    return float(f1_score(truth[target], output[target], labels=metadata['labels'], average='macro', zero_division=0))
