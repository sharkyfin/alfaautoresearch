"""Common dev-only task preparation and in-memory independent evaluation.

Only input CSVs are written into a task. Evaluation answers remain in the
orchestrating evaluator process and are never arguments, environment or prompts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atlas.metrics import Metric, infer_metric
from scripts.kaggle_data import INPUT_FILES
from scripts.source_adapters import beyond_data, openml_data

SEEDS = [42, 53, 71]


def canonical_metric(name):
    metric, _ = infer_metric('Primary metric: ' + name)
    if metric is None:
        raise ValueError('unsupported_metric: ' + name)
    return metric


def columns(value):
    return [] if value is None else [value] if isinstance(value, str) else list(value)


def read_arff(path):
    from scipy.io.arff import loadarff
    values, _ = loadarff(path)
    frame = pd.DataFrame(values)
    for column in frame.select_dtypes(include='object'):
        frame[column] = frame[column].map(lambda value: value.decode() if isinstance(value, bytes) else value)
    return frame


def source_split(record, seed):
    if seed not in SEEDS:
        raise ValueError('Use a predeclared model/split selector seed: 42, 53 or 71')
    slot = SEEDS.index(seed)
    if record['source']=='official_public':
        from scripts.kaggle_data import frozen_splits
        path=Path(record['data_path'])
        frame=pd.read_csv(path).drop(columns=record.get('id_columns',[]))
        fit,development,held=frozen_splits(frame,record['target'],record['problem_type']!='regression',seed)
        training=np.sort(np.r_[fit,development])
        return frame,training,held,{'official_repeat':None,'official_fold':None,
            'split_selector_policy':'train-only 80/20 external holdout, exact feature duplicates grouped; original organizer test labels unavailable',
            'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'official_splits_sha256':None}
    if record['source'] == 'beyondarena':
        path, provenance = beyond_data(record)
        frame = pd.read_parquet(path)
        splits = json.loads((Path(record['cache_directory']) / record['experiment_file']).read_text())['splits']
        repeats = sorted(splits, key=int)
        if slot >= len(repeats):
            raise ValueError('Source has no independent official repeat for this seed selector')
        repeat = repeats[slot]
        fold = sorted(splits[repeat], key=int)[0]
        fit, held = splits[repeat][fold]
        return frame, np.asarray(fit), np.asarray(held), {
            'official_repeat': repeat, 'official_fold': fold,
            'split_selector_policy': '42/53/71 select official repeats 0/1/2; no regeneration',
            'dataset_sha256': provenance['sha256'],
            'official_splits_sha256': record['metadata_provenance'][record['experiment_file']]['sha256']}
    if record['source'] == 'openml':
        dataset, split_file, provenance = openml_data(record)
        frame = read_arff(dataset)
        splits = read_arff(split_file)
        part = splits[(splits.repeat.astype(int) == 0) & (splits.fold.astype(int) == slot)]
        fit = part.loc[part.type.str.upper() == 'TRAIN', 'rowid'].to_numpy(dtype=int)
        held = part.loc[part.type.str.upper() == 'TEST', 'rowid'].to_numpy(dtype=int)
        if not len(fit) or not len(held):
            raise ValueError('Missing original OpenML split')
        return frame, fit, held, {'official_repeat': 0, 'official_fold': slot,
            'split_selector_policy': '42/53/71 select original OpenML folds 0/1/2',
            'dataset_sha256': provenance['dataset']['sha256'],
            'official_splits_sha256': provenance['splits']['sha256']}
    raise ValueError('Unknown adapter source: ' + record['source'])


def prepare_external(record, destination, seed=42, phase='development'):
    if record['reserved_phase'] != phase:
        raise ValueError('Dataset reservation does not permit this evaluation phase')
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    frame, fit, held, provenance = source_split(record, seed)
    if set(fit) & set(held) or not len(fit) or not len(held):
        raise ValueError('Invalid original outer split')
    if np.min(np.r_[fit, held]) < 0 or np.max(np.r_[fit, held]) >= len(frame):
        raise ValueError('Original outer indices do not match source rows')
    target = record['target']
    if frame[target].isna().any():
        raise ValueError('Curated source has missing target; cannot silently change original splits')
    train = frame.iloc[fit].copy()
    test = frame.iloc[held].drop(columns=target).copy()
    task = record.get('task_metadata', {})
    groups = columns(task.get('group_on'))
    times = columns(task.get('time_on'))
    for group in groups:
        if set(train[group].astype(str)) & set(test[group].astype(str)):
            raise ValueError('Original curated group overlap: ' + group)
    for column in times:
        dates = pd.to_datetime(frame[column], utc=True, errors='raise')
        if dates.iloc[fit].max() >= dates.iloc[held].min():
            raise ValueError('Original temporal boundary overlaps: ' + column)
    id_column = '__benchmark_id__'
    while id_column in frame:
        id_column = '_' + id_column
    train.insert(0, id_column, [f'{record["case"]}:{index:09d}' for index in fit])
    test.insert(0, id_column, [f'{record["case"]}:{index:09d}' for index in held])
    classification = record['problem_type'] != 'regression'
    metric_name = canonical_metric(record.get('metric_override', record['metric']))
    labels = sorted(train[target].unique().tolist()) if classification else []
    # This is train-derived vocabulary, not an inspection of evaluation labels.
    metric = Metric(metric_name, len(labels))
    if classification and metric.needs_proba:
        if len(labels) == 2 and metric_name != 'logloss':
            sample = pd.DataFrame({id_column: test[id_column], target: .5})
        else:
            sample = pd.DataFrame({id_column: test[id_column], **{str(label): 1 / len(labels) for label in labels}})
    else:
        sample = pd.DataFrame({id_column: test[id_column], target: labels[0] if classification else 0.})
    dictionary = []
    original_dictionary={}
    if record.get('dictionary_path'):
        original_dictionary=pd.read_csv(record['dictionary_path']).fillna('').set_index('feature').to_dict('index')
    for column in train:
        role = 'id' if column == id_column else 'target' if column == target else 'ignore' if column in groups else 'feature'
        dtype = 'datetime' if column in times or pd.api.types.is_datetime64_any_dtype(train[column]) else (
            'categorical' if isinstance(train[column].dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(train[column]) else
            'numeric' if pd.api.types.is_numeric_dtype(train[column]) else 'string')
        semantic=original_dictionary.get(column,{})
        dictionary.append({'feature': column, 'role': role, 'dtype': semantic.get('dtype') or dtype,
            'validation_role': 'group' if column in groups else 'time' if column in times else '',
            'description_ru': semantic.get('description_ru') or column,'unit':semantic.get('unit','')})
    description = f'Task: {"classification" if classification else "regression"}\nPrimary metric: {metric_name}\n'
    if groups:
        description += 'validation: group; test contains unseen entities. Group identifiers are validation metadata, excluded from model features.\n'
    if times:
        description += 'validation: time; forecast future observations from earlier training rows.\n'
    description += 'Inputs: `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv`.\nWrite `submission.csv`.\n'
    train.to_csv(destination / 'train.csv', index=False)
    test.to_csv(destination / 'test.csv', index=False)
    sample.to_csv(destination / 'sample_submission.csv', index=False)
    pd.DataFrame(dictionary).to_csv(destination / 'data_dictionary.csv', index=False)
    (destination / 'task.md').write_text(description)
    truth = pd.DataFrame({id_column: test[id_column].to_numpy(), target: frame[target].iloc[held].to_numpy()})
    if task.get('group_labels') == 'per_group' and groups:
        truth['__evaluation_group__'] = frame.iloc[held][groups].astype(str).agg('::'.join, axis=1).to_numpy()
    split_hash = hashlib.sha256(np.asarray(fit, dtype='<i8').tobytes() + np.asarray(held, dtype='<i8').tobytes()).hexdigest()
    features = [row['feature'] for row in dictionary if row['role'] == 'feature']
    supports = train[target].value_counts() if classification else None
    metadata = {key: record.get(key) for key in ['case', 'source', 'dataset', 'revision', 'container', 'task_id', 'dataset_id', 'version', 'license', 'source_url', 'split_regime']}
    metadata.update(provenance, target=target, id_column=id_column, metric=metric_name,
        original_source_metric=record['metric'], metric_override=record.get('metric_override'), labels=labels,
        phase=phase, seed=seed, split_sha256=split_hash, raw_sha256=provenance['dataset_sha256'],
        training_rows=len(fit), evaluation_rows=len(held), features=len(features), n_classes=len(labels),
        missingness=float(train[features].isna().to_numpy().mean()) if features else 0.,
        imbalance_ratio=float(supports.max() / supports.min()) if classification else None,
        train_class_support={str(key):int(value) for key,value in supports.items()} if classification else None,
        evaluation_unit='group_mean_prediction' if '__evaluation_group__' in truth else 'row',
        input_sha256={name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in INPUT_FILES},
        labels_storage='parent evaluator memory; no evaluation-label file or child argument')
    return metadata, truth


def score_details(actual,prediction,metric,labels):
    if labels:
        from sklearn.metrics import classification_report,confusion_matrix
        hard=np.asarray(prediction).argmax(1)
        return {'per_class':classification_report(actual,hard,labels=np.arange(len(labels)),
            target_names=labels,output_dict=True,zero_division=0),
            'confusion_matrix':confusion_matrix(actual,hard,labels=np.arange(len(labels))).tolist() if len(labels)<=32 else None}
    residual=np.asarray(prediction)-np.asarray(actual);edges=np.unique(np.quantile(actual,np.linspace(0,1,11)))
    bins=[]
    for i in range(len(edges)-1):
        mask=(actual>=edges[i])&(actual<=edges[i+1] if i==len(edges)-2 else actual<edges[i+1])
        if mask.any():bins.append({'target_range':[float(edges[i]),float(edges[i+1])],
            'support':int(mask.sum()),'rmse':float(np.sqrt(np.mean(residual[mask]**2))),
            'mae':float(np.mean(abs(residual[mask]))),'bias':float(residual[mask].mean())})
    return {'target_deciles':bins,'residual_bias':float(residual.mean())}


def evaluate_memory(task_root, truth, metadata):
    """A single trusted Metric implementation; evaluation is after child exit."""
    task_root = Path(task_root)
    sample = pd.read_csv(task_root / 'sample_submission.csv', dtype=str)
    output = pd.read_csv(task_root / 'submission.csv', dtype=str)
    identity, target = metadata['id_column'], metadata['target']
    if list(output) != list(sample) or not output[identity].equals(sample[identity]) or len(output) != len(truth):
        raise ValueError('Output schema/ID order differs from the task contract')
    if not output[identity].equals(truth[identity].astype(str)) or output.isna().any().any():
        raise ValueError('Evaluation labels/IDs or missing predictions mismatch')
    labels = [str(label) for label in metadata.get('labels', [])]
    metric = Metric(metadata['metric'], len(labels))
    if labels:
        mapping = {label: index for index, label in enumerate(labels)}
        actual = truth[target].astype(str).map(mapping)
        if actual.isna().any():
            raise ValueError('Evaluation contains a class absent from supplied training labels')
        actual = actual.to_numpy(dtype=int)
        if metric.needs_proba:
            if len(labels) == 2 and metadata['metric'] != 'logloss':
                positive = pd.to_numeric(output[target], errors='raise').to_numpy()
                prediction = np.column_stack([1 - positive, positive])
            else:
                prediction = output[labels].apply(pd.to_numeric, errors='raise').to_numpy()
        else:
            hard = output[target].map(mapping)
            if hard.isna().any():
                raise ValueError('Invalid submitted class label')
            prediction = np.eye(len(labels))[hard.to_numpy(dtype=int)]
    else:
        actual = pd.to_numeric(truth[target], errors='raise').to_numpy()
        prediction = pd.to_numeric(output[target], errors='raise').to_numpy()
    row_score = metric.score(actual, prediction)
    if '__evaluation_group__' not in truth:
        return row_score, {'row_metric': row_score,**score_details(actual,prediction,metric,labels)}
    groups = truth['__evaluation_group__'].to_numpy()
    group_actual = []
    group_prediction = []
    for group in pd.unique(groups):
        mask = groups == group
        values = actual[mask]
        if len(np.unique(values)) != 1:
            raise ValueError('per_group target varies inside an evaluation group')
        group_actual.append(values[0])
        group_prediction.append(prediction[mask].mean(axis=0))
    value = metric.score(np.asarray(group_actual), np.asarray(group_prediction))
    return value, {'row_metric': row_score, 'group_count': len(group_actual),
                   'scoring_policy': 'One equally weighted group; mean row predictions; constant group label',
                   **score_details(np.asarray(group_actual),np.asarray(group_prediction),metric,labels)}
