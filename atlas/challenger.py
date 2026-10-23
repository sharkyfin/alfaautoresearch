"""Bounded independent CPU portfolio; search policy is frozen before controls."""
from __future__ import annotations
import copy, importlib.metadata, json, time
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = '1.3.1'
FAMILY = 'autogluon'

def availability():
    try:
        version = importlib.metadata.version('autogluon.tabular')
        # AG 1.3.1 load() imports pkg_resources; this exact dependency is in the official image.
        setuptools = importlib.metadata.version('setuptools')
        return version == VERSION and setuptools == '80.9.0'
    except importlib.metadata.PackageNotFoundError:
        return False


def plan(data, champion, budget, controls, available=None):
    """Account for search, two frozen controls and all-data refit, not one fit."""
    result = {'eligible': False, 'reason': 'no_trained_champion', 'reserve_seconds': 0.}
    if not champion: return result
    if available is None: available = availability()
    if not available: return {**result, 'reason': 'approved_autogluon_1.3.1_unavailable'}
    spec = data.spec
    if not controls or spec.validation_plan.get('kind') not in {'random', 'stratified', 'group', 'duplicate_group'}:
        return {**result, 'reason': 'time_or_unconfirmed_policy_not_verified'}
    try:
        for window in controls: verify_split(data,window.train,window.valid)
    except ValueError as error:
        return {**result,'reason':str(error)}
    if spec.text_candidates or len(data.y) > 200000 or len(spec.features) > 500 or len(data.y) < 100:
        return {**result, 'reason': 'dataset_cost_outside_bounded_classical_policy'}
    if spec.classification and (len(spec.labels) > 50 or any(set(data.y[s.train]) != set(range(len(spec.labels))) for s in controls)):
        return {**result, 'reason': 'class_support_not_comparable'}
    main_cost = max(budget.reserve, champion['runtime_sec'] * (len(controls) + 2) * 1.6)
    allocation = min(1800., budget.total * .30, max(0., budget.remaining - main_cost))
    cap = min(360., allocation / (len(controls) + 2.8))
    if cap < 20.: return {**result, 'reason': 'insufficient_confirmation_and_refit_budget'}
    return {'eligible': True, 'reason': 'bounded_classical_portfolio', 'worker_seconds': cap,
            'reserve_seconds': allocation, 'main_reserve_seconds': main_cost}



def verify_split(data, train_idx, valid_idx):
    train_idx=np.asarray(train_idx);valid_idx=np.asarray(valid_idx)
    if not len(train_idx) or not len(valid_idx) or len(np.intersect1d(train_idx,valid_idx)):
        raise ValueError('challenger_split_overlap_or_empty')
    if not np.array_equal(np.sort(np.concatenate((train_idx,valid_idx))),np.arange(len(data.y))):
        raise ValueError('challenger_split_does_not_cover_all_labels')
    plan=data.spec.validation_plan
    if plan.get('kind')=='group':
        column=plan.get('group_column')
        if column not in data.train: raise ValueError('challenger_group_column_missing')
        groups=data.train[column].fillna('__MISSING__').astype(str).to_numpy()
    elif plan.get('kind')=='duplicate_group':
        groups=pd.util.hash_pandas_object(data.train[data.spec.features],index=False).to_numpy()
    elif plan.get('kind') in {'random','stratified'}:groups=None
    else: raise ValueError('challenger_validation_policy_not_verified')
    if groups is not None and len(np.intersect1d(groups[train_idx],groups[valid_idx])):
        raise ValueError('challenger_validation_group_leakage')
    if data.spec.classification and set(data.y[train_idx])!=set(range(len(data.spec.labels))):
        raise ValueError('challenger_training_fold_lacks_global_classes')


def recipe(seconds, split_id):
    return {'model_family': FAMILY, 'params': {'profile': 'classical_frozen_v1'},
            'feature_ops': [], 'feature_expressions': [], 'target_transform': 'none',
            'max_runtime_sec': float(seconds), 'validation_plan_id': split_id,
            'statement': 'Independent classical AutoML portfolio; freeze selected models and weights before controls'}


def aligned_prediction(value, spec, rows):
    if not spec.classification:
        p = np.asarray(value, dtype=float).reshape(-1)
        if p.shape != (rows,): raise ValueError('Challenger regression size mismatch')
        if spec.metric.name == 'rmsle': p = np.maximum(p, 0.)
    else:
        if not hasattr(value, 'columns') or set(value.columns) != set(range(len(spec.labels))):
            raise ValueError('Challenger probability columns are not global encoded classes')
        p = value.reindex(columns=range(len(spec.labels))).to_numpy(dtype=float)
        if p.shape != (rows, len(spec.labels)) or (p < 0).any(): raise ValueError('Invalid challenger probabilities')
        if not np.isfinite(p).all() or (p.sum(axis=1) <= 0).any(): raise ValueError('Invalid challenger probability mass')
        p = p / p.sum(axis=1, keepdims=True)
    if not np.isfinite(p).all(): raise ValueError('Nonfinite challenger predictions')
    return p


class CanonicalScore:
    """Picklable bridge: the exact trusted TaskSpec scorer, including class order."""
    def __init__(self, metric): self.metric = metric
    def __call__(self, y, prediction, **kwargs):
        p = np.asarray(prediction)
        if self.metric.n_classes == 2 and p.ndim == 1: p = np.column_stack((1-p, p))
        if not self.metric.n_classes and self.metric.name == 'rmsle': p = np.maximum(p, 0.)
        return self.metric.score(y, p)


def freeze_policy(predictor):
    """Only this pinned-version adapter touches AG's private weight accessor."""
    selected = predictor.model_best
    info = predictor.model_info(selected)
    if info['model_type'] == 'WeightedEnsembleModel':
        weights = predictor._trainer.load_model(selected)._get_model_weights()
    else: weights = {selected: 1.}
    mapping = {'LGBModel': 'GBM', 'CatBoostModel': 'CAT', 'RFModel': 'RF', 'XTModel': 'XT'}
    leaves = []
    for name, weight in sorted(weights.items()):
        if not np.isfinite(weight) or weight <= 0: raise ValueError('Invalid frozen ensemble weight')
        leaf = predictor.model_info(name)
        if leaf['model_type'] not in mapping: raise ValueError('Unsupported challenger topology')
        params = {**leaf['hyperparameters'], **leaf.get('hyperparameters_fit', {})}
        params.pop('ag_args', None)
        leaves.append({'source_model': name, 'family': mapping[leaf['model_type']],
                       'params': params, 'weight': float(weight), 'name': 'AtlasFrozen_' + str(len(leaves))})
    if not 1 <= len(leaves) <= 6 or not np.isclose(sum(x['weight'] for x in leaves), 1.):
        raise ValueError('Frozen challenger portfolio exceeds bound')
    policy = {'version': VERSION, 'components': leaves, 'selection': 'search_split_only'}
    json.dumps(policy, allow_nan=False)  # Fail closed on opaque estimator objects.
    return policy


def fit_predict(data, train_idx, valid_idx, root, seconds, seed, threads, memory_mb,
                policy=None, source_artifact=None, refit=False):
    from autogluon.tabular import TabularPredictor
    from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    from autogluon.core.metrics import make_scorer
    from .features import FeatureBuilder
    if not availability(): raise ValueError('Only approved AutoGluon 1.3.1 is supported')
    root = Path(root)
    spec = data.spec
    if refit:
        if not policy or not source_artifact: raise ValueError('Refit requires original frozen challenger artifacts')
        saved = Path(source_artifact)
        predictor = TabularPredictor.load(str(saved/'automl'), verbosity=0)
        manifest = json.loads((saved/'challenger_policy.json').read_text())
        if manifest != policy: raise ValueError('Refit policy does not match original model')
        coverage = np.load(saved/'split.npz', allow_pickle=False)
        combined = np.concatenate((coverage['train'], coverage['valid']))
        if len(combined) != len(data.y) or set(combined) != set(range(len(data.y))):
            raise ValueError('Original challenger does not cover all labels for full refit')
        fb = FeatureBuilder(spec, [])
        fb.fit_transform(data.train.iloc[coverage['train']])
        test = fb.transform(data.test)
        models = {}
        for leaf in policy['components']:
            mapping = predictor.refit_full(model=leaf['source_model'],num_cpus=min(8,threads),num_gpus=0,fit_strategy='sequential')
            models[leaf['name']] = mapping[leaf['source_model']]
        pv = np.empty((0, len(spec.labels))) if spec.classification else np.empty(0)
    else:
        verify_split(data,train_idx,valid_idx)
        if spec.classification and set(data.y[train_idx]) != set(range(len(spec.labels))):
            raise ValueError('Training fold lacks global classes')
        fb = FeatureBuilder(spec, [])
        train = fb.fit_transform(data.train.iloc[train_idx])
        valid = fb.transform(data.train.iloc[valid_idx]); test = fb.transform(data.test)
        generator = AutoMLPipelineFeatureGenerator(verbosity=0, enable_text_ngram_features=False,
                                                   enable_raw_text_features=False)
        # AG normally fits on concatenated train+tuning. Prefit prevents that leakage.
        generator.fit_transform(train, y=pd.Series(data.y[train_idx],index=train.index))
        label = '__atlas_challenger_target__'
        while label in train: label += '_'
        training = train.copy(); training[label] = data.y[train_idx]
        tuning = valid.copy(); tuning[label] = data.y[valid_idx]
        optimum = 0 if spec.metric.direction < 0 else 1
        scorer = make_scorer('atlas_' + spec.metric.name, CanonicalScore(spec.metric),
                             optimum=optimum, greater_is_better=spec.metric.direction > 0,
                             needs_proba=spec.classification)
        if policy:
            if policy.get('version') != VERSION: raise ValueError('Unknown challenger policy version')
            hyperparameters = {}
            for leaf in policy['components']:
                params = bounded_model_params(leaf['family'],leaf['params'],threads)
                params['ag_args'] = {'name': leaf['name']}
                hyperparameters.setdefault(leaf['family'], []).append(params)
        else:
            hyperparameters = {'GBM': [{'extra_trees': True}, {}, {'num_leaves': 63, 'min_data_in_leaf': 10}],
                               'CAT': {'iterations': 800, 'depth': 6},
                               'RF': {'n_estimators': 200, 'max_leaf_nodes': 20000},
                               'XT': {'n_estimators': 200, 'max_leaf_nodes': 20000}}
        hyperparameters={family:[bounded_model_params(family,p,threads) for p in (values if isinstance(values,list) else [values])] for family,values in hyperparameters.items()}
        predictor = TabularPredictor(label=label, problem_type=spec.problem_type, eval_metric=scorer,
            path=str(root/'automl'), verbosity=0,
            learner_kwargs={'random_state': seed, 'label_count_threshold': 1})
        fit_started = time.monotonic()
        predictor.fit(training, tuning_data=tuning, feature_generator=generator,
            time_limit=max(1., seconds * .65), hyperparameters=hyperparameters,
            num_cpus=min(8, threads), num_gpus=0, memory_limit=max(.5, memory_mb/1024*.75),
            fit_strategy='sequential', num_bag_folds=0, num_stack_levels=0,
            dynamic_stacking=False, calibrate_decision_threshold=False, calibrate=False,
            fit_weighted_ensemble=not bool(policy), refit_full=False)
        if policy:
            models = {leaf['name']: leaf['name'] for leaf in policy['components']}
            if not set(models.values()) <= set(predictor.model_names()):
                raise ValueError('A frozen challenger component failed; partial portfolio forbidden')
        else:
            policy = freeze_policy(predictor)
            models = {leaf['name']: leaf['source_model'] for leaf in policy['components']}
        (root/'challenger_policy.json').write_text(json.dumps(policy, allow_nan=False))
        pv = sum(leaf['weight'] * _predict(predictor, valid, spec, models[leaf['name']]) for leaf in policy['components'])
    pt = sum(leaf['weight'] * _predict(predictor, test, spec, models[leaf['name']]) for leaf in policy['components'])
    return pv, pt, {'challenger_policy': policy, 'selected_models': models,
                    'features': len(spec.features), 'iterations': None,
                    'diagnostics': {}, 'train_sample_metric': None}


def _predict(predictor, frame, spec, model):
    value = predictor.predict_proba(frame, model=model, as_multiclass=True) if spec.classification else predictor.predict(frame, model=model)
    return aligned_prediction(value, spec, len(frame))


def bounded_model_params(family, params, threads):
    """AG RF defaults n_jobs=-1 and otherwise ignores num_cpus; bind explicitly."""
    result=copy.deepcopy(params)
    cpus=max(1,min(8,int(threads)))
    if family in {'RF','XT'}:result['n_jobs']=cpus
    elif family=='GBM':result.update(num_threads=cpus,device_type='cpu')
    elif family=='CAT':result.update(thread_count=cpus,task_type='CPU')
    else:raise ValueError('Unapproved challenger model family')
    return result
