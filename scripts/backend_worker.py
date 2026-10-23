"""Dev-only classical AutoML references; no foundation models or downloads.

Run through the benchmark's bounded child process. External answers never enter
this module. These backends are references, not automatically enabled runtime.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from atlas.contract import read_inputs
from atlas.features import FeatureBuilder, MatrixBuilder
from atlas.submit import checkpoint, dummy_prediction
from atlas.task import load
from atlas.utils import code_digest
from atlas.validation import selection_splits


def aligned(model, matrix, classes):
    if not classes:
        return np.asarray(model.predict(matrix)).reshape(-1)
    raw = np.asarray(model.predict_proba(matrix))
    labels = np.asarray(getattr(model, 'classes_', np.arange(raw.shape[1])), dtype=int)
    out = np.zeros((len(matrix), classes))
    out[:, labels] = raw
    out = np.maximum(out, 1e-12)
    return out / out.sum(axis=1, keepdims=True)


def run_backend(args, started):
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=False)
    data = load(read_inputs(args.task), strict=True)
    checkpoint(data, dummy_prediction(data.y, data.spec, len(data.test)))
    split, _ = selection_splits(data, args.seed)
    if not len(split.valid):
        raise ValueError('Reference needs a feasible honest internal holdout')
    features = FeatureBuilder(data.spec).fit(data.train.iloc[split.train])
    train = features.transform(data.train.iloc[split.train])
    valid = features.transform(data.train.iloc[split.valid])
    test = features.transform(data.test)
    metric = data.spec.metric
    cl = data.spec.classification
    classes = len(data.spec.labels)
    y_train, y_valid = data.y[split.train], data.y[split.valid]
    remaining = lambda: max(0., args.budget - (time.monotonic() - started))
    package = 'autogluon.tabular' if args.mode == 'autogluon' else 'flaml'
    summary = {'runtime_code_sha256': code_digest(), 'backend': args.mode,
        'backend_version': importlib.metadata.version(package),
        'backend_wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'mode': args.mode, 'budget_seconds': args.budget, 'validation': split.json(),
        'qwen_calls': 0, 'qwen_tokens': 0, 'experiments': 0, 'fit_success': 0,
        'full_refit_published': False, 'checkpoint_kind': 'valid_dummy',
        'featurizer_fit_rows': len(split.train), 'full_refit_feature_policy': 'Fixed fold-train preprocessing; model refit uses all supplied labels',
        'champion_recipe': {'model_family': args.mode + '_classical_reference'},
        'prediction_provenance': 'Fresh training on this task only; no external models/weights'}
    def save():
        summary.update(elapsed_seconds=time.monotonic() - started,
            output_sha256=hashlib.sha256(data.inputs.output.read_bytes()).hexdigest())
        (root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    save()
    try:
        if args.mode == 'autogluon':
            from autogluon.tabular import TabularPredictor
            from autogluon.features.generators import AutoMLPipelineFeatureGenerator
            generator = AutoMLPipelineFeatureGenerator(verbosity=0)
            # AG normally fits its generator on concatenated train/validation.
            # Its documented prefit path transforms without refitting instead.
            generator.fit_transform(train, pd.Series(y_train, index=train.index))
            label = '__reference_label__'
            while label in train:
                label = '_' + label
            training = train.copy(); training[label] = y_train
            tuning = valid.copy(); tuning[label] = y_valid
            ag_metric = {'rmse': 'root_mean_squared_error', 'mse': 'mean_squared_error',
                'mae': 'mean_absolute_error', 'logloss': 'log_loss'}.get(metric.name, metric.name)
            hyperparameters = {'GBM': {'seed': args.seed}, 'CAT': {'random_seed': args.seed},
                'XGB': {'seed': args.seed}, 'RF': {'n_estimators': 250, 'random_state': args.seed},
                'XT': {'n_estimators': 250, 'random_state': args.seed}, 'LR': {'random_state': args.seed}}
            model = TabularPredictor(label=label, problem_type='multiclass' if classes > 2 else 'binary' if cl else 'regression',
                eval_metric=ag_metric, path=str(root / 'models'), verbosity=0,
                learner_kwargs={'random_state': args.seed,'label_count_threshold':1})
            model.fit(training, tuning_data=tuning, feature_generator=generator,
                time_limit=max(3., remaining() * .65), hyperparameters=hyperparameters,
                num_cpus=args.threads, num_gpus=0, fit_strategy='sequential',
                num_bag_folds=0, num_stack_levels=0, dynamic_stacking=False,
                calibrate_decision_threshold=False)
            def predict(frame):
                if not cl:return np.asarray(model.predict(frame))
                probabilities = model.predict_proba(frame, as_multiclass=True)
                out = probabilities.reindex(columns=range(classes), fill_value=0.).to_numpy()
                out = np.maximum(out, 1e-12)
                return out / out.sum(axis=1, keepdims=True)
            summary.update(allowed_algorithms=list(hyperparameters), experiments=len(model.model_names()),
                fit_success=len(model.model_names()), champion_metric=metric.score(y_valid, predict(valid)))
            checkpoint(data, predict(test)); summary.update(checkpoint_kind='trained_checkpoint', status='trained_checkpoint'); save()
            if remaining() > 8:
                try:
                    model.refit_full(model='best', set_best_to_refit_full=True)
                    checkpoint(data, predict(test)); summary.update(full_refit_published=True,
                        checkpoint_kind='full_refit', status='full_refit', model_refit_rows=len(data.y))
                except Exception as error:
                    summary['refit_error'] = type(error).__name__ + ': ' + str(error)[:300]
            summary['best_model'] = model.model_best
        else:
            from flaml import AutoML
            matrix = MatrixBuilder('lightgbm', features.cat, features.num, data.spec.text_candidates).fit(train)
            xa, xv, xt = matrix.transform(train), matrix.transform(valid), matrix.transform(test)
            def objective(X_val, actual, estimator, labels, X_train, y_train,
                          weight_val=None, weight_train=None, config=None, groups_val=None, groups_train=None):
                score = metric.score(actual, aligned(estimator, X_val, classes))
                return (1 - score if metric.direction > 0 else score), {'primary_metric': score}
            model = AutoML()
            allowed = ['lgbm', 'xgboost', 'catboost', 'extra_tree', 'rf', 'lrl1'] if cl else ['lgbm', 'xgboost', 'catboost', 'extra_tree', 'rf']
            model.fit(X_train=xa, y_train=y_train, X_val=xv, y_val=y_valid,
                task='classification' if cl else 'regression', metric=objective,
                time_budget=max(3., remaining() * .65), estimator_list=allowed,
                eval_method='holdout', n_jobs=args.threads, seed=args.seed,
                retrain_full=False, verbose=0, log_file_name=str(root / 'flaml.log'))
            summary.update(allowed_algorithms=allowed, best_model=model.best_estimator,
                best_config=model.best_config, experiments=model._track_iter + 1,
                fit_success=1, champion_metric=metric.score(y_valid, aligned(model, xv, classes)))
            checkpoint(data, aligned(model, xt, classes)); summary.update(checkpoint_kind='trained_checkpoint', status='trained_checkpoint'); save()
            if remaining() > 8:
                try:
                    full = copy.deepcopy(model.model)
                    combined = pd.concat([xa, xv])
                    full.fit(combined, np.r_[y_train, y_valid], budget=remaining() * .9)
                    checkpoint(data, aligned(full, xt, classes)); summary.update(full_refit_published=True,
                        checkpoint_kind='full_refit', status='full_refit', model_refit_rows=len(data.y))
                except Exception as error:
                    summary['refit_error'] = type(error).__name__ + ': ' + str(error)[:300]
        summary['trained_output'] = True
        save()
        return 0
    except Exception as error:
        summary.update(status='fatal_with_fallback', error=type(error).__name__ + ': ' + str(error)[:500])
        save()
        (root / 'error.txt').write_text(summary['error'])
        raise
