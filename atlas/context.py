"""Bounded, evidence-ranked researcher inputs; no raw rows leave the process."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .expressions import expression_references
from .task import column_profile


def feature_sources(recipe, columns):
    sources = {column: {column} for column in columns}
    for i, operation in enumerate(recipe.get('feature_ops', [])):
        dependencies = set(operation['columns'])
        key = f'__atlas_op_{i}'
        sources[key] = dependencies
        for column in dependencies:
            for suffix in ('', '_len', '_digits', '_prefix'):
                sources[key + '_' + column + suffix] = dependencies
    for feature in recipe.get('feature_expressions', []):
        dependencies = set()
        for reference in expression_references(feature['expression']):
            dependencies.update(sources.get(reference, set()))
        sources[feature['name']] = dependencies
    return sources


def input_importance(spec, recipe, diagnostics):
    sources = feature_sources(recipe, spec.features)
    importance = {}
    for name, value in diagnostics.get('feature_importance', {}).items():
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            continue
        dependencies = sources.get(name)
        if dependencies is None:
            dependencies = {c for c in spec.features if name.startswith(c + '__')}
        for column in dependencies:
            importance[column] = importance.get(column, 0.) + value / max(1, len(dependencies))
    return importance


def context_profiles(data):
    """Cache a bounded sample of schema tails, including columns after #200."""
    existing = data.spec.profile.get('columns', {})
    cached = data.spec.profile.setdefault('context_columns', {})
    if not data.spec.profile.get('context_columns_scanned'):
        remaining = [c for c in data.spec.features if c not in existing]
        train = data.train.iloc[np.linspace(0, len(data.train)-1, min(512, len(data.train)), dtype=int)]
        test = data.test.iloc[np.linspace(0, len(data.test)-1, min(512, len(data.test)), dtype=int)] if len(data.test) else data.test
        for i in np.linspace(0, len(remaining)-1, min(256, len(remaining)), dtype=int) if remaining else ():
            column = remaining[i]
            cached[column] = {**column_profile(train, test, data.spec, column), 'sampled': True}
        data.spec.profile['context_columns_scanned'] = True
    return {**existing, **cached}


def ranked_columns(data, history=(), champion=None):
    spec = data.spec
    profiles = context_profiles(data)
    columns = [c for c in spec.features if isinstance(c, str) and len(c) <= 160]
    scores = {c: 0. for c in columns}
    signals = {c: {} for c in columns}
    evidence = [r for r in history if r.get('status') == 'ok'][-3:]
    if champion:
        evidence.append(champion)
        sources = feature_sources(champion.get('recipe', {}), columns)
        recipe = champion.get('recipe', {})
        active = set().union(*(sources.get(f['name'], set()) for f in recipe.get('feature_expressions', [])))
        active.update(c for op in recipe.get('feature_ops', []) for c in op['columns'])
        for c in active & scores.keys():
            scores[c] += 4.
            signals[c]['accepted_feature_input'] = True
    for item in evidence:
        detail = item.get('diagnostics', {})
        importance = input_importance(spec, item.get('recipe', {}), detail)
        maximum = max(importance.values(), default=1.)
        for c, value in importance.items():
            if c in scores:
                normalized = value / maximum
                signals[c]['importance'] = max(signals[c].get('importance', 0.), normalized)
        for association in detail.get('error_associations', []):
            c = association.get('column')
            if c in scores:
                signals[c]['error_association'] = max(signals[c].get('error_association', 0.), association['association'])
        for group in detail.get('worst_groups', []):
            if group.get('column') in scores:
                signals[group['column']]['error_group'] = True
    for c in columns:
        info = profiles.get(c, {})
        signal = signals[c]
        scores[c] += 6 * signal.get('importance', 0.) + 5 * signal.get('error_association', 0.) + 2 * signal.get('error_group', False)
        shift = max(min(abs(info.get('median_shift_std', 0.)) / 3, 1.), info.get('unseen_fraction', 0.))
        missing = info.get('nonfinite_or_missing_fraction', info.get('missing', 0.))
        scores[c] += 3 * shift + 2 * missing
        if shift: signal['train_test_shift'] = round(float(shift), 4)
        if missing: signal['missing_fraction'] = round(float(missing), 4)
    informed = sorted((c for c in columns if scores[c] > 0), key=lambda c: -scores[c])
    neutral = [c for c in columns if scores[c] == 0]
    # Explore the entire schema when there is no evidence, rather than its prefix.
    if neutral:
        positions = np.linspace(0, len(neutral) - 1, min(20, len(neutral)), dtype=int)
        exploration = [neutral[i] for i in positions]
        neutral = exploration + [c for c in neutral if c not in set(exploration)]
    return informed + neutral, signals


def string_structure(series):
    """Describe separators and numeric fields without transmitting raw values."""
    text = series.dropna().astype('string').str.slice(0, 4096)
    if len(text) > 512:
        text = text.iloc[np.linspace(0, len(text)-1, 512, dtype=int)]
    if not len(text):
        return []
    structures = []
    for separator in ('/', '_', '|', ':', '-', ' '):
        selected = text[text.str.contains(separator, regex=False)]
        coverage = len(selected) / len(text)
        if coverage < .5:
            continue
        parts = selected.str.split(separator, n=16, regex=False)
        sizes = parts.str.len()
        count = int(sizes.median())
        if not 2 <= count <= 12:
            continue
        numeric_parts = []
        for position in range(count):
            values = parts.str.get(position)
            if pd.to_numeric(values, errors='coerce').notna().mean() >= .9:
                numeric_parts.append(position)
        structures.append({'separator': separator, 'coverage': round(coverage, 3),
                           'median_parts': count, 'numeric_parts': numeric_parts})
    return structures[:3]


def selected_profile(data, columns, signals, limit=20):
    profile = {}
    profiles = context_profiles(data)
    # At most a small sample is profiled for columns outside the initial 200.
    train = data.train.iloc[np.linspace(0, len(data.train) - 1, min(4096, len(data.train)), dtype=int)]
    test = data.test.iloc[np.linspace(0, len(data.test) - 1, min(4096, len(data.test)), dtype=int)] if len(data.test) else data.test
    for column in columns[:limit]:
        info = profiles.get(column)
        if info is None:
            info = {**column_profile(train, test, data.spec, column), 'sampled': True}
        profile[column] = {**info, **signals.get(column, {})}
        if column in data.spec.categorical_features:
            structure = string_structure(train[column])
            if structure:
                profile[column]['string_structure'] = structure
                data.spec.profile.setdefault('structured_columns', {})[column] = structure
    return profile


def error_associations(data, indices, prediction, importance=None, max_columns=80, max_rows=2048):
    """Aggregate error associations on a bounded holdout sample, without values/IDs."""
    if len(indices) < 12:
        return []
    positions = np.linspace(0, len(indices) - 1, min(max_rows, len(indices)), dtype=int)
    sample = data.train.iloc[np.asarray(indices)[positions]]
    y = data.y[np.asarray(indices)[positions]]
    p = np.asarray(prediction)[positions]
    error = 1 - p[np.arange(len(y)), y.astype(int)] if data.spec.classification else np.abs(p - y)
    error = np.asarray(error, dtype=float)
    scale = max(float(np.max(np.abs(error))), 1.)
    error = error / scale
    if not np.isfinite(error).all() or np.std(error) < 1e-12:
        return []
    priority = sorted((importance or {}), key=lambda c: -(importance or {})[c])[:16]
    columns, _ = ranked_columns(data)
    columns = list(dict.fromkeys([*priority, *columns[:24], *[data.spec.features[i] for i in np.linspace(0, len(data.spec.features)-1, min(max_columns, len(data.spec.features)), dtype=int)]]))[:max_columns] if data.spec.features else []
    rows = []

    def correlation(values):
        values = np.asarray(values, dtype=float)
        ok = np.isfinite(values)
        if ok.sum() < 10:
            return 0.
        values = values[ok]
        values = values / max(float(np.max(np.abs(values))), 1.)
        if np.std(values) < 1e-12 or np.std(error[ok]) < 1e-12:
            return 0.
        return float(abs(np.corrcoef(values, error[ok])[0, 1]))

    for column in columns:
        values = sample[column]
        best = correlation(values.isna().to_numpy(dtype=float))
        kind = 'missingness'
        if column in data.spec.numeric_features:
            if not pd.api.types.is_numeric_dtype(values):
                values = values.astype('string').str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
            association = correlation(pd.to_numeric(values, errors='coerce').to_numpy(dtype=float, na_value=np.nan))
            if association > best:
                best, kind = association, 'absolute_error_correlation'
        elif column in data.spec.categorical_features:
            groups = values.fillna('__MISSING__').astype(str).str.slice(0, 4096).to_numpy()
            grouped = pd.DataFrame({'group': groups, 'error': error}).groupby('group', sort=False).error.agg(['mean', 'count'])
            grouped = grouped[grouped['count'] >= 5]
            if len(grouped) > 1:
                association = float(np.sqrt(min(1., np.sum(grouped['count'] * (grouped['mean'] - error.mean())**2) / (len(error) * np.var(error)))))
                if association > best:
                    best, kind = association, 'group_error_variation'
        if np.isfinite(best) and best >= .05:
            rows.append({'column': column, 'association': round(best, 4), 'kind': kind, 'support': len(sample)})
    return sorted(rows, key=lambda row: -row['association'])[:8]
