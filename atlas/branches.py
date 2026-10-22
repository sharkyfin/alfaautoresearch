"""Data-dependent, bounded alternatives to an exhausted representation."""
from __future__ import annotations

import copy
import hashlib
import json

from .recipes import merge_hypothesis_features


def feature_name(kind, *columns):
    digest = hashlib.sha256(json.dumps(columns, ensure_ascii=False).encode()).hexdigest()[:10]
    return 'research_' + kind + '_' + digest


def preprocessing_candidates(spec, parent):
    """Propose small groups; inputs, models and validation remain unchanged."""
    source = parent['recipe']
    profiles = spec.profile.get('columns', {})
    importance = parent.get('diagnostics', {}).get('feature_importance', {})
    def priority(column):
        return (-importance.get(column, 0), -profiles.get(column, {}).get('missing', 0))
    numeric = sorted(spec.numeric_features, key=priority)
    categorical = sorted(spec.categorical_features, key=priority)
    groups = []
    missing = sorted(spec.features, key=lambda c: -profiles.get(c, {}).get('missing', 0))
    missing = [c for c in missing if profiles.get(c, {}).get('missing', 0) > .01][:4]
    if missing:
        groups.append(('missing_indicators', [], [
            {'name': feature_name('missing', c), 'expression': 'isna(col(' + json.dumps(c) + '))'}
            for c in missing]))
    if numeric:
        for operation in ['log1p', 'sqrt', 'clip']:
            operations = ([{'op': operation, 'columns': numeric[:4]}] if operation == 'clip'
                          else [{'op': operation, 'columns': [c]} for c in numeric[:4]])
            groups.append((operation, operations, []))
    if categorical:
        groups.append(('category_frequency', [{'op': 'frequency', 'columns': categorical[:4]}], []))
        sparse = [c for c in categorical if profiles.get(c, {}).get('unseen_fraction', 0) >= .5]
        if sparse:
            groups.append(('drop_unseen_categories', [{'op': 'drop', 'columns': sparse[:3]}], []))
    if len(categorical) >= 2:
        groups.append(('category_interaction', [{'op': 'combine', 'columns': categorical[:2]}], []))
    if len(numeric) >= 2:
        for operation in ['ratio', 'difference', 'product']:
            groups.append((operation, [{'op': operation, 'columns': numeric[:2]}], []))
    if categorical and numeric:
        groups.append(('group_scale', [{'op': 'group_mean', 'columns': [categorical[0], numeric[0]]}], []))
    for column, structures in spec.profile.get('structured_columns', {}).items():
        if column not in spec.features:
            continue
        for structure in structures[:1]:
            separator = structure['separator']
            expressions = []
            for position in range(min(4, structure['median_parts'])):
                expression = 'split_part(col(' + json.dumps(column) + '), ' + json.dumps(separator) + ', ' + str(position) + ')'
                if position in structure['numeric_parts']:
                    expression = 'to_number(' + expression + ')'
                expressions.append({'name': feature_name('part_' + str(position), column, separator),
                                    'expression': expression})
            groups.append(('structured_parts', [], expressions))
    for kind, operations, expressions in groups:
        raw = copy.deepcopy(source)
        raw.update(feature_ops=operations, feature_expressions=expressions,
                   parent_experiment=parent['experiment_id'], id='branch_' + kind,
                   statement='New research branch: ' + kind)
        yield kind, merge_hypothesis_features(raw, source)
    if not spec.classification:
        quantiles = spec.profile.get('target', {}).get('quantiles', {})
        if quantiles and min(quantiles.values()) >= 0:
            raw = copy.deepcopy(source)
            raw.update(target_transform='none' if source.get('target_transform') == 'log1p' else 'log1p',
                       parent_experiment=parent['experiment_id'], id='branch_target_transform',
                       statement='New research branch: alternate target scale scored on original units')
            yield 'target_scale', raw


def transfer_candidates(donors, recipients):
    """Transfer complete measured features while holding recipient parameters fixed."""
    for donor in donors:
        features = donor['recipe']
        if not (features.get('feature_ops') or features.get('feature_expressions')):
            continue
        for recipient in recipients:
            if features['model_family'] == recipient['recipe']['model_family']:
                continue
            raw = copy.deepcopy(recipient['recipe'])
            for key in ['feature_ops', 'feature_expressions']:
                raw[key] = copy.deepcopy(features.get(key, []))
            raw.update(parent_experiment=recipient['experiment_id'], id='branch_feature_transfer',
                       statement='Transfer independently measured feature representation to another model family')
            yield raw, {'donor_experiment': donor['experiment_id'],
                        'recipient_experiment': recipient['experiment_id'],
                        'held_constant': ['model_family', 'params', 'target_transform', 'validation', 'seed']}
