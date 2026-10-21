"""Controlled feature trials and bounded, dependency-aware ablations."""
from __future__ import annotations

import copy
import json

from .expressions import expression_references
from .recipes import RecipeError, merge_hypothesis_features, recipe_hash, validate


def _op_key(operation):
    return json.dumps(operation, sort_keys=True, ensure_ascii=False)


def feature_changes(parent, candidate):
    changes = []
    before = {f['name']: f for f in parent.get('feature_expressions', [])}
    after = {f['name']: f for f in candidate.get('feature_expressions', [])}
    for name in dict.fromkeys([*after, *before]):
        if before.get(name, {}).get('expression') != after.get(name, {}).get('expression'):
            changes.append({'kind': 'formula', 'name': name,
                            'before': before.get(name), 'after': after.get(name)})
    old_ops = {_op_key(op): op for op in parent.get('feature_ops', [])}
    new_ops = {_op_key(op): op for op in candidate.get('feature_ops', [])}
    for key in dict.fromkeys([*new_ops, *old_ops]):
        if key not in old_ops or key not in new_ops:
            changes.append({'kind': 'operation', 'name': key,
                            'before': old_ops.get(key), 'after': new_ops.get(key)})
    return changes


def controlled_hypothesis(raw, champion, spec, remaining):
    parent = champion['recipe']
    merged = merge_hypothesis_features(raw, parent)
    # Validate requested fields before freezing controls, so forbidden requests
    # cannot be hidden by normalization.
    candidate = validate(merged, spec, remaining=remaining)
    changes = feature_changes(parent, candidate)
    if not changes:
        return candidate, None
    if len(changes) > 4:
        raise RecipeError('A controlled feature hypothesis may change at most four formulas/operations')
    held = ['model_family', 'params', 'target_transform']
    overrides = [key for key in held if candidate.get(key) != parent.get(key)]
    for key in held:
        candidate[key] = copy.deepcopy(parent[key])
    candidate['parent_experiment'] = champion['experiment_id']
    candidate = validate(candidate, spec, remaining=remaining)
    trial = {'kind': 'feature_group', 'parent_experiment': champion['experiment_id'],
             'parent_metric': champion['metric_value'],
             'parent_utility': champion['standardized_utility'],
             'held_constant': [*held, 'validation', 'seed'],
             'changes': [{'kind': c['kind'], 'name': c['name']} for c in changes],
             'ignored_model_overrides': overrides}
    return candidate, trial


def _ordered_formulas(formulas):
    """Stable topological order after restoring a previously removed formula."""
    pending = list(formulas)
    names = {f['name'] for f in formulas}
    ordered = []
    available = set()
    while pending:
        ready = next((f for f in pending if expression_references(f['expression']) & names <= available), None)
        if ready is None:
            raise RecipeError('Cyclic formula dependencies after feature ablation')
        ordered.append(ready)
        available.add(ready['name'])
        pending.remove(ready)
    return ordered


def ablation_options(candidate, parent, attempted=()):
    """Revert one change; removing a formula also removes its dependants."""
    changes = feature_changes(parent, candidate)
    if len(changes) < 2:
        return []
    options = []
    for change in reversed(changes):
        label = change['kind'] + ':' + change['name']
        if label in attempted:
            continue
        raw = copy.deepcopy(candidate)
        affected = []
        if change['kind'] == 'formula':
            name = change['name']
            if change['before'] is None:
                removed = {name}
                for formula in raw['feature_expressions']:
                    if formula['name'] in removed or expression_references(formula['expression']) & removed:
                        removed.add(formula['name'])
                raw['feature_expressions'] = [f for f in raw['feature_expressions'] if f['name'] not in removed]
                affected = sorted(removed)
            else:
                raw['feature_expressions'] = [f for f in raw['feature_expressions'] if f['name'] != name]
                raw['feature_expressions'].append(copy.deepcopy(change['before']))
                try:
                    raw['feature_expressions'] = _ordered_formulas(raw['feature_expressions'])
                except RecipeError:
                    continue  # Individually restoring an old formula may create a cycle.
                affected = [name]
        else:
            if change['after'] is not None:
                raw['feature_ops'] = [op for op in raw['feature_ops'] if _op_key(op) != change['name']]
            else:
                raw['feature_ops'].append(copy.deepcopy(change['before']))
                order = {_op_key(op): i for i, op in enumerate(parent.get('feature_ops', []))}
                raw['feature_ops'].sort(key=lambda op: order.get(_op_key(op), len(order)))
            affected = [change['name']]
        raw['id'] = 'ablation_' + str(len(options) + 1)
        raw['statement'] = 'Controlled feature ablation: revert ' + label
        if recipe_hash(raw) == recipe_hash(parent):
            continue  # The unchanged parent already has a measured score.
        options.append((raw, {'label': label, 'reverted': affected}))
    return options


def accepts_ablation(utility, current, parent_utility):
    tolerance = max(1e-5, abs(current) * .0005)
    parent_tolerance = max(1e-5, abs(parent_utility) * .0005)
    return utility >= current - tolerance and utility > parent_utility + parent_tolerance
