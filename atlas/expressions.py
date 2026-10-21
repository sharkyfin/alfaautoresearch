"""Named feature formulas with a bounded AST interpreter and fold-fitted statistics.

Formulas are data: they are never passed to eval, exec, pandas.eval or a shell.
Only input feature columns and earlier named formulas can be referenced.
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd


class ExpressionError(ValueError):
    pass


MAX_FEATURES = 16
MAX_NODES = 512
FUNCTIONS = {
    'col': (1, 1), 'abs': (1, 1), 'log1p': (1, 1),
    'signed_log1p': (1, 1), 'sqrt': (1, 1), 'clip': (3, 3),
    'minimum': (2, 2), 'maximum': (2, 2), 'where': (3, 3),
    'fillna': (2, 2), 'isna': (1, 1),
    'year': (1, 1), 'month': (1, 1), 'day': (1, 1),
    'dayofweek': (1, 1), 'hour': (1, 1), 'days_between': (2, 2),
    'length': (1, 1), 'word_count': (1, 1), 'digit_count': (1, 1),
    'lower': (1, 1), 'concat': (2, 3),
    'frequency': (1, 1), 'group_count': (1, 1),
    'group_mean': (2, 2), 'group_median': (2, 2), 'group_std': (2, 2),
}


def expression_references(source):
    """Read formula dependencies without interpreting any expression."""
    references = set()

    class References(ast.NodeVisitor):
        def visit_Name(self, node):
            references.add(node.id)

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == 'col':
                if node.args and isinstance(node.args[0], ast.Constant):
                    references.add(node.args[0].value)
            else:
                for argument in node.args:
                    self.visit(argument)

    References().visit(ast.parse(source, mode='eval'))
    return references


def _validate_node(node, columns, depth=0):
    if depth > 12:
        raise ExpressionError('Formula nesting exceeds 12 levels')
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, (bool, str)):
            if isinstance(value, str) and len(value) > 256:
                raise ExpressionError('Formula string constant is too long')
        elif not isinstance(value, (int, float)) or abs(value) > 1e12 or not math.isfinite(value):
            raise ExpressionError('Invalid formula constant')
        return
    if isinstance(node, ast.Name):
        if node.id not in columns:
            raise ExpressionError('Unknown or forbidden formula column: ' + node.id)
        return
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
            raise ExpressionError('Unknown function, attribute access or keyword arguments')
        low, high = FUNCTIONS[node.func.id]
        if not low <= len(node.args) <= high:
            raise ExpressionError('Invalid argument count for ' + node.func.id)
        if node.func.id == 'col':
            if not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str) or node.args[0].value not in columns:
                raise ExpressionError('col() requires an allowed literal column name')
            return
        children = node.args
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr)):
            raise ExpressionError('Forbidden formula operator')
        if isinstance(node.op, ast.Pow):
            exponent = node.right
            if isinstance(exponent, ast.UnaryOp) and isinstance(exponent.op, (ast.UAdd, ast.USub)):
                exponent = exponent.operand
            if not isinstance(exponent, ast.Constant) or isinstance(exponent.value, bool) or not isinstance(exponent.value, (int, float)) or not -4 <= exponent.value <= 4:
                raise ExpressionError('Power requires a constant exponent between -4 and 4')
        children = [node.left, node.right]
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)):
        children = [node.operand]
    elif isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        children = node.values
    elif isinstance(node, ast.Compare) and all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
        children = [node.left, *node.comparators]
    elif isinstance(node, ast.IfExp):
        children = [node.test, node.body, node.orelse]
    else:
        raise ExpressionError('Forbidden formula syntax: ' + type(node).__name__)
    for child in children:
        _validate_node(child, columns, depth + 1)


def validate_expressions(expressions, columns, forbidden=()):
    if not isinstance(expressions, list) or len(expressions) > MAX_FEATURES:
        raise ExpressionError('At most 16 named feature formulas are allowed')
    available = set(columns) - set(forbidden)
    reserved = set(columns) | set(forbidden) | set(FUNCTIONS)
    result = []
    total_nodes = 0
    total_bytes = 0
    for feature in expressions:
        if not isinstance(feature, dict) or not {'name', 'expression'} <= set(feature) or set(feature) - {'name', 'expression', 'reason'}:
            raise ExpressionError('Formula requires name, expression and optional reason')
        name = feature['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name) or name in reserved:
            raise ExpressionError('Duplicate, reserved or invalid formula name')
        source = feature['expression']
        if not isinstance(source, str) or not 1 <= len(source) <= 768:
            raise ExpressionError('Formula must contain 1 to 768 characters')
        reason = feature.get('reason', '')
        if not isinstance(reason, str) or len(reason) > 240:
            raise ExpressionError('Formula reason must be a short string')
        total_bytes += len(source.encode('utf-8')) + len(reason.encode('utf-8'))
        if total_bytes > 4096:
            raise ExpressionError('Formula payload exceeds 4096 bytes')
        try:
            tree = ast.parse(source, mode='eval')
        except (SyntaxError, ValueError, RecursionError) as error:
            raise ExpressionError('Invalid formula syntax') from error
        nodes = sum(1 for _ in ast.walk(tree))
        total_nodes += nodes
        if nodes > 96 or total_nodes > MAX_NODES:
            raise ExpressionError('Formula computation bound exceeded')
        _validate_node(tree.body, available)
        result.append({'name': name, 'expression': source, **({'reason': reason} if reason else {})})
        available.add(name)
        reserved.add(name)
    return result


def _series(value, index):
    return pd.Series(value, index=index)


def _number(value, index):
    series = _series(value, index)
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype('string').str.strip().str.replace(' ', '', regex=False).str.replace(',', '.', regex=False)
    return pd.to_numeric(series, errors='coerce').astype(float).replace([np.inf, -np.inf], np.nan).clip(-1e30, 1e30)


def _text(value, index):
    return _series(value, index).fillna('__MISSING__').astype(str).str.slice(0, 4096)


def _boolean(value, index):
    return _series(value, index).fillna(False).astype(bool)


class ExpressionFeatures:
    def __init__(self, expressions, columns, forbidden=()):
        self.columns = set(columns)
        self.expressions = validate_expressions(expressions, columns, forbidden)
        self.trees = [(item['name'], ast.parse(item['expression'], mode='eval').body) for item in self.expressions]
        self.learned = {}
        self.fitted = False

    def fit(self, x):
        self.learned = {}
        self._transform(x, fitting=True)
        self.fitted = True
        return self

    def transform(self, x):
        if not self.fitted:
            raise ExpressionError('Feature formulas must be fitted on the training fold')
        return self._transform(x, fitting=False)

    def _transform(self, x, fitting):
        generated = {}
        for name, tree in self.trees:
            try:
                with np.errstate(all='ignore'):
                    value = _series(self._evaluate(tree, x, generated, fitting), x.index)
                if pd.api.types.is_numeric_dtype(value):
                    value = _number(value, x.index)
                else:
                    value = _text(value, x.index)
                generated[name] = value
            except (TypeError, ValueError, KeyError, OverflowError) as error:
                raise ExpressionError('Could not compute formula ' + name + ': ' + type(error).__name__) from error
        return generated

    def _evaluate(self, node, x, generated, fitting):
        index = x.index
        evaluate = lambda child: self._evaluate(child, x, generated, fitting)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return generated[node.id] if node.id in generated else x[node.id]
        if isinstance(node, ast.IfExp):
            return pd.Series(np.where(_boolean(evaluate(node.test), index), evaluate(node.body), evaluate(node.orelse)), index=index)
        if isinstance(node, ast.UnaryOp):
            value = evaluate(node.operand)
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~_boolean(value, index)
            return _number(value, index) * (-1 if isinstance(node.op, ast.USub) else 1)
        if isinstance(node, ast.BoolOp):
            values = [_boolean(evaluate(value), index) for value in node.values]
            result = values[0]
            for other in values[1:]:
                result = result & other if isinstance(node.op, ast.And) else result | other
            return result
        if isinstance(node, ast.Compare):
            left = evaluate(node.left)
            result = pd.Series(True, index=index)
            for operator, child in zip(node.ops, node.comparators):
                right = evaluate(child)
                a, b = _series(left, index), _series(right, index)
                if isinstance(operator, ast.Eq): part = a == b
                elif isinstance(operator, ast.NotEq): part = a != b
                elif isinstance(operator, ast.Lt): part = a < b
                elif isinstance(operator, ast.LtE): part = a <= b
                elif isinstance(operator, ast.Gt): part = a > b
                else: part = a >= b
                result &= part.fillna(False)
                left = right
            return result
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.BitAnd): return _boolean(left, index) & _boolean(right, index)
            if isinstance(node.op, ast.BitOr): return _boolean(left, index) | _boolean(right, index)
            a, b = _number(left, index), _number(right, index)
            if isinstance(node.op, ast.Add): return a + b
            if isinstance(node.op, ast.Sub): return a - b
            if isinstance(node.op, ast.Mult): return a * b
            if isinstance(node.op, ast.Div): return a / b.where(b.abs() > 1e-9)
            if isinstance(node.op, ast.Mod): return a % b.where(b.abs() > 1e-9)
            return a ** b
        function = node.func.id
        if function == 'col':
            column = node.args[0].value
            return generated[column] if column in generated else x[column]
        args = [evaluate(argument) for argument in node.args]
        if function in {'frequency', 'group_count', 'group_mean', 'group_median', 'group_std'}:
            key = ast.dump(node, include_attributes=False)
            groups = _text(args[-1], index)
            if fitting:
                if function in {'frequency', 'group_count'}:
                    mapping = groups.value_counts(normalize=function == 'frequency').to_dict()
                    fallback = 0.
                else:
                    values = _number(args[0], index)
                    grouped = values.groupby(groups, sort=False)
                    stats = grouped.std(ddof=0) if function == 'group_std' else getattr(grouped, function.removeprefix('group_'))()
                    mapping = stats.to_dict()
                    fallback = float(values.median()) if values.notna().any() else 0.
                    if function == 'group_std': fallback = 0.
                self.learned[key] = (mapping, fallback)
            mapping, fallback = self.learned[key]
            return groups.map(mapping).fillna(fallback).astype(float)
        if function in {'year', 'month', 'day', 'dayofweek', 'hour', 'days_between'}:
            a = pd.to_datetime(_series(args[0], index), errors='coerce', utc=True)
            if function == 'days_between':
                b = pd.to_datetime(_series(args[1], index), errors='coerce', utc=True)
                return (a - b).dt.total_seconds() / 86400.
            return getattr(a.dt, function).astype(float)
        if function in {'length', 'word_count', 'digit_count', 'lower', 'concat'}:
            text = _text(args[0], index)
            if function == 'length': return text.str.len().astype(float)
            if function == 'word_count': return text.str.count(r'\S+').astype(float)
            if function == 'digit_count': return text.str.count(r'\d').astype(float)
            if function == 'lower': return text.str.lower()
            for other in args[1:]: text = (text + '::' + _text(other, index)).str.slice(0, 4096)
            return text
        if function == 'isna': return _series(args[0], index).isna()
        if function == 'fillna': return _series(args[0], index).fillna(args[1])
        if function == 'where': return pd.Series(np.where(_boolean(args[0], index), args[1], args[2]), index=index)
        number = _number(args[0], index)
        if function == 'abs': return number.abs()
        if function == 'log1p': return np.log1p(number.where(number > -1))
        if function == 'signed_log1p': return np.sign(number) * np.log1p(number.abs())
        if function == 'sqrt': return np.sqrt(number.where(number >= 0))
        if function == 'clip': return number.clip(lower=_number(args[1], index), upper=_number(args[2], index))
        if function == 'minimum': return np.minimum(number, _number(args[1], index))
        if function == 'maximum': return np.maximum(number, _number(args[1], index))
        raise ExpressionError('Unsupported formula function')
