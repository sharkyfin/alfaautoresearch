"""Small predeclared synthetic A/B of the two exact-upload entrypoints.

This measures search duration and operational behavior. Two synthetic tasks are
insufficient evidence of generalization or a competition score. Outer targets
remain in this evaluator's memory and are scored only after all fits finish.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
    os.environ[name] = '8'

DIRECTORY = ROOT / 'reports/thorough-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inputs(root, kind, seed):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    n, m = 200, 80
    x = rng.normal(size=(n + m, 8))
    category = rng.integers(0, 4, n + m)
    latent = (1.7*x[:, 0] + 1.2*x[:, 1]**2 - .6*x[:, 2]*x[:, 3]
              + .6*np.sin(2*x[:, 4]) + .4*category + rng.normal(0, .35, n + m))
    y = latent if kind == 'regression' else np.array(['red', 'blue', 'green'])[
        np.digitize(latent, np.quantile(latent[:n], [.33, .67]))]
    frame = pd.DataFrame(x, columns=['x'+str(i) for i in range(8)])
    frame['category'] = ['category_'+str(value) for value in category]
    frame.insert(0, 'id', ['row_'+str(i) for i in range(n+m)])
    train, test = frame.iloc[:n].copy(), frame.iloc[n:].copy()
    train['target'] = y[:n]
    sample = pd.DataFrame({'id': test.id, 'target': 0. if kind == 'regression' else y[0]})
    dictionary = pd.DataFrame([{'feature': column,
        'role': 'id' if column == 'id' else 'target' if column == 'target' else 'feature',
        'dtype': 'category' if column in {'category', 'id'} or column == 'target' and kind != 'regression' else 'numeric',
        'description_ru': ''} for column in train])
    root.mkdir(parents=True)
    train.to_csv(root/'train.csv', index=False)
    test.to_csv(root/'test.csv', index=False)
    sample.to_csv(root/'sample_submission.csv', index=False)
    dictionary.to_csv(root/'data_dictionary.csv', index=False)
    (root/'task.md').write_text('Task: '+kind+'\nPrimary metric: '+
        ('RMSE' if kind == 'regression' else 'Macro F1')+
        '\nInputs: `train.csv`, `test.csv`, `data_dictionary.csv`, `sample_submission.csv`.\n'
        'Write `submission.csv` in test row order.\n')
    return y[n:].copy()


def main():
    from atlas.budget import run_process
    from sklearn.metrics import f1_score, root_mean_squared_error
    import numpy as np
    import pandas as pd
    archives = {'standard': ROOT/'submission.zip', 'thorough': ROOT/'submission_thorough.zip'}
    manifest = json.loads((DIRECTORY/'manifest.json').read_text())
    assert sha(archives['standard']) == manifest['parent_zip_sha256']
    assert sha(archives['thorough']) == manifest['sha256']
    plan = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'hypothesis': 'More patience yields more completed fits before natural convergence; external quality may worsen.',
        'purpose': 'Operational duration A/B; no claim of generalization advantage',
        'archives': {arm: sha(path) for arm, path in archives.items()},
        'cases': [{'task': 'regression', 'seed': 831105}, {'task': 'multiclass', 'seed': 831106}],
        'train_rows': 200, 'test_rows': 80, 'runtime_budget_seconds': 10500,
        'external_operational_guard_seconds': 1200, 'threads': 8, 'memory_guard_mb': 8192,
        'qwen_calls': 0, 'command': 'python3 solution.py task.md',
        'driver_sha256': sha(Path(__file__)), 'runtime_tuning_after_scoring': False}
    plan_path = DIRECTORY/'synthetic-plan.json'
    if plan_path.exists():
        raise RuntimeError('Do not replace a predeclared experiment')
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    runroot = ROOT/'runs'/('thorough-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
    truths, rows = {}, []
    for case in plan['cases']:
        kind = case['task']
        for arm, archive in archives.items():
            task = runroot/kind/arm
            target = inputs(task, kind, case['seed'])
            if kind in truths:
                assert np.array_equal(target, truths[kind])
            truths[kind] = target
            fingerprints = {name: sha(task/name) for name in ['train.csv', 'test.csv',
                'sample_submission.csv', 'data_dictionary.csv', 'task.md']}
            with zipfile.ZipFile(archive) as payload:
                payload.extractall(task)
            print(json.dumps({'event': 'start', 'task': kind, 'arm': arm}), flush=True)
            observed = run_process([sys.executable, 'solution.py', 'task.md'], task,
                                   timeout=1200, memory_mb=8192, threads=8)
            summaries = list((task/'.atlas').glob('*/summary.json'))
            summary = json.loads(summaries[0].read_text()) if len(summaries) == 1 else {}
            assert fingerprints == {name: sha(task/name) for name in fingerprints}
            record = {'task': kind, 'seed': case['seed'], 'arm': arm,
                'archive_sha256': sha(archive), 'runtime_source_sha256': summary.get('runtime_code_sha256'),
                'run_dir': str(task), 'inputs': fingerprints,
                'operational': {key: observed[key] for key in ['status', 'returncode', 'runtime_sec',
                    'peak_memory_mb', 'cpu_seconds', 'max_live_threads']},
                'terminal_status': summary.get('status'), 'cancelled': summary.get('cancelled'),
                'full_refit': summary.get('full_refit_done'), 'fit_success': summary.get('fit_success'),
                'experiments': summary.get('experiments'), 'budget_seconds': summary.get('budget_seconds'),
                'control_folds_completed': len(summary.get('confirmation_metrics', [])),
                'qwen_calls': summary.get('qwen_calls'), 'search': summary.get('adaptive_search'),
                'output_sha256': sha(task/'submission.csv') if (task/'submission.csv').exists() else None}
            rows.append(record)
            print(json.dumps({'event': 'finished', **record}), flush=True)
    # No external answers or scores were used to alter either policy.
    for row in rows:
        root = Path(row['run_dir'])
        output = pd.read_csv(root/'submission.csv')
        sample = pd.read_csv(root/'sample_submission.csv')
        valid = (list(output) == list(sample) and output.id.equals(sample.id)
                 and output.notna().all().all() and len(output) == 80)
        row['valid_output'] = bool(valid)
        if row['task'] == 'regression':
            row['metric'] = 'rmse'
            row['outer_score'] = root_mean_squared_error(truths[row['task']], output.target) if valid else None
        else:
            row['metric'] = 'f1_macro'
            row['outer_score'] = f1_score(truths[row['task']], output.target, average='macro') if valid else None
    result = {'status': 'completed', 'plan_sha256': sha(plan_path), 'runs': rows,
        'finished_at_utc': datetime.now(timezone.utc).isoformat(),
        'limitations': ['Only two synthetic tasks and one data seed per task',
            'Not the hackathon scorer, independent real-data lockbox or continuous three-hour soak',
            'No organizer API or full organizer image tested',
            'Longer search can overfit selection folds; do not promote from this small panel alone']}
    (DIRECTORY/'synthetic-comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'event': 'complete', 'report': str(DIRECTORY/'synthetic-comparison.json')}), flush=True)


if __name__ == '__main__':
    main()
