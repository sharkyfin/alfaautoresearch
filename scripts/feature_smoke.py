"""Check named feature generation, real worker execution and optional organizer Qwen.

Offline: python scripts/feature_smoke.py
Live:    python scripts/feature_smoke.py --live
No external datasets are needed. API settings are read from the ignored local file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Make at most two requests to organizer Qwen')
    parser.add_argument('--key-file', type=Path, default=ROOT / '.env.organizer.local')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()

    import pandas as pd
    from atlas.agent import QwenClient, researcher_context
    from atlas.contract import read_inputs
    from atlas.experiments import Executor
    from atlas.judge import evaluate
    from atlas.recipes import RecipeError, merge_hypothesis_features, validate
    from atlas.submit import checkpoint
    from atlas.task import load
    from atlas.utils import code_digest
    from atlas.validation import design
    from scripts.synthetic import create

    runs = ROOT / 'runs'
    runs.mkdir(exist_ok=True)
    directory = args.run_dir or Path(tempfile.mkdtemp(prefix='feature-smoke-', dir=runs))
    directory = directory.resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('run-dir must be a new empty directory')
    directory.mkdir(parents=True, exist_ok=True)
    task = create(directory / 'data', 'iid', n=600, seed=42)
    path = task.parent / 'train.csv'
    train = pd.read_csv(path, dtype={'id': 'string'})
    train['response'] = 3 * train.x0 * train.x1
    train.to_csv(path, index=False)
    data = load(read_inputs(task))
    split = design(data)
    data.spec.validation_plan = split.json()
    executor = Executor(data, directory / 'experiments', threads=1)
    baseline = validate({'model_family': 'linear', 'params': {'alpha': .01}, 'max_runtime_sec': 20}, data.spec)
    info, predictions = executor.run(baseline, split, timeout=20)
    if info['status'] != 'ok':
        raise RuntimeError('Baseline worker failed: ' + info['status'])
    scored = {**info, **evaluate(data, split, predictions, info)}
    baseline_metric = scored['metric_value']
    champion = {**scored, 'recipe': baseline}
    client = None
    proposal = {'model_family': 'linear', 'params': {'alpha': .01},
                'feature_expressions': [{'name': 'interaction', 'expression': 'x0 * x1',
                                         'reason': 'Test an interaction missing from a linear model'}],
                'max_runtime_sec': 20}
    if args.live:
        from scripts.run_organizer import load_environment
        os.environ.update(load_environment(args.key_file))
        client = QwenClient('live', max_calls=2)
        proposal = None
        for _ in range(2):
            context = researcher_context(data, [scored], champion, 60)
            context['capabilities']['models'] = {'linear': context['capabilities']['models']['linear']}
            context['objective']['diagnostic'] = 'For this feature-generation smoke test keep the linear model and its alpha=0.01. Propose 1-3 named feature_expressions, including a useful interaction between x0 and x1; do not use legacy feature_ops. This verifies formula execution, not autonomous discovery quality.'
            proposal = client.propose(context, timeout=20)
            if proposal and proposal.get('feature_expressions'):
                try:
                    proposal = validate(merge_hypothesis_features(proposal, baseline), data.spec)
                    break
                except RecipeError as error:
                    client.events.append({'status': 'recipe_rejected', 'reason': str(error)})
            proposal = None
        if proposal is None:
            report = {'status': 'qwen_proposal_failed', 'runtime_code_sha256': code_digest(),
                      'qwen_calls': client.requests, 'qwen_events': client.events, 'run_dir': str(directory)}
            (directory / 'report.json').write_text(json.dumps(report, indent=2))
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
    candidate = validate(merge_hypothesis_features(proposal, baseline), data.spec)
    info, predictions = executor.run(candidate, split, timeout=30)
    if info['status'] != 'ok':
        raise RuntimeError('Feature worker failed: ' + info['status'])
    enhanced_metric = evaluate(data, split, predictions, info)['metric_value']
    checkpoint(data, predictions[1])
    report = {'status': 'passed', 'runtime_code_sha256': code_digest(),
              'mode': 'organizer-qwen' if args.live else 'offline',
              'metric': data.spec.metric.name, 'baseline_metric': baseline_metric,
              'feature_metric': enhanced_metric, 'improved': enhanced_metric < baseline_metric,
              'feature_expressions': candidate['feature_expressions'],
              'qwen_calls': client.requests if client else 0,
              'qwen_events': client.events if client else [],
              'worker_status': info['status'], 'output_valid': True,
              'run_dir': str(directory),
              'limitation': 'Synthetic plumbing check with an explicitly suggested interaction; not a public leaderboard benchmark'}
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
