"""One bounded benchmark run of the uploaded runtime, without benchmark labels."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task')
    parser.add_argument('--mode', choices=['lightgbm', 'deterministic', 'live-agent','fixed-allocation','autogluon','flaml','catboost'], required=True)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-mb', type=int, default=8192)
    parser.add_argument('--max-calls', type=int, default=2)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--exclude-families', default='')
    parser.add_argument('--fixed-recipe',type=Path)
    parser.add_argument('--ensemble-strategy',choices=['none','weighted','greedy'])
    parser.add_argument('--finalization',choices=['refit','fold_average'])
    args = parser.parse_args()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = '1'
    if args.mode in {'autogluon','flaml','catboost'}:
        from scripts.backend_worker import run_backend
        return run_backend(args,started)
    from atlas.agent import QwenClient
    from atlas.orchestration import run
    from atlas import search
    if args.exclude_families:
        from atlas import task as task_module
        excluded=set(args.exclude_families.split(','))
        original_capabilities=task_module.capabilities
        def restricted_capabilities(strict=False):
            return {key:value for key,value in original_capabilities(strict).items() if key not in excluded}
        task_module.capabilities=restricted_capabilities
    experiments = 480
    if args.fixed_recipe:
        import json
        fixed=json.loads(args.fixed_recipe.read_text())
        if not isinstance(fixed,dict):raise ValueError('One typed recipe object is required')
        def fixed_portfolio(spec,budget):
            import copy
            return [copy.deepcopy(fixed)]
        search.initial_portfolio=fixed_portfolio
        experiments=1
    elif args.mode == 'lightgbm':
        def fixed_portfolio(spec, budget):
            if 'lightgbm' not in spec.available_capabilities:
                raise ValueError('The reference requires the approved LightGBM package')
            params = {'iterations': 500 if budget < 180 else 1800, 'num_leaves': 31,
                      'min_child_samples': 20, 'learning_rate': .05}
            if spec.classification:
                params.update(weight='sqrt', stopping_metric='primary')
            return [search.recipe('lightgbm', params, statement='Fixed local LightGBM reference; no AutoResearch')]
        search.initial_portfolio = fixed_portfolio
        experiments = 1
    elif args.mode=='fixed-allocation':
        original_portfolio=search.initial_portfolio
        def allocated_portfolio(spec,budget):
            import copy
            candidates=original_portfolio(spec,budget)
            extra=[]
            for candidate in candidates:
                other=copy.deepcopy(candidate);family=other['model_family'];params=other['params']
                if family=='lightgbm':params.update(num_leaves=63,min_child_samples=15)
                elif family=='catboost':params.update(depth=8,l2_leaf_reg=10.)
                elif family=='hist':params.update(max_leaf_nodes=63,l2_regularization=3.)
                elif family=='xgboost':params.update(max_depth=4,min_child_weight=5.)
                elif family=='extra_trees':params.update(min_samples_leaf=1,max_features=1.)
                else:params.update({'C':10.} if spec.classification else {'alpha':1.})
                extra.append(other)
            return candidates+extra
        search.initial_portfolio=allocated_portfolio
        # A fixed diverse allocation is a reference, not an oracle with labels.
        from atlas.contract import read_inputs
        from atlas.task import load
        experiments=len(allocated_portfolio(load(read_inputs(args.task)).spec,args.budget))
    live = args.mode == 'live-agent'
    client = QwenClient('live' if live else 'disabled', max_calls=args.max_calls)
    if live and not client.configured:
        raise ValueError('Live benchmark requires organizer API environment variables')
    # Execute the archived source entrypoint. Only this dev shim supplies a
    # shorter budget/seed/mode; the platform's ordinary two-argument CLI is intact.
    import runpy
    from atlas import orchestration
    original_run=orchestration.run
    benchmark_started=started
    def bounded_run(task_path,started=None,**kwargs):
        policies={key:value for key,value in {'ensemble_strategy':args.ensemble_strategy,'finalization':args.finalization}.items() if value is not None}
        return original_run(task_path,started=benchmark_started,budget_seconds=args.budget,
            mode='live-agent' if live else 'deterministic',seed=args.seed,threads=args.threads,
            memory_mb=args.memory_mb,max_experiments=experiments,run_dir=args.run_dir,research_client=client,**policies)
    orchestration.run=bounded_run
    sys.argv=[str(ROOT/'solution.py'),args.task]
    runpy.run_path(str(ROOT/'solution.py'),run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
