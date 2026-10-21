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
    parser.add_argument('--mode', choices=['lightgbm', 'deterministic', 'live-agent'], required=True)
    parser.add_argument('--budget', type=float, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-mb', type=int, default=8192)
    parser.add_argument('--max-calls', type=int, default=2)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        os.environ[name] = str(args.threads)
    from atlas.agent import QwenClient
    from atlas.orchestration import run
    from atlas import search
    experiments = 30
    if args.mode == 'lightgbm':
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
    live = args.mode == 'live-agent'
    client = QwenClient('live' if live else 'disabled', max_calls=args.max_calls)
    if live and not client.configured:
        raise ValueError('Live benchmark requires organizer API environment variables')
    return run(args.task, started=started, budget_seconds=args.budget, mode='live-agent' if live else 'deterministic',
               seed=args.seed, threads=args.threads, memory_mb=args.memory_mb,
               max_experiments=experiments, run_dir=args.run_dir, research_client=client)


if __name__ == '__main__':
    raise SystemExit(main())
