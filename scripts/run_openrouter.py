"""Run the user-authorized free research backend outside the competition ZIP."""
from __future__ import annotations
import argparse,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',nargs='?',default='task.md')
    parser.add_argument('--key-file',type=Path,default=ROOT/'.env.openrouter.local')
    parser.add_argument('--budget',type=float,default=120)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--memory-mb',type=int,default=24576)
    parser.add_argument('--max-experiments',type=int,default=12)
    parser.add_argument('--max-calls',type=int,default=4)
    parser.add_argument('--run-dir')
    args=parser.parse_args()
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]=str(args.threads)
    from scripts.openrouter_client import OpenRouterClient,load_key
    try:client=OpenRouterClient(load_key(args.key_file),max_calls=args.max_calls)
    except ValueError as error:parser.error(str(error))
    from atlas.orchestration import run
    return run(args.task,started=started,budget_seconds=args.budget,mode='live-agent',seed=args.seed,threads=args.threads,memory_mb=args.memory_mb,max_experiments=args.max_experiments,run_dir=args.run_dir,research_client=client)


if __name__=='__main__':raise SystemExit(main())
